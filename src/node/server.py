"""Outbound-only Groove compute node.

The node opens ONE persistent websocket to the signal service (or legacy
relay), registers itself, then processes inbound ENVELOPEs and replies with
ENVELOPEs back. The node never accepts inbound connections — solving
NAT/CGNAT/firewall topology constraints.

M2: nodes register with capabilities only; the scheduler assigns a layer
range via ASSIGN_LAYERS after inspecting the global fleet. The node then
loads its shard on demand and replies with ASSIGNMENT_ACK. A later REBALANCE
swaps the shard in place. A legacy mode (--layers on the CLI) pre-loads the
shard at startup like M1.

M3: --signal flag connects to the Groove signal service (e.g.
signal.groovedev.ai) over TLS. The signal service handles node registration,
scoring, consumer matching, and envelope routing.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import logging
import os
import signal
import sys
import time

import msgpack
import torch
import websockets
from websockets.asyncio.client import connect as ws_connect

from src.common.protocol import (
    ACTIVATIONS,
    ASSIGN_LAYERS,
    LOGITS,
    ASSIGNMENT_ACK,
    AUTH_CHALLENGE,
    ENVELOPE,
    HEARTBEAT,
    ERROR,
    ICE_CANDIDATE,
    MESH_CONNECT,
    P2P_READY,
    PROTOCOL_VERSION,
    REBALANCE,
    REGISTER_ACK,
    SDP_ANSWER,
    SDP_OFFER,
    SESSION_INIT,
    SPEC_WINDOW,
    TOKEN_RESULT,
    decode_message,
    encode_message,
    make_activations,
    make_assignment_ack,
    make_auth_response,
    make_deregister,
    make_envelope,
    make_error,
    make_heartbeat,
    make_logits,
    make_mesh_broken,
    make_mesh_ready,
    make_p2p_ready,
    make_register_node,
    make_sdp_answer,
    make_token_result,
    make_verify_result,
)
from src.common.benchmark import classify_node_role, run_node_benchmark
from src.common.chunking import ChunkedChannel
from src.common.tensor_transfer import deserialize_tensor, serialize_tensor
from src.node.identity import derive_node_id, load_or_create_identity, sign_message
from src.node.kv_cache import KVCacheManager
from src.node.shard_loader import forward_shard, get_model_info, load_model_shard

logger = logging.getLogger("node")

_NODE_START_TIME = time.time()


def _peek_msg_type(data: bytes) -> str | None:
    """Extract the 'type' field from msgpack bytes without full deserialization."""
    try:
        unpacker = msgpack.Unpacker(raw=False, max_buffer_size=256)
        unpacker.feed(data[:256])
        obj = unpacker.unpack()
        if isinstance(obj, dict):
            return obj.get("type")
    except Exception:
        pass
    return None


def _capabilities(
    model_name: str | None,
    device: str,
    max_context: int,
    loaded_layers: tuple[int, int] | None = None,
    model_preferences: list[str] | None = None,
    bench_data: dict | None = None,
) -> dict:
    try:
        import psutil

        ram_mb = int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        ram_mb = 0

    vram_mb = 0
    gpu_model = ""
    if torch.cuda.is_available():
        try:
            vram_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
            gpu_model = torch.cuda.get_device_name(0)
        except Exception:
            pass
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        gpu_model = "Apple Silicon"
        vram_mb = int(ram_mb * 0.75)

    try:
        ram_used_mb = int(psutil.virtual_memory().used / (1024 * 1024))
        ram_pct = round(psutil.virtual_memory().percent, 1)
    except Exception:
        ram_used_mb = 0
        ram_pct = 0.0

    try:
        load = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        load = 0.0

    gpu_utilization_pct = 0
    vram_used_mb = 0
    if torch.cuda.is_available():
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                gpu_utilization_pct = int(parts[0])
                vram_used_mb = int(parts[1])
        except Exception:
            pass
    elif device == "mps":
        try:
            vram_used_mb = int(torch.mps.current_allocated_memory() / (1024 * 1024))
            driver_mb = int(torch.mps.driver_allocated_memory() / (1024 * 1024))
            if vram_mb > 0:
                gpu_utilization_pct = min(100, int(vram_used_mb * 100 / vram_mb))
        except Exception:
            pass

    caps = {
        "ram_mb": ram_mb,
        "ram_used_mb": ram_used_mb,
        "ram_pct": ram_pct,
        "vram_mb": vram_mb,
        "vram_used_mb": vram_used_mb,
        "gpu_utilization_pct": gpu_utilization_pct,
        "load": load,
        "device": device,
        "cpu_cores": os.cpu_count() or 0,
        "gpu_model": gpu_model,
        "max_context_length": max_context,
        "bandwidth_mbps": 0.0,
        "uptime_seconds": int(time.time() - _NODE_START_TIME),
        "models_loaded": [model_name] if (model_name and loaded_layers) else [],
        "model_preferences": model_preferences or ([model_name] if model_name else []),
        "protocol_version": PROTOCOL_VERSION,
    }
    if loaded_layers is not None:
        caps["layer_start"] = loaded_layers[0]
        caps["layer_end"] = loaded_layers[1]
    if bench_data:
        caps["bench_ms_per_layer"] = bench_data.get("bench_ms_per_layer", 0.0)
        caps["bench_mem_bandwidth_gbps"] = bench_data.get("bench_mem_bandwidth_gbps", 0.0)
        caps["node_role"] = bench_data.get("node_role", "writer")
    return caps


class ComputeNodeServer:
    """Outbound-only compute node serving a single model shard.

    In dynamic mode, layer_start/layer_end start as None and are assigned
    by the relay via ASSIGN_LAYERS. In legacy mode (--layers on the CLI),
    the shard is pre-loaded at startup and the ASSIGN_LAYERS path is
    skipped.
    """

    def __init__(
        self,
        model_name: str | None,
        layer_start: int | None = None,
        layer_end: int | None = None,
        device: str = "cpu",
        max_context: int = 4096,
        quantize: bool = False,
        node_id: str | None = None,
        legacy_mode: bool = False,
        identity: dict | None = None,
        model_preferences: list[str] | None = None,
    ):
        self.model_name = model_name
        self.layer_start = layer_start
        self.layer_end = layer_end
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device == "cpu":
            import shutil
            if shutil.which("nvidia-smi"):
                logger.warning(
                    "Running on CPU despite NVIDIA GPU present. "
                    "Reinstall PyTorch with CUDA: pip install torch "
                    "--index-url https://download.pytorch.org/whl/cu124 --force-reinstall"
                )
        self.device = device
        self.max_context = max_context
        self.quantize = quantize
        self.legacy_mode = legacy_mode
        self.identity = identity
        self.model_preferences = model_preferences or ([model_name] if model_name else [])

        self.shard: dict | None = None
        self.kv_manager: KVCacheManager | None = None
        self.bench_data: dict | None = None

        if node_id is not None:
            self.node_id = node_id
        elif identity is not None:
            self.node_id = derive_node_id(identity["address"])
        else:
            self.node_id = f"node-L{layer_start}-{layer_end}-{os.getpid()}"

        self._stop = asyncio.Event()
        self._ws = None
        self._assignment_lock = asyncio.Lock()
        self._inference_executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
            if self.device == "mps" else None
        )

        self.peer_manager = None
        self.upstream_peer: str | None = None
        self.downstream_peer: str | None = None
        self.p2p_channels: dict[str, ChunkedChannel] = {}
        self._p2p_recv_tasks: dict[str, asyncio.Task] = {}
        self._session_sampling: dict[str, dict] = {}

        self._gpu_model_cached: str = ""
        if torch.cuda.is_available():
            try:
                self._gpu_model_cached = torch.cuda.get_device_name(0)
            except Exception:
                pass
        elif device == "mps":
            self._gpu_model_cached = "Apple Silicon"

    def _node_telemetry(self) -> dict:
        """Lightweight resource snapshot attached to every inference response."""
        telem: dict = {
            "node_id": self.node_id[:12],
            "device": self.device,
            "gpu_model": self._gpu_model_cached,
        }
        if self.layer_start is not None and self.layer_end is not None:
            telem["layers"] = [self.layer_start, self.layer_end]

        if torch.cuda.is_available():
            try:
                telem["vram_used_mb"] = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
                telem["vram_peak_mb"] = int(torch.cuda.max_memory_allocated(0) / (1024 * 1024))
                telem["vram_total_mb"] = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
            except Exception:
                pass
        elif self.device == "mps":
            try:
                telem["vram_used_mb"] = int(torch.mps.current_allocated_memory() / (1024 * 1024))
                telem["vram_peak_mb"] = int(torch.mps.driver_allocated_memory() / (1024 * 1024))
            except Exception:
                pass

        try:
            import psutil
            mem = psutil.virtual_memory()
            telem["ram_used_mb"] = int(mem.used / (1024 * 1024))
            telem["ram_total_mb"] = int(mem.total / (1024 * 1024))
            telem["ram_pct"] = round(mem.percent, 1)
            telem["cpu_pct"] = round(psutil.cpu_percent(interval=0), 1)
        except Exception:
            pass

        return telem

    async def load(self) -> None:
        if self.model_name is None or self.layer_start is None or self.layer_end is None:
            raise RuntimeError(
                "load() requires model_name, layer_start, layer_end — "
                "use dynamic assignment instead"
            )
        logger.info(
            "loading shard model=%s layers=[%d,%d) device=%s",
            self.model_name, self.layer_start, self.layer_end, self.device,
        )
        loop = asyncio.get_event_loop()
        self.shard = await loop.run_in_executor(
            None,
            lambda: load_model_shard(
                self.model_name,
                self.layer_start,
                self.layer_end,
                device=self.device,
                quantize=self.quantize,
            ),
        )
        self._init_kv_manager()
        logger.info("shard loaded node_id=%s", self.node_id)

    def _init_kv_manager(self) -> None:
        quantize_kv = self.device == "mps"
        config = self.shard["config"] if self.shard else None
        self.kv_manager = KVCacheManager(config=config, quantize=quantize_kv)
        logger.info("KV cache: %s", "INT4-quantized" if quantize_kv else "FP16")

    def request_stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            asyncio.ensure_future(self._ws.close())

    _BACKOFF_SCHEDULE = (1, 2, 5, 10)

    async def run(self, relay_url: str, use_tls: bool = False) -> None:
        """Main loop: connect to relay with backoff, register, serve until stop."""
        if self.bench_data is None:
            from src.relay.scheduler import MODEL_REGISTRY
            hidden_size = 2560
            if self.model_name and self.model_name in MODEL_REGISTRY:
                hidden_size = MODEL_REGISTRY[self.model_name].get("hidden_size", 2560)
            loop = asyncio.get_event_loop()
            self.bench_data = await loop.run_in_executor(
                None, lambda: run_node_benchmark(hidden_size=hidden_size, device=self.device)
            )
            self.bench_data["node_role"] = classify_node_role(
                self.bench_data.get("bench_ms_per_layer", 0.0),
                self.bench_data.get("bench_mem_bandwidth_gbps", 0.0),
                device=self.device,
            )

        if self.legacy_mode:
            await self.load()

        if relay_url.startswith("wss://") or relay_url.startswith("ws://"):
            uri = relay_url
        else:
            scheme = "wss" if use_tls else "ws"
            uri = f"{scheme}://{relay_url}"
        attempt = 0
        while not self._stop.is_set():
            try:
                async with ws_connect(
                    uri,
                    max_size=10 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    await self._register(ws)
                    attempt = 0
                    hb_task = asyncio.create_task(self._heartbeat_loop(ws))
                    try:
                        await self._receive_loop(ws)
                    finally:
                        self._ws = None
                        hb_task.cancel()
                        try:
                            await hb_task
                        except asyncio.CancelledError:
                            pass
                        try:
                            await asyncio.wait_for(
                                ws.send(encode_message(make_deregister(self.node_id, "shutdown"))),
                                timeout=2.0,
                            )
                        except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError):
                            pass
            except (websockets.ConnectionClosed, OSError, ConnectionError) as e:
                logger.warning("relay connection lost: %s", e)
            except Exception:
                logger.exception("unexpected error in node loop")

            if self._stop.is_set():
                break

            delay = self._BACKOFF_SCHEDULE[min(attempt, len(self._BACKOFF_SCHEDULE) - 1)]
            attempt += 1
            logger.info("reconnecting in %ds", delay)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                pass

    async def _register(self, ws) -> None:
        loaded = (
            (self.layer_start, self.layer_end)
            if self.legacy_mode and self.layer_start is not None
            else None
        )
        public_key = self.identity["public_key"] if self.identity else None
        reg = make_register_node(
            self.node_id,
            layer_start=self.layer_start if self.legacy_mode else None,
            layer_end=self.layer_end if self.legacy_mode else None,
            capabilities=_capabilities(
                self.model_name,
                self.device,
                self.max_context,
                loaded_layers=loaded,
                model_preferences=self.model_preferences,
                bench_data=self.bench_data,
            ),
            public_key=public_key,
        )
        await ws.send(encode_message(reg))

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        except asyncio.TimeoutError as e:
            raise ConnectionError("no response within 10s") from e
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        msg = decode_message(raw)

        if msg.get("type") == AUTH_CHALLENGE:
            if not self.identity:
                raise ConnectionError("relay requires authentication but node has no identity")
            nonce = msg.get("nonce", "")
            signature = sign_message(
                nonce.encode("utf-8"), self.identity["private_key"],
            )
            auth_resp = make_auth_response(
                self.node_id, signature, self.identity["public_key"],
            )
            await ws.send(encode_message(auth_resp))
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            except asyncio.TimeoutError as e:
                raise ConnectionError("no register_ack within 10s") from e
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            msg = decode_message(raw)

        if msg.get("type") != REGISTER_ACK:
            raise ConnectionError(f"unexpected reply from relay: {msg.get('type')!r}")
        if not msg.get("accepted", False):
            raise ConnectionError(f"relay rejected registration: {msg.get('message', '')}")
        logger.info("registered with relay node_id=%s dynamic=%s", self.node_id, not self.legacy_mode)

    async def _heartbeat_loop(self, ws) -> None:
        cleanup_counter = 0
        while True:
            await asyncio.sleep(10)
            cleanup_counter += 1
            if cleanup_counter % 6 == 0 and self.kv_manager is not None:
                expired = self.kv_manager.cleanup_expired(ttl_seconds=300.0)
                if expired:
                    logger.info("cleaned up %d expired KV sessions", len(expired))
            loaded = (
                (self.layer_start, self.layer_end)
                if self.shard is not None and self.layer_start is not None
                else None
            )
            hb = make_heartbeat(
                self.node_id,
                status="active",
                capabilities=_capabilities(
                    self.model_name,
                    self.device,
                    self.max_context,
                    loaded_layers=loaded,
                    model_preferences=self.model_preferences,
                    bench_data=self.bench_data,
                ),
            )
            try:
                await ws.send(encode_message(hb))
            except (websockets.ConnectionClosed, OSError):
                return

    async def _receive_loop(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            try:
                frame = decode_message(raw)
            except Exception:
                logger.exception("frame decode failed")
                continue

            ftype = frame.get("type")

            if ftype == ASSIGN_LAYERS:
                await self._handle_assign_layers(ws, frame)
                continue
            if ftype == REBALANCE:
                await self._handle_rebalance(ws, frame)
                continue

            if ftype == SDP_OFFER:
                await self._handle_sdp_offer(ws, frame)
                continue
            if ftype == SDP_ANSWER:
                await self._handle_sdp_answer(frame)
                continue
            if ftype == ICE_CANDIDATE:
                await self._handle_ice_candidate(frame)
                continue
            if ftype == P2P_READY:
                await self._handle_p2p_ready(ws, frame)
                continue
            if ftype == MESH_CONNECT:
                asyncio.create_task(self._handle_mesh_connect(ws, frame))
                continue

            if ftype != ENVELOPE:
                logger.warning("non-envelope frame from relay: %s", ftype)
                continue

            stream_id = frame.get("stream_id")
            payload = frame.get("payload")
            hop_start = time.perf_counter()

            try:
                inner = msgpack.unpackb(
                    payload, raw=False,
                    max_str_len=10 * 1024 * 1024,
                    max_bin_len=10 * 1024 * 1024,
                    max_array_len=10_000,
                    max_map_len=1_000,
                )
            except Exception:
                logger.exception("inner payload decode failed stream_id=%s", stream_id)
                continue

            deserialize_ms = (time.perf_counter() - hop_start) * 1000.0
            dispatch_start = time.perf_counter()

            try:
                response = await self._dispatch(inner)
            except torch.cuda.OutOfMemoryError:
                logger.error("CUDA OOM during inference — clearing cache and reloading shard")
                torch.cuda.empty_cache()
                self.shard = None
                try:
                    self.shard = await self._load_shard_async(
                        self.model_name, self.layer_start, self.layer_end,
                    )
                    self._init_kv_manager()
                    logger.info("shard reloaded after OOM recovery")
                except Exception:
                    logger.exception("shard reload failed after OOM")
                response = make_error(
                    inner.get("session_id", ""), 503,
                    "GPU out of memory — shard reloading, retry shortly",
                )
                response["seq_pos"] = inner.get("seq_pos")
            except Exception as exc:
                logger.exception("dispatch failed")
                response = make_error(
                    inner.get("session_id", ""), 500,
                    f"internal processing error: {type(exc).__name__}: {exc}",
                )
                response["seq_pos"] = inner.get("seq_pos")

            if response is None:
                continue

            dispatch_ms = (time.perf_counter() - dispatch_start) * 1000.0
            serialize_start = time.perf_counter()
            response_bytes = msgpack.packb(response, use_bin_type=True)
            serialize_ms = (time.perf_counter() - serialize_start) * 1000.0

            send_start = time.perf_counter()
            downstream = self.downstream_peer
            if downstream and downstream in self.p2p_channels:
                try:
                    await self.p2p_channels[downstream].send_message(response_bytes)
                except Exception:
                    logger.warning("P2P send failed, falling back to relay")
                    outbound = make_envelope(
                        stream_id, response_bytes, target_node_id=None,
                    )
                    try:
                        await ws.send(encode_message(outbound))
                    except (websockets.ConnectionClosed, OSError):
                        return
            else:
                outbound = make_envelope(
                    stream_id, response_bytes, target_node_id=None,
                )
                try:
                    await ws.send(encode_message(outbound))
                except (websockets.ConnectionClosed, OSError):
                    return
            send_ms = (time.perf_counter() - send_start) * 1000.0
            total_ms = (time.perf_counter() - hop_start) * 1000.0

            logger.info(
                "relay hop",
                extra={
                    "stream_id": stream_id,
                    "seq": inner.get("seq_pos"),
                    "deserialize_ms": round(deserialize_ms, 2),
                    "dispatch_ms": round(dispatch_ms, 2),
                    "forward_ms": round(response.get("forward_ms", 0), 2),
                    "queue_ms": round(response.get("queue_ms", 0), 2),
                    "serialize_ms": round(serialize_ms, 2),
                    "send_ms": round(send_ms, 2),
                    "total_ms": round(total_ms, 2),
                    "response_bytes": len(response_bytes),
                },
            )

    async def _load_shard_async(self, model_name: str, layer_start: int, layer_end: int) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: load_model_shard(
                model_name,
                layer_start,
                layer_end,
                device=self.device,
                quantize=self.quantize,
            ),
        )

    def _unload_shard(self) -> None:
        if self.shard is None:
            return
        self.shard = None
        try:
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    async def _handle_assign_layers(self, ws, msg: dict) -> None:
        """Receive a layer assignment from the relay and load the shard."""
        if self.legacy_mode:
            logger.info("ignoring ASSIGN_LAYERS in legacy mode")
            await self._send_ack(
                ws, accepted=False, reason="node is in legacy mode"
            )
            return

        model_name = msg.get("model_name") or self.model_name
        layer_start = msg.get("layer_start")
        layer_end = msg.get("layer_end")
        if model_name is None or layer_start is None or layer_end is None:
            await self._send_ack(
                ws, accepted=False, reason="ASSIGN_LAYERS missing fields"
            )
            return

        async with self._assignment_lock:
            logger.info(
                "ASSIGN_LAYERS model=%s layers=[%d,%d)",
                model_name, layer_start, layer_end,
            )

            from src.relay.scheduler import MODEL_REGISTRY, _effective_capacity_mb
            reg = MODEL_REGISTRY.get(model_name)
            if reg:
                mem_needed = (layer_end - layer_start) * reg.get("memory_per_layer_mb", 0)
                caps = _capabilities(
                    model_name, self.device, self.max_context,
                )
                cap_mb = _effective_capacity_mb(caps)
                if mem_needed > 0 and 0 < cap_mb < mem_needed:
                    reason = (
                        f"node has {cap_mb:.0f}MB but shard needs "
                        f"~{mem_needed:.0f}MB"
                    )
                    logger.warning("rejecting assignment: %s", reason)
                    await self._send_ack(
                        ws, accepted=False,
                        model_name=model_name,
                        layer_start=layer_start,
                        layer_end=layer_end,
                        reason=reason,
                    )
                    return

            started = time.perf_counter()
            try:
                shard = await self._load_shard_async(model_name, layer_start, layer_end)
            except (MemoryError, RuntimeError, ValueError, OSError) as e:
                logger.exception("shard load failed")
                await self._send_ack(
                    ws,
                    accepted=False,
                    model_name=model_name,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    reason=str(e),
                )
                return

            self.shard = shard
            self.model_name = model_name
            self.layer_start = layer_start
            self.layer_end = layer_end
            self._init_kv_manager()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.info("shard loaded in %.0fms", elapsed_ms)
            await self._send_ack(
                ws,
                accepted=True,
                model_name=model_name,
                layer_start=layer_start,
                layer_end=layer_end,
                load_time_ms=elapsed_ms,
            )

    async def _handle_rebalance(self, ws, msg: dict) -> None:
        """Swap the current shard for a new layer range."""
        if self.legacy_mode:
            await self._send_ack(
                ws, accepted=False, reason="node is in legacy mode"
            )
            return

        model_name = msg.get("model_name") or self.model_name
        layer_start = msg.get("new_layer_start", msg.get("layer_start"))
        layer_end = msg.get("new_layer_end", msg.get("layer_end"))
        if model_name is None or layer_start is None or layer_end is None:
            await self._send_ack(
                ws, accepted=False, reason="REBALANCE missing fields"
            )
            return

        async with self._assignment_lock:
            logger.info(
                "REBALANCE model=%s layers=[%d,%d)", model_name, layer_start, layer_end,
            )
            self._unload_shard()
            started = time.perf_counter()
            try:
                shard = await self._load_shard_async(model_name, layer_start, layer_end)
            except (MemoryError, RuntimeError, ValueError, OSError) as e:
                logger.exception("rebalance load failed")
                await self._send_ack(
                    ws,
                    accepted=False,
                    model_name=model_name,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    reason=str(e),
                )
                return

            self.shard = shard
            self.model_name = model_name
            self.layer_start = layer_start
            self.layer_end = layer_end
            self._init_kv_manager()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            await self._send_ack(
                ws,
                accepted=True,
                model_name=model_name,
                layer_start=layer_start,
                layer_end=layer_end,
                load_time_ms=elapsed_ms,
            )

    async def _send_ack(
        self,
        ws,
        accepted: bool,
        model_name: str | None = None,
        layer_start: int | None = None,
        layer_end: int | None = None,
        load_time_ms: float = 0.0,
        reason: str = "",
    ) -> None:
        ack = make_assignment_ack(
            node_id=self.node_id,
            model_name=model_name or self.model_name or "",
            layer_start=layer_start if layer_start is not None else -1,
            layer_end=layer_end if layer_end is not None else -1,
            accepted=accepted,
            reason=reason,
            load_time_ms=int(load_time_ms),
        )
        try:
            await ws.send(encode_message(ack))
        except (websockets.ConnectionClosed, OSError):
            pass

    async def _dispatch(self, msg: dict) -> dict | None:
        t = msg.get("type")
        if t == SESSION_INIT:
            return await self._handle_session_init(msg)
        if t == ACTIVATIONS:
            return await self._handle_activations(msg)
        if t == SPEC_WINDOW:
            return await self._handle_spec_window(msg)
        if t == HEARTBEAT:
            return self._handle_heartbeat(msg)
        logger.warning("unhandled inner type: %s", t)
        err = make_error(msg.get("session_id", ""), 400, f"Unhandled message type: {t}")
        err["seq_pos"] = msg.get("seq_pos")
        return err

    async def _handle_session_init(self, msg: dict) -> dict:
        if self.shard is None:
            err = make_error(
                msg.get("session_id", ""), 409,
                "node has no shard loaded yet — awaiting ASSIGN_LAYERS",
            )
            err["seq_pos"] = msg.get("seq_pos")
            return err
        session_id = msg["session_id"]
        num_layers = self.layer_end - self.layer_start
        config = msg.get("config", {})
        max_ctx = config.get("max_context", self.max_context)
        self.kv_manager.create_session(session_id, num_layers, max_ctx)

        if config.get("node_side_sampling"):
            self._session_sampling[session_id] = {
                "temperature": config.get("temperature", 0.7),
                "top_p": config.get("top_p", 0.9),
            }
            logger.info("session %s: node-side sampling enabled (temp=%.2f top_p=%.2f)",
                        session_id,
                        self._session_sampling[session_id]["temperature"],
                        self._session_sampling[session_id]["top_p"])
        else:
            self._session_sampling.pop(session_id, None)

        logger.info("session %s initialized", session_id)
        return make_heartbeat(self.node_id, status="session_ready")

    async def _handle_activations(self, msg: dict) -> dict:
        queue_start = time.perf_counter()
        if self.shard is None:
            err = make_error(
                msg.get("session_id", ""), 409,
                "node has no shard loaded",
            )
            err["seq_pos"] = msg.get("seq_pos")
            return err
        session_id = msg["session_id"]
        hidden_bytes = msg["hidden_states_bytes"]
        hidden_states = deserialize_tensor(hidden_bytes, device=self.device)

        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        shard_device = str(next(self.shard["layers"].parameters()).device)
        input_device = str(hidden_states.device)
        if shard_device != input_device:
            logger.warning(
                "device mismatch: shard on %s, input on %s — moving input",
                shard_device, input_device,
            )
            hidden_states = hidden_states.to(shard_device)

        session_cache = self.kv_manager.get_session(session_id)
        if session_cache is None:
            num_layers = self.layer_end - self.layer_start
            session_cache = self.kv_manager.create_session(session_id, num_layers, self.max_context)
        kv_cache = session_cache.get_cache()

        queue_ms = (time.perf_counter() - queue_start) * 1000.0
        forward_start = time.perf_counter()

        loop = asyncio.get_event_loop()
        output, kv_cache = await loop.run_in_executor(
            self._inference_executor, lambda: forward_shard(self.shard, hidden_states, kv_cache=kv_cache)
        )

        forward_ms = (time.perf_counter() - forward_start) * 1000.0

        is_last_shard = self.shard["lm_head"] is not None

        telemetry = self._node_telemetry()

        if is_last_shard:
            sampling = self._session_sampling.get(session_id)
            temperature = msg.get("temperature")
            if temperature is not None:
                sampling = {
                    "temperature": float(temperature),
                    "top_p": float(msg.get("top_p", 0.9)),
                }
            if not sampling:
                logger.warning(
                    "last shard but no sampling params — temp=%s session_sampling=%s keys=%s",
                    temperature, bool(self._session_sampling.get(session_id)),
                    list(msg.keys()),
                )
            if sampling:
                sample_start = time.perf_counter()
                token_id = self._sample_from_logits(
                    output, sampling["temperature"], sampling["top_p"],
                )
                sample_ms = (time.perf_counter() - sample_start) * 1000.0
                resp = make_token_result(
                    session_id=session_id,
                    seq_pos=msg["seq_pos"],
                    token_id=token_id,
                    forward_ms=round(forward_ms, 2),
                    queue_ms=round(queue_ms, 2),
                    sample_ms=round(sample_ms, 2),
                )
                resp["node_telemetry"] = telemetry
                return resp

            output_bytes = serialize_tensor(output)
            dtype = str(output.dtype).replace("torch.", "")
            resp = make_logits(
                session_id=session_id,
                seq_pos=msg["seq_pos"],
                logits_bytes=output_bytes,
                shape=tuple(output.shape),
                dtype=dtype,
                forward_ms=round(forward_ms, 2),
                queue_ms=round(queue_ms, 2),
            )
            resp["node_telemetry"] = telemetry
            return resp

        output_bytes = serialize_tensor(output)
        dtype = str(output.dtype).replace("torch.", "")
        resp = make_activations(
            session_id=session_id,
            seq_pos=msg["seq_pos"],
            hidden_states_bytes=output_bytes,
            shape=tuple(output.shape),
            dtype=dtype,
            forward_ms=round(forward_ms, 2),
            queue_ms=round(queue_ms, 2),
        )
        resp["node_telemetry"] = telemetry

        if self.downstream_peer and self.downstream_peer in self.p2p_channels:
            if "temperature" in msg:
                resp["temperature"] = msg["temperature"]
                resp["top_p"] = msg.get("top_p", 0.9)
            if hasattr(self, "_session_temperature"):
                resp["temperature"] = self._session_temperature
                resp["top_p"] = self._session_top_p
            sampling = self._session_sampling.get(session_id)
            if sampling:
                resp["temperature"] = sampling["temperature"]
                resp["top_p"] = sampling["top_p"]
            try:
                fwd_bytes = msgpack.packb(resp, use_bin_type=True)
                await self.p2p_channels[self.downstream_peer].send_message(fwd_bytes)
                logger.debug(
                    "mesh forwarded to %s: seq=%s %d bytes",
                    self.downstream_peer[:12], msg["seq_pos"], len(fwd_bytes),
                )
                return resp
            except Exception:
                logger.warning("mesh forward failed, returning to consumer")
                broken = make_mesh_broken(
                    session_id=session_id,
                    from_node_id=self.node_id,
                    peer_node_id=self.downstream_peer,
                    reason="forward failed",
                )
                if self._ws:
                    try:
                        await self._ws.send(encode_message(broken))
                    except Exception:
                        pass
                self.downstream_peer = None

        return resp

    def _sample_from_logits(
        self, logits: torch.Tensor, temperature: float, top_p: float,
    ) -> int:
        logits = logits[:, -1, :].float() if logits.dim() == 3 else logits[-1].float()
        logits = logits.squeeze(0) if logits.dim() > 1 else logits
        if temperature <= 0:
            return int(logits.argmax().item())
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()
        idx = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_indices[idx.item()].item())

    @torch.inference_mode()
    async def _handle_spec_window(self, msg: dict) -> dict:
        if self.shard is None:
            err = make_error(
                msg.get("session_id", ""), 409,
                "node has no shard loaded",
            )
            err["seq_pos"] = msg.get("seq_pos")
            return err
        session_id = msg["session_id"]
        candidate_ids = msg["candidate_ids"]
        num_candidates = len(candidate_ids)

        if not self.shard["lm_head"]:
            err = make_error(
                session_id, 400,
                "SPEC_WINDOW can only be handled by the final shard (has lm_head)",
            )
            err["seq_pos"] = msg.get("seq_pos")
            return err

        session_cache = self.kv_manager.get_session(session_id)
        if session_cache is None:
            err = make_error(session_id, 404, f"No session found: {session_id}")
            err["seq_pos"] = msg.get("seq_pos")
            return err

        if not self.shard["embed_tokens"]:
            err = make_error(
                session_id, 400,
                "SPEC_WINDOW requires the shard to have embed_tokens (first shard) "
                "or pre-computed hidden states.",
            )
            err["seq_pos"] = msg.get("seq_pos")
            return err

        candidate_tensor = torch.tensor(
            [candidate_ids], dtype=torch.long, device=self.device
        )

        kv_cache = session_cache.get_cache()
        loop = asyncio.get_event_loop()
        logits, kv_cache = await loop.run_in_executor(
            self._inference_executor, lambda: forward_shard(self.shard, candidate_tensor, kv_cache=kv_cache)
        )

        accepted_tokens: list[int] = []
        correction_token: int | None = None
        num_accepted = 0

        for i in range(num_candidates - 1):
            predicted_token = logits[0, i].argmax(dim=-1).item()
            next_candidate = candidate_ids[i + 1]
            if predicted_token == next_candidate:
                accepted_tokens.append(candidate_ids[i])
                num_accepted += 1
            else:
                accepted_tokens.append(candidate_ids[i])
                num_accepted += 1
                correction_token = predicted_token
                break
        else:
            accepted_tokens.append(candidate_ids[-1])
            num_accepted = num_candidates
            correction_token = logits[0, -1].argmax(dim=-1).item()

        if num_accepted < num_candidates:
            trim_count = num_candidates - num_accepted
            keep = max(0, kv_cache.get_seq_length() - trim_count)
            kv_cache.crop(keep)

        result = make_verify_result(
            session_id=session_id,
            accepted_tokens=accepted_tokens,
            correction_token=correction_token,
            num_accepted=num_accepted,
        )
        result["seq_pos"] = msg.get("seq_pos")
        return result

    def _handle_heartbeat(self, msg: dict) -> dict:
        return make_heartbeat(node_id=self.node_id, status="active")

    def _get_peer_manager(self, turn_servers: list[dict] | None = None):
        """Lazy-init PeerConnectionManager (backend agent's module)."""
        if self.peer_manager is not None:
            return self.peer_manager
        try:
            from src.common.peer import PeerConnectionManager
        except ImportError:
            logger.warning(
                "src.common.peer not available — P2P disabled. "
                "Install aiortc and ensure peer.py is present."
            )
            return None
        rtc_config = None
        if turn_servers:
            try:
                from src.common.peer import build_rtc_config
                rtc_config = build_rtc_config(turn_servers)
            except (ImportError, Exception):
                logger.warning("build_rtc_config unavailable, using default config")
        self.peer_manager = PeerConnectionManager(rtc_config=rtc_config)
        return self.peer_manager

    async def _handle_sdp_offer(self, ws, msg: dict) -> None:
        """Received via signal WS. Create answer, send back via signal.

        Handles re-offers: if a peer already has a connection (reconnect
        scenario), the old connection is closed and replaced.
        """
        turn_servers = msg.get("turn_servers")
        pm = self._get_peer_manager(turn_servers=turn_servers)
        if pm is None:
            return
        from_id = msg.get("from_node_id", "")

        if pm.is_connected(from_id) or from_id in self.p2p_channels:
            logger.info("re-offer from %s, replacing old connection", from_id[:12])
            old_task = self._p2p_recv_tasks.pop(from_id, None)
            if old_task and not old_task.done():
                old_task.cancel()
            self.p2p_channels.pop(from_id, None)
            try:
                await pm.close(from_id)
            except Exception:
                pass

        try:
            answer_sdp = await pm.handle_offer(from_id, msg["sdp"])
        except Exception:
            logger.exception("failed to handle SDP offer from %s", from_id[:12])
            return

        answer = make_sdp_answer(
            session_id=msg.get("session_id", ""),
            from_node_id=self.node_id,
            to_node_id=from_id,
            sdp=answer_sdp,
        )
        try:
            await ws.send(encode_message(answer))
        except (websockets.ConnectionClosed, OSError):
            return

        logger.info("SDP answer sent to %s", from_id[:12])

        asyncio.create_task(self._activate_p2p(ws, from_id, msg.get("session_id", "")))

    async def _handle_mesh_connect(self, ws, msg: dict) -> None:
        """Consumer tells us about our mesh peer."""
        downstream_id = msg.get("downstream_node_id", "")
        upstream_id = msg.get("upstream_node_id", "")
        session_id = msg.get("session_id", "")

        if upstream_id:
            self.upstream_peer = upstream_id
            logger.info("MESH_CONNECT: upstream peer set to %s", upstream_id[:12])
            return

        if not downstream_id:
            logger.warning("MESH_CONNECT missing both downstream and upstream node_id")
            return

        logger.info("MESH_CONNECT: initiating P2P to downstream %s", downstream_id[:12])
        turn_servers = msg.get("turn_servers")
        pm = self._get_peer_manager(turn_servers=turn_servers)
        if pm is None:
            return

        try:
            offer_sdp = await pm.create_offer(downstream_id)
        except Exception:
            logger.exception("failed to create mesh offer for %s", downstream_id[:12])
            return

        from src.common.protocol import make_sdp_offer
        offer = make_sdp_offer(
            session_id=session_id,
            from_node_id=self.node_id,
            to_node_id=downstream_id,
            sdp=offer_sdp,
        )
        try:
            await ws.send(encode_message(offer))
        except (websockets.ConnectionClosed, OSError):
            return

        self.downstream_peer = downstream_id
        logger.info("mesh SDP offer sent to %s via relay", downstream_id[:12])
        asyncio.create_task(self._activate_mesh_p2p(ws, downstream_id, session_id))

    async def _activate_mesh_p2p(self, ws, peer_id: str, session_id: str) -> None:
        """Wait for mesh P2P to become ready, then notify consumer."""
        pm = self._get_peer_manager()
        if pm is None:
            return
        if not await pm.wait_connected(peer_id, timeout=5.0):
            logger.warning("mesh P2P to %s timed out", peer_id[:12])
            self.downstream_peer = None
            return

        async def _mesh_send(data: bytes):
            await pm.send(peer_id, data)

        self.p2p_channels[peer_id] = ChunkedChannel(_mesh_send)

        old_task = self._p2p_recv_tasks.pop(peer_id, None)
        if old_task and not old_task.done():
            old_task.cancel()
        self._p2p_recv_tasks[peer_id] = asyncio.create_task(
            self._p2p_receive_loop(ws, peer_id)
        )

        logger.info("mesh P2P active: downstream=%s", peer_id[:12])

        ready = make_mesh_ready(
            session_id=session_id,
            from_node_id=self.node_id,
            peer_node_id=peer_id,
        )
        try:
            await ws.send(encode_message(ready))
        except (websockets.ConnectionClosed, OSError):
            pass

    async def _handle_sdp_answer(self, msg: dict) -> None:
        """Apply remote SDP answer to complete handshake."""
        pm = self._get_peer_manager()
        if pm is None:
            return
        from_id = msg.get("from_node_id", "")
        try:
            await pm.accept_answer(from_id, msg["sdp"])
        except Exception:
            logger.exception("failed to accept SDP answer from %s", from_id[:12])

    async def _handle_ice_candidate(self, msg: dict) -> None:
        """Add trickled ICE candidate to connection."""
        pm = self._get_peer_manager()
        if pm is None:
            return
        from_id = msg.get("from_node_id", "")
        try:
            await pm.add_ice_candidate(from_id, {
                "candidate": msg.get("candidate", ""),
                "sdpMid": msg.get("sdp_mid"),
                "sdpMLineIndex": msg.get("sdp_m_line_index"),
            })
        except Exception:
            logger.exception("failed to add ICE candidate from %s", from_id[:12])

    async def _handle_p2p_ready(self, ws, msg: dict) -> None:
        """Mark peer connection as ready, create ChunkedChannel wrapper."""
        pm = self._get_peer_manager()
        if pm is None:
            return
        peer_id = msg.get("node_id") or msg.get("peer_node_id", "")
        if not peer_id:
            return

        async def _p2p_send(data: bytes):
            await pm.send(peer_id, data)

        self.p2p_channels[peer_id] = ChunkedChannel(_p2p_send)
        logger.info("P2P channel ready with %s", peer_id[:12])

        asyncio.create_task(self._monitor_peer_state(peer_id))

    async def _monitor_peer_state(self, peer_id: str) -> None:
        """Watch a peer connection and clean up its ChunkedChannel on drop."""
        try:
            while True:
                await asyncio.sleep(2.0)
                if self.peer_manager is None:
                    break
                if not self.peer_manager.is_connected(peer_id):
                    if peer_id in self.p2p_channels:
                        del self.p2p_channels[peer_id]
                        logger.warning(
                            "P2P channel dropped for %s, falling back to relay",
                            peer_id[:12],
                        )
                    break
        except asyncio.CancelledError:
            return

    async def _activate_p2p(self, ws, peer_id: str, session_id: str) -> None:
        """Wait for data channels to open, send P2P_READY, start receive loop."""
        pm = self.peer_manager
        if pm is None:
            return
        if not await pm.wait_connected(peer_id, timeout=10.0):
            logger.warning("P2P activation timeout for %s", peer_id[:12])
            return

        async def _p2p_send(data: bytes):
            await pm.send(peer_id, data)

        self.p2p_channels[peer_id] = ChunkedChannel(_p2p_send)

        old_task = self._p2p_recv_tasks.pop(peer_id, None)
        if old_task and not old_task.done():
            old_task.cancel()
        self._p2p_recv_tasks[peer_id] = asyncio.create_task(
            self._p2p_receive_loop(ws, peer_id)
        )

        ready_msg = make_p2p_ready(
            session_id=session_id,
            node_id=self.node_id,
            peer_node_id=peer_id,
        )
        try:
            await ws.send(encode_message(ready_msg))
        except (websockets.ConnectionClosed, OSError):
            pass

        logger.info("P2P activated with %s", peer_id[:12])
        asyncio.create_task(self._monitor_peer_state(peer_id))

    async def _p2p_receive_loop(self, ws, peer_id: str) -> None:
        """Read incoming P2P data, dispatch, respond via P2P."""
        pm = self.peer_manager
        channel = self.p2p_channels.get(peer_id)
        if pm is None or channel is None:
            return
        try:
            while peer_id in self.p2p_channels:
                try:
                    raw = await asyncio.wait_for(pm.recv(peer_id), timeout=5.0)
                except asyncio.TimeoutError:
                    if not pm.is_connected(peer_id):
                        break
                    continue

                complete = channel.on_chunk_received(raw)
                if complete is None:
                    continue

                hop_start = time.perf_counter()

                if peer_id == self.downstream_peer:
                    peeked_type = _peek_msg_type(complete)
                    if peeked_type in (TOKEN_RESULT, LOGITS, ERROR):
                        consumer_peers = [
                            pid for pid in self.p2p_channels
                            if pid != self.upstream_peer and pid != self.downstream_peer
                        ]
                        sent = False
                        if consumer_peers:
                            try:
                                await self.p2p_channels[consumer_peers[0]].send_message(complete)
                                sent = True
                            except Exception:
                                logger.warning("mesh passthrough via P2P failed, falling back to relay")
                        if not sent:
                            inner = msgpack.unpackb(
                                complete, raw=False,
                                max_str_len=10 * 1024 * 1024,
                                max_bin_len=10 * 1024 * 1024,
                                max_array_len=10_000,
                                max_map_len=1_000,
                            )
                            fwd_bytes = msgpack.packb(inner, use_bin_type=True)
                            outbound = make_envelope(
                                inner.get("stream_id", ""), fwd_bytes,
                                target_node_id=None,
                            )
                            try:
                                await ws.send(encode_message(outbound))
                            except (websockets.ConnectionClosed, OSError):
                                return
                        logger.debug("mesh passthrough %s to consumer (zero-copy)", peeked_type)
                        continue

                try:
                    inner = msgpack.unpackb(
                        complete, raw=False,
                        max_str_len=10 * 1024 * 1024,
                        max_bin_len=10 * 1024 * 1024,
                        max_array_len=10_000,
                        max_map_len=1_000,
                    )
                except Exception:
                    logger.exception("P2P payload decode failed from %s", peer_id[:12])
                    continue

                deserialize_ms = (time.perf_counter() - hop_start) * 1000.0

                if inner.get("type") == MESH_CONNECT:
                    asyncio.create_task(self._handle_mesh_connect(ws, inner))
                    continue

                dispatch_start = time.perf_counter()

                try:
                    response = await self._dispatch(inner)
                except torch.cuda.OutOfMemoryError:
                    logger.error("CUDA OOM during P2P inference — clearing cache")
                    torch.cuda.empty_cache()
                    self.shard = None
                    try:
                        self.shard = await self._load_shard_async(
                            self.model_name, self.layer_start, self.layer_end,
                        )
                        self._init_kv_manager()
                    except Exception:
                        logger.exception("shard reload failed after OOM")
                    response = make_error(
                        inner.get("session_id", ""), 503,
                        "GPU out of memory — shard reloading",
                    )
                    response["seq_pos"] = inner.get("seq_pos")
                except Exception as exc:
                    logger.exception("P2P dispatch failed")
                    response = make_error(
                        inner.get("session_id", ""), 500,
                        f"internal error: {type(exc).__name__}: {exc}",
                    )
                    response["seq_pos"] = inner.get("seq_pos")

                if response is None:
                    continue

                dispatch_ms = (time.perf_counter() - dispatch_start) * 1000.0
                serialize_start = time.perf_counter()
                response_bytes = msgpack.packb(response, use_bin_type=True)
                serialize_ms = (time.perf_counter() - serialize_start) * 1000.0

                send_start = time.perf_counter()
                reply_channel = channel
                if peer_id == self.upstream_peer:
                    consumer_peers = [
                        pid for pid in self.p2p_channels
                        if pid != self.upstream_peer and pid != self.downstream_peer
                    ]
                    if consumer_peers:
                        reply_channel = self.p2p_channels[consumer_peers[0]]

                try:
                    await reply_channel.send_message(response_bytes)
                except Exception:
                    logger.warning(
                        "P2P response failed for %s, falling back to relay",
                        peer_id[:12],
                    )
                    outbound = make_envelope(
                        inner.get("stream_id", ""), response_bytes,
                        target_node_id=None,
                    )
                    try:
                        await ws.send(encode_message(outbound))
                    except (websockets.ConnectionClosed, OSError):
                        return
                send_ms = (time.perf_counter() - send_start) * 1000.0
                total_ms = (time.perf_counter() - hop_start) * 1000.0

                logger.info(
                    "P2P hop",
                    extra={
                        "peer": peer_id[:12],
                        "seq": inner.get("seq_pos"),
                        "deserialize_ms": round(deserialize_ms, 2),
                        "dispatch_ms": round(dispatch_ms, 2),
                        "forward_ms": round(response.get("forward_ms", 0), 2),
                        "queue_ms": round(response.get("queue_ms", 0), 2),
                        "serialize_ms": round(serialize_ms, 2),
                        "send_ms": round(send_ms, 2),
                        "total_ms": round(total_ms, 2),
                        "response_bytes": len(response_bytes),
                    },
                )
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("P2P receive loop crashed for %s", peer_id[:12])


def parse_layer_range(layer_str: str) -> tuple[int, int]:
    """Parse 'START-END' (inclusive end) into (start, end_exclusive)."""
    parts = layer_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Layer range must be 'START-END', got '{layer_str}'")
    return int(parts[0]), int(parts[1]) + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Groove Compute Node — outbound-only worker that registers with a relay"
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="HuggingFace model name. Required in legacy mode (--layers). "
        "In dynamic mode acts as a hint / preferred model.",
    )
    parser.add_argument(
        "--layers",
        required=False,
        default=None,
        help="Legacy mode override: pre-load this layer range at startup (e.g. 0-15). "
        "If omitted, the node runs in dynamic mode and waits for ASSIGN_LAYERS.",
    )
    conn_group = parser.add_mutually_exclusive_group(required=True)
    conn_group.add_argument(
        "--signal",
        help="Signal service hostname (e.g. signal.groovedev.ai). Uses wss:// automatically.",
    )
    conn_group.add_argument(
        "--relay",
        help="Legacy relay address, format HOST:PORT (use --signal for production)",
    )
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--node-id", default=None, help="Override the node id (rarely needed)")
    parser.add_argument(
        "--key-path",
        default="~/.groove/node_key.json",
        help="Path to the persisted node keypair",
    )
    parser.add_argument("--tls", action="store_true", help="Use wss:// (automatic with --signal)")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    if args.signal:
        connect_url = args.signal
        use_tls = True
    else:
        connect_url = args.relay
        use_tls = args.tls

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    identity = load_or_create_identity(args.key_path)
    logger.info("Node identity: %s", identity["address"])

    legacy_mode = args.layers is not None
    layer_start: int | None = None
    layer_end: int | None = None

    connect_label = f"signal={args.signal}" if args.signal else f"relay={args.relay}"

    if legacy_mode:
        if args.model is None:
            parser.error("--model is required when --layers is provided")
        layer_start, layer_end = parse_layer_range(args.layers)
        model_info = get_model_info(args.model)
        total_layers = model_info["total_layers"]
        if layer_end > total_layers:
            parser.error(
                f"Layer range 0-{layer_end - 1} exceeds model's {total_layers} layers"
            )
        logger.info(
            "starting compute node (legacy mode) model=%s layers=[%d,%d) device=%s %s",
            args.model, layer_start, layer_end, args.device, connect_label,
        )
    else:
        logger.info(
            "starting compute node (dynamic mode) device=%s %s",
            args.device, connect_label,
        )

    if args.model:
        from src.node.shard_loader import _validate_model_name
        _validate_model_name(args.model)
        try:
            from huggingface_hub import snapshot_download, try_to_load_from_cache
            from huggingface_hub.utils import LocalEntryNotFoundError

            cached = try_to_load_from_cache(args.model, "config.json")
            if cached is None or isinstance(cached, str) and not os.path.exists(cached):
                logger.info(
                    "Model %s not in local cache — downloading (~8 GB). "
                    "This is a one-time download, please wait...",
                    args.model,
                )
            else:
                logger.info("Verifying cached model weights for %s...", args.model)

            snapshot_download(
                args.model,
                allow_patterns=["*.safetensors", "*.json"],
            )
            logger.info("Model weights ready")
        except Exception as e:
            logger.error("Failed to download model weights: %s", e)
            sys.exit(1)

    server = ComputeNodeServer(
        model_name=args.model,
        layer_start=layer_start,
        layer_end=layer_end,
        device=args.device,
        max_context=args.max_context,
        quantize=args.quantize,
        node_id=args.node_id,
        legacy_mode=legacy_mode,
        identity=identity,
    )

    async def _run():
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, server.request_stop)
            except NotImplementedError:
                pass
        await server.run(connect_url, use_tls=use_tls)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
