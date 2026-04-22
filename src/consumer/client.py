"""Groove Decentralized Inference - Consumer Client.

The consumer holds a single websocket to the signal service (or legacy
relay). All work is sent in ENVELOPEs addressed by stream_id (returned in
PIPELINE_CONFIG) and target_node_id (the node the consumer wants to reach).
Responses come back via the same socket — a single receive task fans them
out to per-seq_pos futures. The consumer never learns or contacts a node
directly.

M3: --signal flag connects to the Groove signal service over TLS. The signal
service scores and matches nodes, then brokers the connection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import AsyncGenerator, Optional

try:
    from src.common.peer import build_rtc_config
except ImportError:
    build_rtc_config = None

import msgpack
import numpy as np
import torch
import websockets

logger = logging.getLogger("consumer")

from src.common.chunking import ChunkedChannel
from src.common.protocol import (
    ACTIVATIONS,
    ENVELOPE,
    ERROR,
    ICE_CANDIDATE,
    LOGITS,
    P2P_READY,
    PIPELINE_CONFIG,
    PROTOCOL_VERSION,
    SDP_ANSWER,
    SDP_OFFER,
    TOKEN_RESULT,
    VERIFY_RESULT,
    decode_message,
    encode_message,
    make_activations,
    make_envelope,
    make_sdp_offer,
    make_session_init,
    new_session_id,
)
from src.common.tensor_transfer import deserialize_tensor, serialize_tensor


def _logits_from_response(response: dict) -> np.ndarray:
    blob = response["logits_bytes"]
    tensor = deserialize_tensor(blob)
    return tensor.cpu().float().numpy()


def _last_token_logits(logits: np.ndarray) -> np.ndarray:
    while logits.ndim > 1:
        logits = logits[-1]
    return logits


class InferenceClient:
    def __init__(
        self,
        relay_host: str = "localhost",
        relay_port: int = 8770,
        json_mode: bool = False,
        use_tls: bool = False,
        signal_mode: bool = False,
        node_timeout: float = 30.0,
    ):
        self.relay_host = relay_host
        self.relay_port = relay_port
        self.use_tls = use_tls
        self.signal_mode = signal_mode
        self.relay_ws = None
        self.session_id: Optional[str] = None
        self.stream_id: Optional[str] = None
        self.pipeline: list[dict] = []
        self.tokenizer = None
        self.model_name: Optional[str] = None
        self._waiters: dict[int, asyncio.Future] = {}
        self._recv_task: Optional[asyncio.Task] = None
        self._closed = False
        self.envelope_count: int = 0
        self.json_mode = json_mode
        self.tokens_generated: int = 0
        self._node_gone: bool = False
        self.max_retries: int = 3
        self.node_timeout: float = node_timeout
        self.timing_stats: dict = {}
        self._stage_times: dict[int, list[float]] = {}
        self._stage_forward_times: dict[int, list[float]] = {}
        self._generation_start: float = 0.0
        self._ttft_ms: float = 0.0
        self._first_token_emitted: bool = False

        self.peer_manager = None
        self.p2p_channels: dict[str, ChunkedChannel] = {}
        self.p2p_connected: set[str] = set()
        self._turn_servers: list[dict] | None = None
        self._p2p_monitor_task: asyncio.Task | None = None
        self._reconnect_attempts: dict[str, int] = {}
        self._p2p_send_count: int = 0
        self._relay_send_count: int = 0
        self._reconnect_count: int = 0
        self._max_reconnect_per_peer: int = 3
        self._token_traces: list[dict] = []
        self._last_token_timing: dict = {}
        self._live_trace_path: str = ""
        self._live_trace_file = None

    def emit_event(self, event: dict) -> None:
        """Print one JSON event line to stdout. No-op when json_mode is off."""
        if not self.json_mode:
            return
        print(json.dumps(event), flush=True)

    async def connect(self) -> None:
        if self.relay_host.startswith("wss://") or self.relay_host.startswith("ws://"):
            uri = self.relay_host
        elif self.signal_mode:
            uri = f"wss://{self.relay_host}"
        else:
            scheme = "wss" if self.use_tls else "ws"
            uri = f"{scheme}://{self.relay_host}:{self.relay_port}"
        self.relay_ws = await websockets.connect(uri, max_size=10 * 1024 * 1024)
        if self.signal_mode:
            self.emit_event({
                "type": "signal_connected",
                "signal": self.relay_host,
            })

    async def start_session(
        self, model_name: str, config: Optional[dict] = None,
        temperature: float = 0.7, top_p: float = 0.9,
    ) -> str:
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.session_id = new_session_id()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=False,
        )
        self._session_temperature = temperature
        self._session_top_p = top_p

        session_config = dict(config or {})
        session_config["node_side_sampling"] = True
        session_config["temperature"] = temperature
        session_config["top_p"] = top_p

        init = make_session_init(self.session_id, model_name, 0, -1, session_config)
        await self.relay_ws.send(encode_message(init))

        raw = await self.relay_ws.recv()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        response = decode_message(raw)

        if response["type"] == ERROR:
            msg = response.get("message", "")
            code = response.get("code")
            if code in ("COVERAGE_INCOMPLETE", "NO_NODES") or "coverage" in msg.lower():
                raise RuntimeError(
                    "Network does not have full model coverage yet. "
                    "Waiting for more nodes to join..."
                )
            raise RuntimeError(f"Session init failed: {msg}")
        if response["type"] != PIPELINE_CONFIG:
            raise RuntimeError(f"Unexpected response type: {response['type']}")

        self.pipeline = response["nodes"]
        self.stream_id = response.get("stream_id")
        self._turn_servers = response.get("turn_servers")
        if not self.stream_id:
            raise RuntimeError("Relay did not return a stream_id in PIPELINE_CONFIG")

        if self.signal_mode:
            self.emit_event({
                "type": "matched",
                "nodes": [
                    {
                        "node_id": n["node_id"],
                        "score": n.get("score"),
                        "device": n.get("device"),
                        "layers": [n.get("layer_start"), n.get("layer_end")],
                    }
                    for n in self.pipeline
                ],
            })
        self.emit_event({
            "type": "connected",
            "signal" if self.signal_mode else "relay": (
                self.relay_host if self.signal_mode
                else f"{self.relay_host}:{self.relay_port}"
            ),
            "session_id": self.session_id,
        })
        self.emit_event({
            "type": "pipeline",
            "nodes": [
                {
                    "node_id": n["node_id"],
                    "layers": [n.get("layer_start"), n.get("layer_end")],
                }
                for n in self.pipeline
            ],
        })

        self._recv_task = asyncio.create_task(self._receive_loop())

        await self._establish_p2p()

        return self.session_id

    async def _establish_p2p(self) -> None:
        """Attempt P2P WebRTC connections to all pipeline nodes."""
        start_time = time.monotonic()
        try:
            from src.common.peer import PeerConnectionManager
        except ImportError:
            logger.info("P2P unavailable (peer.py not found), using relay")
            return

        rtc_config = None
        if build_rtc_config is not None and self._turn_servers:
            try:
                rtc_config = build_rtc_config(self._turn_servers)
            except Exception:
                logger.warning("build_rtc_config failed, using default config")

        self.peer_manager = PeerConnectionManager(
            session_id=self.session_id, rtc_config=rtc_config,
        )
        session_id = self.session_id or ""

        for node in self.pipeline:
            node_id = node["node_id"]
            try:
                offer_sdp = await self.peer_manager.create_offer(node_id)
            except Exception:
                logger.warning("failed to create SDP offer for %s", node_id[:12])
                continue

            offer_msg = make_sdp_offer(
                session_id=session_id,
                from_node_id="consumer",
                to_node_id=node_id,
                sdp=offer_sdp,
            )
            try:
                await self.relay_ws.send(encode_message(offer_msg))
            except (websockets.ConnectionClosed, OSError):
                logger.warning("failed to send SDP offer for %s", node_id[:12])
                continue

        try:
            await asyncio.wait_for(self._wait_for_p2p(), timeout=5.0)
        except asyncio.TimeoutError:
            missing = [
                n["node_id"][:12] for n in self.pipeline
                if n["node_id"] not in self.p2p_connected
            ]
            if missing:
                logger.warning(
                    "P2P timeout for %d node(s) (%s), using relay fallback",
                    len(missing), ", ".join(missing),
                )

        logger.info(
            "P2P setup complete",
            extra={
                "p2p_nodes": len(self.p2p_connected),
                "relay_nodes": len(self.pipeline) - len(self.p2p_connected),
                "setup_time_ms": round((time.monotonic() - start_time) * 1000),
            },
        )

        self._p2p_monitor_task = asyncio.create_task(self._monitor_p2p())

    async def _wait_for_p2p(self) -> None:
        """Wait until all pipeline nodes are P2P-connected."""
        expected = {n["node_id"] for n in self.pipeline}
        while not expected.issubset(self.p2p_connected):
            await asyncio.sleep(0.05)

    async def _monitor_p2p(self) -> None:
        """Background task: detect dropped P2P peers and attempt reconnection."""
        try:
            while True:
                await asyncio.sleep(2.0)
                if self.peer_manager is None:
                    continue
                for node_id in list(self.p2p_connected):
                    if self.peer_manager.is_connected(node_id):
                        continue
                    self.p2p_connected.discard(node_id)
                    self.p2p_channels.pop(node_id, None)
                    logger.warning("P2P dropped for %s, falling back to relay", node_id[:12])

                    attempts = self._reconnect_attempts.get(node_id, 0)
                    if attempts >= self._max_reconnect_per_peer:
                        logger.warning(
                            "max reconnection attempts reached for %s", node_id[:12],
                        )
                        continue

                    self._reconnect_attempts[node_id] = attempts + 1
                    self._reconnect_count += 1

                    try:
                        await self.peer_manager.close(node_id)
                    except Exception:
                        pass

                    try:
                        offer_sdp = await self.peer_manager.create_offer(node_id)
                        offer_msg = make_sdp_offer(
                            session_id=self.session_id or "",
                            from_node_id="consumer",
                            to_node_id=node_id,
                            sdp=offer_sdp,
                        )
                        await self.relay_ws.send(encode_message(offer_msg))
                    except Exception:
                        logger.warning("reconnect offer failed for %s", node_id[:12])
                        continue

                    try:
                        await asyncio.wait_for(
                            self._wait_for_peer_reconnect(node_id), timeout=5.0,
                        )
                        logger.info("P2P reconnected to %s", node_id[:12])
                    except asyncio.TimeoutError:
                        logger.warning(
                            "P2P reconnect timeout for %s, staying on relay", node_id[:12],
                        )
        except asyncio.CancelledError:
            return

    async def _wait_for_peer_reconnect(self, node_id: str) -> None:
        while node_id not in self.p2p_connected:
            await asyncio.sleep(0.05)

    async def _receive_loop(self) -> None:
        try:
            async for raw in self.relay_ws:
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                try:
                    msg = decode_message(raw)
                except Exception:
                    try:
                        msg = msgpack.unpackb(
                            raw, raw=False,
                            max_str_len=10 * 1024 * 1024,
                            max_bin_len=10 * 1024 * 1024,
                        )
                    except Exception:
                        logger.warning("failed to decode message from relay, skipping")
                        continue
                    if not isinstance(msg, dict) or "type" not in msg:
                        logger.warning("failed to decode message from relay, skipping")
                        continue

                mtype = msg.get("type", "")

                if mtype == SDP_ANSWER:
                    await self._handle_sdp_answer(msg)
                    continue
                if mtype == ICE_CANDIDATE:
                    await self._handle_ice_candidate(msg)
                    continue
                if mtype == P2P_READY:
                    await self._handle_p2p_ready(msg)
                    continue

                if msg["type"] == ENVELOPE:
                    try:
                        inner = msgpack.unpackb(
                            msg["payload"], raw=False,
                            max_str_len=10 * 1024 * 1024,
                            max_bin_len=10 * 1024 * 1024,
                            max_array_len=10_000,
                            max_map_len=1_000,
                        )
                    except Exception:
                        logger.warning("failed to decode envelope payload, skipping")
                        continue
                    if inner.get("type") == ERROR:
                        seq = inner.get("seq_pos")
                        err = RuntimeError(
                            f"Node error [{inner.get('code')}]: "
                            f"{inner.get('message', '')}"
                        )
                        if seq is not None:
                            fut = self._waiters.pop(seq, None)
                            if fut is not None and not fut.done():
                                fut.set_exception(err)
                        else:
                            for fut in list(self._waiters.values()):
                                if not fut.done():
                                    fut.set_exception(err)
                            self._waiters.clear()
                        continue
                    seq = inner.get("seq_pos")
                    fut = self._waiters.pop(seq, None)
                    if fut is not None and not fut.done():
                        fut.set_result(inner)
                elif msg["type"] == ERROR:
                    code = msg.get("code", "")
                    if code == "NODE_GONE":
                        self._node_gone = True
                    err = RuntimeError(
                        f"Relay error [{code}]: {msg.get('message', '')}"
                    )
                    for fut in list(self._waiters.values()):
                        if not fut.done():
                            fut.set_exception(err)
                    self._waiters.clear()
        except websockets.ConnectionClosed:
            pass
        except Exception:
            logger.exception("receive loop crashed")
        finally:
            err = ConnectionError("Relay connection closed")
            for fut in list(self._waiters.values()):
                if not fut.done():
                    fut.set_exception(err)
            self._waiters.clear()

    async def _handle_sdp_answer(self, msg: dict) -> None:
        if self.peer_manager is None:
            return
        from_id = msg.get("from_node_id", "")
        try:
            await self.peer_manager.accept_answer(from_id, msg["sdp"])
        except Exception:
            logger.warning("failed to accept SDP answer from %s", from_id[:12])

    async def _handle_ice_candidate(self, msg: dict) -> None:
        if self.peer_manager is None:
            return
        from_id = msg.get("from_node_id", "")
        try:
            await self.peer_manager.add_ice_candidate(from_id, {
                "candidate": msg.get("candidate", ""),
                "sdpMid": msg.get("sdp_mid"),
                "sdpMLineIndex": msg.get("sdp_m_line_index"),
            })
        except Exception:
            logger.warning("failed to add ICE candidate from %s", from_id[:12])

    async def _handle_p2p_ready(self, msg: dict) -> None:
        node_id = msg.get("node_id") or msg.get("peer_node_id", "")
        if not node_id:
            return

        if self.peer_manager is not None:
            async def _p2p_send(data: bytes):
                await self.peer_manager.send(node_id, data)

            self.p2p_channels[node_id] = ChunkedChannel(_p2p_send)

        self.p2p_connected.add(node_id)
        logger.info("P2P channel ready with %s", node_id[:12])
        asyncio.create_task(self._p2p_receive_loop(node_id))

    async def _p2p_receive_loop(self, node_id: str) -> None:
        """Read P2P responses from a node and resolve waiters."""
        pm = self.peer_manager
        channel = self.p2p_channels.get(node_id)
        if pm is None or channel is None:
            return
        try:
            while node_id in self.p2p_connected:
                try:
                    raw = await asyncio.wait_for(pm.recv(node_id), timeout=5.0)
                except asyncio.TimeoutError:
                    if not pm.is_connected(node_id):
                        break
                    continue

                complete = channel.on_chunk_received(raw)
                if complete is None:
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
                    logger.warning("P2P payload decode failed from %s", node_id[:12])
                    continue

                if inner.get("type") == ERROR:
                    seq = inner.get("seq_pos")
                    err = RuntimeError(
                        f"Node error [{inner.get('code')}]: "
                        f"{inner.get('message', '')}"
                    )
                    if seq is not None:
                        fut = self._waiters.pop(seq, None)
                        if fut is not None and not fut.done():
                            fut.set_exception(err)
                    continue

                seq = inner.get("seq_pos")
                if seq is not None:
                    fut = self._waiters.pop(seq, None)
                    if fut is not None and not fut.done():
                        fut.set_result(inner)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("P2P receive loop crashed for %s", node_id[:12])

    async def _send_to_node(
        self, node_id: str, inner: dict, timeout: float | None = None,
        stage_idx: int | None = None,
    ) -> dict:
        if timeout is None:
            timeout = self.node_timeout
        if self.relay_ws is None or self.stream_id is None:
            raise RuntimeError("No active session — call connect()/start_session() first")
        if self._recv_task is not None and self._recv_task.done():
            exc = self._recv_task.exception() if not self._recv_task.cancelled() else None
            raise ConnectionError(
                f"Receive loop died before response: {exc or 'cancelled'}"
            )
        seq = inner["seq_pos"]
        if seq in self._waiters:
            raise RuntimeError(f"In-flight request already exists for seq_pos={seq}")
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._waiters[seq] = fut

        if hasattr(self, "_session_temperature"):
            inner["temperature"] = self._session_temperature
            inner["top_p"] = self._session_top_p

        serialize_start = time.perf_counter()
        inner_bytes = msgpack.packb(inner, use_bin_type=True)
        serialize_ms = (time.perf_counter() - serialize_start) * 1000.0

        rtt_start = time.perf_counter()
        via = "p2p"

        if node_id in self.p2p_connected and node_id in self.p2p_channels:
            try:
                send_start = time.perf_counter()
                await self.p2p_channels[node_id].send_message(inner_bytes)
                send_ms = (time.perf_counter() - send_start) * 1000.0
                self._p2p_send_count += 1
            except Exception:
                logger.warning("P2P send to %s failed, falling back to relay", node_id[:12])
                self.p2p_connected.discard(node_id)
                via = "relay"
                self.envelope_count += 1
                env = make_envelope(
                    self.stream_id, inner_bytes,
                    target_node_id=node_id,
                    envelope_count=self.envelope_count,
                )
                try:
                    send_start = time.perf_counter()
                    await self.relay_ws.send(encode_message(env))
                    send_ms = (time.perf_counter() - send_start) * 1000.0
                    self._relay_send_count += 1
                except Exception:
                    self._waiters.pop(seq, None)
                    raise
        else:
            via = "relay"
            self.envelope_count += 1
            env = make_envelope(
                self.stream_id, inner_bytes,
                target_node_id=node_id,
                envelope_count=self.envelope_count,
            )
            try:
                send_start = time.perf_counter()
                await self.relay_ws.send(encode_message(env))
                send_ms = (time.perf_counter() - send_start) * 1000.0
                self._relay_send_count += 1
            except Exception:
                self._waiters.pop(seq, None)
                raise

        wait_start = time.perf_counter()
        try:
            result = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._waiters.pop(seq, None)
            raise RuntimeError(
                f"Pipeline node {node_id[:10]}… did not respond within "
                f"{timeout}s (seq_pos={seq})"
            )
        wait_ms = (time.perf_counter() - wait_start) * 1000.0

        rtt_ms = (time.perf_counter() - rtt_start) * 1000.0
        fwd = result.get("forward_ms")
        fwd_ms = float(fwd) if isinstance(fwd, (int, float)) and fwd > 0 else 0.0
        queue = result.get("queue_ms")
        queue_ms = float(queue) if isinstance(queue, (int, float)) and queue > 0 else 0.0

        if stage_idx is not None:
            self._stage_times.setdefault(stage_idx, []).append(rtt_ms)
            if fwd_ms > 0:
                self._stage_forward_times.setdefault(stage_idx, []).append(fwd_ms)

        resp_type = result.get("type", "unknown")
        resp_data = result.get("hidden_states_bytes") or result.get("data", b"")
        resp_bytes = len(resp_data) if isinstance(resp_data, (bytes, bytearray)) else 0

        trace_entry = {
            "seq": seq,
            "stage": stage_idx,
            "node": node_id[:12],
            "via": via,
            "serialize_ms": round(serialize_ms, 2),
            "send_ms": round(send_ms, 2),
            "wait_ms": round(wait_ms, 2),
            "forward_ms": round(fwd_ms, 2),
            "queue_ms": round(queue_ms, 2),
            "rtt_ms": round(rtt_ms, 2),
            "request_bytes": len(inner_bytes),
            "response_type": resp_type,
            "response_bytes": resp_bytes,
        }
        if resp_type == TOKEN_RESULT:
            trace_entry["node_sampled"] = True
            s_ms = result.get("sample_ms")
            if s_ms is not None:
                trace_entry["sample_ms"] = round(float(s_ms), 2)
        node_telem = result.get("node_telemetry")
        if node_telem:
            trace_entry["node_telemetry"] = node_telem
        self._token_traces.append(trace_entry)

        if self._live_trace_file:
            try:
                self._live_trace_file.write(json.dumps(trace_entry) + "\n")
                self._live_trace_file.flush()
            except Exception:
                pass

        return result

    async def send_to_pipeline(self, message: dict) -> dict:
        """Forward a message through every node in pipeline order.

        For single-node pipelines, sends directly. For multi-node, uses
        serial forwarding (pipeline parallelism is handled at the generation
        loop level by send_to_pipeline_parallel).

        On NODE_GONE or timeout, retries by re-establishing the session
        up to max_retries times.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                msg = message
                last_response: dict = {}
                for node in self.pipeline:
                    response = await self._send_to_node(node["node_id"], msg)
                    t = response.get("type")
                    if t in (LOGITS, TOKEN_RESULT, VERIFY_RESULT, ERROR):
                        return response
                    if t == ACTIVATIONS:
                        msg = response
                        continue
                    return response
                return last_response
            except RuntimeError as e:
                is_node_gone = self._node_gone
                is_timeout = "did not respond within" in str(e)
                if not (is_node_gone or is_timeout):
                    raise
                if attempt >= self.max_retries:
                    raise
                reason = "NODE_GONE" if is_node_gone else "TIMEOUT"
                self.emit_event({
                    "type": "retry",
                    "reason": reason,
                    "attempt": attempt,
                })
                logger.warning(
                    "%s — retrying session (%d/%d)",
                    reason, attempt, self.max_retries,
                )
                self._node_gone = False
                model = self.model_name
                await self.close_session()
                self._closed = False
                await self.connect()
                await self.start_session(model)
        return {}

    async def _send_through_stages(self, message: dict) -> dict:
        """Send a single message through all pipeline stages serially."""
        msg = message
        for idx, node in enumerate(self.pipeline):
            response = await self._send_to_node(node["node_id"], msg, stage_idx=idx)
            t = response.get("type")
            if t in (LOGITS, TOKEN_RESULT, VERIFY_RESULT, ERROR):
                return response
            if t == ACTIVATIONS:
                msg = response
                continue
            return response
        return {}

    async def send_to_pipeline_parallel(
        self,
        messages: AsyncGenerator[dict, None],
    ) -> AsyncGenerator[dict, None]:
        """Pipeline-parallel generation: overlap stages across tokens.

        Each pipeline stage has an asyncio.Queue. Stage workers pull from
        their input queue, send to their node, and push results to the
        next stage's queue. After a warmup period equal to len(pipeline)-1
        tokens, throughput approaches single-node latency.

        Yields final-stage responses (LOGITS) in order.
        """
        num_stages = len(self.pipeline)
        if num_stages <= 1:
            async for msg in messages:
                yield await self.send_to_pipeline(msg)
            return

        _SENTINEL = object()
        queues: list[asyncio.Queue] = [asyncio.Queue() for _ in range(num_stages + 1)]
        errors: list[Exception] = []

        async def stage_worker(stage_idx: int) -> None:
            node = self.pipeline[stage_idx]
            in_q = queues[stage_idx]
            out_q = queues[stage_idx + 1]
            try:
                while True:
                    item = await in_q.get()
                    if item is _SENTINEL:
                        await out_q.put(_SENTINEL)
                        return
                    msg, token_idx = item
                    try:
                        response = await self._send_to_node(node["node_id"], msg)
                    except Exception as exc:
                        errors.append(exc)
                        await out_q.put(_SENTINEL)
                        return
                    t = response.get("type")
                    if t in (LOGITS, VERIFY_RESULT, ERROR):
                        await out_q.put((response, token_idx))
                    elif t == ACTIVATIONS:
                        await out_q.put((response, token_idx))
                    else:
                        await out_q.put((response, token_idx))
            except Exception as exc:
                errors.append(exc)
                await out_q.put(_SENTINEL)

        workers = [
            asyncio.create_task(stage_worker(i)) for i in range(num_stages)
        ]

        async def feed_input() -> None:
            token_idx = 0
            async for msg in messages:
                await queues[0].put((msg, token_idx))
                token_idx += 1
            await queues[0].put(_SENTINEL)

        feeder = asyncio.create_task(feed_input())

        try:
            out_q = queues[num_stages]
            while True:
                item = await out_q.get()
                if item is _SENTINEL:
                    break
                response, token_idx = item
                if errors:
                    raise errors[0]
                yield response
        finally:
            feeder.cancel()
            for w in workers:
                w.cancel()
            for w in workers:
                try:
                    await w
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                await feeder
            except (asyncio.CancelledError, Exception):
                pass

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        use_speculative: bool | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = "You are a helpful assistant.",
        raw_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        if not self.session_id or not self.tokenizer:
            raise RuntimeError("No active session. Call start_session() first.")

        if use_speculative is None:
            use_speculative = len(self.pipeline) == 1

        if raw_mode:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        elif hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0].tolist()

        if use_speculative:
            from src.consumer.speculative import SpeculativeDecoder
            from src.consumer.draft_model import DraftModel

            draft = DraftModel(device="cpu")
            await asyncio.to_thread(draft.load)
            decoder = SpeculativeDecoder(draft_model=draft, client=self, window_size=8)
            async for token_batch in decoder.full_generate(
                input_ids, max_tokens, temperature=temperature, top_p=top_p
            ):
                text = self.tokenizer.decode(token_batch, skip_special_tokens=True)
                self.tokens_generated += 1
                self.emit_event({
                    "type": "token",
                    "text": text,
                    "tokens_generated": self.tokens_generated,
                })
                yield text
        else:
            async for token_text in self._autoregressive_generate(
                input_ids, max_tokens, temperature, top_p
            ):
                self.tokens_generated += 1
                elapsed_s = time.perf_counter() - self._generation_start
                event = {
                    "type": "token",
                    "text": token_text,
                    "tokens_generated": self.tokens_generated,
                    "tps": round(self.tokens_generated / elapsed_s, 2) if elapsed_s > 0 else 0,
                    **self._last_token_timing,
                }
                if self.tokens_generated == 1:
                    event["ttft_ms"] = round(self._ttft_ms, 2)
                self.emit_event(event)
                yield token_text

    async def _prefill_pipelined(
        self, input_ids: list[int], chunk_size: int = 128,
    ) -> dict:
        """Pipeline the prompt prefill across nodes by chunking.

        Sends prompt chunks so that while Node1 processes chunk K's
        activations, Node0 can already process chunk K+1's embeddings.
        Falls back to serial for single-node or short prompts.
        """
        num_nodes = len(self.pipeline)
        prompt_tensor = torch.tensor(input_ids, dtype=torch.int64)
        prompt_len = len(input_ids)

        if num_nodes <= 1 or prompt_len <= chunk_size:
            msg = make_activations(
                self.session_id,
                seq_pos=0,
                hidden_states_bytes=serialize_tensor(prompt_tensor),
                shape=tuple(prompt_tensor.shape),
                dtype="int64",
            )
            msg["is_prompt"] = True
            return await self.send_to_pipeline(msg)

        chunks = []
        for start in range(0, prompt_len, chunk_size):
            end = min(start + chunk_size, prompt_len)
            chunks.append(input_ids[start:end])

        last_response = None
        pending_tasks: list[asyncio.Task] = []

        for chunk_idx, chunk in enumerate(chunks):
            chunk_tensor = torch.tensor(chunk, dtype=torch.int64)
            msg = make_activations(
                self.session_id,
                seq_pos=chunk_idx * chunk_size,
                hidden_states_bytes=serialize_tensor(chunk_tensor),
                shape=tuple(chunk_tensor.shape),
                dtype="int64",
            )
            msg["is_prompt"] = True

            if pending_tasks:
                done_response = await pending_tasks.pop(0)
                if done_response.get("type") == ERROR:
                    for t in pending_tasks:
                        t.cancel()
                    return done_response

            task = asyncio.create_task(self._send_through_stages(msg))
            pending_tasks.append(task)

        for task in pending_tasks:
            last_response = await task
            if last_response.get("type") == ERROR:
                return last_response

        return last_response

    def _finalize_timing(self) -> dict:
        """Build a JSON timing breakdown from collected stage times."""
        stats: dict = {}
        total_rtt_ms = 0.0
        total_compute_ms = 0.0
        for stage_idx, times in sorted(self._stage_times.items()):
            avg = sum(times) / len(times) if times else 0.0
            stats[f"stage_{stage_idx}_avg_ms"] = round(avg, 2)
            stats[f"stage_{stage_idx}_count"] = len(times)
            total_rtt_ms += sum(times)
            fwd_times = self._stage_forward_times.get(stage_idx, [])
            total_compute_ms += sum(fwd_times)
        stats["total_network_ms"] = round(total_rtt_ms - total_compute_ms, 2)
        stats["total_compute_ms"] = round(total_compute_ms, 2)
        stats["ttft_ms"] = round(self._ttft_ms, 2)
        gen_elapsed = (time.perf_counter() - self._generation_start) * 1000.0 if self._generation_start else 0
        stats["tokens_generated"] = self.tokens_generated
        if gen_elapsed > 0 and self.tokens_generated > 0:
            stats["tps"] = round(self.tokens_generated / (gen_elapsed / 1000.0), 2)
        else:
            stats["tps"] = 0.0
        stats["p2p_sends"] = self._p2p_send_count
        stats["relay_sends"] = self._relay_send_count
        self.timing_stats = stats

        if self._token_traces:
            self._write_trace_file(stats)

        return stats

    def _open_live_trace(self) -> None:
        """Open a trace file for live streaming. Hops are appended in real time."""
        import os
        if self._live_trace_file:
            try:
                self._live_trace_file.close()
            except Exception:
                pass
        trace_dir = os.path.expanduser("~/.groove/traces")
        os.makedirs(trace_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._live_trace_path = os.path.join(trace_dir, f"trace_{ts}.jsonl")
        try:
            self._live_trace_file = open(self._live_trace_path, "w")
        except Exception:
            self._live_trace_file = None

    def _write_trace_file(self, summary: dict) -> None:
        """Finalize trace: rewrite file with summary header + all hops."""
        if self._live_trace_file:
            try:
                self._live_trace_file.close()
            except Exception:
                pass
            self._live_trace_file = None

        pipeline_meta = []
        for node in self.pipeline:
            pipeline_meta.append({
                "node_id": node.get("node_id", "")[:12],
                "device": node.get("device"),
                "layers": [node.get("layer_start"), node.get("layer_end")],
                "gpu_model": node.get("gpu_model", ""),
                "score": node.get("score"),
            })

        node_resources = {}
        for t in self._token_traces:
            telem = t.get("node_telemetry")
            if telem and telem.get("node_id"):
                node_resources[telem["node_id"]] = telem

        summary["pipeline"] = pipeline_meta
        summary["node_count"] = len(self.pipeline)
        summary["model"] = self.model_name
        summary["node_resources"] = list(node_resources.values())

        path = self._live_trace_path
        if not path:
            import os
            trace_dir = os.path.expanduser("~/.groove/traces")
            os.makedirs(trace_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(trace_dir, f"trace_{ts}.jsonl")

        try:
            with open(path, "w") as f:
                f.write(json.dumps(summary) + "\n")
                for t in self._token_traces:
                    f.write(json.dumps(t) + "\n")
            logger.info("trace written to %s (%d hops)", path, len(self._token_traces))
        except Exception:
            logger.warning("failed to write trace file")

    async def _autoregressive_generate(
        self,
        input_ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncGenerator[str, None]:
        generated = list(input_ids)
        eos_id = self.tokenizer.eos_token_id
        stop_ids = {eos_id}
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != self.tokenizer.unk_token_id:
            stop_ids.add(im_end_id)
        use_pipeline = len(self.pipeline) > 1

        self._generation_start = time.perf_counter()
        self._stage_times.clear()
        self._stage_forward_times.clear()
        self._token_traces.clear()
        self._first_token_emitted = False
        self._open_live_trace()

        trace_idx = len(self._token_traces)
        prefill_start = time.perf_counter()
        response = await self._prefill_pipelined(input_ids)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        if response.get("type") == ERROR:
            raise RuntimeError(f"Pipeline error: {response.get('message', '')}")

        logits_deser_ms = 0.0
        sample_ms = 0.0
        if response.get("type") == TOKEN_RESULT:
            next_token = response["token_id"]
            sample_ms = response.get("sample_ms", 0.0)
        else:
            logits_start = time.perf_counter()
            logits = _logits_from_response(response)
            logits_deser_ms = (time.perf_counter() - logits_start) * 1000.0

            sample_start = time.perf_counter()
            next_token = self._sample_token(
                _last_token_logits(logits), temperature, top_p
            )
            sample_ms = (time.perf_counter() - sample_start) * 1000.0

        generated.append(next_token)
        self._ttft_ms = (time.perf_counter() - self._generation_start) * 1000.0
        self._first_token_emitted = True

        decode_start = time.perf_counter()
        text = self.tokenizer.decode([next_token], skip_special_tokens=True)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        self._last_token_timing = {
            "token_ms": round(prefill_ms + logits_deser_ms + sample_ms + decode_ms, 2),
            "prefill_ms": round(prefill_ms, 2),
            "logits_deser_ms": round(logits_deser_ms, 2),
            "sample_ms": round(sample_ms, 2),
            "decode_ms": round(decode_ms, 2),
            "is_prefill": True,
            "node_sampled": response.get("type") == TOKEN_RESULT,
            "stages": self._token_traces[trace_idx:],
        }
        yield text

        if use_pipeline:
            async for text in self._pipelined_token_generate(
                generated, next_token, max_tokens - 1, temperature, top_p
            ):
                yield text
        else:
            for _ in range(max_tokens - 1):
                if next_token in stop_ids:
                    break

                trace_idx = len(self._token_traces)
                token_start = time.perf_counter()

                token_tensor = torch.tensor([next_token], dtype=torch.int64)
                msg = make_activations(
                    self.session_id,
                    seq_pos=len(generated) - 1,
                    hidden_states_bytes=serialize_tensor(token_tensor),
                    shape=tuple(token_tensor.shape),
                    dtype="int64",
                )
                msg["is_prompt"] = False

                pipeline_start = time.perf_counter()
                response = await self.send_to_pipeline(msg)
                pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0
                if response.get("type") == ERROR:
                    raise RuntimeError(f"Pipeline error: {response.get('message', '')}")

                logits_deser_ms = 0.0
                sample_ms = 0.0
                if response.get("type") == TOKEN_RESULT:
                    next_token = response["token_id"]
                    sample_ms = response.get("sample_ms", 0.0)
                else:
                    logits_start = time.perf_counter()
                    logits = _logits_from_response(response)
                    logits_deser_ms = (time.perf_counter() - logits_start) * 1000.0

                    sample_start = time.perf_counter()
                    next_token = self._sample_token(
                        _last_token_logits(logits), temperature, top_p
                    )
                    sample_ms = (time.perf_counter() - sample_start) * 1000.0

                generated.append(next_token)

                decode_start = time.perf_counter()
                text = self.tokenizer.decode([next_token], skip_special_tokens=True)
                decode_ms = (time.perf_counter() - decode_start) * 1000.0

                token_ms = (time.perf_counter() - token_start) * 1000.0
                self._last_token_timing = {
                    "token_ms": round(token_ms, 2),
                    "pipeline_ms": round(pipeline_ms, 2),
                    "logits_deser_ms": round(logits_deser_ms, 2),
                    "sample_ms": round(sample_ms, 2),
                    "decode_ms": round(decode_ms, 2),
                    "is_prefill": False,
                    "node_sampled": response.get("type") == TOKEN_RESULT,
                    "stages": self._token_traces[trace_idx:],
                }
                yield text

        timing = self._finalize_timing()
        logger.info("generation timing: %s", json.dumps(timing))
        self.emit_event({"type": "timing", **timing})

    async def _pipelined_token_generate(
        self,
        generated: list[int],
        first_token: int,
        remaining_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncGenerator[str, None]:
        """Token generation for multi-node pipelines using stage queues.

        Uses the stage-queue architecture: each pipeline stage has its own
        asyncio queue. Stage workers process items concurrently, so when
        multiple tokens are in flight (e.g. via speculative candidates fed
        into the pipeline), stages overlap. For pure autoregressive mode,
        tokens are sequential but still benefit from the direct stage-to-stage
        handoff without returning to the consumer between stages.
        """
        eos_id = self.tokenizer.eos_token_id
        stop_ids = {eos_id}
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != self.tokenizer.unk_token_id:
            stop_ids.add(im_end_id)
        next_token = first_token
        num_stages = len(self.pipeline)

        _SENTINEL = object()
        stage_queues: list[asyncio.Queue] = [
            asyncio.Queue(maxsize=num_stages + 1) for _ in range(num_stages + 1)
        ]
        stage_errors: list[Exception] = []

        async def stage_worker(idx: int) -> None:
            node = self.pipeline[idx]
            in_q = stage_queues[idx]
            out_q = stage_queues[idx + 1]
            try:
                while True:
                    item = await in_q.get()
                    if item is _SENTINEL:
                        await out_q.put(_SENTINEL)
                        return
                    msg, seq_idx = item
                    response = await self._send_to_node(
                        node["node_id"], msg, stage_idx=idx,
                    )
                    t = response.get("type")
                    if t in (LOGITS, TOKEN_RESULT, VERIFY_RESULT, ERROR):
                        await out_q.put((response, seq_idx))
                    elif t == ACTIVATIONS:
                        await out_q.put((response, seq_idx))
                    else:
                        await out_q.put((response, seq_idx))
            except Exception as exc:
                stage_errors.append(exc)
                await out_q.put(_SENTINEL)

        workers = [asyncio.create_task(stage_worker(i)) for i in range(num_stages)]

        try:
            for _ in range(remaining_tokens):
                if next_token in stop_ids:
                    break

                trace_idx = len(self._token_traces)
                token_start = time.perf_counter()

                token_tensor = torch.tensor([next_token], dtype=torch.int64)
                msg = make_activations(
                    self.session_id,
                    seq_pos=len(generated) - 1,
                    hidden_states_bytes=serialize_tensor(token_tensor),
                    shape=tuple(token_tensor.shape),
                    dtype="int64",
                )
                msg["is_prompt"] = False

                await stage_queues[0].put((msg, len(generated) - 1))

                item = await stage_queues[num_stages].get()
                if item is _SENTINEL or stage_errors:
                    if stage_errors:
                        raise stage_errors[0]
                    raise RuntimeError("Pipeline stage terminated unexpectedly")
                response, seq_idx = item
                pipeline_ms = (time.perf_counter() - token_start) * 1000.0

                if response.get("type") == ERROR:
                    raise RuntimeError(f"Pipeline error: {response.get('message', '')}")

                logits_deser_ms = 0.0
                sample_ms = 0.0
                if response.get("type") == TOKEN_RESULT:
                    next_token = response["token_id"]
                    sample_ms = response.get("sample_ms", 0.0)
                else:
                    logits_start = time.perf_counter()
                    logits = _logits_from_response(response)
                    logits_deser_ms = (time.perf_counter() - logits_start) * 1000.0

                    sample_start = time.perf_counter()
                    next_token = self._sample_token(
                        _last_token_logits(logits), temperature, top_p
                    )
                    sample_ms = (time.perf_counter() - sample_start) * 1000.0

                generated.append(next_token)

                decode_start = time.perf_counter()
                text = self.tokenizer.decode([next_token], skip_special_tokens=True)
                decode_ms = (time.perf_counter() - decode_start) * 1000.0

                token_ms = (time.perf_counter() - token_start) * 1000.0
                self._last_token_timing = {
                    "token_ms": round(token_ms, 2),
                    "pipeline_ms": round(pipeline_ms, 2),
                    "logits_deser_ms": round(logits_deser_ms, 2),
                    "sample_ms": round(sample_ms, 2),
                    "decode_ms": round(decode_ms, 2),
                    "is_prefill": False,
                    "node_sampled": response.get("type") == TOKEN_RESULT,
                    "stages": self._token_traces[trace_idx:],
                }
                yield text
        finally:
            await stage_queues[0].put(_SENTINEL)
            for w in workers:
                w.cancel()
            for w in workers:
                try:
                    await w
                except (asyncio.CancelledError, Exception):
                    pass

    def _sample_token(
        self, logits: np.ndarray, temperature: float, top_p: float
    ) -> int:
        if temperature <= 0:
            return int(np.argmax(logits))

        logits = logits.astype(np.float64)
        logits /= temperature
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)

        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)

        cutoff_idx = int(np.searchsorted(cumulative, top_p)) + 1
        top_indices = sorted_indices[:cutoff_idx]
        top_probs = probs[top_indices]
        top_probs /= np.sum(top_probs)

        return int(np.random.choice(top_indices, p=top_probs))

    async def close_session(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._p2p_monitor_task is not None:
            self._p2p_monitor_task.cancel()
            try:
                await self._p2p_monitor_task
            except (asyncio.CancelledError, Exception):
                pass
            self._p2p_monitor_task = None
        logger.info(
            "session transport stats",
            extra={
                "p2p_sends": self._p2p_send_count,
                "relay_sends": self._relay_send_count,
                "p2p_reconnects": self._reconnect_count,
            },
        )
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None
        if self.relay_ws:
            try:
                await self.relay_ws.close()
            except Exception:
                pass
            self.relay_ws = None
        if self.peer_manager is not None:
            try:
                await self.peer_manager.close_all()
            except Exception:
                pass
            self.peer_manager = None
        self.p2p_channels.clear()
        self.p2p_connected.clear()
        self.session_id = None
        self.stream_id = None
        self.pipeline = []


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Groove Decentralized Inference Client"
    )
    conn_group = parser.add_mutually_exclusive_group(required=True)
    conn_group.add_argument(
        "--signal", type=str,
        help="Signal service hostname (e.g. signal.groovedev.ai). Uses wss:// automatically.",
    )
    conn_group.add_argument(
        "--relay", type=str,
        help="Legacy relay host:port (use --signal for production)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B", help="Model name"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=200)
    spec_group = parser.add_mutually_exclusive_group()
    spec_group.add_argument(
        "--speculative", action="store_true", default=None,
        help="Force speculative decoding on (default: auto, on for multi-node)",
    )
    spec_group.add_argument(
        "--no-speculative", action="store_true",
        help="Force speculative decoding off",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--json", action="store_true",
        help="Emit structured JSON events (one per line) on stdout",
    )
    parser.add_argument("--tls", action="store_true", help="Use wss:// (automatic with --signal)")
    args = parser.parse_args()

    if args.signal:
        host = args.signal
        port = 443
        use_tls = True
        signal_mode = True
    else:
        host, port_str = args.relay.rsplit(":", 1)
        port = int(port_str)
        use_tls = args.tls
        signal_mode = False
    client = InferenceClient(
        relay_host=host, relay_port=port, json_mode=args.json,
        use_tls=use_tls, signal_mode=signal_mode,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    start_ms = time.time() * 1000.0

    connect_label = f"signal={args.signal}" if args.signal else f"relay={args.relay}"

    try:
        try:
            await client.connect()
        except (OSError, websockets.WebSocketException) as e:
            client.emit_event({
                "type": "error",
                "message": f"Failed to connect to {connect_label}: {e}",
                "code": "CONNECTION_FAILED",
            })
            logger.error("connect failed: %s", e)
            sys.exit(1)
        logger.info("connected to %s (protocol v%d)", connect_label, PROTOCOL_VERSION)

        session_id = await client.start_session(args.model)
        logger.info(
            "session=%s stream=%s pipeline=%d nodes",
            session_id, client.stream_id, len(client.pipeline),
        )

        if not args.json:
            sys.stdout.write("Output: ")
            sys.stdout.flush()

        if args.no_speculative:
            spec_flag = False
        elif args.speculative:
            spec_flag = True
        else:
            spec_flag = None

        async for text in client.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            use_speculative=spec_flag,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            if not args.json:
                sys.stdout.write(text)
                sys.stdout.flush()

        if not args.json:
            sys.stdout.write("\n")
        client.emit_event({
            "type": "done",
            "tokens_generated": client.tokens_generated,
            "total_time_ms": int(time.time() * 1000.0 - start_ms),
        })
        logger.info("generation complete")
    except asyncio.TimeoutError as e:
        client.emit_event({
            "type": "error",
            "message": f"Operation timed out: {e}",
            "code": "TIMEOUT",
        })
        logger.error("inference timed out: %s", e)
        sys.exit(1)
    except RuntimeError as e:
        msg = str(e)
        low = msg.lower()
        if "coverage" in low:
            code = "COVERAGE_INCOMPLETE"
        elif "no nodes" in low or "no active compute" in low:
            code = "NO_NODES"
        elif "lost node" in low or "node_gone" in low or "pipeline" in low:
            code = "PIPELINE_BROKEN"
        elif "node error" in low or "no shard" in low:
            code = "NODE_ERROR"
        else:
            code = "SESSION_ERROR"
        client.emit_event({"type": "error", "message": msg, "code": code})
        logger.error("inference failed: %s", e)
        sys.exit(1)
    except ConnectionError as e:
        client.emit_event({
            "type": "error",
            "message": str(e),
            "code": "CONNECTION_FAILED",
        })
        logger.error("connection lost: %s", e)
        sys.exit(1)
    except Exception as e:
        client.emit_event({
            "type": "error",
            "message": str(e),
            "code": "SESSION_ERROR",
        })
        logger.error("inference failed: %s", e)
        sys.exit(1)
    finally:
        await client.close_session()


if __name__ == "__main__":
    asyncio.run(main())
