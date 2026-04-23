"""Groove Relay - the only public endpoint.

The relay terminates two kinds of websocket connections:
  - nodes: a persistent outbound socket from each compute node, identified
           by an opening REGISTER_NODE frame.
  - consumers: a single socket per inference session, opened with a
               SESSION_INIT frame; the relay mints a stream_id and
               returns a PIPELINE_CONFIG referencing node_ids only.

After the opening handshake every consumer<->node frame is wrapped in an
ENVELOPE addressed by stream_id (and target_node_id when consumer→node).
The relay is a pure router; it does not decode envelope payloads.
"""

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import ssl
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from websockets.asyncio.server import serve, ServerConnection
from websockets.exceptions import ConnectionClosed

from src.common.protocol import (
    ASSIGN_LAYERS,
    ASSIGNMENT_ACK,
    AUTH_CHALLENGE,
    AUTH_RESPONSE,
    DEREGISTER,
    ENVELOPE,
    ERROR,
    HEARTBEAT,
    ICE_CANDIDATE,
    KV_TRIM,
    MESH_BROKEN,
    MESH_READY,
    P2P_READY,
    PIPELINE_CONFIG,
    PIPELINE_MESH,
    PROTOCOL_VERSION,
    REBALANCE,
    REGISTER_NODE,
    SDP_ANSWER,
    SDP_OFFER,
    SESSION_INIT,
    decode_message,
    encode_message,
    make_assign_layers,
    make_auth_challenge,
    make_error,
    make_pipeline_config,
    make_rebalance,
    make_register_ack,
    normalize_capabilities,
)
from src.node.identity import address_from_public_key, verify_signature
from src.relay.scheduler import (
    MODEL_REGISTRY,
    assign_layers,
    calculate_rebalance,
    get_model_info,
    minimize_hops_assign,
    validate_coverage,
)


_RESERVED_LOG_FIELDS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
})


class JsonFormatter(logging.Formatter):
    """One JSON object per log record. No external deps."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k in _RESERVED_LOG_FIELDS:
                continue
            payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


logger = logging.getLogger("relay")


@dataclass
class NodeEntry:
    node_id: str
    ws: ServerConnection
    layer_start: int | None
    layer_end: int | None
    capabilities: dict = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "active"
    assigned_model: str = ""
    # 'pending' | 'loading' | 'active' | 'rebalancing'
    assignment_status: str = "pending"
    # Set when an ASSIGN_LAYERS / REBALANCE has been sent and the relay is
    # awaiting an ASSIGNMENT_ACK. Cleared on ack receipt.
    pending_ack: asyncio.Future | None = field(default=None, repr=False)
    pending_range: tuple[int, int] | None = None


@dataclass
class ConsumerEntry:
    stream_id: str
    ws: ServerConnection
    session_id: str
    pipeline: list[str]
    model_name: str = ""
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    # M4 prep: per-node envelope counters for this session.
    envelope_counts: dict[str, int] = field(default_factory=dict)
    last_envelope_count: dict[str, int] = field(default_factory=dict)
    # P2P connection metrics.
    p2p_established: set[str] = field(default_factory=set)
    p2p_failed: set[str] = field(default_factory=set)
    connection_start_time: float = field(default_factory=time.time)


class _RateLimiter:
    """Simple token-bucket rate limiter."""

    __slots__ = ("_rate", "_burst", "_tokens", "_last")

    def __init__(self, rate: float = 100.0, burst: int = 200):
        self._rate = rate
        self._burst = float(burst)
        self._tokens = float(burst)
        self._last = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self._tokens = min(
            self._burst, self._tokens + (now - self._last) * self._rate
        )
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


def validate_layer_coverage(nodes: list[dict], total_layers: int) -> bool:
    """Check that nodes cover layers [0, total_layers) contiguously."""
    if not nodes:
        return False
    sorted_nodes = sorted(nodes, key=lambda n: n["layer_start"])
    if sorted_nodes[0]["layer_start"] != 0:
        return False
    for i in range(len(sorted_nodes) - 1):
        if sorted_nodes[i]["layer_end"] + 1 != sorted_nodes[i + 1]["layer_start"]:
            return False
    if sorted_nodes[-1]["layer_end"] != total_layers - 1:
        return False
    return True


def get_node_for_layer(nodes: list[dict], layer_idx: int) -> dict | None:
    """Return the node dict responsible for a given layer index."""
    for node in nodes:
        if node["layer_start"] <= layer_idx <= node["layer_end"]:
            return node
    return None


class RelayNode:
    # How long to wait for ASSIGNMENT_ACK after sending ASSIGN_LAYERS.
    # Generous because the node may be downloading the shard from HF.
    ASSIGNMENT_TIMEOUT_S = 120.0

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8770,
        http_port: int = 8771,
        require_auth: bool = True,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        cors_origin: str = "*",
        max_connections_per_ip: int = 20,
        max_nodes: int = 1000,
        max_streams: int = 5000,
        max_message_size: int = 10 * 1024 * 1024,
        session_idle_timeout: float = 300.0,
        turn_secret: str | None = None,
    ):
        self.host = host
        self.port = port
        self.http_port = http_port
        self.require_auth = require_auth
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self.cors_origin = cors_origin
        self.max_connections_per_ip = max_connections_per_ip
        self.max_nodes = max_nodes
        self.max_streams = max_streams
        self.max_message_size = max_message_size
        self.session_idle_timeout = session_idle_timeout
        self._turn_secret = turn_secret or os.environ.get("GROOVE_TURN_SECRET", "")
        self.nodes: dict[str, NodeEntry] = {}
        self.streams: dict[str, ConsumerEntry] = {}
        self.heartbeat_timeout = 120.0
        self.start_time: float = time.time()
        self._sched_lock = asyncio.Lock()
        self._connections_per_ip: dict[str, int] = {}
        self._node_streams: dict[str, set[str]] = {}

    def _generate_turn_credentials(self, username: str, ttl: int = 86400) -> dict:
        """Generate HMAC-based time-limited TURN credentials (coturn shared-secret model)."""
        expiry = int(time.time()) + ttl
        turn_username = f"{expiry}:{username}"
        turn_password = base64.b64encode(
            hmac.new(
                self._turn_secret.encode(),
                turn_username.encode(),
                hashlib.sha1,
            ).digest()
        ).decode()
        return {
            "urls": ["turn:turn.groovedev.ai:3478"],
            "username": turn_username,
            "credential": turn_password,
        }

    def _get_turn_servers(self, username: str) -> list[dict] | None:
        """Return TURN server list if secret is configured, else None."""
        if not self._turn_secret:
            return None
        return [self._generate_turn_credentials(username)]

    def _active_assignments_for_model(self, model_name: str) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for nid, entry in self.nodes.items():
            if (
                entry.assignment_status == "active"
                and entry.assigned_model == model_name
                and entry.layer_start is not None
                and entry.layer_end is not None
            ):
                out[nid] = (entry.layer_start, entry.layer_end)
        return out

    def _find_consumer_by_session(self, session_id: str) -> ConsumerEntry | None:
        """Look up a consumer entry by session_id."""
        for ce in self.streams.values():
            if ce.session_id == session_id:
                return ce
        return None

    def assemble_pipeline(self, model_name: str, session_id: str) -> list[NodeEntry]:
        active = [
            n for n in self.nodes.values()
            if n.status == "active"
            and n.assignment_status == "active"
            and n.layer_start is not None
            and n.layer_end is not None
            and (not model_name or not n.assigned_model or n.assigned_model == model_name)
        ]
        if not active:
            raise RuntimeError("No active compute nodes available")

        if model_name in MODEL_REGISTRY:
            assignments = {n.node_id: (n.layer_start, n.layer_end) for n in active}
            if not validate_coverage(assignments, model_name):
                raise RuntimeError(
                    f"Layer coverage incomplete for {model_name!r}: "
                    f"{sorted(assignments.values())}"
                )
        active.sort(key=lambda n: n.layer_start)
        return active

    async def _safe_send(self, ws: ServerConnection, data: bytes) -> bool:
        try:
            await ws.send(data)
            return True
        except (ConnectionClosed, OSError) as e:
            logger.warning("ws send failed", extra={"err": str(e)})
            return False

    async def _send_error(self, ws: ServerConnection, session_id: str, code: str, message: str) -> None:
        try:
            await ws.send(encode_message(make_error(session_id, code, message)))
        except (ConnectionClosed, OSError):
            pass

    async def _forward_signaling(self, from_node_id: str, msg: dict) -> None:
        """Forward a WebRTC signaling message from a node to its target."""
        t = msg.get("type")
        session_id = msg.get("session_id")

        claimed = msg.get("from_node_id") or msg.get("node_id")
        if claimed != from_node_id:
            logger.warning(
                "signaling sender mismatch",
                extra={"claimed": claimed, "actual": from_node_id},
            )
            return

        ce = self._find_consumer_by_session(session_id)
        if ce is None:
            logger.warning(
                "signaling for unknown session",
                extra={"type": t, "session_id": session_id, "node_id": from_node_id},
            )
            return

        if from_node_id not in ce.pipeline:
            logger.warning(
                "signaling sender not in pipeline",
                extra={"node_id": from_node_id, "session_id": session_id},
            )
            return

        if t == P2P_READY:
            peer_node_id = msg.get("peer_node_id")
            if peer_node_id:
                ce.p2p_established.add(peer_node_id)
                elapsed = time.time() - ce.connection_start_time
                logger.info(
                    "p2p established",
                    extra={
                        "session_id": session_id,
                        "node_id": from_node_id,
                        "peer_node_id": peer_node_id,
                        "elapsed_s": round(elapsed, 3),
                    },
                )
            await self._safe_send(ce.ws, encode_message(msg))
            return

        if t == SDP_OFFER:
            turn = self._get_turn_servers(from_node_id)
            if turn:
                msg["turn_servers"] = turn

        to_node_id = msg.get("to_node_id")
        target_node = self.nodes.get(to_node_id)
        if target_node is not None:
            if to_node_id not in ce.pipeline:
                logger.warning(
                    "signaling target not in pipeline",
                    extra={"to_node_id": to_node_id, "session_id": session_id},
                )
                return
            await self._safe_send(target_node.ws, encode_message(msg))
            return

        await self._safe_send(ce.ws, encode_message(msg))

    async def _forward_consumer_signaling(self, ce: ConsumerEntry, msg: dict) -> None:
        """Forward a signaling message from a consumer to a pipeline node."""
        to_node_id = msg.get("to_node_id")
        if to_node_id not in ce.pipeline:
            logger.warning(
                "consumer signaling target not in pipeline",
                extra={"stream_id": ce.stream_id, "to_node_id": to_node_id},
            )
            return
        target = self.nodes.get(to_node_id)
        if target is None or target.status != "active":
            await self._send_error(
                ce.ws, ce.session_id, "NODE_UNAVAILABLE",
                "signaling target node not available",
            )
            return
        if msg.get("type") == SDP_OFFER:
            turn = self._get_turn_servers(ce.session_id)
            if turn:
                msg["turn_servers"] = turn
        await self._safe_send(target.ws, encode_message(msg))

    async def _teardown_streams_using_node(self, node_id: str) -> None:
        affected = [sid for sid, ce in self.streams.items() if node_id in ce.pipeline]
        for sid in affected:
            ce = self.streams.pop(sid, None)
            if ce is None:
                continue
            await self._send_error(ce.ws, ce.session_id, "NODE_GONE",
                                   f"Pipeline node {node_id} disconnected")
            try:
                await ce.ws.close()
            except (ConnectionClosed, OSError):
                pass
            logger.info("stream torn down", extra={"stream_id": sid, "reason": "node_gone", "node_id": node_id})

    async def _authenticate_node(
        self, ws: ServerConnection, node_id: str, public_key: str | None,
    ) -> bool:
        """Challenge-response authentication. Returns True on success."""
        if not self.require_auth:
            return True
        if not public_key:
            await self._send_error(ws, "", "AUTH_REQUIRED",
                                   "registration requires public_key for authentication")
            return False
        try:
            derived = address_from_public_key(public_key)
        except (ValueError, Exception):
            await self._send_error(ws, "", "AUTH_FAILED", "invalid public key")
            return False
        if derived.lower() != node_id.lower():
            await self._send_error(ws, "", "AUTH_FAILED",
                                   "public key does not match claimed node_id")
            return False
        nonce = secrets.token_hex(32)
        challenge = make_auth_challenge(node_id, nonce)
        if not await self._safe_send(ws, encode_message(challenge)):
            return False
        try:
            raw_auth = await asyncio.wait_for(ws.recv(), timeout=10.0)
        except (asyncio.TimeoutError, ConnectionClosed):
            return False
        if isinstance(raw_auth, str):
            raw_auth = raw_auth.encode("utf-8")
        try:
            auth_msg = decode_message(raw_auth)
        except Exception:
            await self._send_error(ws, "", "AUTH_FAILED", "invalid auth response")
            return False
        if auth_msg.get("type") != AUTH_RESPONSE:
            await self._send_error(ws, "", "AUTH_FAILED", "expected auth_response")
            return False
        signature = auth_msg.get("signature", "")
        if not verify_signature(nonce.encode("utf-8"), signature, public_key):
            await self._send_error(ws, "", "AUTH_FAILED",
                                   "signature verification failed")
            return False
        logger.info("node authenticated", extra={"node_id": node_id})
        return True

    async def _handle_node(self, ws: ServerConnection, first: dict) -> None:
        node_id = first["node_id"]

        pv = first.get("protocol_version")
        if pv != PROTOCOL_VERSION:
            await self._send_error(
                ws, "", "BAD_PROTOCOL",
                f"Unsupported protocol version {pv!r}; expected {PROTOCOL_VERSION}",
            )
            return

        if len(self.nodes) >= self.max_nodes:
            await self._send_error(ws, "", "CAPACITY", "node limit reached")
            return

        if not await self._authenticate_node(ws, node_id, first.get("public_key")):
            try:
                await ws.close()
            except (ConnectionClosed, OSError):
                pass
            return

        if node_id in self.nodes:
            old = self.nodes.pop(node_id)
            try:
                await old.ws.close()
            except (ConnectionClosed, OSError):
                pass
            logger.info("evicted stale node entry", extra={"node_id": node_id})

        caps = normalize_capabilities(first.get("capabilities"))
        layer_start = first.get("layer_start")
        layer_end = first.get("layer_end")
        loaded_models = caps.get("models_loaded") or []
        legacy_model = loaded_models[0] if loaded_models else ""

        entry = NodeEntry(
            node_id=node_id,
            ws=ws,
            layer_start=layer_start,
            layer_end=layer_end,
            capabilities=caps,
            assigned_model=legacy_model,
        )

        is_legacy = layer_start is not None and layer_end is not None
        if is_legacy:
            entry.assignment_status = "active"
            self.nodes[node_id] = entry
            if not await self._safe_send(
                ws, encode_message(make_register_ack(node_id, True, "ok"))
            ):
                self.nodes.pop(node_id, None)
                return
            logger.info(
                "node registered (legacy/static)",
                extra={
                    "node_id": node_id,
                    "layer_start": layer_start,
                    "layer_end": layer_end,
                },
            )
        else:
            entry.assignment_status = "pending"
            self.nodes[node_id] = entry
            if not await self._safe_send(
                ws, encode_message(make_register_ack(node_id, True, "registered; awaiting assignment"))
            ):
                self.nodes.pop(node_id, None)
                return
            logger.info(
                "node registered (dynamic)",
                extra={"node_id": node_id, "capabilities": caps},
            )
            asyncio.create_task(self._dynamic_assign(entry))

        limiter = _RateLimiter()
        try:
            async for raw in ws:
                if not limiter.allow():
                    logger.warning("rate limit exceeded", extra={"node_id": node_id})
                    continue
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                try:
                    msg = decode_message(raw)
                except Exception:
                    logger.exception("decode failed on node frame", extra={"node_id": node_id})
                    continue

                t = msg.get("type")
                if t == HEARTBEAT:
                    entry.last_heartbeat = time.time()
                    hb_caps = msg.get("capabilities")
                    if isinstance(hb_caps, dict):
                        entry.capabilities = normalize_capabilities(hb_caps)
                    entry.status = msg.get("status", "active")
                elif t == ENVELOPE:
                    sid = msg.get("stream_id")
                    ce = self.streams.get(sid)
                    if ce is None:
                        logger.warning(
                            "envelope for unknown stream dropped",
                            extra={"stream_id": sid, "from_node": node_id},
                        )
                        continue
                    if node_id not in ce.pipeline:
                        logger.warning(
                            "node not in stream pipeline, dropping envelope",
                            extra={"stream_id": sid, "node_id": node_id},
                        )
                        continue
                    ce.envelope_counts[node_id] = ce.envelope_counts.get(node_id, 0) + 1
                    ce.last_activity = time.time()
                    ok = await self._safe_send(ce.ws, encode_message(msg))
                    if not ok:
                        self.streams.pop(sid, None)
                elif t == ASSIGNMENT_ACK:
                    self._handle_assignment_ack(entry, msg)
                elif t in (PIPELINE_MESH, KV_TRIM):
                    sid = msg.get("stream_id") or msg.get("session_id")
                    ce = self.streams.get(sid) if sid else None
                    if ce is not None:
                        await self._safe_send(ce.ws, encode_message(msg))
                elif t in (SDP_OFFER, SDP_ANSWER, ICE_CANDIDATE, P2P_READY,
                          MESH_READY, MESH_BROKEN):
                    await self._forward_signaling(node_id, msg)
                elif t == DEREGISTER:
                    logger.info(
                        "node deregistered",
                        extra={"node_id": node_id, "reason": msg.get("reason", "")},
                    )
                    break
                else:
                    logger.warning(
                        "unexpected node frame type",
                        extra={"node_id": node_id, "type": t},
                    )
        except ConnectionClosed:
            pass
        finally:
            current = self.nodes.get(node_id)
            if current is entry:
                self.nodes.pop(node_id, None)
            self._node_streams.pop(node_id, None)
            await self._teardown_streams_using_node(node_id)
            if entry.assigned_model:
                asyncio.create_task(self._rebalance_after_leave(entry.assigned_model))
            logger.info("node connection closed", extra={"node_id": node_id})

    def _handle_assignment_ack(self, entry: NodeEntry, msg: dict) -> None:
        """Resolve the pending future for an outstanding ASSIGN_LAYERS / REBALANCE."""
        fut = entry.pending_ack
        entry.pending_ack = None
        accepted = bool(msg.get("accepted", False))
        log_extra = {
            "node_id": entry.node_id,
            "model": msg.get("model_name"),
            "layer_start": msg.get("layer_start"),
            "layer_end": msg.get("layer_end"),
            "accepted": accepted,
            "reason": msg.get("reason", ""),
            "load_time_ms": msg.get("load_time_ms", 0),
        }
        if accepted:
            logger.info("assignment accepted", extra=log_extra)
        else:
            logger.warning("assignment rejected", extra=log_extra)
        if fut is not None and not fut.done():
            fut.set_result(msg)

    async def _send_assignment(
        self,
        entry: NodeEntry,
        model_name: str,
        layer_start: int,
        layer_end: int,
        kind: str,
        reason: str = "",
    ) -> dict | None:
        """Send ASSIGN_LAYERS or REBALANCE and wait for ASSIGNMENT_ACK.

        Returns the ack dict on success, None on timeout / send failure /
        rejection.
        """
        try:
            info = get_model_info(model_name)
        except ValueError:
            logger.error("send_assignment: unknown model", extra={"model": model_name})
            return None

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        entry.pending_ack = fut
        entry.pending_range = (layer_start, layer_end)
        entry.assignment_status = "rebalancing" if kind == "rebalance" else "loading"

        if kind == "rebalance":
            frame = make_rebalance(
                entry.node_id, model_name, layer_start, layer_end, reason=reason,
            )
        else:
            frame = make_assign_layers(
                entry.node_id, model_name, layer_start, layer_end,
                total_layers=info["total_layers"],
                hidden_size=info["hidden_size"],
            )

        if not await self._safe_send(entry.ws, encode_message(frame)):
            entry.pending_ack = None
            entry.pending_range = None
            return None

        try:
            ack = await asyncio.wait_for(fut, timeout=self.ASSIGNMENT_TIMEOUT_S)
        except asyncio.TimeoutError:
            entry.pending_ack = None
            entry.pending_range = None
            entry.assignment_status = "pending"
            logger.warning(
                "assignment ack timeout",
                extra={"node_id": entry.node_id, "kind": kind, "model": model_name},
            )
            return None

        if ack.get("accepted"):
            entry.assigned_model = model_name
            entry.layer_start = layer_start
            entry.layer_end = layer_end
            entry.assignment_status = "active"
        else:
            entry.assignment_status = "pending"
            entry.pending_range = None
        return ack

    async def _dynamic_assign(self, entry: NodeEntry) -> None:
        """Compute assignments for a newly joined dynamic node and dispatch."""
        async with self._sched_lock:
            model_name = self._pick_model_for_new_node(entry)
            if not model_name:
                logger.warning(
                    "no model in registry; cannot assign",
                    extra={"node_id": entry.node_id},
                )
                return

            current = self._active_assignments_for_model(model_name)
            all_caps = [
                {"node_id": nid, **self.nodes[nid].capabilities}
                for nid in current
                if nid in self.nodes
            ]
            for nid, node_entry in self.nodes.items():
                if (node_entry.assignment_status == "pending"
                        and nid not in current
                        and node_entry is not entry):
                    all_caps.append({"node_id": nid, **node_entry.capabilities})
            all_caps.append({"node_id": entry.node_id, **entry.capabilities})
            try:
                new_assignments, affected = calculate_rebalance(
                    current, all_caps, model_name,
                )
            except ValueError as e:
                logger.warning(
                    "scheduler refused to assign",
                    extra={"node_id": entry.node_id, "err": str(e)},
                )
                return

            await self._apply_assignments(model_name, new_assignments, affected, reason="join")

    def _pick_model_for_new_node(self, entry: NodeEntry) -> str:
        # Prefer a model that already has active nodes; otherwise cold start
        # with the first registry entry. (M3 will replace this with real
        # multi-model scheduling.)
        for n in self.nodes.values():
            if n is entry:
                continue
            if n.assignment_status == "active" and n.assigned_model in MODEL_REGISTRY:
                return n.assigned_model
        if MODEL_REGISTRY:
            return next(iter(MODEL_REGISTRY))
        return ""

    async def _apply_assignments(
        self,
        model_name: str,
        new_assignments: dict[str, tuple[int, int]],
        affected: list[str],
        reason: str,
    ) -> None:
        """Send ASSIGN_LAYERS / REBALANCE to each affected node, in parallel."""
        tasks = []
        for nid in affected:
            entry = self.nodes.get(nid)
            if entry is None:
                continue
            new = new_assignments.get(nid)
            if new is None:
                # Node no longer in plan — leave it alone (active sessions
                # can continue; a future cycle will tidy up).
                continue
            ls, le = new
            kind = "assign" if entry.assignment_status == "pending" else "rebalance"
            tasks.append(self._send_assignment(entry, model_name, ls, le, kind, reason=reason))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _rebalance_after_leave(self, model_name: str) -> None:
        async with self._sched_lock:
            current = self._active_assignments_for_model(model_name)
            if not current:
                return
            try:
                new_assignments = assign_layers(
                    [
                        {"node_id": nid, **self.nodes[nid].capabilities}
                        for nid in current
                        if nid in self.nodes
                    ],
                    model_name,
                )
            except ValueError as e:
                logger.warning(
                    "post-leave rebalance impossible",
                    extra={"model": model_name, "err": str(e)},
                )
                return
            affected = [
                nid for nid, span in new_assignments.items() if current.get(nid) != span
            ]
            await self._apply_assignments(model_name, new_assignments, affected, reason="leave")

    async def _handle_consumer(self, ws: ServerConnection, first: dict) -> None:
        raw_sid = first.get("session_id") or ""
        session_id = raw_sid if re.match(r'^[a-f0-9]{32}$', raw_sid) else uuid.uuid4().hex
        model_name = first.get("model_name", "")
        pv = first.get("protocol_version")
        if pv != PROTOCOL_VERSION:
            await self._send_error(
                ws, session_id, "BAD_PROTOCOL",
                f"Unsupported protocol version {pv!r}; expected {PROTOCOL_VERSION}",
            )
            return

        if len(self.streams) >= self.max_streams:
            await self._send_error(ws, session_id, "CAPACITY", "session limit reached")
            return

        try:
            active = self.assemble_pipeline(model_name, session_id)
        except RuntimeError:
            await self._send_error(ws, session_id, "NO_NODES",
                                   "no suitable pipeline available")
            return

        stream_id = uuid.uuid4().hex
        nodes_payload = [
            {"node_id": n.node_id, "layer_start": n.layer_start, "layer_end": n.layer_end}
            for n in active
        ]
        turn_servers = self._get_turn_servers(session_id)
        cfg = make_pipeline_config(
            session_id, nodes_payload, stream_id=stream_id,
            turn_servers=turn_servers,
        )
        pipeline_ids = [n.node_id for n in active]
        ce = ConsumerEntry(
            stream_id=stream_id,
            ws=ws,
            session_id=session_id,
            pipeline=pipeline_ids,
            model_name=model_name,
        )
        self.streams[stream_id] = ce

        for nid in pipeline_ids:
            self._node_streams.setdefault(nid, set()).add(stream_id)

        if not await self._safe_send(ws, encode_message(cfg)):
            self.streams.pop(stream_id, None)
            for nid in pipeline_ids:
                self._node_streams.get(nid, set()).discard(stream_id)
            return

        logger.info(
            "session opened",
            extra={
                "stream_id": stream_id,
                "session_id": session_id,
                "model": model_name,
                "pipeline": ce.pipeline,
            },
        )

        limiter = _RateLimiter()
        try:
            async for raw in ws:
                if not limiter.allow():
                    continue
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                try:
                    msg = decode_message(raw)
                except Exception:
                    logger.exception("decode failed on consumer frame", extra={"stream_id": stream_id})
                    continue

                t = msg.get("type")
                if t in (SDP_OFFER, SDP_ANSWER, ICE_CANDIDATE):
                    await self._forward_consumer_signaling(ce, msg)
                    continue
                if t != ENVELOPE:
                    await self._send_error(
                        ws, session_id, "BAD_FRAME",
                        "expected envelope or signaling message",
                    )
                    continue

                target = msg.get("target_node_id")
                if not target:
                    await self._send_error(
                        ws, session_id, "BAD_TARGET",
                        "envelope missing target_node_id",
                    )
                    continue

                if target not in ce.pipeline:
                    await self._send_error(
                        ws, session_id, "BAD_TARGET",
                        "target node is not in this session's pipeline",
                    )
                    continue

                node = self.nodes.get(target)
                if node is None or node.status != "active":
                    await self._send_error(
                        ws, session_id, "NODE_UNAVAILABLE",
                        "target node is not available",
                    )
                    continue

                count = msg.get("envelope_count", 0)
                last = ce.last_envelope_count.get(target, -1)
                if count <= last:
                    await self._send_error(
                        ws, session_id, "REPLAY",
                        "envelope_count not monotonically increasing",
                    )
                    continue
                ce.last_envelope_count[target] = count
                msg["stream_id"] = stream_id
                ce.envelope_counts[target] = ce.envelope_counts.get(target, 0) + 1
                ce.last_activity = time.time()
                ok = await self._safe_send(node.ws, encode_message(msg))
                if not ok:
                    self.nodes.pop(target, None)
                    await self._send_error(
                        ws, session_id, "NODE_UNAVAILABLE",
                        "send to target node failed",
                    )
        except ConnectionClosed:
            pass
        finally:
            self.streams.pop(stream_id, None)
            for nid in pipeline_ids:
                self._node_streams.get(nid, set()).discard(stream_id)
            total_nodes = len(ce.pipeline)
            p2p_count = len(ce.p2p_established)
            p2p_failed_count = len(ce.p2p_failed)
            relay_only = total_nodes - p2p_count - p2p_failed_count
            logger.info(
                "session closed",
                extra={
                    "stream_id": stream_id,
                    "session_id": session_id,
                    "model": ce.model_name,
                    "envelope_counts": dict(ce.envelope_counts),
                    "duration_s": round(time.time() - ce.created_at, 3),
                    "p2p_established": list(ce.p2p_established),
                    "p2p_failed": list(ce.p2p_failed),
                    "p2p_success_rate": round(p2p_count / total_nodes, 2) if total_nodes else 0,
                    "relay_only_count": relay_only,
                },
            )

    async def handle_connection(self, ws: ServerConnection) -> None:
        remote = ws.remote_address
        ip = remote[0] if remote else "unknown"

        count = self._connections_per_ip.get(ip, 0)
        if count >= self.max_connections_per_ip:
            await self._send_error(ws, "", "RATE_LIMITED",
                                   "too many connections from this address")
            try:
                await ws.close()
            except (ConnectionClosed, OSError):
                pass
            return
        self._connections_per_ip[ip] = count + 1

        try:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except (ConnectionClosed, asyncio.TimeoutError):
                return
            if isinstance(raw, str):
                raw = raw.encode("utf-8")

            try:
                msg = decode_message(raw)
            except Exception:
                await self._send_error(ws, "", "BAD_HELLO",
                                       "failed to decode first frame")
                try:
                    await ws.close()
                except (ConnectionClosed, OSError):
                    pass
                return

            t = msg.get("type")
            if t == REGISTER_NODE:
                await self._handle_node(ws, msg)
            elif t == SESSION_INIT:
                await self._handle_consumer(ws, msg)
            else:
                await self._send_error(
                    ws, msg.get("session_id", ""), "BAD_HELLO",
                    "first frame must be register_node or session_init",
                )
                try:
                    await ws.close()
                except (ConnectionClosed, OSError):
                    pass
        finally:
            self._connections_per_ip[ip] = max(
                0, self._connections_per_ip.get(ip, 1) - 1
            )

    def build_status(self) -> dict:
        """Snapshot of network state for the HTTP /status endpoint.

        Node IDs are truncated and hardware details omitted to avoid
        leaking wallet addresses and detailed hardware profiles.
        """
        nodes_payload = []
        active_node_count = 0
        for nid, entry in self.nodes.items():
            if entry.layer_start is not None and entry.layer_end is not None:
                layers = [entry.layer_start, entry.layer_end]
            else:
                layers = []
            if entry.assignment_status == "active":
                active_node_count += 1
            redacted_id = nid[:8] + "..." if len(nid) > 8 else nid
            nodes_payload.append({
                "node_id": redacted_id,
                "device": (entry.capabilities or {}).get("device", "cpu"),
                "layers": layers,
                "status": entry.assignment_status,
            })

        models_payload = []
        for model_name, info in MODEL_REGISTRY.items():
            total = info["total_layers"]
            assignments = self._active_assignments_for_model(model_name)
            covered = sum(end - start for (start, end) in assignments.values())
            covered = max(0, min(covered, total))
            available = bool(assignments) and validate_coverage(assignments, model_name)
            models_payload.append({
                "name": model_name,
                "total_layers": total,
                "covered_layers": covered,
                "available": available,
            })

        return {
            "nodes": nodes_payload,
            "models": models_payload,
            "total_nodes": active_node_count,
            "active_sessions": len(self.streams),
            "uptime_seconds": int(time.time() - self.start_time),
        }

    async def _http_respond(
        self,
        writer: asyncio.StreamWriter,
        status_code: int,
        status_text: str,
        body: bytes,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        headers = (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            f"Access-Control-Allow-Origin: {self.cors_origin}\r\n"
            "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "\r\n"
        )
        writer.write(headers.encode("ascii") + body)
        try:
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass

    async def _handle_http(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await asyncio.wait_for(
                reader.readline(), timeout=10.0,
            )
            if not request_line:
                return
            if len(request_line) > 8192:
                await self._http_respond(writer, 414, "URI Too Long", b"uri too long")
                return
            try:
                method, path, _ = request_line.decode("ascii", errors="replace").rstrip("\r\n").split(" ", 2)
            except ValueError:
                await self._http_respond(writer, 400, "Bad Request", b"bad request")
                return
            header_count = 0
            while header_count < 100:
                line = await asyncio.wait_for(reader.readline(), timeout=10.0)
                if not line or line in (b"\r\n", b"\n"):
                    break
                header_count += 1

            if method == "OPTIONS":
                await self._http_respond(writer, 204, "No Content", b"")
                return
            if method != "GET":
                await self._http_respond(writer, 405, "Method Not Allowed", b"method not allowed")
                return

            clean_path = path.split("?", 1)[0]
            if clean_path != "/status":
                await self._http_respond(writer, 404, "Not Found", b"not found")
                return

            body = json.dumps(self.build_status()).encode("utf-8")
            await self._http_respond(
                writer, 200, "OK", body, content_type="application/json",
            )
        except Exception:
            logger.exception("http handler error")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, OSError):
                pass

    async def monitor_heartbeats(self) -> None:
        while True:
            now = time.time()
            for node_id, entry in list(self.nodes.items()):
                elapsed = now - entry.last_heartbeat
                if elapsed > self.heartbeat_timeout and entry.status == "active":
                    entry.status = "inactive"
                    logger.warning(
                        "node inactive — no heartbeat",
                        extra={"node_id": node_id, "elapsed_s": round(elapsed, 1)},
                    )
                    try:
                        await entry.ws.close()
                    except (ConnectionClosed, OSError):
                        pass
                    if self.nodes.get(node_id) is entry:
                        self.nodes.pop(node_id, None)
                    await self._teardown_streams_using_node(node_id)
            await asyncio.sleep(5.0)

    async def monitor_idle_sessions(self) -> None:
        """Close consumer sessions that have been idle too long."""
        while True:
            now = time.time()
            for sid, ce in list(self.streams.items()):
                idle = now - ce.last_activity
                if idle > self.session_idle_timeout:
                    logger.info(
                        "closing idle session",
                        extra={"stream_id": sid, "idle_s": round(idle, 1)},
                    )
                    self.streams.pop(sid, None)
                    try:
                        await ce.ws.close()
                    except (ConnectionClosed, OSError):
                        pass
            await asyncio.sleep(30.0)

    async def start(self) -> None:
        asyncio.create_task(self.monitor_heartbeats())
        asyncio.create_task(self.monitor_idle_sessions())
        logger.info("relay starting", extra={"host": self.host, "port": self.port})

        ssl_ctx = None
        if self.tls_cert and self.tls_key:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ssl_ctx.load_cert_chain(self.tls_cert, self.tls_key)
            logger.info("TLS enabled for WebSocket server")

        http_server = await asyncio.start_server(
            self._handle_http, self.host, self.http_port, ssl=ssl_ctx,
        )
        scheme = "https" if ssl_ctx else "http"
        logger.info(
            "status http listening",
            extra={
                "host": self.host,
                "http_port": self.http_port,
                "url": f"{scheme}://{self.host}:{self.http_port}/status",
            },
        )
        async with serve(
            self.handle_connection,
            self.host,
            self.port,
            max_size=self.max_message_size,
            ssl=ssl_ctx,
            ping_interval=30,
            ping_timeout=10,
        ), http_server:
            ws_scheme = "wss" if ssl_ctx else "ws"
            logger.info("relay listening",
                        extra={"host": self.host, "port": self.port,
                               "scheme": ws_scheme})
            await asyncio.Future()


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Groove Relay")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--http-port", type=int, default=8771,
        help="Port for the HTTP /status endpoint",
    )
    parser.add_argument(
        "--no-auth", action="store_true",
        help="Disable challenge-response node authentication (dev only)",
    )
    parser.add_argument("--tls-cert", default=None, help="Path to TLS certificate")
    parser.add_argument("--tls-key", default=None, help="Path to TLS private key")
    parser.add_argument(
        "--cors-origin", default="*",
        help="Access-Control-Allow-Origin header value",
    )
    parser.add_argument(
        "--max-connections-per-ip", type=int, default=20,
        help="Maximum WebSocket connections per IP address",
    )
    parser.add_argument(
        "--max-message-size", type=int, default=10 * 1024 * 1024,
        help="Maximum WebSocket message size in bytes (default 10 MB)",
    )
    parser.add_argument(
        "--turn-secret", default=None,
        help="Shared secret for HMAC-based TURN credential generation (coturn)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    configure_logging(args.log_level)

    relay = RelayNode(
        host=args.host,
        port=args.port,
        http_port=args.http_port,
        require_auth=not args.no_auth,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        cors_origin=args.cors_origin,
        max_connections_per_ip=args.max_connections_per_ip,
        max_message_size=args.max_message_size,
        turn_secret=args.turn_secret,
    )
    await relay.start()


if __name__ == "__main__":
    asyncio.run(main())
