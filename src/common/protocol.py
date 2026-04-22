"""Message protocol for the Groove decentralized inference pipeline.

Wire format (v2): every consumer<->node frame travels inside an ENVELOPE
that carries a relay-minted stream_id. Nodes maintain one persistent
outbound websocket to the relay; consumers maintain one websocket to the
relay. The relay alone knows node addresses.
"""

from __future__ import annotations

import re
import struct
import time
import uuid

import msgpack
import numpy as np

PROTOCOL_VERSION = 2

# Message type constants
SESSION_INIT = "session_init"
ACTIVATIONS = "activations"
LOGITS = "logits"
SPEC_WINDOW = "spec_window"
VERIFY_RESULT = "verify_result"
HEARTBEAT = "heartbeat"
PIPELINE_CONFIG = "pipeline_config"
ERROR = "error"
REGISTER_NODE = "register_node"
REGISTER_ACK = "register_ack"
DEREGISTER = "deregister"
ENVELOPE = "envelope"
ASSIGN_LAYERS = "assign_layers"
ASSIGNMENT_ACK = "assignment_ack"
REBALANCE = "rebalance"
AUTH_CHALLENGE = "auth_challenge"
AUTH_RESPONSE = "auth_response"
PIPELINE_MESH = "pipeline_mesh"
KV_TRIM = "kv_trim"
TOKEN_RESULT = "token_result"

# WebRTC P2P signaling message types.
SDP_OFFER = "sdp_offer"
SDP_ANSWER = "sdp_answer"
ICE_CANDIDATE = "ice_candidate"
P2P_READY = "p2p_ready"

# M3 signal-specific message types.
SIGNAL_REGISTER = "signal_register"
SIGNAL_ACK = "signal_ack"
SIGNAL_HEARTBEAT = "signal_heartbeat"
SIGNAL_QUERY = "signal_query"
SIGNAL_MATCH = "signal_match"
SIGNAL_DEREGISTER = "signal_deregister"

ALL_MESSAGE_TYPES = frozenset({
    SESSION_INIT, ACTIVATIONS, LOGITS, SPEC_WINDOW,
    VERIFY_RESULT, HEARTBEAT, PIPELINE_CONFIG, ERROR,
    REGISTER_NODE, REGISTER_ACK, DEREGISTER, ENVELOPE,
    ASSIGN_LAYERS, ASSIGNMENT_ACK, REBALANCE,
    AUTH_CHALLENGE, AUTH_RESPONSE,
    PIPELINE_MESH, KV_TRIM, TOKEN_RESULT,
    SDP_OFFER, SDP_ANSWER, ICE_CANDIDATE, P2P_READY,
    SIGNAL_REGISTER, SIGNAL_ACK, SIGNAL_HEARTBEAT,
    SIGNAL_QUERY, SIGNAL_MATCH, SIGNAL_DEREGISTER,
})

# Default capability fields (M2). Nodes may report a subset; missing keys
# are filled with these defaults by normalize_capabilities().
CAPABILITY_DEFAULTS = {
    "ram_mb": 0,
    "vram_mb": 0,
    "device": "cpu",
    "bandwidth_mbps": 0.0,
    "cpu_cores": 0,
    "gpu_model": "",
    "max_context_length": 4096,
}


CAPABILITY_KEYS = frozenset(CAPABILITY_DEFAULTS.keys()) | frozenset({
    "models_loaded", "model_preferences", "protocol_version",
    "layer_start", "layer_end", "load",
    "bench_ms_per_layer", "bench_mem_bandwidth_gbps", "node_role",
})

NODE_ID_RE = re.compile(r'^0x[0-9a-fA-F]{40}$')

def validate_node_id(node_id: str) -> bool:
    return bool(node_id and NODE_ID_RE.match(node_id))


def normalize_capabilities(caps: dict | None) -> dict:
    """Return a dict with all M2 capability fields populated.

    Only whitelisted keys are preserved to prevent injection of
    unexpected fields from untrusted input. Numeric fields are
    coerced and clamped; non-finite values are rejected.
    """
    import math
    out = dict(CAPABILITY_DEFAULTS)
    if caps:
        for k, v in caps.items():
            if k in CAPABILITY_KEYS:
                out[k] = v
    for k in ("ram_mb", "vram_mb", "bandwidth_mbps", "cpu_cores", "max_context_length"):
        try:
            v = float(out[k])
            if math.isnan(v) or math.isinf(v) or v < 0:
                out[k] = CAPABILITY_DEFAULTS.get(k, 0)
            else:
                out[k] = v
        except (TypeError, ValueError):
            out[k] = CAPABILITY_DEFAULTS.get(k, 0)
    if not isinstance(out.get("device"), str):
        out["device"] = "cpu"
    if not isinstance(out.get("gpu_model"), str):
        out["gpu_model"] = ""
    return out

# Binary header format for tensor data embedded in messages:
#   shape_len (uint32) | shape_bytes | dtype_str_len (uint16) | dtype_str | data_len (uint64) | data_bytes
SHAPE_LEN_FMT = "!I"       # 4 bytes, big-endian uint32
DTYPE_STR_LEN_FMT = "!H"   # 2 bytes, big-endian uint16
DATA_LEN_FMT = "!Q"        # 8 bytes, big-endian uint64


def pack_tensor_header(shape: tuple, dtype_str: str, data: bytes) -> bytes:
    """Pack tensor metadata + raw bytes into a single binary blob."""
    shape_bytes = msgpack.packb(shape)
    dtype_bytes = dtype_str.encode("ascii")
    parts = [
        struct.pack(SHAPE_LEN_FMT, len(shape_bytes)),
        shape_bytes,
        struct.pack(DTYPE_STR_LEN_FMT, len(dtype_bytes)),
        dtype_bytes,
        struct.pack(DATA_LEN_FMT, len(data)),
        data,
    ]
    return b"".join(parts)


def unpack_tensor_header(blob: bytes) -> tuple[tuple, str, bytes]:
    """Unpack a binary blob into (shape, dtype_str, raw_data)."""
    offset = 0

    (shape_len,) = struct.unpack_from(SHAPE_LEN_FMT, blob, offset)
    offset += struct.calcsize(SHAPE_LEN_FMT)
    shape = tuple(msgpack.unpackb(
        blob[offset : offset + shape_len],
        max_array_len=100,
        max_map_len=10,
        max_str_len=256,
        max_bin_len=256,
    ))
    offset += shape_len

    (dtype_str_len,) = struct.unpack_from(DTYPE_STR_LEN_FMT, blob, offset)
    offset += struct.calcsize(DTYPE_STR_LEN_FMT)
    dtype_str = blob[offset : offset + dtype_str_len].decode("ascii")
    offset += dtype_str_len

    (data_len,) = struct.unpack_from(DATA_LEN_FMT, blob, offset)
    offset += struct.calcsize(DATA_LEN_FMT)
    data = blob[offset : offset + data_len]
    offset += data_len

    return shape, dtype_str, data


def encode_message(msg: dict) -> bytes:
    """Encode a protocol message to bytes using msgpack."""
    if "type" not in msg:
        raise ValueError("Message must contain a 'type' field")
    if msg["type"] not in ALL_MESSAGE_TYPES:
        raise ValueError(f"Unknown message type: {msg['type']}")
    return msgpack.packb(msg, use_bin_type=True)


def decode_message(data: bytes) -> dict:
    """Decode a msgpack-encoded protocol message."""
    msg = msgpack.unpackb(
        data,
        raw=False,
        max_str_len=10 * 1024 * 1024,
        max_bin_len=10 * 1024 * 1024,
        max_array_len=10_000,
        max_map_len=1_000,
    )
    if not isinstance(msg, dict) or "type" not in msg:
        raise ValueError("Invalid message: must be a dict with a 'type' field")
    if msg["type"] not in ALL_MESSAGE_TYPES:
        raise ValueError(f"Unknown message type: {msg['type']}")
    return msg


def make_session_init(
    session_id: str,
    model_name: str,
    layer_start: int,
    layer_end: int,
    config: dict | None = None,
    protocol_version: int = PROTOCOL_VERSION,
    grammar_mode: str | None = None,
) -> dict:
    msg = {
        "type": SESSION_INIT,
        "protocol_version": protocol_version,
        "session_id": session_id,
        "model_name": model_name,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "config": config or {},
    }
    if grammar_mode is not None:
        msg["grammar_mode"] = grammar_mode
    return msg


def make_activations(
    session_id: str,
    seq_pos: int,
    hidden_states_bytes: bytes,
    shape: tuple,
    dtype: str,
    forward_ms: float | None = None,
    queue_ms: float | None = None,
) -> dict:
    msg = {
        "type": ACTIVATIONS,
        "session_id": session_id,
        "seq_pos": seq_pos,
        "hidden_states_bytes": hidden_states_bytes,
        "shape": list(shape),
        "dtype": dtype,
    }
    if forward_ms is not None:
        msg["forward_ms"] = forward_ms
    if queue_ms is not None:
        msg["queue_ms"] = queue_ms
    return msg


def make_logits(
    session_id: str,
    seq_pos: int,
    logits_bytes: bytes,
    shape: tuple,
    dtype: str,
    forward_ms: float | None = None,
    queue_ms: float | None = None,
) -> dict:
    msg = {
        "type": LOGITS,
        "session_id": session_id,
        "seq_pos": seq_pos,
        "logits_bytes": logits_bytes,
        "shape": list(shape),
        "dtype": dtype,
    }
    if forward_ms is not None:
        msg["forward_ms"] = forward_ms
    if queue_ms is not None:
        msg["queue_ms"] = queue_ms
    return msg


def make_token_result(
    session_id: str,
    seq_pos: int,
    token_id: int,
    forward_ms: float | None = None,
    queue_ms: float | None = None,
    sample_ms: float | None = None,
) -> dict:
    msg = {
        "type": TOKEN_RESULT,
        "session_id": session_id,
        "seq_pos": seq_pos,
        "token_id": token_id,
    }
    if forward_ms is not None:
        msg["forward_ms"] = forward_ms
    if queue_ms is not None:
        msg["queue_ms"] = queue_ms
    if sample_ms is not None:
        msg["sample_ms"] = sample_ms
    return msg


def make_spec_window(
    session_id: str,
    candidate_ids: list[int],
    turn: int,
) -> dict:
    return {
        "type": SPEC_WINDOW,
        "session_id": session_id,
        "candidate_ids": candidate_ids,
        "turn": turn,
    }


def make_verify_result(
    session_id: str,
    accepted_tokens: list[int],
    correction_token: int | None,
    num_accepted: int,
) -> dict:
    return {
        "type": VERIFY_RESULT,
        "session_id": session_id,
        "accepted_tokens": accepted_tokens,
        "correction_token": correction_token,
        "num_accepted": num_accepted,
    }


def make_heartbeat(
    node_id: str,
    status: str = "alive",
    timestamp: float | None = None,
    capabilities: dict | None = None,
) -> dict:
    msg = {
        "type": HEARTBEAT,
        "node_id": node_id,
        "timestamp": timestamp if timestamp is not None else time.time(),
        "status": status,
    }
    if capabilities is not None:
        msg["capabilities"] = capabilities
    return msg


def make_pipeline_config(
    session_id: str,
    nodes: list[dict],
    stream_id: str | None = None,
    protocol_version: int = PROTOCOL_VERSION,
    turn_servers: list[dict] | None = None,
) -> dict:
    msg = {
        "type": PIPELINE_CONFIG,
        "protocol_version": protocol_version,
        "session_id": session_id,
        "nodes": nodes,
    }
    if stream_id is not None:
        msg["stream_id"] = stream_id
    if turn_servers:
        msg["turn_servers"] = turn_servers
    return msg


def make_error(
    session_id: str,
    code,
    message: str,
) -> dict:
    return {
        "type": ERROR,
        "session_id": session_id,
        "code": code,
        "message": message,
    }


def make_register_node(
    node_id: str,
    layer_start: int | None = None,
    layer_end: int | None = None,
    capabilities: dict | None = None,
    public_key: str | None = None,
) -> dict:
    """Register a node with the relay.

    layer_start/layer_end may be None to request dynamic assignment by the
    relay's scheduler. Providing both is a legacy/static-override mode.
    public_key is the node's secp256k1 public key for challenge-response auth.
    """
    msg = {
        "type": REGISTER_NODE,
        "protocol_version": PROTOCOL_VERSION,
        "node_id": node_id,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "capabilities": normalize_capabilities(capabilities),
    }
    if public_key is not None:
        msg["public_key"] = public_key
    return msg


def make_register_ack(
    node_id: str,
    accepted: bool,
    message: str = "",
) -> dict:
    return {
        "type": REGISTER_ACK,
        "node_id": node_id,
        "accepted": accepted,
        "message": message,
    }


def make_deregister(node_id: str, reason: str = "") -> dict:
    return {
        "type": DEREGISTER,
        "node_id": node_id,
        "reason": reason,
    }


def make_envelope(
    stream_id: str,
    payload: bytes,
    target_node_id: str | None = None,
    envelope_count: int = 0,
    protocol_version: int = PROTOCOL_VERSION,
) -> dict:
    """Wrap a frame for relay routing.

    envelope_count is a per-session monotonic counter (uint64 on the wire),
    incremented by the sender for each envelope it emits. Used by the relay
    for M4 token metering.
    """
    return {
        "type": ENVELOPE,
        "protocol_version": protocol_version,
        "stream_id": stream_id,
        "target_node_id": target_node_id,
        "envelope_count": int(envelope_count) & 0xFFFFFFFFFFFFFFFF,
        "payload": payload,
    }


def make_assign_layers(
    node_id: str,
    model_name: str,
    layer_start: int,
    layer_end: int,
    total_layers: int,
    hidden_size: int,
) -> dict:
    return {
        "type": ASSIGN_LAYERS,
        "node_id": node_id,
        "model_name": model_name,
        "layer_start": int(layer_start),
        "layer_end": int(layer_end),
        "total_layers": int(total_layers),
        "hidden_size": int(hidden_size),
    }


def make_assignment_ack(
    node_id: str,
    model_name: str,
    layer_start: int,
    layer_end: int,
    accepted: bool,
    reason: str = "",
    load_time_ms: int = 0,
) -> dict:
    return {
        "type": ASSIGNMENT_ACK,
        "node_id": node_id,
        "model_name": model_name,
        "layer_start": int(layer_start),
        "layer_end": int(layer_end),
        "accepted": bool(accepted),
        "reason": reason,
        "load_time_ms": int(load_time_ms),
    }


def make_rebalance(
    node_id: str,
    model_name: str,
    new_layer_start: int,
    new_layer_end: int,
    reason: str = "",
) -> dict:
    return {
        "type": REBALANCE,
        "node_id": node_id,
        "model_name": model_name,
        "new_layer_start": int(new_layer_start),
        "new_layer_end": int(new_layer_end),
        "reason": reason,
    }


def new_session_id() -> str:
    return uuid.uuid4().hex


def new_stream_id() -> str:
    return uuid.uuid4().hex


def make_auth_challenge(node_id: str, nonce: str) -> dict:
    return {
        "type": AUTH_CHALLENGE,
        "node_id": node_id,
        "nonce": nonce,
    }


def make_auth_response(
    node_id: str, signature: str, public_key: str,
) -> dict:
    return {
        "type": AUTH_RESPONSE,
        "node_id": node_id,
        "signature": signature,
        "public_key": public_key,
    }


# ---------------------------------------------------------------------------
# M3 signal-specific factory functions.
# ---------------------------------------------------------------------------
def _normalize_location(location: dict | None) -> dict | None:
    """Keep only whitelisted location fields."""
    if not location:
        return None
    allowed = {"lat", "lon", "city", "region", "country"}
    return {k: v for k, v in location.items() if k in allowed}


def make_signal_register(
    node_id: str,
    capabilities: dict | None = None,
    location: dict | None = None,
    models_supported: list[str] | None = None,
    public_key: str | None = None,
    protocol_version: int = PROTOCOL_VERSION,
) -> dict:
    msg = {
        "type": SIGNAL_REGISTER,
        "protocol_version": protocol_version,
        "node_id": node_id,
        "capabilities": normalize_capabilities(capabilities),
        "location": _normalize_location(location),
        "models_supported": list(models_supported or []),
    }
    if public_key is not None:
        msg["public_key"] = public_key
    return msg


def make_signal_ack(
    node_id: str,
    accepted: bool,
    signal_id: str,
    message: str = "",
) -> dict:
    return {
        "type": SIGNAL_ACK,
        "node_id": node_id,
        "accepted": bool(accepted),
        "signal_id": signal_id,
        "message": message,
    }


def make_signal_heartbeat(
    node_id: str,
    timestamp: float | None = None,
    status: str = "alive",
    capabilities: dict | None = None,
    active_sessions: int = 0,
    load: float = 0.0,
) -> dict:
    msg = {
        "type": SIGNAL_HEARTBEAT,
        "node_id": node_id,
        "timestamp": timestamp if timestamp is not None else time.time(),
        "status": status,
        "active_sessions": int(active_sessions),
        "load": float(load),
    }
    if capabilities is not None:
        msg["capabilities"] = normalize_capabilities(capabilities)
    return msg


def make_signal_query(
    session_id: str,
    model_name: str,
    consumer_location: dict | None = None,
    requirements: dict | None = None,
    top_n: int = 10,
    protocol_version: int = PROTOCOL_VERSION,
) -> dict:
    return {
        "type": SIGNAL_QUERY,
        "protocol_version": protocol_version,
        "session_id": session_id,
        "model_name": model_name,
        "consumer_location": _normalize_location(consumer_location),
        "requirements": dict(requirements or {}),
        "top_n": int(top_n),
    }


def make_signal_match(
    session_id: str,
    nodes: list[dict],
    stream_id: str | None = None,
    protocol_version: int = PROTOCOL_VERSION,
) -> dict:
    msg = {
        "type": SIGNAL_MATCH,
        "protocol_version": protocol_version,
        "session_id": session_id,
        "nodes": nodes,
    }
    if stream_id is not None:
        msg["stream_id"] = stream_id
    return msg


def make_signal_deregister(node_id: str, reason: str = "") -> dict:
    return {
        "type": SIGNAL_DEREGISTER,
        "node_id": node_id,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Pipeline mesh and KV management messages (prep for Phase 2).
# ---------------------------------------------------------------------------

def make_pipeline_mesh(
    session_id: str,
    nodes: list[dict],
    consumer: dict | None = None,
) -> dict:
    """Direct P2P mesh topology for a session.

    nodes: [{node_id, host, port, position}]
    consumer: {host, port}
    """
    return {
        "type": PIPELINE_MESH,
        "session_id": session_id,
        "nodes": nodes,
        "consumer": consumer or {},
    }


def make_kv_trim(
    session_id: str,
    trim_count: int,
) -> dict:
    """Request KV cache trimming for rejected speculative tokens."""
    return {
        "type": KV_TRIM,
        "session_id": session_id,
        "trim_count": int(trim_count),
    }


# ---------------------------------------------------------------------------
# WebRTC P2P signaling factory functions.
# ---------------------------------------------------------------------------

def make_sdp_offer(
    session_id: str,
    from_node_id: str,
    to_node_id: str,
    sdp: str,
) -> dict:
    return {
        "type": SDP_OFFER,
        "session_id": session_id,
        "from_node_id": from_node_id,
        "to_node_id": to_node_id,
        "sdp": sdp,
    }


def make_sdp_answer(
    session_id: str,
    from_node_id: str,
    to_node_id: str,
    sdp: str,
) -> dict:
    return {
        "type": SDP_ANSWER,
        "session_id": session_id,
        "from_node_id": from_node_id,
        "to_node_id": to_node_id,
        "sdp": sdp,
    }


def make_ice_candidate(
    session_id: str,
    from_node_id: str,
    to_node_id: str,
    candidate: str,
    sdp_mid: str,
    sdp_m_line_index: int,
) -> dict:
    return {
        "type": ICE_CANDIDATE,
        "session_id": session_id,
        "from_node_id": from_node_id,
        "to_node_id": to_node_id,
        "candidate": candidate,
        "sdp_mid": sdp_mid,
        "sdp_m_line_index": int(sdp_m_line_index),
    }


def make_p2p_ready(
    session_id: str,
    node_id: str,
    peer_node_id: str,
) -> dict:
    return {
        "type": P2P_READY,
        "session_id": session_id,
        "node_id": node_id,
        "peer_node_id": peer_node_id,
    }
