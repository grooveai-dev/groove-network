"""WebRTC peer connection manager for direct P2P data channels.

Uses aiortc to establish encrypted data channels between pipeline peers,
bypassing the signal relay for inference data transfer.

Dual-channel design:
  tensor channel  — ordered=True,  maxRetransmits=None (reliable, for inference data)
  control channel — ordered=False, maxRetransmits=0    (unreliable, low-latency control)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from aiortc import (
    RTCConfiguration,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

logger = logging.getLogger("peer")


def _default_rtc_config() -> RTCConfiguration:
    """STUN-only configuration for NAT traversal."""
    return RTCConfiguration(iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ])


def build_rtc_config(turn_servers: list[dict] | None = None) -> RTCConfiguration:
    """Build RTC config with optional dynamic TURN credentials."""
    servers = [
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ]
    if turn_servers:
        for ts in turn_servers:
            servers.append(RTCIceServer(
                urls=ts["urls"],
                username=ts.get("username", ""),
                credential=ts.get("credential", ""),
            ))
    return RTCConfiguration(iceServers=servers)


@dataclass
class _PeerState:
    pc: RTCPeerConnection
    tensor_channel: object | None = None
    control_channel: object | None = None
    tensor_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    control_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    tensor_connected: bool = False
    control_connected: bool = False
    connected_event: asyncio.Event = field(default_factory=asyncio.Event)


class PeerConnectionManager:
    """Manages WebRTC data channels between pipeline peers."""

    def __init__(
        self,
        session_id: str | None = None,
        rtc_config: RTCConfiguration | None = None,
    ):
        self._session_id = session_id or uuid.uuid4().hex
        self._config = rtc_config or _default_rtc_config()
        self._peers: dict[str, _PeerState] = {}
        self.on_ice_candidate: Callable[[str, dict], Awaitable[None]] | None = None

    def _setup_channel_events(self, remote_node_id: str, channel, kind: str = "tensor") -> None:
        state = self._peers[remote_node_id]
        queue = state.tensor_queue if kind == "tensor" else state.control_queue

        @channel.on("open")
        def on_open():
            if kind == "tensor":
                state.tensor_connected = True
            else:
                state.control_connected = True
            if state.tensor_connected and state.control_connected:
                state.connected_event.set()
            logger.info("%s channel open", kind, extra={"peer": remote_node_id})

        @channel.on("close")
        def on_close():
            if kind == "tensor":
                state.tensor_connected = False
            else:
                state.control_connected = False
            state.connected_event.clear()
            logger.info("%s channel closed", kind, extra={"peer": remote_node_id})

        @channel.on("message")
        def on_message(data):
            queue.put_nowait(data)

        if getattr(channel, "readyState", None) == "open":
            if kind == "tensor":
                state.tensor_connected = True
            else:
                state.control_connected = True
            if state.tensor_connected and state.control_connected:
                state.connected_event.set()

    async def create_offer(self, remote_node_id: str) -> str:
        """Create RTCPeerConnection + dual data channels, return SDP offer."""
        pc = RTCPeerConnection(configuration=self._config)
        state = _PeerState(pc=pc)
        self._peers[remote_node_id] = state

        tensor_ch = pc.createDataChannel(
            f"groove-tensor-{self._session_id}",
            ordered=True,
            maxRetransmits=5,
        )
        state.tensor_channel = tensor_ch
        self._setup_channel_events(remote_node_id, tensor_ch, "tensor")

        control_ch = pc.createDataChannel(
            f"groove-control-{self._session_id}",
            ordered=False,
            maxRetransmits=0,
        )
        state.control_channel = control_ch
        self._setup_channel_events(remote_node_id, control_ch, "control")

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed"):
                state.tensor_connected = False
                state.control_connected = False

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        return pc.localDescription.sdp

    async def handle_offer(self, remote_node_id: str, sdp_offer: str) -> str:
        """Receive SDP offer, create answer, return SDP answer string."""
        pc = RTCPeerConnection(configuration=self._config)
        state = _PeerState(pc=pc)
        self._peers[remote_node_id] = state

        @pc.on("datachannel")
        def on_datachannel(channel):
            label = channel.label
            if label.startswith("groove-tensor-"):
                state.tensor_channel = channel
                self._setup_channel_events(remote_node_id, channel, "tensor")
            elif label.startswith("groove-control-"):
                state.control_channel = channel
                self._setup_channel_events(remote_node_id, channel, "control")
            else:
                state.tensor_channel = channel
                self._setup_channel_events(remote_node_id, channel, "tensor")
                state.control_connected = True

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed"):
                state.tensor_connected = False
                state.control_connected = False

        offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription.sdp

    async def accept_answer(self, remote_node_id: str, sdp_answer: str) -> None:
        """Apply remote SDP answer to complete the handshake."""
        state = self._peers.get(remote_node_id)
        if state is None:
            raise ValueError(f"No pending connection for {remote_node_id}")
        answer = RTCSessionDescription(sdp=sdp_answer, type="answer")
        await state.pc.setRemoteDescription(answer)

    async def add_ice_candidate(self, remote_node_id: str, candidate: dict) -> None:
        """Add a trickled ICE candidate to the connection."""
        state = self._peers.get(remote_node_id)
        if state is None:
            raise ValueError(f"No connection for {remote_node_id}")
        candidate_str = candidate.get("candidate", "")
        if not candidate_str:
            return
        sdp_mid = candidate.get("sdpMid")
        sdp_m_line_index = candidate.get("sdpMLineIndex")
        ice = _parse_ice_candidate(candidate_str, sdp_mid, sdp_m_line_index)
        await state.pc.addIceCandidate(ice)

    async def send(self, remote_node_id: str, data: bytes) -> None:
        """Send binary data over the tensor data channel."""
        state = self._peers.get(remote_node_id)
        if state is None or state.tensor_channel is None:
            raise ValueError(f"No data channel for {remote_node_id}")
        state.tensor_channel.send(data)

    async def send_control(self, remote_node_id: str, data: bytes) -> None:
        """Send binary data over the control data channel."""
        state = self._peers.get(remote_node_id)
        if state is None or state.control_channel is None:
            raise ValueError(f"No control channel for {remote_node_id}")
        state.control_channel.send(data)

    async def recv(self, remote_node_id: str) -> bytes:
        """Receive binary data from the tensor data channel."""
        state = self._peers.get(remote_node_id)
        if state is None:
            raise ValueError(f"No connection for {remote_node_id}")
        return await state.tensor_queue.get()

    async def recv_control(self, remote_node_id: str) -> bytes:
        """Receive binary data from the control data channel."""
        state = self._peers.get(remote_node_id)
        if state is None:
            raise ValueError(f"No connection for {remote_node_id}")
        return await state.control_queue.get()

    def is_connected(self, remote_node_id: str) -> bool:
        """Check if both data channels are open."""
        state = self._peers.get(remote_node_id)
        if state is None:
            return False
        return state.tensor_connected and state.control_connected

    async def wait_connected(self, remote_node_id: str, timeout: float = 10.0) -> bool:
        """Wait until both data channels are open, or timeout."""
        state = self._peers.get(remote_node_id)
        if state is None:
            return False
        if state.tensor_connected and state.control_connected:
            return True
        try:
            await asyncio.wait_for(state.connected_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def close(self, remote_node_id: str) -> None:
        """Close a specific peer connection and clean up."""
        state = self._peers.pop(remote_node_id, None)
        if state is None:
            return
        state.tensor_connected = False
        state.control_connected = False
        await state.pc.close()

    async def close_all(self) -> None:
        """Close all peer connections."""
        for peer_id in list(self._peers.keys()):
            await self.close(peer_id)


def _parse_ice_candidate(
    candidate_str: str,
    sdp_mid: str | None,
    sdp_m_line_index: int | None,
) -> RTCIceCandidate:
    """Parse an ICE candidate SDP string into an RTCIceCandidate."""
    parts = candidate_str.split()
    foundation = parts[0].split(":", 1)[-1] if ":" in parts[0] else parts[0]
    component = int(parts[1])
    protocol = parts[2]
    priority = int(parts[3])
    ip = parts[4]
    port = int(parts[5])
    cand_type = parts[7]
    related_address = None
    related_port = None
    tcp_type = None
    i = 8
    while i < len(parts) - 1:
        if parts[i] == "raddr":
            related_address = parts[i + 1]
            i += 2
        elif parts[i] == "rport":
            related_port = int(parts[i + 1])
            i += 2
        elif parts[i] == "tcptype":
            tcp_type = parts[i + 1]
            i += 2
        else:
            i += 1
    return RTCIceCandidate(
        component=component,
        foundation=foundation,
        ip=ip,
        port=port,
        priority=priority,
        protocol=protocol,
        type=cand_type,
        relatedAddress=related_address,
        relatedPort=related_port,
        sdpMid=sdp_mid,
        sdpMLineIndex=sdp_m_line_index,
        tcpType=tcp_type,
    )
