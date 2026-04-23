"""KV cache management for compute node sessions.

Each compute node maintains KV caches for its assigned layer range,
keyed by session_id. Wraps transformers DynamicCache with session
management (TTL cleanup, context limits).

Supports INT4-quantized cache via optimum-quanto to reduce memory
footprint by ~75% on RAM-constrained devices (e.g. Apple Silicon).
"""

import logging
import os
import sys
import time

from transformers import DynamicCache

logger = logging.getLogger(__name__)

_QUANTIZED_AVAILABLE = False
_QuantizedCache = None

try:
    venv_bin = os.path.join(sys.prefix, "bin")
    if venv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

    from transformers.cache_utils import QuantizedCache as _QuantizedCache
    import optimum.quanto  # noqa: F401 — required backend
    _QUANTIZED_AVAILABLE = True
except ImportError:
    pass


def _make_cache(config=None, quantize: bool = False):
    if quantize and _QUANTIZED_AVAILABLE and config is not None:
        try:
            cache = _QuantizedCache("quanto", config, nbits=4, residual_length=128)
            logger.info("using INT4 quantized KV cache (quanto backend)")
            return cache
        except Exception as e:
            logger.warning("INT4 KV cache init failed (%s), falling back to FP16", e)
    elif quantize and not _QUANTIZED_AVAILABLE:
        logger.warning("quantized KV cache requested but optimum-quanto not installed, falling back to FP16")
    return DynamicCache()


class SessionKVCache:
    """Manages the KV cache for a single session using DynamicCache."""

    def __init__(self, num_layers: int, max_context: int = 4096,
                 config=None, quantize: bool = False):
        self.num_layers = num_layers
        self.max_context = max_context
        self._config = config
        self._quantize = quantize
        self.cache = _make_cache(config, quantize)
        self.last_access = time.time()

    @property
    def context_length(self) -> int:
        return self.cache.get_seq_length()

    def get_cache(self) -> DynamicCache:
        self.last_access = time.time()
        return self.cache

    def clear(self) -> None:
        self.cache = _make_cache(self._config, self._quantize)


class KVCacheManager:
    """Manages multiple SessionKVCache instances keyed by session_id."""

    def __init__(self, max_sessions: int = 100, config=None, quantize: bool = False):
        self._sessions: dict[str, SessionKVCache] = {}
        self._max_sessions = max_sessions
        self._config = config
        self._quantize = quantize

    def create_session(
        self,
        session_id: str,
        num_layers: int,
        max_context: int = 4096,
    ) -> SessionKVCache:
        if len(self._sessions) >= self._max_sessions:
            self.cleanup_expired(ttl_seconds=60.0)
        if len(self._sessions) >= self._max_sessions:
            oldest_sid = min(
                self._sessions, key=lambda s: self._sessions[s].last_access,
            )
            self.remove_session(oldest_sid)
        cache = SessionKVCache(
            num_layers=num_layers, max_context=max_context,
            config=self._config, quantize=self._quantize,
        )
        self._sessions[session_id] = cache
        return cache

    def get_session(self, session_id: str) -> SessionKVCache | None:
        cache = self._sessions.get(session_id)
        if cache is not None:
            cache.last_access = time.time()
        return cache

    def remove_session(self, session_id: str) -> None:
        cache = self._sessions.pop(session_id, None)
        if cache is not None:
            cache.clear()

    def cleanup_expired(self, ttl_seconds: float = 300.0) -> list[str]:
        now = time.time()
        expired = [
            sid
            for sid, cache in self._sessions.items()
            if (now - cache.last_access) > ttl_seconds
        ]
        for sid in expired:
            self.remove_session(sid)
        return expired

    def trim(self, session_id: str, count: int) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            logger.warning("trim requested for unknown session %s", session_id)
            return
        if count <= 0:
            return
        cache = session.cache
        seq_len = cache.get_seq_length()
        if seq_len == 0:
            return
        keep = max(0, seq_len - count)
        cache.crop(keep)

    @property
    def active_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def __len__(self) -> int:
        return len(self._sessions)
