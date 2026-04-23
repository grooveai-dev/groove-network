"""Fast tensor serialization for inter-node activation transfer.

Uses raw numpy byte buffers — no pickle (security), no torch.save (slow).
Target: <1ms round-trip for 8KB tensors on local network.

Wire format v2 adds an optional LZ4 compression flag byte:
  compression_flag (uint8: 0x00=raw, 0x01=lz4) | original_payload
"""

import io
import struct

import numpy as np
import torch

try:
    import lz4.frame as _lz4
except ImportError:
    _lz4 = None

# Header: shape_ndim (uint8) | shape (ndim x int32) | dtype_len (uint8) | dtype_str | data
_NDIM_FMT = "!B"
_DIM_FMT = "!i"
_DTYPE_LEN_FMT = "!B"

_COMPRESS_FLAG_FMT = "!B"
_COMPRESS_RAW = 0x00
_COMPRESS_LZ4 = 0x01

_COMPRESSION_THRESHOLD = 16 * 1024  # 16KB

# Mapping between torch and numpy dtypes
_TORCH_TO_NP = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

_NP_TO_TORCH = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _torch_to_numpy_safe(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to numpy, handling bfloat16 specially."""
    t = tensor.detach()
    if not t.is_cpu:
        t = t.cpu()
    if t.dtype == torch.bfloat16:
        return t.to(torch.float16).numpy()
    return t.numpy()


def serialize_tensor(tensor: torch.Tensor, compress: bool = True) -> bytes:
    """Serialize a PyTorch tensor to a compact binary blob.

    Format: compression_flag | ndim | shape dims | dtype string | raw data bytes
    When compress=True and the payload exceeds 16KB, LZ4 is applied.
    """
    shape = tensor.shape
    dtype_str = _TORCH_TO_NP.get(tensor.dtype)
    if dtype_str is None:
        raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")

    arr = _torch_to_numpy_safe(tensor)
    if tensor.dtype == torch.bfloat16:
        dtype_str = "bfloat16"

    dtype_bytes = dtype_str.encode("ascii")
    ndim = len(shape)

    header_fmt = f"!B{ndim}iB"
    header = struct.pack(header_fmt, ndim, *shape, len(dtype_bytes))

    buf = io.BytesIO()
    buf.write(header)
    buf.write(dtype_bytes)
    buf.write(arr.tobytes())

    payload = buf.getvalue()

    if compress and _lz4 is not None and len(payload) >= _COMPRESSION_THRESHOLD:
        compressed = _lz4.compress(payload)
        return struct.pack(_COMPRESS_FLAG_FMT, _COMPRESS_LZ4) + compressed
    else:
        return struct.pack(_COMPRESS_FLAG_FMT, _COMPRESS_RAW) + payload


_MAX_TENSOR_NDIM = 8
_MAX_TENSOR_ELEMENTS = 100_000_000


def deserialize_tensor(data: bytes, device: str = "cpu") -> torch.Tensor:
    """Reconstruct a PyTorch tensor from a binary blob.

    Supports both v1 (no compression flag) and v2 (compression flag) formats.
    Validates shape, dtype, and data length to prevent OOM, crashes,
    or unexpected behavior from malicious payloads.
    """
    if len(data) < 2:
        raise ValueError("tensor data too short")

    flag_byte = data[0]
    if flag_byte in (_COMPRESS_RAW, _COMPRESS_LZ4):
        payload_data = data[1:]
        if flag_byte == _COMPRESS_LZ4:
            if _lz4 is None:
                raise RuntimeError("lz4 not installed but received LZ4-compressed tensor")
            payload_data = _lz4.decompress(payload_data)
    else:
        payload_data = data

    return _deserialize_payload(payload_data, device)


def _deserialize_payload(data: bytes, device: str = "cpu") -> torch.Tensor:
    offset = 0

    (ndim,) = struct.unpack_from(_NDIM_FMT, data, offset)
    offset += struct.calcsize(_NDIM_FMT)

    if ndim > _MAX_TENSOR_NDIM:
        raise ValueError(f"tensor ndim {ndim} exceeds maximum {_MAX_TENSOR_NDIM}")

    if ndim > 0:
        shape_fmt = f"!{ndim}i"
        shape = list(struct.unpack_from(shape_fmt, data, offset))
        offset += struct.calcsize(shape_fmt)
    else:
        shape = []

    for dim in shape:
        if dim <= 0:
            raise ValueError(f"invalid tensor dimension: {dim}")

    total_elements = 1
    for dim in shape:
        total_elements *= dim
        if total_elements > _MAX_TENSOR_ELEMENTS:
            raise ValueError(
                f"tensor too large: exceeds {_MAX_TENSOR_ELEMENTS} element limit"
            )

    (dtype_len,) = struct.unpack_from(_DTYPE_LEN_FMT, data, offset)
    offset += struct.calcsize(_DTYPE_LEN_FMT)

    dtype_str = data[offset : offset + dtype_len].decode("ascii")
    offset += dtype_len

    if dtype_str != "bfloat16" and dtype_str not in _NP_TO_TORCH:
        raise ValueError(f"unsupported tensor dtype: {dtype_str!r}")

    if dtype_str == "bfloat16":
        expected_bytes = total_elements * 2
    else:
        expected_bytes = total_elements * np.dtype(dtype_str).itemsize

    if len(data) - offset < expected_bytes:
        raise ValueError(
            f"tensor data too short: got {len(data) - offset} bytes, expected {expected_bytes}"
        )
    raw_data = data[offset : offset + expected_bytes]

    if dtype_str == "bfloat16":
        arr = np.frombuffer(raw_data, dtype=np.float16).reshape(shape)
        tensor = torch.from_numpy(arr.copy()).to(torch.bfloat16)
    else:
        torch_dtype = _NP_TO_TORCH[dtype_str]
        tensor = torch.frombuffer(bytearray(raw_data), dtype=torch_dtype).reshape(shape)

    if device != "cpu":
        tensor = tensor.to(device, non_blocking=True)

    return tensor
