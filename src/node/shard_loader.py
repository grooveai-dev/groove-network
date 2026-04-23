"""Model shard loading for distributed inference.

Loads a subset of transformer layers from a HuggingFace causal LM,
enabling layer-wise sharding across multiple compute nodes.

Supported architectures: Qwen3, Qwen2.5 (all use model.model.layers).
"""

import logging
import shutil
import subprocess

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


ALLOWED_MODELS = frozenset({
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
})


def _validate_model_name(model_name: str) -> None:
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            f"model {model_name!r} is not in the allowed model list. "
            f"Allowed: {sorted(ALLOWED_MODELS)}"
        )


def get_model_info(model_name: str) -> dict:
    """Get model metadata without loading the full model.

    Returns:
        Dict with total_layers, hidden_size, num_heads, dtype.
    """
    _validate_model_name(model_name)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    return {
        "total_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_heads": getattr(config, "num_attention_heads", None),
        "num_kv_heads": getattr(config, "num_key_value_heads", None),
        "vocab_size": config.vocab_size,
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "dtype": str(getattr(config, "torch_dtype", "float32")),
        "model_type": config.model_type,
    }


def _detect_device(requested: str) -> str:
    """Auto-detect the best device, or honor an explicit request."""
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split("\n")[0]
                logger.warning(
                    "NVIDIA GPU detected (%s) but torch.cuda.is_available() is False. "
                    "PyTorch was likely installed without CUDA support. "
                    "Run: pip install torch torchvision torchaudio "
                    "--index-url https://download.pytorch.org/whl/cu121",
                    gpu_name,
                )
        except Exception:
            pass
    return "cpu"


def load_model_shard(
    model_name: str,
    layer_start: int,
    layer_end: int,
    device: str = "auto",
    quantize: bool = False,
) -> dict:
    """Load a specific layer range from a HuggingFace causal LM.

    Args:
        model_name: HuggingFace model identifier (e.g. "Qwen/Qwen2.5-7B").
        layer_start: First layer index (inclusive, 0-based).
        layer_end: Last layer index (exclusive).
        device: Target device ("cpu", "cuda", "mps"). Auto-detects GPU if "cpu".
        quantize: If True, load in 4-bit quantization via bitsandbytes.

    Returns:
        Dict with:
            layers: nn.ModuleList of the requested transformer layers
            embed_tokens: Embedding layer (only if layer_start == 0, else None)
            norm: Final layer norm (only if layer_end == total_layers, else None)
            lm_head: Language model head (only if layer_end == total_layers, else None)
            config: Model config object
            layer_start: int
            layer_end: int
            total_layers: int
    """
    _validate_model_name(model_name)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    total_layers = config.num_hidden_layers

    if layer_start < 0 or layer_end > total_layers or layer_start >= layer_end:
        raise ValueError(
            f"Invalid layer range [{layer_start}, {layer_end}) for model "
            f"with {total_layers} layers"
        )

    target_device = _detect_device(device)

    is_first_shard = layer_start == 0
    is_last_shard = layer_end == total_layers

    logger.info(
        "Loading shard [%d, %d) of %d layers from %s (first=%s, last=%s, device=%s)",
        layer_start, layer_end, total_layers, model_name,
        is_first_shard, is_last_shard, target_device,
    )

    load_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "config": config,
        "trust_remote_code": False,
        "torch_dtype": torch.float16,
    }

    if quantize:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = "auto"
        except ImportError:
            logger.warning("bitsandbytes not available, loading without quantization")

    if not quantize:
        load_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    inner_model = _get_inner_model(model)
    all_layers = inner_model.layers

    shard_layers = nn.ModuleList(
        [all_layers[i] for i in range(layer_start, layer_end)]
    )

    embed_tokens = inner_model.embed_tokens if is_first_shard else None
    norm = inner_model.norm if is_last_shard else None
    lm_head = model.lm_head if is_last_shard else None
    rotary_emb = inner_model.rotary_emb if hasattr(inner_model, "rotary_emb") else None

    shard = {
        "layers": shard_layers,
        "embed_tokens": embed_tokens,
        "norm": norm,
        "lm_head": lm_head,
        "rotary_emb": rotary_emb,
        "config": config,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "total_layers": total_layers,
    }

    del model, inner_model, all_layers

    if target_device != "cpu":
        _move_shard_to_device(shard, target_device)
        if target_device.startswith("cuda"):
            torch.cuda.empty_cache()

    actual_device = str(next(shard["layers"].parameters()).device)

    if actual_device.startswith("cuda"):
        vram_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        logger.info(
            "Shard loaded: %d layers, device=%s (verified), mem=%.1fMB, cuda_alloc=%.1fMB",
            layer_end - layer_start, actual_device,
            _estimate_shard_memory_mb(shard), vram_alloc,
        )
    else:
        logger.info(
            "Shard loaded: %d layers, device=%s (verified), mem=%.1fMB",
            layer_end - layer_start, actual_device,
            _estimate_shard_memory_mb(shard),
        )

    return shard


def _get_inner_model(model) -> nn.Module:
    """Navigate the model wrapper to get the inner transformer model.

    Works for Qwen3, Qwen2.5, and similar architectures where
    the structure is model.model.layers.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        raise NotImplementedError(
            "GPT-style architectures (transformer.h) not yet supported. "
            "Use Llama or Qwen2.5."
        )
    raise ValueError(
        f"Unsupported model architecture: cannot find transformer layers in "
        f"{type(model).__name__}"
    )


def _move_shard_to_device(shard: dict, device: str) -> None:
    """Move shard components to the target device."""
    if device not in ("cpu", "cuda", "mps") and not device.startswith("cuda:"):
        device = "cpu"
    shard["layers"] = shard["layers"].to(device)
    if shard["embed_tokens"] is not None:
        shard["embed_tokens"] = shard["embed_tokens"].to(device)
    if shard["norm"] is not None:
        shard["norm"] = shard["norm"].to(device)
    if shard["lm_head"] is not None:
        shard["lm_head"] = shard["lm_head"].to(device)
    if shard.get("rotary_emb") is not None:
        shard["rotary_emb"] = shard["rotary_emb"].to(device)


def _cleanup_unused_params(model, shard: dict) -> None:
    """Delete references to layers not in the shard to free memory."""
    del model


def _estimate_shard_memory_mb(shard: dict) -> float:
    """Rough estimate of shard memory usage in MB."""
    total_params = 0
    for key in ["layers", "embed_tokens", "norm", "lm_head", "rotary_emb"]:
        component = shard.get(key)
        if component is not None and hasattr(component, "parameters"):
            total_params += sum(p.numel() for p in component.parameters())
    bytes_per_param = 2  # assume float16
    return (total_params * bytes_per_param) / (1024 * 1024)


@torch.inference_mode()
def forward_shard(
    shard: dict,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    kv_cache=None,
) -> tuple[torch.Tensor, object]:
    """Run a forward pass through a model shard.

    Args:
        shard: Shard dict from load_model_shard.
        hidden_states: Input tensor. If this is the first shard and
            embed_tokens is present, this should be token IDs (long tensor).
            Otherwise, hidden state activations.
        position_ids: Position IDs tensor, shape (batch, seq_len).
            If None, auto-computed from cache length.
        attention_mask: Attention mask, shape (batch, seq_len).
        kv_cache: A transformers DynamicCache instance, or None.

    Returns:
        (output_tensor, kv_cache) where output_tensor is logits if this is the
        last shard, or hidden states otherwise.
    """
    from transformers import DynamicCache

    is_first_shard = shard["embed_tokens"] is not None
    is_last_shard = shard["lm_head"] is not None

    if is_first_shard:
        hidden_states = shard["embed_tokens"](hidden_states)

    if kv_cache is None:
        kv_cache = DynamicCache()

    if position_ids is None:
        seq_len = hidden_states.shape[1] if hidden_states.dim() >= 2 else 1
        cache_len = kv_cache.get_seq_length()
        position_ids = torch.arange(
            cache_len, cache_len + seq_len, device=hidden_states.device
        ).unsqueeze(0)

    position_embeddings = None
    rotary_emb = shard.get("rotary_emb")
    if rotary_emb is not None:
        position_embeddings = rotary_emb(hidden_states, position_ids)

    for layer in shard["layers"]:
        layer_kwargs = {"hidden_states": hidden_states}
        if position_embeddings is not None:
            layer_kwargs["position_embeddings"] = position_embeddings
        if position_ids is not None:
            layer_kwargs["position_ids"] = position_ids
        if attention_mask is not None:
            layer_kwargs["attention_mask"] = attention_mask
        layer_kwargs["past_key_values"] = kv_cache
        layer_kwargs["use_cache"] = True

        output = layer(**layer_kwargs)
        hidden_states = output[0] if isinstance(output, tuple) else output

    _mps_sync = hasattr(torch, 'mps') and str(hidden_states.device).startswith('mps')

    if is_last_shard:
        hidden_states = shard["norm"](hidden_states)
        logits = shard["lm_head"](hidden_states)
        if _mps_sync:
            torch.mps.synchronize()
        return logits, kv_cache

    if _mps_sync:
        torch.mps.synchronize()
    return hidden_states, kv_cache
