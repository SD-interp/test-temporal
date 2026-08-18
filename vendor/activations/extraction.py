"""Load a Hugging Face causal LM and read one residual-stream activation.

The extraction contract is fixed, exactly as in the source project: the output
of transformer layer 21 (`layer_out/21`) at the final token of the formatted
prompt (position -1).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .tokenizer import ChatTemplateTokenizer

TARGET_LAYER = 21
TARGET_COMPONENT = "layer_out"
TARGET_LAYER_COMPONENT = f"{TARGET_COMPONENT}/{TARGET_LAYER}"
PROMPT_TOKEN_POSITION = -1


def get_default_device() -> str:
    """Return cuda if available, else mps, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def load_model_tokenizer(
    model_name: str,
    *,
    hf_token: str | None = None,
    device: str | None = None,
    dtype: str | torch.dtype | None = None,
    padding_side: str = "left",
    attn_type: str = "sdpa",
    system_prompt: str = "",
    suffix: str = "",
) -> tuple[PreTrainedModel, ChatTemplateTokenizer]:
    """Download (or reuse the cache for) a model and its chat tokenizer."""
    if device is None:
        device = get_default_device()

    auth = {"token": hf_token} if hf_token else {}

    config = AutoConfig.from_pretrained(model_name, **auth)
    config._attn_implementation = attn_type

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side=padding_side, **auth
    )
    chat_tokenizer = ChatTemplateTokenizer(
        tokenizer, suffix=suffix, system_prompt=system_prompt
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, **auth).eval()

    if dtype is not None:
        if isinstance(dtype, str):
            dtype_str = dtype.replace("torch.", "")
            if not hasattr(torch, dtype_str):
                raise ValueError(
                    f"Invalid dtype string: {dtype}. Expected e.g. 'float16', 'bfloat16'."
                )
            dtype = getattr(torch, dtype_str)
        model = model.to(dtype=dtype, device=device)
    else:
        model = model.to(device)

    _validate_target_layer(model)
    return model, chat_tokenizer


def _validate_target_layer(model: PreTrainedModel) -> None:
    """Fail early if the model has no layer 21 to hook."""
    num_layers = int(model.config.num_hidden_layers)
    if num_layers <= TARGET_LAYER:
        raise ValueError(
            f"Model has {num_layers} layers; extraction targets layer {TARGET_LAYER}."
        )


def _target_module(model: PreTrainedModel) -> torch.nn.Module:
    """Return the module whose output is `layer_out/21`."""
    module_name = f"model.layers.{TARGET_LAYER}"
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise ValueError(f"Module {module_name!r} not found in model.")
    return modules[module_name]


def _first_tensor(value) -> torch.Tensor:
    """Hooks receive tensors or tuples; take the first tensor payload."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item.detach()
    raise TypeError(f"Expected tensor or sequence containing a tensor, got {type(value)!r}")


@contextmanager
def _cache_target_layer(model: PreTrainedModel, cache: dict):
    """Register a forward hook that caches layer 21's output, then remove it."""
    def hook(module, inputs, output):
        cache[TARGET_LAYER_COMPONENT] = _first_tensor(output).clone()

    handle = _target_module(model).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def extract_last_position_activation(
    model: PreTrainedModel,
    tokenizer: ChatTemplateTokenizer,
    prompt: str,
    *,
    thinking: bool = False,
) -> torch.Tensor:
    """Return the `layer_out/21` activation at prompt token -1 as a 1-D tensor.

    A single prompt means no padding, so position -1 is the last real token.
    """
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string.")

    inputs = tokenizer(prompt, thinking=thinking)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    cache: dict[str, torch.Tensor] = {}
    with torch.no_grad(), _cache_target_layer(model, cache):
        model(**inputs)

    if TARGET_LAYER_COMPONENT not in cache:
        raise RuntimeError(f"Forward pass did not produce {TARGET_LAYER_COMPONENT}.")

    activation = cache[TARGET_LAYER_COMPONENT]
    if activation.ndim != 3:
        raise ValueError(
            f"{TARGET_LAYER_COMPONENT} must be batch x position x hidden; "
            f"got {tuple(activation.shape)}."
        )
    return activation[0, PROMPT_TOKEN_POSITION, :].to("cpu", dtype=torch.float32)
