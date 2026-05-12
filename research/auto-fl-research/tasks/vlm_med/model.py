# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn

DEFAULT_MODEL_ARCH = "qwen3vl_lora_adapter"
DEFAULT_MAX_MODEL_PARAMS = 64_000_000

_ADAPTER_KEY_DOT = "__dot__"
QWEN3VL_ADAPTER_SHAPE_FIELDS = (
    "adapter_num_hidden_layers",
    "adapter_hidden_size",
    "adapter_num_key_value_heads",
    "adapter_head_dim",
)
DEFAULT_QWEN3VL_ADAPTER_SHAPE = {
    "adapter_num_hidden_layers": 36,
    "adapter_hidden_size": 4096,
    "adapter_num_key_value_heads": 8,
    "adapter_head_dim": 128,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _capture_rng_state():
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": cuda_state,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]


def _safe_adapter_key(key: str) -> str:
    return key.replace(".", _ADAPTER_KEY_DOT)


def _original_adapter_key(key: str) -> str:
    return key.replace(_ADAPTER_KEY_DOT, ".")


def qwen3vl_adapter_shape_from_config(model_name_or_path: str) -> dict[str, int]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path)
    text_config = getattr(config, "text_config", None) or getattr(config, "llm_config", None) or config

    num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    num_key_value_heads = getattr(text_config, "num_key_value_heads", None)
    head_dim = getattr(text_config, "head_dim", None)

    if head_dim is None:
        num_attention_heads = getattr(text_config, "num_attention_heads", None)
        if hidden_size is not None and num_attention_heads:
            head_dim = hidden_size // num_attention_heads
    if num_key_value_heads is None:
        num_key_value_heads = getattr(text_config, "num_attention_heads", None)

    shape = {
        "adapter_num_hidden_layers": num_hidden_layers,
        "adapter_hidden_size": hidden_size,
        "adapter_num_key_value_heads": num_key_value_heads,
        "adapter_head_dim": head_dim,
    }
    missing = [name for name, value in shape.items() if value is None]
    if missing:
        raise ValueError(
            f"Could not infer Qwen3-VL LoRA adapter shape from {model_name_or_path!r}; "
            f"missing: {', '.join(missing)}"
        )
    return {name: int(value) for name, value in shape.items()}


def resolve_qwen3vl_adapter_shape(model_name_or_path: str | None = None, **overrides) -> dict[str, int]:
    shape = {
        name: int(overrides.get(name, 0) or 0)
        for name in QWEN3VL_ADAPTER_SHAPE_FIELDS
    }
    if any(value <= 0 for value in shape.values()):
        inferred = (
            qwen3vl_adapter_shape_from_config(model_name_or_path)
            if model_name_or_path
            else DEFAULT_QWEN3VL_ADAPTER_SHAPE
        )
        for name, value in shape.items():
            if value <= 0:
                shape[name] = inferred[name]

    invalid = [f"{name}={value}" for name, value in shape.items() if value <= 0]
    if invalid:
        raise ValueError("Invalid Qwen3-VL LoRA adapter shape values: " + ", ".join(invalid))
    return shape


def qwen3vl_lora_keys(
    *,
    num_hidden_layers: int = 36,
    hidden_size: int = 4096,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
):
    kv_size = num_key_value_heads * head_dim
    for layer_idx in range(num_hidden_layers):
        prefix = f"base_model.model.model.language_model.layers.{layer_idx}.self_attn"
        for proj, out_features in (
            ("q_proj", hidden_size),
            ("k_proj", kv_size),
            ("v_proj", kv_size),
            ("o_proj", hidden_size),
        ):
            yield f"{prefix}.{proj}.lora_A.weight", (None, hidden_size)
            yield f"{prefix}.{proj}.lora_B.weight", (out_features, None)


class Qwen3VLLoRAAdapterState(nn.Module):
    """Adapter-only state model for NVFlare aggregation.

    NVFlare's recipe persists and broadcasts a PyTorch ``state_dict``. Sending
    a full Qwen3-VL model would be wasteful, so this module exposes only the
    PEFT LoRA adapter tensors with sanitized parameter names. Clients map these
    tensors into their local Qwen3-VL PEFT model before training and map them
    back before computing the DIFF upload.
    """

    is_vlm_adapter_state = True

    def __init__(
        self,
        *,
        seed: int = 42,
        lora_r: int = 32,
        num_hidden_layers: int = 36,
        hidden_size: int = 4096,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
    ):
        super().__init__()
        if lora_r <= 0:
            raise ValueError("lora_r must be positive")

        rng_state = _capture_rng_state()
        set_seed(seed)
        try:
            self.adapter_key_map: dict[str, str] = {}
            for original_key, shape in qwen3vl_lora_keys(
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
            ):
                resolved_shape = tuple(lora_r if dim is None else dim for dim in shape)
                tensor = torch.empty(resolved_shape, dtype=torch.float32)
                if ".lora_A." in original_key:
                    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
                else:
                    nn.init.zeros_(tensor)
                safe_key = _safe_adapter_key(original_key)
                self.adapter_key_map[safe_key] = original_key
                self.register_parameter(safe_key, nn.Parameter(tensor))
        finally:
            _restore_rng_state(rng_state)

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            self.adapter_key_map[safe_key]: tensor.detach().cpu()
            for safe_key, tensor in self.state_dict().items()
        }


def _cpu_float_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value).detach().cpu().float()


def adapter_state_to_peft_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {_original_adapter_key(key): _cpu_float_tensor(value) for key, value in state_dict.items()}


def peft_state_to_adapter_state(peft_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {_safe_adapter_key(key): _cpu_float_tensor(value) for key, value in peft_state.items()}


MODEL_ARCHITECTURES = {
    "qwen3vl_lora_adapter": Qwen3VLLoRAAdapterState,
}


def available_model_architectures():
    return tuple(sorted(MODEL_ARCHITECTURES))


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def build_model(
    model_arch: str = DEFAULT_MODEL_ARCH,
    seed: int = 42,
    max_model_params: int | None = DEFAULT_MAX_MODEL_PARAMS,
    lora_r: int = 32,
    adapter_num_hidden_layers: int = DEFAULT_QWEN3VL_ADAPTER_SHAPE["adapter_num_hidden_layers"],
    adapter_hidden_size: int = DEFAULT_QWEN3VL_ADAPTER_SHAPE["adapter_hidden_size"],
    adapter_num_key_value_heads: int = DEFAULT_QWEN3VL_ADAPTER_SHAPE["adapter_num_key_value_heads"],
    adapter_head_dim: int = DEFAULT_QWEN3VL_ADAPTER_SHAPE["adapter_head_dim"],
) -> nn.Module:
    if model_arch not in MODEL_ARCHITECTURES:
        choices = ", ".join(available_model_architectures())
        raise ValueError(f"Unknown model_arch={model_arch!r}; expected one of: {choices}")

    rng_state = _capture_rng_state()
    try:
        model = MODEL_ARCHITECTURES[model_arch](
            seed=seed,
            lora_r=lora_r,
            num_hidden_layers=adapter_num_hidden_layers,
            hidden_size=adapter_hidden_size,
            num_key_value_heads=adapter_num_key_value_heads,
            head_dim=adapter_head_dim,
        )
    finally:
        _restore_rng_state(rng_state)

    param_count = count_parameters(model)
    if max_model_params is not None and max_model_params > 0 and param_count > max_model_params:
        raise ValueError(
            f"model_arch={model_arch} has {param_count:,} parameters, "
            f"which exceeds max_model_params={max_model_params:,}"
        )
    return model
