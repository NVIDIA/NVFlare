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
"""Adapter checkpoint helpers for federated LoRA PEFT examples."""

from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict
from enum import Enum
from typing import Any, Mapping

import torch

PERSISTENCE_KEY_MODEL = "model"
PERSISTENCE_KEY_TRAIN_CONF = "train_conf"
PERSISTENCE_KEY_META_PROPS = "meta_props"
ADAPTER_CONFIG_KEY = "adapter_config"
NVFLARE_MODEL_PREFIX = "model."
HF_PEFT_BASE_MODEL_PREFIX = "base_model.model."


def _metadata_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _metadata_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_metadata_safe(v) for v in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _torch_load(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_safetensors(path: str) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise RuntimeError("safetensors is required to load Hugging Face adapter_model.safetensors files.") from e
    return load_file(path, device="cpu")


def _find_adapter_file(path: str) -> str:
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Adapter path does not exist: {path}")

    candidates = [
        os.path.join(path, "adapter_model.safetensors"),
        os.path.join(path, "pytorch_model.bin"),
        os.path.join(path, "adapter_model.bin"),
        os.path.join(path, "model.pt"),
    ]
    for pattern in ("*.safetensors", "*.pt", "*.pth", "*.bin"):
        candidates.extend(sorted(glob.glob(os.path.join(path, pattern))))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No adapter checkpoint file found in {path}.")


def _unwrap_state_dict(data: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected a mapping checkpoint, got {type(data).__name__}.")

    for key in (PERSISTENCE_KEY_MODEL, "state_dict", "adapter_state_dict"):
        value = data.get(key)
        if isinstance(value, Mapping):
            return value
    return data


def load_adapter_state(path: str) -> OrderedDict[str, torch.Tensor]:
    """Load a LoRA adapter state dict from NVFlare, PyTorch, or Hugging Face adapter formats."""
    adapter_file = _find_adapter_file(path)
    if adapter_file.endswith(".safetensors"):
        data = _load_safetensors(adapter_file)
    else:
        data = _torch_load(adapter_file)

    state = _unwrap_state_dict(data)
    tensors = OrderedDict()
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensors[str(key)] = value.detach().cpu()
    if not tensors:
        raise ValueError(f"No tensor adapter weights found in {adapter_file}.")
    return tensors


def load_adapter_config(path: str) -> dict[str, Any] | None:
    """Load adapter_config metadata from a directory or NVFlare checkpoint when available."""
    if os.path.isdir(path):
        config_path = os.path.join(path, "adapter_config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                return json.load(f)
        return None

    data = _torch_load(path)
    if isinstance(data, Mapping):
        adapter_config = data.get(ADAPTER_CONFIG_KEY)
        if isinstance(adapter_config, Mapping):
            return dict(adapter_config)
        meta_props = data.get(PERSISTENCE_KEY_META_PROPS)
        if isinstance(meta_props, Mapping) and isinstance(meta_props.get(ADAPTER_CONFIG_KEY), Mapping):
            return dict(meta_props[ADAPTER_CONFIG_KEY])
    return None


def save_nvflare_adapter_checkpoint(
    state_dict: Mapping[str, torch.Tensor],
    path: str,
    train_conf: Mapping[str, Any] | None = None,
    adapter_config: Mapping[str, Any] | None = None,
) -> None:
    """Save adapter tensors in the format consumed by PTFileModelPersistor."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = OrderedDict()
    data[PERSISTENCE_KEY_MODEL] = OrderedDict((key, value.detach().cpu()) for key, value in state_dict.items())
    data[PERSISTENCE_KEY_TRAIN_CONF] = dict(train_conf or {"train": {"model": "Nemotron3NanoLoRA"}})
    if adapter_config:
        safe_adapter_config = _metadata_safe(adapter_config)
        data[ADAPTER_CONFIG_KEY] = safe_adapter_config
        data[PERSISTENCE_KEY_META_PROPS] = {ADAPTER_CONFIG_KEY: safe_adapter_config}
    torch.save(data, path)


def strip_model_prefix(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (key[len(NVFLARE_MODEL_PREFIX) :] if key.startswith(NVFLARE_MODEL_PREFIX) else key, value)
        for key, value in state_dict.items()
    )


def add_model_prefix(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (key if key.startswith(NVFLARE_MODEL_PREFIX) else f"{NVFLARE_MODEL_PREFIX}{key}", value)
        for key, value in state_dict.items()
    )


def canonical_adapter_key(key: str) -> str:
    """Return an adapter key without NVFlare or Hugging Face PEFT wrapper prefixes."""
    if key.startswith(NVFLARE_MODEL_PREFIX):
        key = key[len(NVFLARE_MODEL_PREFIX) :]
    if key.startswith(HF_PEFT_BASE_MODEL_PREFIX):
        key = key[len(HF_PEFT_BASE_MODEL_PREFIX) :]
    return key


def match_adapter_state_to_reference(
    state_dict: Mapping[str, torch.Tensor],
    reference_state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Map adapter tensors to the reference key namespace using canonical LoRA parameter names."""
    state_by_canonical_key = {canonical_adapter_key(key): value for key, value in state_dict.items()}
    matched = OrderedDict()
    for reference_key in reference_state_dict:
        canonical_key = canonical_adapter_key(reference_key)
        if canonical_key in state_by_canonical_key:
            matched[reference_key] = state_by_canonical_key[canonical_key]
    return matched


def state_dict_size_mb(state_dict: Mapping[str, torch.Tensor]) -> float:
    total_bytes = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total_bytes += value.numel() * value.element_size()
    return total_bytes / (1024 * 1024)


def save_hf_adapter_state_dir(
    state_dict: Mapping[str, torch.Tensor],
    output_dir: str,
    adapter_config: Mapping[str, Any] | None = None,
) -> str:
    """Save adapter weights in a Hugging Face PEFT-style directory."""
    os.makedirs(output_dir, exist_ok=True)
    if adapter_config:
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(_metadata_safe(adapter_config), f, indent=2, sort_keys=True)

    try:
        from safetensors.torch import save_file

        adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
        save_file({key: value.detach().cpu() for key, value in state_dict.items()}, adapter_file)
    except ImportError:
        adapter_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(OrderedDict((key, value.detach().cpu()) for key, value in state_dict.items()), adapter_file)
    return adapter_file
