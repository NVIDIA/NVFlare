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
"""Full-model checkpoint helpers for federated NeMo AutoModel SFT."""

from __future__ import annotations

import glob
import os
from collections import OrderedDict
from typing import Any, Mapping

import torch

PERSISTENCE_KEY_MODEL = "model"
PERSISTENCE_KEY_TRAIN_CONF = "train_conf"
PERSISTENCE_KEY_META_PROPS = "meta_props"


def _torch_load(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_safetensors(path: str) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise RuntimeError("safetensors is required to load model.safetensors checkpoint files.") from e
    return load_file(path, device="cpu")


def _state_files_in_dir(path: str) -> list[str]:
    candidates = [
        os.path.join(path, "model.safetensors"),
        os.path.join(path, "pytorch_model.bin"),
        os.path.join(path, "model.pt"),
        os.path.join(path, "FL_global_model.pt"),
    ]
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        candidates.extend(sorted(glob.glob(os.path.join(path, pattern))))

    result = []
    seen = set()
    for candidate in candidates:
        if os.path.isfile(candidate) and candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def _candidate_state_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Model checkpoint path does not exist: {path}")

    preferred_dirs = [
        path,
        os.path.join(path, "model", "consolidated"),
        os.path.join(path, "model"),
    ]
    for directory in preferred_dirs:
        if os.path.isdir(directory):
            candidates = _state_files_in_dir(directory)
            if candidates:
                return candidates

    candidates = []
    for root, _dirs, files in os.walk(path):
        if any(name.endswith(".safetensors") for name in files):
            candidates.extend(_state_files_in_dir(root))
    return candidates


def _unwrap_state_dict(data: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected a mapping checkpoint, got {type(data).__name__}.")

    for key in (PERSISTENCE_KEY_MODEL, "state_dict", "model_state_dict"):
        value = data.get(key)
        if isinstance(value, Mapping):
            return value
    return data


def load_model_state(path: str) -> OrderedDict[str, torch.Tensor]:
    """Load full-model tensors from an NVFlare, PyTorch, or safetensors checkpoint path."""
    files = _candidate_state_files(path)
    if not files:
        raise FileNotFoundError(f"No model checkpoint file found in {path}.")

    tensors = OrderedDict()
    for file_path in files:
        if file_path.endswith(".safetensors"):
            state = _load_safetensors(file_path)
        else:
            state = _unwrap_state_dict(_torch_load(file_path))
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                tensors[str(key)] = value.detach().cpu()
    if not tensors:
        raise ValueError(f"No tensor model weights found in {path}.")
    return tensors


def save_nvflare_model_checkpoint(
    state_dict: Mapping[str, torch.Tensor],
    path: str,
    train_conf: Mapping[str, Any] | None = None,
) -> None:
    """Save full-model tensors in the format consumed by PTFileModelPersistor."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = OrderedDict()
    data[PERSISTENCE_KEY_MODEL] = OrderedDict((key, value.detach().cpu()) for key, value in state_dict.items())
    data[PERSISTENCE_KEY_TRAIN_CONF] = dict(train_conf or {"train": {"model": "Nemotron3NanoSFT"}})
    torch.save(data, path)


def match_model_state_to_reference(
    state_dict: Mapping[str, torch.Tensor],
    reference_state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Return updated tensors in the reference key namespace."""
    matched = OrderedDict()
    for key in reference_state_dict:
        if key in state_dict:
            matched[key] = state_dict[key]
    return matched


def state_dict_size_mb(state_dict: Mapping[str, torch.Tensor]) -> float:
    total_bytes = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total_bytes += value.numel() * value.element_size()
    return total_bytes / (1024 * 1024)
