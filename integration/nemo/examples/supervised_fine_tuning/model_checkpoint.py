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
STATE_PREVIEW_LIMIT = 5


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


def count_model_state_tensors(state_dict: Mapping[str, Any]) -> int:
    return sum(1 for value in state_dict.values() if isinstance(value, torch.Tensor))


def _preview_values(values: list[str]) -> str:
    preview = ", ".join(values[:STATE_PREVIEW_LIMIT])
    if len(values) > STATE_PREVIEW_LIMIT:
        preview += f", ... ({len(values) - STATE_PREVIEW_LIMIT} more)"
    return preview


def _preview_shape_mismatches(shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]]) -> str:
    values = [
        f"{key}: candidate {candidate_shape}, reference {reference_shape}"
        for key, candidate_shape, reference_shape in shape_mismatches
    ]
    return _preview_values(values)


def validate_model_state_coverage(
    candidate_state_dict: Mapping[str, torch.Tensor],
    reference_state_dict: Mapping[str, torch.Tensor],
    candidate_name: str = "candidate model state",
    reference_name: str = "reference model state",
) -> None:
    """Verify that every reference tensor exists in the candidate state with the same shape."""
    missing = []
    shape_mismatches = []
    for key, reference_value in reference_state_dict.items():
        if not isinstance(reference_value, torch.Tensor):
            continue
        candidate_value = candidate_state_dict.get(key)
        if not isinstance(candidate_value, torch.Tensor):
            missing.append(key)
            continue
        if candidate_value.shape != reference_value.shape:
            shape_mismatches.append((key, tuple(candidate_value.shape), tuple(reference_value.shape)))

    if not missing and not shape_mismatches:
        return

    reference_tensor_count = count_model_state_tensors(reference_state_dict)
    matched = reference_tensor_count - len(missing) - len(shape_mismatches)
    details = [f"matched {matched}/{reference_tensor_count} tensors"]
    if missing:
        details.append(f"missing keys: {_preview_values(missing)}")
    if shape_mismatches:
        details.append(f"shape mismatches: {_preview_shape_mismatches(shape_mismatches)}")
    raise RuntimeError(f"{candidate_name} does not cover all tensors from {reference_name}; " + "; ".join(details))


def match_model_state_to_reference(
    state_dict: Mapping[str, torch.Tensor],
    reference_state_dict: Mapping[str, torch.Tensor],
    require_all: bool = False,
    candidate_name: str = "candidate model state",
    reference_name: str = "reference model state",
) -> OrderedDict[str, torch.Tensor]:
    """Return updated tensors in the reference key namespace."""
    if require_all:
        validate_model_state_coverage(state_dict, reference_state_dict, candidate_name, reference_name)

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
