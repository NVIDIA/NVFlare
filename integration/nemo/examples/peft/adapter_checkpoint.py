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
import hashlib
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
ADAPTER_MANIFEST_KEY = "adapter_manifest"
ADAPTER_MANIFEST_FILE = "nvflare_adapter_manifest.json"
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


def load_adapter_manifest(path: str) -> dict[str, Any] | None:
    """Load the strict NVFlare adapter manifest from a directory or PyTorch checkpoint."""
    if os.path.isdir(path):
        manifest_path = os.path.join(path, ADAPTER_MANIFEST_FILE)
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                return json.load(f)
        model_manifest_path = os.path.join(path, "model", ADAPTER_MANIFEST_FILE)
        if os.path.isfile(model_manifest_path):
            with open(model_manifest_path) as f:
                return json.load(f)
        return None

    data = _torch_load(path)
    if isinstance(data, Mapping):
        manifest = data.get(ADAPTER_MANIFEST_KEY)
        if isinstance(manifest, Mapping):
            return dict(manifest)
        meta_props = data.get(PERSISTENCE_KEY_META_PROPS)
        if isinstance(meta_props, Mapping) and isinstance(meta_props.get(ADAPTER_MANIFEST_KEY), Mapping):
            return dict(meta_props[ADAPTER_MANIFEST_KEY])
    return None


def save_nvflare_adapter_checkpoint(
    state_dict: Mapping[str, torch.Tensor],
    path: str,
    train_conf: Mapping[str, Any] | None = None,
    adapter_config: Mapping[str, Any] | None = None,
    adapter_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Save adapter tensors in the format consumed by PTFileModelPersistor."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = OrderedDict()
    data[PERSISTENCE_KEY_MODEL] = OrderedDict((key, value.detach().cpu()) for key, value in state_dict.items())
    data[PERSISTENCE_KEY_TRAIN_CONF] = dict(train_conf or {"train": {"model": "Nemotron3NanoLoRA"}})
    meta_props = {}
    if adapter_config:
        safe_adapter_config = _metadata_safe(adapter_config)
        data[ADAPTER_CONFIG_KEY] = safe_adapter_config
        meta_props[ADAPTER_CONFIG_KEY] = safe_adapter_config
    if adapter_manifest:
        safe_manifest = _metadata_safe(adapter_manifest)
        data[ADAPTER_MANIFEST_KEY] = safe_manifest
        meta_props[ADAPTER_MANIFEST_KEY] = safe_manifest
    if meta_props:
        data[PERSISTENCE_KEY_META_PROPS] = meta_props
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


def _canonical_state(state_dict: Mapping[str, torch.Tensor], normalize_peft_prefixes: bool) -> OrderedDict:
    canonical = {}
    for key, value in state_dict.items():
        normalized = canonical_adapter_key(key) if normalize_peft_prefixes else key
        if normalized in canonical:
            raise ValueError(f"Duplicate adapter key after normalization: {normalized}")
        canonical[normalized] = (key, value)
    return OrderedDict((key, canonical[key]) for key in sorted(canonical))


def align_adapter_state_strict(
    state_dict: Mapping[str, torch.Tensor],
    reference_state_dict: Mapping[str, torch.Tensor],
    *,
    normalize_peft_prefixes: bool = False,
) -> OrderedDict[str, torch.Tensor]:
    """Return ``state_dict`` in the reference namespace after complete validation."""
    state = _canonical_state(state_dict, normalize_peft_prefixes)
    reference = _canonical_state(reference_state_dict, normalize_peft_prefixes)
    missing = sorted(set(reference) - set(state))
    unexpected = sorted(set(state) - set(reference))
    if missing or unexpected:
        raise ValueError(
            f"Adapter key mismatch: missing={len(missing)} {missing[:5]}, "
            f"unexpected={len(unexpected)} {unexpected[:5]}"
        )

    aligned = OrderedDict()
    for normalized_key, (reference_key, reference_value) in reference.items():
        _source_key, value = state[normalized_key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Adapter value for {reference_key} is not a tensor.")
        if tuple(value.shape) != tuple(reference_value.shape):
            raise ValueError(
                f"Adapter shape mismatch for {reference_key}: got {tuple(value.shape)}, "
                f"expected {tuple(reference_value.shape)}"
            )
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise ValueError(f"Adapter tensor contains non-finite values: {reference_key}")
        aligned[reference_key] = value.detach().cpu()
    if not aligned:
        raise ValueError("Adapter state is empty.")
    return aligned


def tensor_specs(state_dict: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
        for key, value in sorted(state_dict.items())
    }


def state_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensors in sorted key order after conversion to canonical little-endian bytes."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def update_norm(incoming_state: Mapping[str, torch.Tensor], outgoing_state: Mapping[str, torch.Tensor]) -> float:
    outgoing = align_adapter_state_strict(outgoing_state, incoming_state)
    squared_norm = 0.0
    for key, incoming in incoming_state.items():
        delta = outgoing[key].float() - incoming.detach().cpu().float()
        squared_norm += float(torch.sum(delta * delta).item())
    return squared_norm**0.5


def build_adapter_manifest(
    state_dict: Mapping[str, torch.Tensor],
    *,
    model_profile: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    profile_settings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_profile": model_profile,
        "base_model_name_or_path": model_name_or_path,
        "base_model_revision": model_revision,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "tokenizer_revision": tokenizer_revision,
        "profile_settings": _metadata_safe(profile_settings),
        "adapter_hash": state_hash(state_dict),
        "tensor_count": len(state_dict),
        "expected_tensors": tensor_specs(state_dict),
    }


def validate_adapter_manifest(
    manifest: Mapping[str, Any] | None,
    state_dict: Mapping[str, torch.Tensor],
    expected: Mapping[str, Any] | None = None,
) -> None:
    if not manifest:
        raise ValueError("Adapter checkpoint is missing its NVFlare adapter manifest.")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported adapter manifest schema: {manifest.get('schema_version')}")
    if manifest.get("adapter_hash") != state_hash(state_dict):
        raise ValueError("Adapter manifest hash does not match the adapter tensors.")
    if manifest.get("tensor_count") != len(state_dict):
        raise ValueError("Adapter manifest tensor count does not match the adapter tensors.")
    if manifest.get("expected_tensors") != tensor_specs(state_dict):
        raise ValueError("Adapter manifest tensor names, shapes, or dtypes do not match the adapter tensors.")
    for key, value in (expected or {}).items():
        if value is not None and manifest.get(key) != value:
            raise ValueError(f"Adapter manifest conflict for {key}: got {manifest.get(key)!r}, expected {value!r}.")


def validate_adapter_contract(
    contract: Mapping[str, Any] | None,
    state_dict: Mapping[str, torch.Tensor],
    expected: Mapping[str, Any] | None = None,
) -> None:
    """Validate an adapter against the immutable identity and tensor schema from initialization."""
    if not contract:
        raise ValueError("Adapter contract is missing.")
    if contract.get("schema_version") != 1:
        raise ValueError(f"Unsupported adapter contract schema: {contract.get('schema_version')}")
    state_dict = align_adapter_state_strict(state_dict, state_dict)
    if contract.get("tensor_count") != len(state_dict):
        raise ValueError("Adapter contract tensor count does not match the adapter tensors.")
    if contract.get("expected_tensors") != tensor_specs(state_dict):
        raise ValueError("Adapter contract tensor names, shapes, or dtypes do not match the adapter tensors.")
    for key, value in (expected or {}).items():
        if value is not None and contract.get(key) != value:
            raise ValueError(f"Adapter contract conflict for {key}: got {contract.get(key)!r}, expected {value!r}.")


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
    adapter_manifest: Mapping[str, Any] | None = None,
) -> str:
    """Save adapter weights in a Hugging Face PEFT-style directory."""
    os.makedirs(output_dir, exist_ok=True)
    if adapter_config:
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(_metadata_safe(adapter_config), f, indent=2, sort_keys=True)
    if adapter_manifest:
        with open(os.path.join(output_dir, ADAPTER_MANIFEST_FILE), "w") as f:
            json.dump(_metadata_safe(adapter_manifest), f, indent=2, sort_keys=True)

    try:
        from safetensors.torch import save_file

        adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
        save_file({key: value.detach().cpu() for key, value in state_dict.items()}, adapter_file)
    except ImportError:
        adapter_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(OrderedDict((key, value.detach().cpu()) for key, value in state_dict.items()), adapter_file)
    return adapter_file


def make_adapter_persistor_class():
    """Create the example-local persistor without importing NVFlare for checkpoint-only tools."""
    from nvflare.app_common.abstract.model import ModelLearnable
    from nvflare.app_common.app_constant import AppConstants
    from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

    class AdapterPTFileModelPersistor(PTFileModelPersistor):
        def __init__(self, *args, manifest_template: dict | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.manifest_template = dict(manifest_template or {})

        def save_model(self, ml: ModelLearnable, fl_ctx):
            manager = self._get_persistence_manager(fl_ctx)
            manager.update(ml)
            state = strip_model_prefix(manager.var_dict)
            state = align_adapter_state_strict(state, state)
            manifest = build_adapter_manifest(
                state,
                model_profile=self.manifest_template.get("model_profile", "nano"),
                model_name_or_path=self.manifest_template.get("base_model_name_or_path", ""),
                tokenizer_name_or_path=self.manifest_template.get("tokenizer_name_or_path", ""),
                model_revision=self.manifest_template.get("base_model_revision"),
                tokenizer_revision=self.manifest_template.get("tokenizer_revision"),
                profile_settings=self.manifest_template.get("profile_settings", {}),
            )
            manager.other_props[ADAPTER_MANIFEST_KEY] = manifest
            if manager.meta is None:
                manager.meta = {}
            manager.meta[ADAPTER_MANIFEST_KEY] = manifest
            self.save_model_file(self._ckpt_save_path)

            current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
            if current_round is None:
                return
            round_dir = os.path.join(self.log_dir, "server_rounds", f"round_{int(current_round)}")
            os.makedirs(round_dir, exist_ok=True)
            round_checkpoint = os.path.join(round_dir, self.global_model_file_name)
            self.save_model_file(round_checkpoint)
            round_manifest = {
                "schema_version": 1,
                "round": int(current_round),
                "aggregate_adapter_hash": manifest["adapter_hash"],
                "tensor_count": manifest["tensor_count"],
                "checkpoint_location": os.path.abspath(round_checkpoint),
                "aggregation_stats": _metadata_safe(fl_ctx.get_prop(AppConstants.AGGREGATION_STATS) or {}),
            }
            with open(os.path.join(round_dir, "round_manifest.json"), "w") as f:
                json.dump(round_manifest, f, indent=2, sort_keys=True)

    AdapterPTFileModelPersistor.__module__ = __name__
    AdapterPTFileModelPersistor.__qualname__ = "AdapterPTFileModelPersistor"
    return AdapterPTFileModelPersistor


AdapterPTFileModelPersistor = make_adapter_persistor_class()
