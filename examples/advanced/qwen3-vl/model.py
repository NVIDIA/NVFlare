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
"""
Qwen3-VL model for federated SFT.
Used by job.py (initial_model) and client.py (local training).
Provides an nn.Module interface so the PT persistor can save/load state_dict.
"""

import glob
import os
import warnings
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

# LoRA config must match Qwen train_qwen.py (argument.py / train_qwen.py)
DEFAULT_LORA_R = 64
DEFAULT_LORA_ALPHA = 128
DEFAULT_LORA_DROPOUT = 0.0
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
_PEFT_ADAPTER_MARKERS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


def load_state_dict_from_checkpoint(checkpoint_dir: str, lora_only: bool = False) -> dict:
    """Load state_dict from a HuggingFace-style checkpoint dir without loading the full model.

    Reads .safetensors (or pytorch_model.bin) so the client can send weights back to the server
    quickly and return to flare.receive(), avoiding cell_pipe send timeouts.

    If lora_only=True, loads only adapter weights (adapter_model.safetensors if present,
    otherwise filters full checkpoint to keys containing "lora").
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    state_dict = {}

    if lora_only:
        adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        if os.path.isfile(adapter_path):
            try:
                from safetensors.torch import load_file

                state_dict = load_file(adapter_path, device="cpu")
                return state_dict
            except Exception as e:
                warnings.warn(
                    f"Failed to load adapter weights from {adapter_path}: {e}. Falling back to full checkpoint."
                )
        # No adapter file: load full checkpoint and keep only LoRA keys
        state_dict = load_state_dict_from_checkpoint(checkpoint_dir, lora_only=False)
        state_dict = {k: v for k, v in state_dict.items() if "lora" in k.lower()}
        if not state_dict:
            raise FileNotFoundError(f"No adapter_model.safetensors and no lora keys in {checkpoint_dir}")
        return state_dict

    # Prefer safetensors (faster, no pickle)
    safetensor_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if safetensor_files:
        try:
            from safetensors.torch import load_file

            for path in safetensor_files:
                state_dict.update(load_file(path, device="cpu"))
            return state_dict
        except Exception:
            pass

    # Fallback: pytorch_model.bin
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        return state_dict

    raise FileNotFoundError(f"No .safetensors or pytorch_model.bin in {checkpoint_dir}")


def _get_qwen_vl_model_class(model_name_or_path: str):
    """Return the correct ForConditionalGeneration class for the model's config."""
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_vl_moe":
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration
        except ImportError as e:
            raise ValueError(
                "Installed transformers package does not provide Qwen3-VL-MoE support. "
                f"Cannot load model_type={model_type!r} from {model_name_or_path}."
            ) from e

        return Qwen3VLMoeForConditionalGeneration
    if model_type == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        except ImportError as e:
            raise ValueError(
                "Installed transformers package does not provide Qwen3-VL support. "
                f"Cannot load model_type={model_type!r} from {model_name_or_path}."
            ) from e
    if model_type == "qwen2_5_vl":
        return Qwen2_5_VLForConditionalGeneration
    if model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration
    raise ValueError(f"Unsupported Qwen VL model_type: {model_type!r} for {model_name_or_path}")


def load_qwen_vl_from_pretrained(model_name_or_path: str, **kwargs):
    """Load Qwen VL model from path or HF ID using the class that matches the checkpoint config."""
    model_cls = _get_qwen_vl_model_class(model_name_or_path)
    return model_cls.from_pretrained(model_name_or_path, **kwargs)


def _get_peft_adapter_state_dict(peft_model) -> dict:
    """Return only the LoRA adapter state dict (trainable params) from a PEFT model."""
    from peft import get_peft_model_state_dict

    adapter_state = get_peft_model_state_dict(peft_model)
    return adapter_state


def normalize_peft_adapter_key(key: str) -> str:
    for marker in _PEFT_ADAPTER_MARKERS:
        token = f".{marker}."
        default_token = f".{marker}.default."
        if token in key and default_token not in key:
            return key.replace(token, default_token, 1)
    return key


def is_peft_adapter_key(key: str) -> bool:
    return any(f".{marker}." in key for marker in _PEFT_ADAPTER_MARKERS)


def get_expected_peft_adapter_keys(peft_model) -> set:
    return {key for key in peft_model.state_dict().keys() if is_peft_adapter_key(key)}


def _adapter_key_to_model_key(adapter_key: str, model_keys: set) -> Optional[str]:
    """Map a single adapter state key to the peft_model's expected key (handles module. and .default)."""
    if adapter_key in model_keys:
        return adapter_key
    alt = normalize_peft_adapter_key(adapter_key)
    if alt in model_keys:
        return alt
    # Multi-GPU: saved state may have "module." prefix; loading process expects keys without it
    strip = adapter_key[7:] if adapter_key.startswith("module.") else adapter_key
    if strip in model_keys:
        return strip
    alt_strip = normalize_peft_adapter_key(strip)
    if alt_strip in model_keys:
        return alt_strip
    return None


def map_adapter_state_dict_for_peft_model(peft_model, adapter_state: dict) -> tuple[dict, list]:
    model_keys = set(peft_model.state_dict().keys())
    mapped = {}
    unmatched = []
    for key, value in adapter_state.items():
        model_key = _adapter_key_to_model_key(key, model_keys)
        if model_key is not None:
            mapped[model_key] = value
        else:
            unmatched.append(key)
    return mapped, unmatched


class Qwen3VLModel(nn.Module):
    """Qwen3-VL model wrapper for use as initial_model in FedAvgRecipe.

    Model configuration uses the HuggingFace model ID (e.g. Qwen/Qwen3-VL-2B-Instruct).
    Supports both Qwen2.5-VL and Qwen3-VL checkpoints; the correct class is chosen from config.
    Use a Qwen3-VL model when training with the Qwen3-VL repo's train_qwen.py.

    Loaded with dtype=torch.bfloat16 so the server sends the global model in bf16
    (~4651 MB for 2B); from_pretrained can otherwise default to float32 (~9302 MB).
    """

    def __init__(self, model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct", **kwargs):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.model = load_qwen_vl_from_pretrained(model_name_or_path, dtype=torch.bfloat16, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class Qwen3VLLoRAModel(nn.Module):
    """Initial model that exposes only LoRA adapter weights for FedAvg.

    Used when --lora is set so the server and clients exchange only LoRA parameters
    instead of the full model. state_dict() returns only adapter weights with "model."
    prefix to match the wrapper format expected by the client.

    This reduces the communicated payload, but not the server-side base-model footprint:
    the server still instantiates the underlying Qwen3-VL model in bf16 and attaches
    LoRA modules before exposing adapter-only state_dict values for FL exchange.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        lora_r: int = DEFAULT_LORA_R,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
        lora_target_modules: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model

        self.model_name_or_path = model_name_or_path
        base = load_qwen_vl_from_pretrained(model_name_or_path, dtype=torch.bfloat16, **kwargs)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules or DEFAULT_LORA_TARGET_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(base, lora_config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False, **kwargs):
        """Return only LoRA adapter weights with "model." prefix for FL exchange."""
        adapter = _get_peft_adapter_state_dict(self.model)
        if destination is None:
            destination = OrderedDict()
        for key, value in adapter.items():
            destination[prefix + "model." + key] = value if keep_vars else value.detach()
        return destination

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load only LoRA adapter weights to avoid base-model missing-keys noise."""
        adapter_state = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                adapter_state[key[6:]] = value
            elif "lora" in key.lower():
                adapter_state[key] = value

        if not adapter_state:
            if strict:
                raise RuntimeError("No LoRA adapter keys found in provided state_dict.")
            warnings.warn(
                "No LoRA adapter keys found in provided state_dict; skipping adapter load because strict=False."
            )
            try:
                incompatible = self.model.load_state_dict({}, strict=False, assign=assign)
            except TypeError:
                incompatible = self.model.load_state_dict({}, strict=False)
            return _IncompatibleKeys(
                incompatible.missing_keys, list(incompatible.unexpected_keys) + list(state_dict.keys())
            )

        mapped_state, unmatched = map_adapter_state_dict_for_peft_model(self.model, adapter_state)
        missing_adapter_keys = sorted(get_expected_peft_adapter_keys(self.model) - set(mapped_state.keys()))
        if strict and not mapped_state:
            raise RuntimeError(
                "No LoRA adapter keys matched target model parameters; refusing to continue with stale adapter weights."
            )
        if strict and unmatched:
            sample = ", ".join(unmatched[:3])
            raise RuntimeError(
                f"Failed to map {len(unmatched)}/{len(adapter_state)} LoRA adapter keys. Example unmatched keys: {sample}"
            )
        if strict and missing_adapter_keys:
            sample = ", ".join(missing_adapter_keys[:3])
            raise RuntimeError(
                f"Missing {len(missing_adapter_keys)} required LoRA adapter keys. Example missing key: {sample}"
            )

        try:
            incompatible = self.model.load_state_dict(mapped_state, strict=False, assign=assign)
        except TypeError:
            # Older torch versions do not support the `assign` argument.
            incompatible = self.model.load_state_dict(mapped_state, strict=False)

        combined_missing_keys = list(dict.fromkeys(list(incompatible.missing_keys) + missing_adapter_keys))
        combined_unexpected_keys = list(dict.fromkeys(list(incompatible.unexpected_keys) + unmatched))
        if missing_adapter_keys:
            warnings.warn(
                f"Ignoring {len(missing_adapter_keys)} missing LoRA adapter keys because strict=False. "
                f"Example missing key: {missing_adapter_keys[0]}"
            )
        if unmatched:
            warnings.warn(
                f"Ignoring {len(unmatched)} unmatched LoRA adapter keys because strict=False. "
                f"Example unmatched key: {unmatched[0]}"
            )
        if missing_adapter_keys or unmatched:
            return _IncompatibleKeys(combined_missing_keys, combined_unexpected_keys)
        return incompatible
