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
MedGemma model helpers for federated LoRA fine-tuning.
"""

from __future__ import annotations

import glob
import os
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from data_utils import DEFAULT_MODEL_NAME_OR_PATH
from torch.nn.modules.module import _IncompatibleKeys
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_R = 16
DEFAULT_MODULES_TO_SAVE = ["lm_head", "embed_tokens"]
DEFAULT_TARGET_MODULES = "all-linear"
MEDGEMMA_IMAGE_TOKEN_ID = 262144


def create_lora_config():
    from peft import LoraConfig

    return LoraConfig(
        lora_alpha=DEFAULT_LORA_ALPHA,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        r=DEFAULT_LORA_R,
        bias="none",
        target_modules=DEFAULT_TARGET_MODULES,
        task_type="CAUSAL_LM",
        modules_to_save=list(DEFAULT_MODULES_TO_SAVE),
    )


def strip_model_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    stripped = {}
    for key, value in state_dict.items():
        stripped[key[6:] if key.startswith("model.") else key] = value
    return stripped


def load_adapter_state_dict_from_checkpoint(checkpoint_dir: str) -> dict[str, Any]:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    if os.path.isfile(adapter_path):
        from safetensors.torch import load_file

        return load_file(adapter_path, device="cpu")

    safetensor_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if safetensor_files:
        from safetensors.torch import load_file

        state_dict = {}
        for path in safetensor_files:
            state_dict.update(load_file(path, device="cpu"))
        return state_dict

    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        return state_dict

    raise FileNotFoundError(f"No adapter checkpoint files found in {checkpoint_dir}")


def get_model_device_map(local_rank: int | None = None):
    if not torch.cuda.is_available():
        return None
    if local_rank is None:
        return {"": 0}
    return {"": local_rank}


def load_medgemma_base_model(
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH,
    *,
    quantized: bool = False,
    device_map=None,
):
    model_kwargs = {
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if quantized:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    else:
        model_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    return model


def create_peft_medgemma_model(
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH,
    *,
    quantized: bool = False,
    device_map=None,
):
    from peft import get_peft_model, prepare_model_for_kbit_training

    base_model = load_medgemma_base_model(model_name_or_path, quantized=quantized, device_map=device_map)
    if quantized:
        base_model = prepare_model_for_kbit_training(base_model)
    return get_peft_model(base_model, create_lora_config())


def apply_adapter_state(model, adapter_state: dict[str, Any]) -> None:
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model, strip_model_prefix(adapter_state))


def get_adapter_state_dict(model) -> dict[str, torch.Tensor]:
    from peft import get_peft_model_state_dict

    adapter_state = get_peft_model_state_dict(model)
    return {key: value.detach().cpu().contiguous() for key, value in adapter_state.items()}


class MedGemmaLoRAModel(nn.Module):
    """Server-side initial model that exposes only LoRA adapter weights."""

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = create_peft_medgemma_model(model_name_or_path=model_name_or_path, quantized=False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False, **kwargs):
        adapter_state = get_adapter_state_dict(self.model)
        if destination is None:
            destination = OrderedDict()
        for key, value in adapter_state.items():
            destination[prefix + "model." + key] = value if keep_vars else value.detach()
        return destination

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        adapter_state = strip_model_prefix(state_dict)
        if not adapter_state:
            if strict:
                raise RuntimeError("No LoRA adapter keys found in provided state_dict.")
            return _IncompatibleKeys([], list(state_dict.keys()))

        apply_adapter_state(self.model, adapter_state)
        return _IncompatibleKeys([], [])
