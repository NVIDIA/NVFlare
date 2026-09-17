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
"""NeMo AutoModel factory that warm-starts LoRA modules from a received adapter."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Mapping

import adapter_checkpoint
import torch


def _has_adapter_weights(adapter_dir: str | None) -> bool:
    if not adapter_dir or not os.path.isdir(adapter_dir):
        return False
    return any(
        os.path.isfile(os.path.join(adapter_dir, name))
        for name in ("adapter_model.safetensors", "adapter_model.bin", "pytorch_model.bin", "model.pt")
    )


def _compatible_adapter_state(
    model_state: Mapping[str, torch.Tensor], adapter_state: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    compatible = OrderedDict()
    adapter_by_canonical_key = {
        adapter_checkpoint.canonical_adapter_key(key): value for key, value in adapter_state.items()
    }
    for key, target in model_state.items():
        value = adapter_by_canonical_key.get(adapter_checkpoint.canonical_adapter_key(key))
        if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
            continue
        compatible[key] = value.detach().to(device=target.device, dtype=target.dtype)
    return compatible


def _patch_lora_loader_once(incoming_adapter_dir: str | None) -> None:
    if not _has_adapter_weights(incoming_adapter_dir):
        return

    import nemo_automodel._transformers.infrastructure as infrastructure

    original_apply_lora = infrastructure.apply_lora_to_linear_modules

    def wrapped_apply_lora(model, peft_config, *args, **kwargs):
        try:
            num_patched = original_apply_lora(model, peft_config, *args, **kwargs)
            model_state = model.state_dict()
            adapter_state = adapter_checkpoint.load_adapter_state(incoming_adapter_dir)
            compatible_state = _compatible_adapter_state(model_state, adapter_state)
            if not compatible_state:
                raise RuntimeError(
                    f"No incoming adapter tensors from {incoming_adapter_dir} matched the AutoModel LoRA state dict."
                )

            model.load_state_dict(compatible_state, strict=False)
            print(
                f"Loaded {len(compatible_state)}/{len(adapter_state)} incoming adapter tensors from {incoming_adapter_dir}"
            )
            return num_patched
        finally:
            infrastructure.apply_lora_to_linear_modules = original_apply_lora

    infrastructure.apply_lora_to_linear_modules = wrapped_apply_lora


def from_pretrained_with_adapter(
    pretrained_model_name_or_path: str,
    incoming_adapter_dir: str | None = None,
    trust_remote_code: bool = True,
    **kwargs,
):
    """Build a causal LM and arrange for AutoModel to warm-start LoRA weights.

    AutoModel applies PEFT after custom model factories return. This factory installs a
    one-shot wrapper around AutoModel's LoRA injection helper so the incoming federated
    adapter is loaded immediately after the LoRA modules are created.
    """
    from transformers import AutoModelForCausalLM

    _patch_lora_loader_once(incoming_adapter_dir)

    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
