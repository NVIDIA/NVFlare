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
"""NeMo AutoModel factory that warm-starts from a received full-model checkpoint."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import model_checkpoint
import torch


def _compatible_model_state(
    model_state: Mapping[str, torch.Tensor], incoming_state: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    compatible = OrderedDict()
    for key, target in model_state.items():
        value = incoming_state.get(key)
        if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
            continue
        if value.shape != target.shape:
            continue
        compatible[key] = value.detach().to(device=target.device, dtype=target.dtype)
    return compatible


def _resolve_torch_dtype(torch_dtype):
    if isinstance(torch_dtype, str):
        if torch_dtype in ("bfloat16", "bf16"):
            return torch.bfloat16
        if torch_dtype in ("float16", "fp16", "half"):
            return torch.float16
        if torch_dtype in ("float32", "fp32"):
            return torch.float32
    return torch_dtype


def from_pretrained_with_global_state(
    pretrained_model_name_or_path: str,
    incoming_model_ckpt: str | None = None,
    trust_remote_code: bool = True,
    torch_dtype=torch.bfloat16,
    **kwargs,
):
    """Build a causal LM and load the current federated global weights when provided."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=_resolve_torch_dtype(torch_dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    if incoming_model_ckpt:
        incoming_state = model_checkpoint.load_model_state(incoming_model_ckpt)
        model_state = model.state_dict()
        model_checkpoint.validate_model_state_coverage(
            model_state,
            incoming_state,
            candidate_name=f"AutoModel state dict for {pretrained_model_name_or_path}",
            reference_name=f"incoming global model checkpoint {incoming_model_ckpt}",
        )
        compatible_state = _compatible_model_state(model_state, incoming_state)
        model.load_state_dict(compatible_state, strict=False)
        incoming_tensor_count = model_checkpoint.count_model_state_tensors(incoming_state)
        print(f"Loaded {len(compatible_state)}/{incoming_tensor_count} incoming global model tensors")
    return model
