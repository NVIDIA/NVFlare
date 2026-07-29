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

"""Server-side model definition for the standard simulator benchmark."""

import torch
from transformers import AutoModelForCausalLM


class CausalLMModel(torch.nn.Module):
    def __init__(self, model_name_or_path: str, precision: str):
        super().__init__()
        bf16_supported = torch.cuda.is_available() and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if precision == "bfloat16" and not bf16_supported:
            raise RuntimeError("bfloat16 precision requires a CUDA device with BF16 support")
        if precision not in ("float32", "bfloat16"):
            raise ValueError(f"unsupported precision: {precision}")
        dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=dtype,
        )

    def forward(self, input_ids):
        return self.model(input_ids=input_ids)
