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

"""Shared Qwen model definition for the server and FSDP2 clients."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"


class CausalLMModel(torch.nn.Module):
    def __init__(self, model_name=DEFAULT_MODEL, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float32, attn_implementation="sdpa"
            )
        else:
            # Clients receive pretrained weights from the server after sharding.
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_config(config, dtype=torch.float32, attn_implementation="sdpa")
        self.model.config.use_cache = False

    def forward(self, input_ids):
        # Hugging Face shifts the labels internally for next-token prediction.
        return self.model(input_ids=input_ids, labels=input_ids).loss
