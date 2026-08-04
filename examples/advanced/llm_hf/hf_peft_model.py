# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from collections import OrderedDict

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn.modules.module import _IncompatibleKeys
from transformers import AutoModelForCausalLM


class CausalLMPEFTModel(torch.nn.Module):
    """
    PEFT Model wrapper for federated learning with NVFlare.

    This model is used to define the initial model structure and weights.
    The PEFT configuration matches what's used in client.py for consistency.
    """

    def __init__(self, model_name_or_path):
        super(CausalLMPEFTModel, self).__init__()
        self.model_name_or_path = model_name_or_path
        # PEFT configs - must match the config in client.py
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
        )
        # Wrap with PEFT for initial model structure
        self.model = get_peft_model(full_model, peft_config)

    def forward(self, input_id):
        output = self.model(input_ids=input_id, return_dict=False)
        return output

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False, **kwargs):
        """Return only LoRA adapter weights with the wrapper prefix used by the server."""
        adapter_state = get_peft_model_state_dict(self.model)
        if destination is None:
            destination = OrderedDict()
        for key, value in adapter_state.items():
            destination[prefix + "model." + key] = value if keep_vars else value.detach()
        return destination

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load an adapter-only server state into the wrapped PEFT model."""
        adapter_state = {
            key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")
        }
        if not adapter_state:
            if strict:
                raise RuntimeError("No LoRA adapter keys found in provided state_dict.")
            return _IncompatibleKeys([], list(state_dict.keys()))

        set_peft_model_state_dict(self.model, adapter_state)
        return _IncompatibleKeys([], [])
