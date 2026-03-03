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

"""Server-side model wrapper for PTFileModelPersistor.

Exposes only the LoRA adapter parameters through state_dict() / load_state_dict()
so the persistor stores a compact checkpoint (~2 MB) instead of the full base
model (~1 GB), and the aggregator only accumulates adapter diffs.

The LoRA hyperparameters here MUST match those in client.py.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM

from nvflare.fuel.utils.log_utils import get_obj_logger

# Must match client.py
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


class QwenLoRAModelWrapper(nn.Module):
    """LoRA-adapted Qwen2.5 wrapper for use with PTFileModelPersistor.

    state_dict() returns only the trainable LoRA adapter weights so that
    PTFileModelPersistor, SimpleModelShareableGenerator, and the aggregator
    all operate on the compact adapter space rather than the full model.
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-0.5B"):

        # note: this attribute is must have to make it work !
        self.model_path = model_path

        super().__init__()
        self.logger = get_obj_logger(self)
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._peft_model = get_peft_model(base, lora_cfg)

    # Override state_dict / load_state_dict so the persistor and shareable
    # generator only see (and exchange) the LoRA adapter parameters.

    def state_dict(self, *args, **kwargs):
        return get_peft_model_state_dict(self._peft_model)

    def load_state_dict(self, state_dict, strict: bool = False):
        current = get_peft_model_state_dict(self._peft_model)
        mismatched = [k for k in state_dict if k in current and state_dict[k].shape != current[k].shape]
        if mismatched:
            self.logger.warning(
                f"Checkpoint shape mismatch for {len(mismatched)} adapter param(s) "
                f"(e.g. '{mismatched[0]}': checkpoint {tuple(state_dict[mismatched[0]].shape)} "
                f"vs model {tuple(current[mismatched[0]].shape)}). "
                "Ignoring checkpoint and starting from fresh adapter weights. "
                "This is expected when switching --model_size between runs."
            )
            return
        set_peft_model_state_dict(self._peft_model, state_dict)

    def forward(self, *args, **kwargs):
        return self._peft_model(*args, **kwargs)
