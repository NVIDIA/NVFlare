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

import torch

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05


class QwenCausalLMModel(torch.nn.Module):
    def __init__(self, model_name_or_path: str = DEFAULT_MODEL_NAME):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        try:
            return self.model.load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return self.model.load_state_dict(state_dict, strict=strict)


class QwenLoRAModel(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        lora_r: int = DEFAULT_LORA_R,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
    ):
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=DEFAULT_LORA_TARGET_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(base_model, peft_config)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def state_dict(self, *args, **kwargs):
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(self.model)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        from peft import set_peft_model_state_dict

        return set_peft_model_state_dict(self.model, state_dict)
