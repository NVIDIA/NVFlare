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

import torch
from transformers import AutoModelForCausalLM


class CausalLMModel(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super(CausalLMModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
        )

    def forward(self, input_id):
        output = self.model(input_ids=input_id, return_dict=False)
        return output
