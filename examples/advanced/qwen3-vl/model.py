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

import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration

class Qwen3VLModel(nn.Module):
    """Qwen3-VL model wrapper for use as initial_model in FedAvgRecipe.

    Model configuration uses the HuggingFace model ID (e.g. Qwen/Qwen2.5-VL-3B-Instruct).
    """

    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct", **kwargs):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            #torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            #**{k: v for k, v in kwargs.items() if k != "torch_dtype"},
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
