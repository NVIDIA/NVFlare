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
from torch import nn
from transformers import AutoModelForSequenceClassification


def train(model_name, batches):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for batch in batches:
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"])
        loss = loss_fn(outputs.logits, batch["labels"])
        loss.backward()
        optimizer.step()
