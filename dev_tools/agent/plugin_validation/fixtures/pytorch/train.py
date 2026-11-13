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

"""Small CPU-only source workload for plugin conversion validation."""

import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def create_model():
    return nn.Linear(2, 1)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    features = torch.arange(64, dtype=torch.float32).reshape(32, 2) / 64
    labels = 2 * features[:, :1] - features[:, 1:] + 0.25
    loader = DataLoader(TensorDataset(features, labels), batch_size=8, shuffle=False)
    model = create_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(2):
        for batch, target in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(batch), target)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        loss = nn.functional.mse_loss(model(features), labels).item()
    print(json.dumps({"framework": "pytorch", "loss": loss, "examples": len(features)}))


if __name__ == "__main__":
    main()
