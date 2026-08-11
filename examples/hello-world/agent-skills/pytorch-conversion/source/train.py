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
from model import TinyClassifier


def make_data(seed=7):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(16, 4, generator=generator), torch.randint(0, 2, (16,), generator=generator)


def train_one_epoch(model, features, labels):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for start in range(0, len(labels), 4):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(features[start : start + 4]), labels[start : start + 4])
        loss.backward()
        optimizer.step()


def evaluate(model, features, labels):
    with torch.no_grad():
        return (model(features).argmax(dim=1) == labels).float().mean().item()


if __name__ == "__main__":
    x, y = make_data()
    network = TinyClassifier()
    train_one_epoch(network, x, y)
    print(f"accuracy={evaluate(network, x, y):.3f}")
