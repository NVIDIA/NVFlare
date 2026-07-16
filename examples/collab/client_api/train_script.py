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

"""Standalone Client API training script run in a subprocess.

The script is self-contained (the framework ships and imports it by module
path on each site) and uses the standard Client API pattern:

    flare.init(); while flare.is_running(): receive / train / send
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def main():
    flare.init()
    site_name = flare.get_site_name() or "unknown"
    print(f"[{site_name}] subprocess training loop started")

    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    criterion = nn.MSELoss()

    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break

        model = SimpleModel()
        if input_model.params is not None:
            model.load_state_dict(input_model.params)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        loss = None
        for _epoch in range(2):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()

        print(f"[{site_name}] round {input_model.current_round}: loss={loss.item():.4f}")
        flare.send(FLModel(params=model.state_dict(), metrics={"loss": loss.item()}))

    print(f"[{site_name}] subprocess training loop completed")


if __name__ == "__main__":
    main()
