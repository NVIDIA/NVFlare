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

"""ModerateCNN and model-state helpers for CIFAR-10 federated learning."""

import torch
import torch.nn as nn


# ModerateCNN is the model used by
# examples/advanced/cifar10/pt/cifar10-sim/cifar10_fedavg. Its architecture is
# derived from IBM FedMA (MIT license; see that example's src/model.py).
def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class ModerateCNN(nn.Module):
    """CIFAR-10 model used by the existing NVFlare FedAvg example."""

    def __init__(self, seed: int = 42):
        set_seed(seed)
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return self.fc_layer(x)


def get_model_params(model: nn.Module, target_device=None) -> dict[str, torch.Tensor]:
    params = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if target_device is not None:
        params = {name: value.to(target_device) for name, value in params.items()}
    return params


def load_model_params(model: nn.Module, params: dict[str, torch.Tensor], target_device=None) -> nn.Module:
    model.load_state_dict(params)
    if target_device is not None:
        model.to(target_device)
    return model


def add_update_to_params(
    params: dict[str, torch.Tensor], update: dict[str, torch.Tensor], scale: float = 1.0
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return {
            name: value.detach().clone() + scale * update[name] if name in update else value.detach().clone()
            for name, value in params.items()
        }
