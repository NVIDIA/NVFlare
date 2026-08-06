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

"""ResNet-18 variant for CIFAR-10 federated learning."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class StatTracker(nn.Module):
    """Track channel statistics without normalizing activations."""

    def __init__(self, num_channels: int, momentum: float = 0.1):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                mean = x.mean(dim=(0, 2, 3))
                variance = x.var(dim=(0, 2, 3), unbiased=False)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * variance)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.stat_track1 = StatTracker(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stat_track2 = StatTracker(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                        ),
                        ("stat_track", StatTracker(self.expansion * planes)),
                    ]
                )
            )

    def forward(self, x):
        out = F.relu(self.stat_track1(self.conv1(x)))
        out = self.stat_track2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_blocks: list[int], num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.stat_track1 = StatTracker(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes, bias=False)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.stat_track1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return self.linear(out.view(out.size(0), -1))


def resnet18_local(num_classes: int = 10) -> ResNet:
    """Create the CIFAR-10 ResNet-18 model used by this example."""

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    reset_model_state(model, reset_norm_stats=True)
    return model


def reset_model_state(model: nn.Module, reset_norm_stats: bool = False) -> None:
    model.zero_grad(set_to_none=True)
    if reset_norm_stats:
        for module in model.modules():
            if isinstance(module, StatTracker):
                module.running_mean.zero_()
                module.running_var.fill_(1.0)
    model.train()


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
