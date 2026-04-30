# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from torch.optim import Optimizer


def evaluate(model, data_loader, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def compute_model_diff(model, global_model):
    local_weights = model.state_dict()
    global_weights = global_model.state_dict()
    missing_params = []
    model_diff = {}
    diff_norm = 0.0

    for name in global_weights:
        if name not in local_weights:
            missing_params.append(name)
            continue
        model_diff[name] = (local_weights[name] - global_weights[name]).cpu()
        diff_norm += torch.linalg.norm(model_diff[name])

    if len(model_diff) == 0 or len(missing_params) > 0:
        raise ValueError(f"No weight differences computed or missing parameters: {missing_params}")

    if torch.isnan(diff_norm) or torch.isinf(diff_norm):
        raise ValueError(f"Diff norm is NaN or Inf: {diff_norm}")

    return model_diff, diff_norm


def get_lr_values(optimizer: Optimizer):
    return [group["lr"] for group in optimizer.state_dict()["param_groups"]]
