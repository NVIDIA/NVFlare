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

import numpy as np
import torch


def load_weights(model, global_weights, device="cpu"):
    """Helper function to load global weights to local model"""

    local_var_dict = model.state_dict()

    # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
    n_loaded = 0
    for var_name in global_weights:
        if var_name not in local_var_dict:
            continue
        weights = torch.as_tensor(global_weights[var_name], device=device)
        try:
            # update the local dict
            local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
            n_loaded += 1
        except BaseException as e:
            raise ValueError(f"Convert weight from {var_name} failed!") from e
    model.load_state_dict(local_var_dict)
    if n_loaded == 0:
        raise ValueError(f"No weights loaded! Received weight dict is {global_weights}")

    return n_loaded


def compute_model_diff(model, global_weights):
    """Helper function to compute the weight difference with respect to global weights"""
    local_var_dict = model.state_dict()
    # compute delta model, global model has the primary key set
    model_diff = {}
    n_diff = 0
    for var_name in global_weights:
        if var_name not in local_var_dict:
            continue
        model_diff[var_name] = np.subtract(
            local_var_dict[var_name].cpu().numpy(), global_weights[var_name], dtype=np.float32
        )
        n_diff += 1
        if np.any(np.isnan(model_diff[var_name])):
            raise ValueError(f"{var_name} weights became NaN!")
    if n_diff == 0:
        raise ValueError("No weight differences computed!")

    return model_diff
