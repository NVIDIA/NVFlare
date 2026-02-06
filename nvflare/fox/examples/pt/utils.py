# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


def parse_array_def(array_def):
    if array_def is None:
        return array_def

    if isinstance(array_def, torch.Tensor):
        return array_def

    if isinstance(array_def, list):
        return torch.tensor(array_def)
    else:
        raise ValueError(f"unsupported array def: {array_def}")


def parse_state_dict(d):
    result = {}
    for k, v in d.items():
        result[k] = parse_array_def(v)
    return result


def parse_model_def(model_def):
    if isinstance(model_def, dict):
        return parse_state_dict(model_def)
    else:
        return parse_array_def(model_def)


def add(value: dict, to_model: dict):
    """Add value to a specified model in-place.

    Args:
        value:
        to_model:

    Returns:

    """
    for k, v in value.items():
        if k not in to_model:
            to_model[k] = v
        else:
            to_model[k] += v
    return to_model


def div(model: dict, value):
    """Divide the model in-place by a specified value.

    Args:
        model:
        value:

    Returns:

    """
    for k, v in model.items():
        model[k] = torch.div(v, value)
    return model
