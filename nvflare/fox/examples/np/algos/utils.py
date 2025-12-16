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

import numpy as np


def parse_array_def(array_def):
    if array_def is None:
        return array_def

    if isinstance(array_def, np.ndarray):
        return array_def

    if isinstance(array_def, str):
        # this is base name of the file that contains NP array
        return array_def

    if isinstance(array_def, list):
        return np.array(array_def, dtype=np.float32)
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


def save_np_model(model: np.ndarray, file_name: str):
    np.save(file_name, model)


def load_np_model(file_name: str):
    return np.load(file_name)


def add(model: dict, to_model: dict):
    """Add specified model to another model

    Args:
        model: the model to be added
        to_model: the model to be added to.

    Returns: the updated model
    Notes: the to_model is updated

    """
    for k, v in model.items():
        if k not in to_model:
            to_model[k] = v
        else:
            to_model[k] += v
    return to_model


def div(model: dict, value):
    """Divide a model by a specified value

    Args:
        model: the model to be divided
        value: the value to divide the model with

    Returns: the updated model

    """
    for k, v in model.items():
        model[k] = v / value
    return model
