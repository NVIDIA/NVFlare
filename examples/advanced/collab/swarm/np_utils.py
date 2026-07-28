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

import numpy as np


def parse_array_def(array_def):
    if array_def is None or isinstance(array_def, (np.ndarray, str)):
        return array_def
    if isinstance(array_def, list):
        return np.array(array_def, dtype=np.float32)
    raise ValueError(f"unsupported array def: {array_def}")


def save_np_model(model: np.ndarray, file_name: str):
    np.save(file_name, model)
