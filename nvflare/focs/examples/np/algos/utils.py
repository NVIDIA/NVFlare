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

    if isinstance(array_def, list):
        return np.array(array_def, dtype=np.float32)
    else:
        raise ValueError(f"unsupported array def: {array_def}")
