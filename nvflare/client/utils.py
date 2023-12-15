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

from typing import Dict

from .config import ExchangeFormat


def numerical_params_diff(original: Dict, new: Dict) -> Dict:
    """Calculates the numerical parameter difference.

    Args:
        original: A dict of numerical values.
        new: A dict of numerical values.

    Returns:
        A dict with common keys that exist in both original dict and new dict,
        values are the difference between original and new.
    """
    diff_dict = {}
    for k in original:
        if k not in new:
            continue
        if isinstance(new[k], list) and isinstance(original[k], list):
            diff = [new[k][i] - original[k][i] for i in range(len(new[k]))]
        else:
            diff = new[k] - original[k]

        diff_dict[k] = diff
    if diff_dict == {}:
        raise RuntimeError("no common keys between original and new dict, parameters difference are empty.")
    return diff_dict


DIFF_FUNCS = {ExchangeFormat.PYTORCH: numerical_params_diff, ExchangeFormat.NUMPY: numerical_params_diff}
