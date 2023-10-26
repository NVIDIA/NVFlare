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

from nvflare.app_common.data_exchange.constants import ExchangeFormat
import numpy as np
import torch


def numerical_params_diff(original: Dict, new: Dict) -> Dict:
    """Calculates the numerical parameter difference.

    Args:
        original: A dict of numerical values.
        new: A dict of numerical values.

    Returns:
        A dict with same key as original dict,
        value are the difference between original and new.
    """
    diff_dict = {}
    for k in original:
        if k not in new:
            continue
        print("#### new", k, type(new[k]))
        print("#### original", k, type(original[k]))
        print("#### new", k, torch.min(new[k]), torch.max(new[k]), torch.any(torch.isnan(new[k])))
        print("#### original", k, torch.min(original[k]), torch.max(original[k]), torch.any(torch.isnan(original[k])))
        if isinstance(new[k], list) and isinstance(original[k], list):
            diff = [new[k][i] - original[k][i] for i in range(len(new[k]))]
        else:
            diff = new[k] - original[k]

        print("#### diff", k, torch.min(diff), torch.max(diff), torch.any(torch.isnan(diff)))
        diff_dict[k] = diff
    return diff_dict


DIFF_FUNCS = {ExchangeFormat.PYTORCH: numerical_params_diff, ExchangeFormat.NUMPY: numerical_params_diff}
