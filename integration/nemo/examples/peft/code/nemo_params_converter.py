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

import torch

from nvflare.app_common.utils.fl_model_utils import ParamsConverter


class NumpyToNEMOParamsConverter(ParamsConverter):
    def convert(self, params: Dict) -> Dict:
        result = {}
        for var_name in params:
            if "___" in var_name:
                _var_name_split = var_name.split("___")
                if len(_var_name_split) != 2:
                    raise ValueError(f"Expected list of two strings after split at '___' but got list of length {len(_var_name_split)}")
                encoder_key = _var_name_split[0]
                if encoder_key not in result:
                    result[encoder_key] = {}
                local_var_name = _var_name_split[1]
                result[encoder_key][local_var_name] = torch.as_tensor(params[var_name])
            else:
                result[var_name] = torch.as_tensor(params[var_name])
        return result


class NEMOToNumpyParamsConverter(ParamsConverter):
    def convert(self, params: Dict) -> Dict:
        state_dict = {}
        for encoder_key, prompt_state_dict in params.items():
            if isinstance(prompt_state_dict, dict):
                for k, v in prompt_state_dict.items():
                    state_dict[f"{encoder_key}___{k}"] = v.detach().cpu().numpy()
            else:
                state_dict[encoder_key] = prompt_state_dict.detach().cpu().numpy()
        return state_dict
