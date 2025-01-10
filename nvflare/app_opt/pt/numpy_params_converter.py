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

import numpy as np
import torch

from nvflare.app_common.abstract.params_converter import ParamsConverter


class NumpyToPTParamsConverter(ParamsConverter):
    def convert(self, params: Dict, fl_ctx) -> Dict:
        tensor_shapes = fl_ctx.get_prop("tensor_shapes")
        exclude_vars = fl_ctx.get_prop("exclude_vars")

        return_params = {}
        if tensor_shapes:
            return_params = {
                k: torch.as_tensor(np.reshape(v, tensor_shapes[k])) if k in tensor_shapes else torch.as_tensor(v)
                for k, v in params.items()
            }
        else:
            return_params = {k: torch.as_tensor(v) for k, v in params.items()}

        if exclude_vars:
            for k, v in exclude_vars.items():
                return_params[k] = v

        return return_params


class PTToNumpyParamsConverter(ParamsConverter):
    def convert(self, params: Dict, fl_ctx) -> Dict:
        return_tensors = {}
        tensor_shapes = {}
        exclude_vars = {}
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                return_tensors[k] = v.cpu().numpy()
                tensor_shapes[k] = v.shape
            else:
                exclude_vars[k] = v

        if tensor_shapes:
            fl_ctx.set_prop("tensor_shapes", tensor_shapes)
        if exclude_vars:
            fl_ctx.set_prop("exclude_vars", exclude_vars)
            self.logger.warning(
                f"{len(exclude_vars)} vars excluded as they were non-tensor type: " f"{list(exclude_vars.keys())}"
            )

        return return_tensors
