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

import traceback
from typing import Any, Dict

import numpy as np
import torch

from nvflare.apis.dxo import DataKind, from_file
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.formatter import Formatter
from nvflare.app_common.app_constant import AppConstants


def array_to_list(data: Any) -> Any:
    if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
        return data.tolist()
    return data


def simplify_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {k: array_to_list(v) for k, v in metrics}


class SimpleFormatter(Formatter):
    def __init__(self) -> None:
        super().__init__()
        self.results = {}

    def format(self, fl_ctx: FLContext) -> str:
        # Get validation result
        validation_shareables_dict = fl_ctx.get_prop(AppConstants.VALIDATION_RESULT, {})
        result = {}

        try:
            # Extract results from all clients
            for data_client in validation_shareables_dict.keys():
                validation_dict = validation_shareables_dict[data_client]
                if validation_dict:
                    result[data_client] = {}
                    for model_name in validation_dict.keys():
                        dxo_path = validation_dict[model_name]

                        # Load the shareable
                        metric_dxo = from_file(dxo_path)

                        # Get metrics from shareable
                        if metric_dxo and metric_dxo.data_kind == DataKind.METRICS:
                            metrics = simplify_metrics(metric_dxo.data)
                            result[data_client][model_name] = metrics
            # add any results
            self.results.update(result)
        except Exception as e:
            traceback.print_exc()

        return repr(result)
