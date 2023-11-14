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

from enum import Enum
from typing import Any, Dict, Optional, Union

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.fuel.utils.validation_utils import check_object_type


class ParamsType(str, Enum):
    FULL = "FULL"
    DIFF = "DIFF"


class FLModelConst:
    PARAMS_TYPE = "params_type"
    PARAMS = "params"
    OPTIMIZER_PARAMS = "optimizer_params"
    METRICS = "metrics"
    CURRENT_ROUND = "current_round"
    TOTAL_ROUNDS = "total_rounds"
    META = "meta"


class MetaKey(FLMetaKey):
    pass


class FLModel:
    def __init__(
        self,
        params_type: Union[None, str, ParamsType] = None,
        params: Any = None,
        optimizer_params: Any = None,
        metrics: Optional[Dict] = None,
        current_round: Optional[int] = None,
        total_rounds: Optional[int] = None,
        meta: Optional[Dict] = None,
    ):
        """FLModel is a standardize data structure for NVFlare to communicate with external systems.

        Args:
            params_type: type of the parameters. It only describes the "params".
                If params_type is None, params need to be None.
                If params is provided but params_type is not provided, then it will be treated as FULL.
            params: model parameters, for example: model weights for deep learning.
            optimizer_params: optimizer parameters.
                For many cases, the optimizer parameters don't need to be transferred during FL training.
            metrics: evaluation metrics such as loss and scores.
            current_round: the current FL rounds. A round means round trip between client/server during training.
                None for inference.
            total_rounds: total number of FL rounds. A round means round trip between client/server during training.
                None for inference.
            meta: metadata dictionary used to contain any key-value pairs to facilitate the process.
        """
        if params_type is None:
            if params is not None:
                params_type = ParamsType.FULL
        else:
            params_type = ParamsType(params_type)

        if params_type == ParamsType.FULL or params_type == ParamsType.DIFF:
            if params is None:
                raise ValueError(f"params must be provided when params_type is {params_type}")

        self.params_type = params_type
        self.params = params
        self.optimizer_params = optimizer_params
        self.metrics = metrics
        self.current_round = current_round
        self.total_rounds = total_rounds

        if meta is not None:
            check_object_type("meta", meta, dict)
        else:
            meta = {}
        self.meta = meta

    def __str__(self):
        return (
            f"FLModel(params:{self.params}, params_type: {self.params_type},"
            f" optimizer_params: {self.optimizer_params}, metrics: {self.metrics},"
            f" current_round: {self.current_round}, meta: {self.meta})"
        )
