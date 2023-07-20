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
from typing import Any, Dict, Optional


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
    AGGREGATION = "aggregation"


class MetaKey:
    CONFIGS = "configs"
    VALIDATE_TYPE = "validate_type"
    CURRENT_ROUND = "current_round"
    TOTAL_ROUNDS = "total_rounds"
    JOB_ID = "job_id"
    SITE_NAME = "site_name"


class FLModel:
    def __init__(
        self,
        params_type: Optional[ParamsType] = None,
        params: Any = None,
        optimizer_params: Any = None,
        metrics: Optional[Dict] = None,
        current_round: Optional[int] = None,
        total_rounds: Optional[int] = None,
        meta: Optional[Dict] = None,
    ):
        """
        Args:
            params_type: type of the parameters. It only describes the "params".
                If params_type is None, params need to be None.
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
        FLModel.validate_params_type(params, params_type)

        self.params_type = params_type
        self.params = params
        self.optimizer_params = optimizer_params
        self.metrics = metrics
        self.current_round = current_round
        self.total_rounds = total_rounds
        self.meta = {} if meta is None else meta

    @staticmethod
    def validate_params_type(params: Any, params_type: Optional[ParamsType]) -> None:
        if params_type == ParamsType.FULL or params_type == ParamsType.DIFF:
            if params is None:
                raise ValueError(f"params must be provided when params_type is {params_type.value}")
        if params is not None and params_type is None:
            raise ValueError("params_type must be provided when params is not None.")

    def __str__(self):
        return (
            f"FLModel(params:{self.params}, params_type: {self.params_type},"
            f" optimizer_params: {self.optimizer_params}, metrics: {self.metrics},"
            f" current_round: {self.current_round}, meta: {self.meta})"
        )
