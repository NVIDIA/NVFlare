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
    CLIENT_WEIGHTS = "client_weights"
    CURRENT_ROUND = "current_round"
    TOTAL_ROUNDS = "total_rounds"
    META = "meta"
    AGGREGATION = "aggregation"


class FLModel:
    def __init__(
        self,
        params_type: Optional[ParamsType] = None,
        params: Any = None,
        optimizer_params: Any = None,
        metrics: Optional[Dict] = None,
        client_weights: Optional[Dict] = None,
        current_round: Optional[int] = None,
        total_rounds: Optional[int] = None,
        meta: Optional[Dict] = None,
    ):
        """
        Args:
            params_type: type of the parameters. It only describes the "params".
                If params_type is None, params need to be None. Usually, metrics will be provided.
            params: model parameters, for example: model weights for deep learning.
            optimizer_params: optimizer parameters.
                For many cases, the optimizer parameters don't need to be transferred during FL training.
            metrics: evaluation metrics such as loss and scores.
            client_weights: contains AGGREGATION and METRICS client specific weights, The client_weights will be used
                in weighted aggregation and weighted metrics during training and evaluation process.
            current_round: the current FL rounds. A round means round trip between client/server during training.
                None for inference.
            total_rounds: total number of FL rounds. A round means round trip between client/server during training.
                None for inference.
            meta: metadata dictionary used to contain any key-value pairs to facilitate the process.
        """
        if client_weights is None:
            client_weights = {FLModelConst.AGGREGATION: 1.0, FLModelConst.METRICS: 1.0}
        FLModel.validate_params_type(params, params_type)
        FLModel.validate_client_weights(client_weights)
        for key in [FLModelConst.AGGREGATION, FLModelConst.METRICS]:
            if key not in client_weights:
                client_weights[key] = 1.0

        self.params_type = params_type
        self.params = params
        self.optimizer_params = optimizer_params
        self.metrics = metrics
        self.client_weights = client_weights
        self.current_round = current_round
        self.total_rounds = total_rounds
        self.meta = meta

    @staticmethod
    def validate_params_type(params, params_type):
        if params_type == ParamsType.FULL or params_type == ParamsType.DIFF:
            if params is None:
                raise ValueError(f"params must be provided when params_type is {params_type.value}")
        if params is not None and params_type is None:
            raise ValueError("params_type must be provided when params is not None.")

    @staticmethod
    def validate_client_weights(client_weights: dict):
        if not isinstance(client_weights, dict):
            raise ValueError(f"client_weights need to be a dict but get {type(client_weights)}")
        acceptable_keys = [FLModelConst.AGGREGATION, FLModelConst.METRICS]

        for key in client_weights.keys():
            if key not in acceptable_keys:
                raise ValueError(f"key {key} not recognized, acceptable keys: {acceptable_keys}")
