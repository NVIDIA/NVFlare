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
from typing import Dict, Optional


class TransferType(Enum):
    MODEL = "MODEL"
    MODEL_DIFF = "MODEL_DIFF"
    METRICS_ONLY = "METRICS_ONLY"


class FLModelConst:
    AGGREGATION = "aggregation"
    METRICS = "metrics"


class FLModel:
    def __init__(
        self,
        transfer_type: TransferType,
        model: Optional[Dict] = None,
        optimizer: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        configs: Optional[Dict] = None,
        client_weights: Optional[Dict] = None,
        round: Optional[int] = None,
        meta: Optional[Dict] = None,
    ):
        """
        Args:
            transfer_type:
                how the model will be transferred: as whole model (such as weight) or model_diff (weight_diff).
                when transfer_type is METRICS_ONLY, model is None.
            model:  machine learning model, for Deep learning, this could be weights or others depending on type of model
            optimizer: optionally provided optimizer. For many cases, this optimizer doesn't need to be transferred during FL training.
            metrics: evaluation metrics such as loss and scores
            configs: training configurations that is dynamically changes during training and need to be passed around.
                   In many cases, the statics configurations that can be exchanged before the actually training starts.
                   This configs here should only contains the dynamics configs.
            client_weights: contains AGGREGATION and METRICS client specific weights, The client_weights will be used
                   in weighted aggregation and weighted metrics during training and evaluation process
            round:  one round trip between client/server during training. None for inference
            meta: metadata dictionary used to contain any key-value pairs to facilitate the process.
        """
        FLModel.validate_transfer_type(metrics, model, transfer_type)
        FLModel.init_and_validate_client_weights(client_weights)

        self.transfer_type = transfer_type
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics
        self.configs = configs
        self.client_weights = client_weights
        self.round = round
        self.meta = meta

    @staticmethod
    def validate_transfer_type(metrics, model, transfer_type):
        if transfer_type == TransferType.MODEL or transfer_type == TransferType.MODEL_DIFF:
            if model is None:
                raise ValueError("model must be provided when transfer type is MODEL or MODEL_DIFF")

        if transfer_type == TransferType.METRICS_ONLY:
            if metrics is None:
                raise ValueError("metrics must be provided wehn transfer type is METRICS_ONLY")
            if model is not None:
                raise ValueError("model should not provided wehn transfer type is METRICS_ONLY")

    @staticmethod
    def init_and_validate_client_weights(client_weights):
        if client_weights is None:
            client_weights = {FLModelConst.AGGREGATION: 1.0, FLModelConst.METRICS: 1.0}
        else:
            for key in client_weights.keys():
                if key not in [FLModelConst.AGGREGATION, FLModelConst.METRICS]:
                    raise ValueError(
                        f"key {key} not recognized, acceptable keys: {FLModelConst.AGGREGATION} {FLModelConst.METRICS}"
                    )

            for key in [FLModelConst.AGGREGATION, FLModelConst.METRICS]:
                if key not in client_weights:
                    client_weights[key] = 1.0
