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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

from nvflare.apis.fl_constant import FLMetaKey
from pydantic import BaseModel, Field, field_validator, PositiveInt
from nvflare.fuel.utils.validation_utils import check_non_negative_int, check_object_type


class ParamsType(str, Enum):
    FULL = "FULL"
    DIFF = "DIFF"


class FLModelConst:
    PARAMS_TYPE = "params_type"
    PARAMS = "params"
    OPTIMIZER_PARAMS = "optimizer_params"
    METRICS = "metrics"
    CURRENT_ROUND = "current_round"
    START_ROUND = "start_round"
    TOTAL_ROUNDS = "total_rounds"
    META = "meta"


class MetaKey(FLMetaKey):
    pass


class FLModel(BaseModel):
    params_type: Optional[Union[str, ParamsType]] = Field("FULL", description="Type of model weights: FULL or DIFF.")
    params: Any = Field(default_factory=dict, description="Model weights.")
    optimizer_params: Any = Field(default_factory=dict, description="Optimizer state.")
    metrics: Optional[Dict] = Field(default_factory=dict, description="Model metrics.")
    start_round: Optional[PositiveInt] = Field(0, description="Starting round index.")
    current_round: Optional[PositiveInt] = Field(0, description="Current round index.")
    total_rounds: Optional[PositiveInt] = Field(1, description="Total number of FL rounds.")
    context: Optional[Dict] = Field(
        default_factory=dict, description="Task-specific, semantic, or workflow-related info."
    )
    meta: Optional[Dict] = Field(default_factory=dict, description="Protocol-level or transmission-related metadata.")

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def _add_to_summary(self, kvs: Dict):
        for key, value in kvs.items():
            if value:
                if isinstance(value, dict):
                    self._summary[key] = len(value)
                elif isinstance(value, ParamsType):
                    self._summary[key] = value
                elif isinstance(value, int):
                    self._summary[key] = value
                else:
                    self._summary[key] = type(value)

    def summary(self):
        kvs = dict(
            params=self.params,
            optimizer_params=self.optimizer_params,
            metrics=self.metrics,
            meta=self.meta,
            params_type=self.params_type,
            start_round=self.start_round,
            current_round=self.current_round,
            total_rounds=self.total_rounds,
            context=self.context,
        )
        self._add_to_summary(kvs)
        return self._summary

    def __repr__(self):
        return str(self.summary())

    def __str__(self):
        return str(self.summary())
