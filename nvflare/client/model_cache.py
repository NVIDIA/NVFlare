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

from typing import Any, Callable, Dict, Optional

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.model_exchange.model_exchanger import ModelExchanger

from .config import ClientConfig
from .constants import MODEL_ATTRS, SYS_ATTRS
from .utils import copy_fl_model_attributes, get_meta_from_fl_model, set_fl_model_with_meta


class Cache:
    """This class is used to remember attributes that need to share for a user code.

    For example, after "global_evaluate" we should remember the "metrics" value.
    And set that into the model that we want to submit after "train".

    For each user file:
        - we only need 1 model exchanger.
        - we only need to pull global model once

    """

    def __init__(self, model_exchanger: ModelExchanger, config: ClientConfig, params_diff_func: Callable):
        self.model_exchanger = model_exchanger
        self.input_model: Optional[FLModel] = None
        self.output_params: Any = None
        self.meta = None
        self.sys_meta = None

        self.config = config
        self.params_diff_func = params_diff_func
        self.metrics: Optional[Dict] = None  # get from evaluate on "global model"
        self._get_model()

    def _get_model(self):
        self.input_model = self.model_exchanger.receive_model()
        self.meta = get_meta_from_fl_model(self.input_model, MODEL_ATTRS)
        self.sys_meta = get_meta_from_fl_model(self.input_model, SYS_ATTRS)

    def construct_fl_model(self, params) -> FLModel:
        """Constructs an FLModel objects using params.

        Args:
            params: the parameters to be set into FLModel's params.
        """
        if self.input_model is None or self.meta is None:
            raise RuntimeError("needs to get model first.")
        fl_model = FLModel(params_type=ParamsType.FULL, params=params)
        if self.metrics is not None:
            fl_model.metrics = self.metrics

        # model difference
        params_type = self.config.get_params_type()
        if params_type == ParamsType.DIFF:
            fl_model.params = self.params_diff_func(self.input_model.params, fl_model.params)
            fl_model.params_type = ParamsType.DIFF

        set_fl_model_with_meta(fl_model, self.meta, MODEL_ATTRS)
        copy_fl_model_attributes(self.input_model, fl_model)
        fl_model.meta = self.meta
        return fl_model

    def __str__(self):
        return f"Cache(config: {self.config.get_config()}, metrics: {self.metrics})"
