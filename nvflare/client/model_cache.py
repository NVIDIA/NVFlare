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

from typing import Optional

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.model_exchange.model_exchanger import ModelExchanger

from .config import ClientConfig
from .constants import SYS_ATTRS
from .utils import DIFF_FUNCS, get_meta_from_fl_model


class Cache:
    """This class is used to remember attributes that need to share for a user code.

    For example, after "global_evaluate" we should remember the "metrics" value.
    And set that into the model that we want to submit after "train".

    For each user file:
        - we only need 1 model exchanger.
        - we only need to pull global model once

    """

    def __init__(self, model_exchanger: ModelExchanger, config: ClientConfig):
        self.model_exchanger = model_exchanger
        self.config = config

        self.input_model: Optional[FLModel] = None
        self.metrics = None
        self.sys_info = None
        self.output_meta = {}

    def receive(self, sys_info_receive=False):
        if sys_info_receive and self.input_model is not None:
            return
        self.input_model = self.model_exchanger.receive_model()
        self.sys_info = get_meta_from_fl_model(self.input_model, SYS_ATTRS)

    def send(self, model: FLModel) -> None:
        if self.config.get_transfer_type() == "DIFF":
            exchange_format = self.config.get_exchange_format()
            diff_func = DIFF_FUNCS.get(exchange_format, None)
            if diff_func is None:
                raise RuntimeError(f"no default params diff function for {exchange_format}")
            elif self.input_model is None:
                raise RuntimeError("no received model")
            try:
                model.params = diff_func(original=self.input_model.params, new=model.params)
                model.params_type = ParamsType.DIFF
            except Exception as e:
                raise RuntimeError(f"params diff function failed: {e}")
        self.model_exchanger.submit_model(model=model)

    def __str__(self):
        return f"Cache(config: {self.config.get_config()})"
