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

import copy
from typing import Dict, Optional

from nvflare.app_common.abstract.exchange_task import ExchangeTask
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.data_exchange.data_exchanger import DataExchanger

from .config import ClientConfig
from .constants import SYS_ATTRS
from .utils import DIFF_FUNCS


class ModelRegistry:
    """This class is used to remember attributes that need to share for a user code.

    For example, after "global_evaluate" we should remember the "metrics" value.
    And set that into the model that we want to submit after "train".

    For each user file:
        - we only need 1 model exchanger.
        - we only need to pull global model once

    """

    def __init__(self, config: ClientConfig, rank, data_exchanger: Optional[DataExchanger] = None):
        self.data_exchanger = data_exchanger
        self.config = config

        self.received_task: Optional[ExchangeTask] = None
        self.task_name: str = ""
        self.cache_loaded = False
        self.metrics = None
        self.sys_info = {}
        for k, v in self.config.config.items():
            if k in SYS_ATTRS:
                self.sys_info[k] = v
        self.rank = rank

    def receive(self, timeout: Optional[float] = None):
        if not self.data_exchanger:
            return None

        task_name, task = self.data_exchanger.receive_data(timeout)

        if task is None:
            raise RuntimeError("no received task")

        if task.data is None:
            raise RuntimeError("no received model")

        self.received_task = task
        self.task_name = task_name
        self.cache_loaded = True

    def get_task(self, timeout: Optional[float] = None) -> Optional[ExchangeTask]:
        try:
            if not self.cache_loaded:
                self.receive()
            return copy.deepcopy(self.received_task)
        except:
            return None

    def get_model(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        try:
            if not self.cache_loaded:
                self.receive()
            if self.received_task:
                return copy.deepcopy(self.received_task.data)
            return None
        except:
            return None

    def get_sys_info(self) -> Dict:
        return self.sys_info

    def send_task(self, data, meta: dict, return_code: str) -> None:
        if not self.data_exchanger or not self.task_name or self.received_task is None:
            return None
        task = ExchangeTask(self.task_name, self.received_task.task_id, meta, data, return_code)
        self.data_exchanger.submit_data(data=task)

    def send_model(self, model: FLModel) -> None:
        if not self.data_exchanger:
            return None
        if self.config.get_transfer_type() == "DIFF":
            exchange_format = self.config.get_exchange_format()
            diff_func = DIFF_FUNCS.get(exchange_format, None)
            if diff_func is None:
                raise RuntimeError(f"no default params diff function for {exchange_format}")
            elif self.received_task is None:
                raise RuntimeError("no received task")
            elif self.received_task.data is None:
                raise RuntimeError("no received model")
            elif model.params is not None:
                if model.params_type == ParamsType.FULL:
                    try:
                        model.params = diff_func(original=self.received_task.data.params, new=model.params)
                        model.params_type = ParamsType.DIFF
                    except Exception as e:
                        raise RuntimeError(f"params diff function failed: {e}")
            elif model.metrics is None:
                raise RuntimeError("the model to send does not have either params or metrics")
        self.data_exchanger.submit_data(data=model)

    def clear(self):
        self.received_task = None
        self.cache_loaded = False
        self.metrics = None

    def __str__(self):
        return f"{self.__class__.__name__}(config: {self.config.get_config()})"

    def __del__(self):
        if self.data_exchanger:
            self.data_exchanger.finalize(close_pipe=False)
