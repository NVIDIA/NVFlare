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

from typing import Any, Dict, Optional

from nvflare.app_common.abstract.exchange_task import ExchangeTask
from nvflare.app_common.data_exchange.data_exchanger import DataExchanger

from .config import ClientConfig
from .constants import SYS_ATTRS


class TaskRegistry:
    """This class is used to remember attributes that need to share for a user code."""

    def __init__(
        self, config: ClientConfig, rank: Optional[str] = None, data_exchanger: Optional[DataExchanger] = None
    ):
        self.data_exchanger = data_exchanger
        self.config = config

        self.received_task: Optional[ExchangeTask] = None
        self.task_name: str = ""
        self.cache_loaded = False
        self.sys_info = {}
        for k, v in self.config.config.items():
            if k in SYS_ATTRS:
                self.sys_info[k] = v
        self.rank = rank

    def _receive(self, timeout: Optional[float] = None):
        if not self.data_exchanger:
            return None

        _, task = self.data_exchanger.receive_data(timeout)

        if not isinstance(task, ExchangeTask):
            raise RuntimeError("received data is not an ExchangeTask")

        if task.data is None:
            raise RuntimeError("no received task.data")

        self.received_task = task
        self.task_name = task.task_name
        self.cache_loaded = True

    def set_task_name(self, task_name: str):
        self.task_name = task_name

    def get_task(self, timeout: Optional[float] = None) -> Optional[ExchangeTask]:
        try:
            if not self.cache_loaded:
                self._receive()
            return self.received_task
        except:
            return None

    def get_sys_info(self) -> Dict:
        return self.sys_info

    def submit_task(self, data: Any, meta: dict, return_code: str) -> None:
        if not self.data_exchanger or not self.task_name or self.received_task is None:
            return None
        task = ExchangeTask(
            task_name=self.task_name, task_id=self.received_task.task_id, data=data, meta=meta, return_code=return_code
        )
        self.data_exchanger.submit_data(data=task)

    def clear(self):
        self.received_task = None
        self.cache_loaded = False

    def __str__(self):
        return f"{self.__class__.__name__}(config: {self.config.get_config()})"

    def __del__(self):
        if self.data_exchanger:
            self.data_exchanger.finalize(close_pipe=False)
