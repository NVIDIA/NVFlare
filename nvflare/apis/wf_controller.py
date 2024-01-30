# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from abc import abstractmethod, ABC
from typing import Optional, Callable, List

from nvflare.app_common import wf_comm


class WFController(ABC):

    def __init__(self):
        self.communicator = wf_comm.get_wf_comm_api()

    @abstractmethod
    def run(self):
        pass

    def broadcast_and_wait(
            self,
            task_name: str,
            min_responses: int,
            data: any,
            meta: dict = None,
            targets: Optional[List[str]] = None,
            callback: Callable = None,
    ):
        return self.communicator.broadcast_and_wait(task_name, min_responses, data, meta, targets, callback)

    def send_and_wait(self, task_name: str, min_responses: int, data: any, meta: dict = None,
                      targets: Optional[List[str]] = None, send_order: str = "sequential", callback: Callable = None):
        return self.communicator.send_and_wait(task_name, min_responses, data, meta, targets, send_order, callback)

    def relay_and_wait(self, task_name: str, min_responses: int, data: any, meta: dict = None,
                       targets: Optional[List[str]] = None, relay_order: str = "sequential", callback: Callable = None):
        return self.communicator.relay_and_wait(task_name, min_responses, data, meta, targets, relay_order, callback)

    def broadcast(self, task_name: str, data: any, meta: dict = None, targets: Optional[List[str]] = None):
        return self.communicator.broadcast(task_name, data, meta, targets)

    def send(self, task_name: str, data: any, meta: dict = None, targets: Optional[str] = None,
             send_order: str = "sequential"):
        return self.communicator.send(task_name, data, meta, targets, send_order)

    def relay(self, task_name: str, data: any, meta: dict = None, targets: Optional[List[str]] = None,
              relay_order: str = "sequential"):
        return self.communicator.send(task_name, data, meta, targets, relay_order)

    def get_site_names(self) -> List[str]:
        return self.communicator.get_site_names()

