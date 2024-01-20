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

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

SITE_NAMES = "SITE_NAMES"
TASK_NAME = "TASK_NAME"

# note same as app_constant constant (todo: we only need one constant definition)
MIN_RESPONSES = "min_responses"
RESP_MAX_WAIT_TIME = "resp_max_wait_time"
START_ROUND = "start_round"
CURRENT_ROUND = "current_round"
CONTRIBUTION_ROUND = "contribution_round"
CONTRIBUTION_CLIENT = "contribution_client"
NUM_ROUNDS = "num_rounds"

STATUS = "status"
RESULT = "result"
DATA = "data"
TARGET_SITES = "target_sizes"


class WFCommAPISpec(ABC):
    @abstractmethod
    def broadcast_and_wait(
        self,
        task_name: str,
        min_responses: int,
        data: any,
        meta: dict = None,
        targets: Optional[List[str]] = None,
        callback: Callable = None,
    ):
        pass

    @abstractmethod
    def send_and_wait(
        self,
        task_name: str,
        min_responses: int,
        data: any,
        meta: dict = None,
        targets: Optional[List[str]] = None,
        send_order: str = "sequential",
        callback: Callable = None,
    ):
        pass

    @abstractmethod
    def relay_and_wait(
        self,
        task_name: str,
        min_responses: int,
        data: any,
        meta: dict = None,
        relay_order: str = "sequential",
        targets: Optional[List[str]] = None,
        callback: Callable = None,
    ):
        pass

    @abstractmethod
    def broadcast(self, task_name: str, data: any, meta: dict = None, targets: Optional[List[str]] = None):
        pass

    @abstractmethod
    def send(
        self,
        task_name: str,
        data: any,
        meta: dict = None,
        send_order: str = "sequential",
        targets: Optional[str] = None,
    ):
        pass

    @abstractmethod
    def relay(
        self,
        task_name: str,
        data: any,
        meta: dict = None,
        relay_order: str = "sequential",
        targets: Optional[List[str]] = None,
    ):
        pass

    @abstractmethod
    def get_site_names(self) -> List[str]:
        pass
