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
from typing import Dict, List, Optional, Tuple

from nvflare.app_common.abstract.fl_model import FLModel

CMD = "COMMAND"
CMD_STOP = "STOP"
CMD_ABORT = "ABORT"
PAYLOAD = "PAYLOAD"
SEND_ORDER = "SEND_ORDER"
SITE_NAMES = "SITE_NAMES"

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
    def broadcast_and_wait(self, msg_payload: Dict):
        pass

    @abstractmethod
    def send_and_wait(self, msg_payload: Dict):
        pass

    @abstractmethod
    def relay_and_wait(self, msg_payload: Dict):
        pass

    @abstractmethod
    def broadcast(self, msg_payload: Dict):
        pass

    @abstractmethod
    def send(self, msg_payload: Dict):
        pass

    @abstractmethod
    def relay(self, msg_payload: Dict):
        pass

    @abstractmethod
    def get_site_names(self) -> List[str]:
        pass

    @abstractmethod
    def wait_all(self, min_responses: int, resp_max_wait_time: Optional[float]) -> Dict[str, Dict[str, FLModel]]:
        """
        wait for result
        Args:
            min_responses: if min_responses or more sites are received, then the result will return
            resp_max_wait_time: the max wait time after the 1st site response is received. This is used to deal
            with really late site result arrival, instead of waiting forever, we set a timeout.
            if resp_max_wait_time is None, it will not timeout

        Returns:
            all results with min_response
        """
        pass

    @abstractmethod
    def wait_one(self, resp_max_wait_time: Optional[float] = None) -> Tuple[str, str, FLModel]:
        """
        wait for result
        Args:
            resp_max_wait_time: the max wait time after the 1st site response is received. This is used to deal
            with really late site result arrival, instead of waiting forever, we set a timeout.
            if resp_max_wait_time is None, it will not timeout

        Returns:
            Tuple of task_name, site_name, FLModel
        """

        pass
