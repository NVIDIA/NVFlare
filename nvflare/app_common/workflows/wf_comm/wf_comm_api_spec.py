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
from typing import Dict

CMD = "COMMAND"
CMD_SEND = "SEND"
CMD_STOP = "STOP"
CMD_ABORT = "ABORT"
CMD_BROADCAST = "BROADCAST"
PAYLOAD = "PAYLOAD"
SITE_NAMES = "SITE_NAMES"

# note same as app_constant constant (todo: we only need one constant definition)
MIN_RESPONSES = "min_responses"
START_ROUND = "start_round"
CURRENT_ROUND = "current_round"
CONTRIBUTION_ROUND = "contribution_round"
CONTRIBUTION_CLIENT = "contribution_client"
NUM_ROUNDS = "num_rounds"

STATUS = "status"
RESULT = "result"
DATA = "data"


class WFCommAPISpec(ABC):
    @abstractmethod
    def broadcast_and_wait(self, msg_payload: Dict):
        pass

    @abstractmethod
    def broadcast(self, msg_payload):
        pass

    @abstractmethod
    def send(self, msg_payload: Dict):
        pass

    @abstractmethod
    def send_and_wait(self, msg_payload: Dict):
        pass

    @abstractmethod
    def get_site_names(self):
        pass

    @abstractmethod
    def wait(self, min_responses):
        pass
