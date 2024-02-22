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

from abc import ABC, abstractmethod
from typing import Dict, Optional

from nvflare.apis.controller_spec import SendOrder


class WFCommunicatorSpec(ABC):
    def __init__(self):
        self.controller_config: Optional[Dict] = None

    @abstractmethod
    def broadcast_to_peers_and_wait(self, pay_load: Dict):
        """Convert pay_load and call Controller's 'broadcast_and_wait' method.

        Args:
            pay_load: the name of the task to be sent.
        """
        pass

    @abstractmethod
    def broadcast_to_peers(self, pay_load: Dict):
        """Convert pay_load and call Controller's 'broadcast' method.

        Args:
            pay_load: the name of the task to be sent.
        """
        pass

    @abstractmethod
    def send_to_peers(self, pay_load: Dict, send_order: SendOrder = SendOrder.SEQUENTIAL):
        """Convert pay_load and call Controller's 'send' method.

        Args:
            pay_load: the name of the task to be sent.
            send_order: order for choosing the next client.
        """
        pass

    @abstractmethod
    def send_to_peers_and_wait(self, pay_load: Dict, send_order: SendOrder = SendOrder.SEQUENTIAL):
        """Convert pay_load and call Controller's 'send_and_wait' method.

        Args:
            pay_load: the name of the task to be sent.
            send_order: order for choosing the next client.
        """
        pass

    @abstractmethod
    def relay_to_peers_and_wait(self, pay_load: Dict, send_order: SendOrder = SendOrder.SEQUENTIAL):
        """Convert pay_load and call Controller's 'relay_and_wait' method.

        Args:
            pay_load: the name of the task to be sent.
            send_order: order for choosing the next client.
        """
        pass

    @abstractmethod
    def relay_to_peers(self, pay_load: Dict, send_order: SendOrder = SendOrder.SEQUENTIAL):
        """Convert pay_load and call Controller's 'relay' method.

        Args:
            pay_load: the name of the task to be sent.
            send_order: order for choosing the next client.
        """
        pass
