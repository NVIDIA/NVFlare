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
from typing import Callable, Dict, List, Optional

SITE_NAMES = "SITE_NAMES"
TASK_NAME = "TASK_NAME"

MIN_RESPONSES = "min_responses"
RESP_MAX_WAIT_TIME = "resp_max_wait_time"

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
    ) -> Dict[str, any]:
        """Communication interface for the blocking version of the 'broadcast' method.

        First, the task is scheduled for broadcast (see the broadcast method);
        It then waits until the task is completed.

        Args:
            task_name: the name of the task to be sent.
            min_responses: the min number of responses expected.  If 0, must get responses from
              all clients that the task has been sent to.
            data: the data to be sent in the task.
            meta: the meta to be sent in the task.
            targets: list of destination clients. If None, all clients.
            callback: callback to be registered.

        Returns:
            result dict if callback is None
        """
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
    ) -> Dict[str, any]:
        """Communication interface for the blocking version of the 'send' method.

        First, the task is scheduled for send (see the 'send' method);
        It then waits until the task is completed and returns the task completion status and collected result.

        Args:
            task_name: the name of the task to be sent.
            min_responses: the min number of responses expected.  If 0, must get responses from
              all clients that the task has been sent to.
            data: the data to be sent in the task.
            meta: the meta to be sent in the task.
            targets: list of destination clients.
            send_order: order for choosing the next client.
            callback: callback to be registered.

        Returns:
            result dict if callback is None
        """
        pass

    @abstractmethod
    def relay_and_wait(
        self,
        task_name: str,
        min_responses: int,
        data: any,
        meta: dict = None,
        targets: Optional[List[str]] = None,
        relay_order: str = "sequential",
        callback: Callable = None,
    ) -> Dict[str, any]:
        """Communication interface to schedule a task to be done sequentially by the clients in the targets list. This is a non-blocking call.

        Args:
            task_name: the name of the task to be sent.
            min_responses: the min number of responses expected.  If 0, must get responses from
              all clients that the task has been sent to.
            data: the data to be sent in the task.
            meta: the meta to be sent in the task.
            targets: list of destination clients. If None, all clients.
            relay_order: order for choosing the next client.
            callback: callback to be registered.

        Returns:
            result dict if callback is None
        """
        pass

    @abstractmethod
    def broadcast(self, task_name: str, data: any, meta: dict = None, targets: Optional[List[str]] = None):
        """Communication interface to schedule to broadcast the task to specified targets.

        This is a non-blocking call.

        The task is standing until one of the following conditions comes true:
            - if timeout is specified (> 0), and the task has been standing for more than the specified time
            - the controller has received the specified min_responses results for this task, and all target clients
              are done.
            - the controller has received the specified min_responses results for this task, and has waited
              for wait_time_after_min_received.

        While the task is standing:
            - Before sending the task to a client, the before_task_sent CB (if specified) is called;
            - When a result is received from a client, the result_received CB (if specified) is called;

        After the task is done, the task_done CB (if specified) is called:
            - If result_received CB is specified, the 'result' in the ClientTask of each
              client is produced by the result_received CB;
            - Otherwise, the 'result' contains the original result submitted by the clients;

        NOTE: if the targets is None, the actual broadcast target clients will be dynamic, because the clients
        could join/disconnect at any moment. While the task is standing, any client that joins automatically
        becomes a target for this broadcast.

        Args:
            task_name: the name of the task to be sent.
            data: the data to be sent in the task.
            meta: the meta to be sent in the task.
            targets: list of destination clients. If None, all clients.
        """
        pass

    @abstractmethod
    def send(
        self,
        task_name: str,
        data: any,
        meta: dict = None,
        targets: Optional[str] = None,
        send_order: str = "sequential",
    ):
        """Communication interface to schedule to send the task to a single target client.

        This is a non-blocking call.

        In ANY order, the target client is the first target that asks for task.
        In SEQUENTIAL order, the controller will try its best to send the task to the first client
        in the targets list. If can't, it will try the next target, and so on.

        NOTE: if the 'targets' is None, the actual target clients will be dynamic, because the clients
        could join/disconnect at any moment. While the task is standing, any client that joins automatically
        becomes a target for this task.

        If the send_order is SEQUENTIAL, the targets must be a non-empty list of client names.

        Args:
            task_name: the name of the task to be sent.
            data: the data to be sent in the task.
            meta: the meta to be sent in the task.
            targets: list of destination clients. If None, all clients.
            send_order: order for choosing the next client.
        """
        pass

    @abstractmethod
    def relay(
        self,
        task_name: str,
        data: any,
        meta: dict = None,
        targets: Optional[List[str]] = None,
        relay_order: str = "sequential",
    ):
        """Communication interface to schedule a task to be done sequentially by the clients in the targets list. This is a non-blocking call.

        Args:
            task_name: the name of the task to be sent.
            data: the data to be sent in the task.
            meta: the meta to be sent in the task.
            targets: list of destination clients.
            relay_order: order for choosing the next client.
        """
        pass

    @abstractmethod
    def get_site_names(self) -> List[str]:
        """Get list of site names."""
        pass
