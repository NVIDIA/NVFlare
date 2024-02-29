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

from abc import ABC
from typing import List, Union

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import SendOrder, Task
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal


class WFCommSpec(ABC):
    def broadcast(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = 0,
        wait_time_after_min_received: int = 0,
    ):
        """Schedule to broadcast the task to specified targets.

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
            task: the task to be sent
            fl_ctx: the FL context
            targets: list of destination clients. None means all clients are determined dynamically;
            min_responses: the min number of responses expected. If == 0, must get responses from
              all clients that the task has been sent to;
            wait_time_after_min_received: how long (secs) to wait after the min_responses is received.
              If == 0, end the task immediately after the min responses are received;

        """
        pass

    def broadcast_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = 0,
        wait_time_after_min_received: int = 0,
        abort_signal: Signal = None,
    ):
        """This is the blocking version of the 'broadcast' method.

        First, the task is scheduled for broadcast (see the broadcast method);
        It then waits until the task is completed.

        Args:
            task: the task to be sent
            fl_ctx: the FL context
            targets: list of destination clients. None means all clients are determined dynamically.
            min_responses: the min number of responses expected. If == 0, must get responses from
              all clients that the task has been sent to;
            wait_time_after_min_received: how long (secs) to wait after the min_responses is received.
              If == 0, end the task immediately after the min responses are received;
            abort_signal: the abort signal. If triggered, this method stops waiting and returns to the caller.

        """
        pass

    def broadcast_forever(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
    ):
        """Schedule a broadcast task that never ends until timeout or explicitly cancelled.

        All clients will get the task every time it asks for a new task.
        This is a non-blocking call.

        NOTE: you can change the content of the task in the before_task_sent function.

        Args:
            task: the task to be sent
            fl_ctx: the FL context
            targets: list of destination clients. None means all clients are determined dynamically.

        """
        pass

    def send(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
    ):
        """Schedule to send the task to a single target client.

        This is a non-blocking call.

        In ANY order, the target client is the first target that asks for task.
        In SEQUENTIAL order, the controller will try its best to send the task to the first client
        in the targets list. If can't, it will try the next target, and so on.

        NOTE: if the 'targets' is None, the actual target clients will be dynamic, because the clients
        could join/disconnect at any moment. While the task is standing, any client that joins automatically
        becomes a target for this task.

        If the send_order is SEQUENTIAL, the targets must be a non-empty list of client names.

        Args:
            task: the task to be sent
            fl_ctx: the FL context
            targets: list of candidate target clients.
            send_order: how to choose the client to send the task.
            task_assignment_timeout: in SEQUENTIAL order, this is the wait time for trying a target client, before trying next target.

        """
        pass

    def send_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
        abort_signal: Signal = None,
    ):
        """This is the blocking version of the 'send' method.

        First, the task is scheduled for send (see the 'send' method);
        It then waits until the task is completed and returns the task completion status and collected result.

        Args:
            task: the task to be performed by each client
            fl_ctx: the FL context for scheduling the task
            targets: list of clients. If None, all clients.
            send_order: how to choose the next client
            task_assignment_timeout: how long to wait for the expected client to get assigned
            before assigning to next client.
            abort_signal: the abort signal. If triggered, this method stops waiting and returns to the caller.

        """
        pass

    def relay(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
        task_result_timeout: int = 0,
        dynamic_targets: bool = True,
    ):
        """Schedules a task to be done sequentially by the clients in the targets list. This is a non-blocking call.

        Args:
            task: the task to be performed by each client
            fl_ctx: the FL context for scheduling the task
            targets: list of clients. If None, all clients.
            send_order: how to choose the next client
            task_assignment_timeout: how long to wait for the expected client to get assigned
            before assigning to next client.
            task_result_timeout: how long to wait for result from the assigned client before giving up.
            dynamic_targets: whether to dynamically grow the target list. If True, then the target list is
            expanded dynamically when a new client joins.

        """
        pass

    def relay_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order=SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
        task_result_timeout: int = 0,
        dynamic_targets: bool = True,
        abort_signal: Signal = None,
    ):
        """This is the blocking version of 'relay'."""
        pass
