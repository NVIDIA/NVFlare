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
from typing import List, Optional, Tuple, Union

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import SendOrder, Task, TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def get_num_standing_tasks(self) -> int:
        """Gets tasks that are currently standing.

        Returns: length of the list of standing tasks

        """
        raise NotImplementedError

    def cancel_task(
        self,
        task: Task,
        completion_status: TaskCompletionStatus = TaskCompletionStatus.CANCELLED,
        fl_ctx: Optional[FLContext] = None,
    ):
        """Cancels the specified task.

        If the task is standing, the task is cancelled immediately (and removed from job queue) and calls
        the task_done CB (if specified);

        If the task is not standing, this method has no effect.

        Args:
            task: the task to be cancelled
            completion_status: the TaskCompletionStatus of the task
            fl_ctx: the FL context

        """
        raise NotImplementedError

    def cancel_all_tasks(self, completion_status=TaskCompletionStatus.CANCELLED, fl_ctx: Optional[FLContext] = None):
        """Cancels all standing tasks.

        Args:
            completion_status: the TaskCompletionStatus of the task
            fl_ctx: the FL context
        """
        raise NotImplementedError

    def check_tasks(self):
        """Checks if tasks should be exited."""
        raise NotImplementedError

    def process_task_request(self, client: Client, fl_ctx: FLContext) -> Tuple[str, str, Shareable]:
        """Called by the Engine when a task request is received from a client.

        Args:
            client: the Client that the task request is from
            fl_ctx: the FLContext

        Returns: task name, task id, and task data

        """
        raise NotImplementedError

    def handle_exception(self, task_id: str, fl_ctx: FLContext):
        """Called after process_task_request returns, but exception occurs before task is sent out."""
        raise NotImplementedError

    def process_submission(self, client: Client, task_name: str, task_id: str, result: Shareable, fl_ctx: FLContext):
        """Called by the Engine to process the submitted result from a client.

        Args:
            client: the Client that the submitted result is from
            task_name: the name of the task
            task_id: the id of the task
            result: the Shareable result from the Client
            fl_ctx: the FLContext

        """
        raise NotImplementedError

    def get_client_disconnect_time(self, client_name):
        """Get the time that the client is deemed disconnected.

        Args:
            client_name: the name of the client

        Returns: time at which the client was deemed disconnected; or None if the client is not disconnected.

        """
        raise NotImplementedError

    def process_dead_client_report(self, client_name: str, fl_ctx: FLContext):
        """Called by the Engine to process dead client report.

        Args:
            client_name: name of the client that dead report is received
            fl_ctx: the FLContext

        """
        raise NotImplementedError

    def client_is_active(self, client_name: str, reason: str, fl_ctx: FLContext):
        """Called by the Engine to notify us that the client is active .

        Args:
            client_name: name of the client that is active
            reason: why client is considered active
            fl_ctx: the FLContext
        """
        raise NotImplementedError

    def process_task_check(self, task_id: str, fl_ctx: FLContext):
        """Called by the Engine to check whether a specified task still exists.
        Args:
            task_id: the id of the task
            fl_ctx: the FLContext
        Returns: the ClientTask object if exists; None otherwise
        """
        raise NotImplementedError

    def initialize_run(self, fl_ctx: FLContext):
        """Called when a new RUN is about to start.

        Args:
            fl_ctx: FL context. It must contain 'job_id' that is to be initialized

        """
        raise NotImplementedError

    def finalize_run(self, fl_ctx: FLContext):
        """Called when a new RUN is finished.

        Args:
            fl_ctx: the FL context

        """
        raise NotImplementedError
