# Copyright (c) 2021, NVIDIA CORPORATION.
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

import threading
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union

from nvflare.apis.signal import Signal

from .client import Client
from .fl_context import FLContext
from .shareable import Shareable


class TaskCompletionStatus(Enum):

    OK = "ok"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"
    ABORTED = "aborted"
    IGNORED = "ignored"


class Task(object):
    def __init__(
        self,
        name: str,
        data: Shareable,
        props: Optional[Dict] = None,
        timeout: int = 0,
        before_task_sent_cb=None,
        after_task_sent_cb=None,
        result_received_cb=None,
        task_done_cb=None,
    ):
        """

        Args:
            name (str): name of the task
            data (Shareable): data of the task
            props: Any additional properties of the task
            timeout: How long this task will last. If == 0, the task never time out.
            before_task_sent_cb: If provided, this callback would be called before controller sends the tasks to clients.
                It needs to follow the before_task_sent_cb_signature.
            after_task_sent_cb: If provided, this callback would be called after controller sends the tasks to clients.
                It needs to follow the after_task_sent_cb_signature.
            result_received_cb: If provided, this callback would be called when controller receives results from clients.
                It needs to follow the result_received_cb_signature.
            task_done_cb: If provided, this callback would be called when task is done.
                It needs to follow the task_done_cb_signature.

        """
        if not isinstance(name, str):
            raise TypeError("name must be an instance of str.")

        if not isinstance(data, Shareable):
            raise TypeError("data must be an instance of Shareable.")

        self.name = name  # name of the task
        self.data = data  # task data to be sent to client(s)
        self.cb_lock = threading.Lock()

        if props is None:
            self.props = {}
        else:
            if not isinstance(props, dict):
                raise TypeError("props must be None or dict.")
            self.props = props

        if not isinstance(timeout, int):
            raise TypeError("timeout must be an instance of int.")

        if timeout < 0:
            raise ValueError("timeout must >= 0.")

        if before_task_sent_cb is not None and not callable(before_task_sent_cb):
            raise TypeError("before_task_sent must be a callable function.")

        if after_task_sent_cb is not None and not callable(after_task_sent_cb):
            raise TypeError("after_task_sent_cb must be a callable function.")

        if result_received_cb is not None and not callable(result_received_cb):
            raise TypeError("result_received must be a callable function.")

        if task_done_cb is not None and not callable(task_done_cb):
            raise TypeError("task_done must be a callable function.")

        self.timeout = timeout
        self.before_task_sent_cb = before_task_sent_cb
        self.after_task_sent_cb = after_task_sent_cb
        self.result_received_cb = result_received_cb
        self.task_done_cb = task_done_cb

        self.targets = None
        self.client_tasks = []  # list of ClientTasks sent
        self.last_client_task_map = {}  # dict of: client name => last ClientTask of the client
        self.completion_status = None  # task completion status
        self.is_standing = False  # whether the task is still standing
        self.schedule_time = None  # when the task was scheduled
        self.create_time = time.time()

    def set_prop(self, key, value):
        if key.startswith("__"):
            raise ValueError("Keys start with __ is reserved. Please use other key.")
        self.props[key] = value

    def get_prop(self, key):
        if key.startswith("__"):
            raise ValueError("Keys start with __ is reserved. Please use other key.")
        return self.props.get(key)


class ClientTask(object):

    """
    ClientTask records the processing information of a task for a client.

    """

    def __init__(self, client: Client, task: Task):
        self.client = client
        self.task = task
        self.id = str(uuid.uuid4())
        self.task_send_count = 0  # number of times the task is sent to the client
        self.task_sent_time = None  # last time the task was sent to the client
        self.result_received_time = None  # time when the result was received from the client
        self.result = None  # result submitted by the client, or processed result
        self.props = {}  # callbacks can use this dict to keep additional processing info


class SendOrder(Enum):

    ANY = "any"
    SEQUENTIAL = "sequential"


def before_task_sent_cb_signature(client_task: ClientTask, fl_ctx: FLContext):
    """
    Signature of the before_task_sent CB.

    Called before sending a task to a client.
    Usually used to prepare the FL Context, which is created to process client's task req
    You can also use this CB to alter the data of the task to be sent.

    Args:
        client_task: the client task that is about to be sent
        fl_ctx: the FL context that comes with the client's task request.
        Public properties you set to this context will be sent to the client!

    Returns:

    """
    pass


def after_task_sent_cb_signature(client_task: ClientTask, fl_ctx: FLContext):
    """
    Signature of the after_task_sent CB.

    Called after sending a task to a client.
    Usually used to clean up the FL Context or the Task data

    Args:
        client_task: the client task that has been sent
        fl_ctx: the FL context that comes with the client's task request.

    Returns:

    """
    pass


def result_received_cb_signature(client_task: ClientTask, fl_ctx: FLContext):
    """

    Signature of result_received CB

    Called after a result is received from a client

    Args:
        client_task: the client task that the result is for
        fl_ctx: the FL context that comes with the client's result submission

    Returns:

    """
    pass


def task_done_cb_signature(task: Task, fl_ctx: FLContext):
    """
    Signature of task_done CB.

    Called when the task is completed.

    Args:
        task: the task that is completed
        fl_ctx: an instance of FL Context used for this call only.

    Returns:

    """
    pass


class ControllerSpec(ABC):
    @abstractmethod
    def start_controller(self, fl_ctx: FLContext):
        """Starts the controller.

        This method is called at the beginning of the RUN.

        Args:
            fl_ctx: the FL context. You can use this context to access services provided by the
            framework. For example, you can get Command Register from it and register your
            admin command modules.

        Returns:

        """
        pass

    @abstractmethod
    def stop_controller(self, fl_ctx: FLContext):
        """Stops the controller

        This method is called right before the RUN is ended.

        Args:
            fl_ctx: the FL context. You can use this context to access services provided by the
            framework. For example, you can get Command Register from it and unregister your
            admin command modules.

        Returns:

        """
        pass

    @abstractmethod
    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        """
        This is called when a result submission is received from a client, but no standing
        task can be found for it (from the task queue)

        This could happen when:
        - the client's submission is too late - the task is already completed
        - the Controller lost the task, e.g. the Server is restarted

        Args:
            client: the client that the result comes from
            task_name: the name of the task
            client_task_id: ID of the task
            result: the result from the client
            fl_ctx: the FL context that comes with the client's submission

        Returns:

        """
        pass

    def broadcast(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = 0,
        wait_time_after_min_received: int = 0,
    ):
        """
        Schedule to broadcast the task to specified targets.
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

        Returns:

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
        """
        This is the blocking version of the 'broadcast' method.

        First, the task is scheduled for broadcast (see the broadcast method);
        It then waits until the task is completed.

        Args:
            task:
            fl_ctx:
            targets:
            min_responses:
            wait_time_after_min_received:
            abort_signal: the abort signal. If triggered, this method stops waiting and returns to the caller.

        Returns:

        """
        pass

    def broadcast_forever(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
    ):
        """
        Schedule a broadcast task that never ends until timeout or explicitly cancelled.
        All clients will get the task every time it asks for a new task.

        This is a non-blocking call.

        NOTE: you can change the content of the task in the before_task_sent function.

        Args:
            task:
            fl_ctx:
            targets:

        Returns:

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
        """
        Schedule to send the task to a single target client.
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

        Returns:

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
            task:
            fl_ctx:
            targets:
            send_order:
            task_assignment_timeout:
            abort_signal:

        Returns:

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

        Returns:

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

    def get_num_standing_tasks(self) -> int:
        """Gets tasks that are currently standing.

        Returns: length of the list of standing tasks

        """
        pass

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
            task:
            completion_status:
            fl_ctx

        Returns:

        """
        pass

    def cancel_all_tasks(self, completion_status=TaskCompletionStatus.CANCELLED, fl_ctx: Optional[FLContext] = None):
        """Cancels all standing tasks.

        Returns:

        """
        pass

    def abort_task(self, task: Task, fl_ctx: FLContext):
        """Asks all clients to abort the execution of the specified task.

        Args:
            task: the task to be aborted
            fl_ctx: the FL context

        Returns:

        """
        pass

    def abort_all_tasks(self, fl_ctx: FLContext):
        """Asks clients to abort the execution of all tasks.

        NOTE: the server should send a notification to all clients, regardless of whether the server
        has any standing tasks.

        Returns:

        """
        pass
