# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import time
from typing import Tuple

from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable

from .task_manager import TaskCheckStatus, TaskManager

_KEY_DYNAMIC_TARGETS = "__dynamic_targets"
_KEY_TASK_RESULT_TIMEOUT = "__task_result_timeout"
_KEY_SEND_TARGET_COUNTS = "__sent_target_count"
_KEY_PENDING_CLIENT = "__pending_client"


class AnyRelayTaskManager(TaskManager):
    def __init__(self, task: Task, task_result_timeout, dynamic_targets):
        """Task manager for relay controller on SendOrder.ANY.

        Args:
            task (Task): an instance of Task
            task_result_timeout (int): timeout value on reply of one client
            dynamic_targets (bool): allow clients to join after this task starts
        """
        TaskManager.__init__(self)

        if task_result_timeout is None:
            task_result_timeout = 0

        task.props[_KEY_DYNAMIC_TARGETS] = dynamic_targets
        task.props[_KEY_TASK_RESULT_TIMEOUT] = task_result_timeout
        task.props[_KEY_SEND_TARGET_COUNTS] = {}  # target name => times sent
        task.props[_KEY_PENDING_CLIENT] = None

    def check_task_send(self, client_task: ClientTask, fl_ctx: FLContext) -> TaskCheckStatus:
        """Determine whether the task should be sent to the client.

        Args:
            client_task (ClientTask): the task processing state of the client
            fl_ctx (FLContext): fl context that comes with the task request

        Raises:
            RuntimeError: when a client asking for a task while the same client_task has already been dispatched to it

        Returns:
            TaskCheckStatus: NO_BLOCK for not sending the task, BLOCK for waiting, SEND for OK to send
        """
        client_name = client_task.client.name
        task = client_task.task
        if task.props[_KEY_DYNAMIC_TARGETS]:
            if task.targets is None:
                task.targets = []
            if client_name not in task.targets:
                task.targets.append(client_name)

        # is this client eligible?
        if client_name not in task.targets:
            # this client is not a target
            return TaskCheckStatus.NO_BLOCK

        client_occurrences = task.targets.count(client_name)
        sent_target_count = task.props[_KEY_SEND_TARGET_COUNTS]
        send_count = sent_target_count.get(client_name, 0)
        if send_count >= client_occurrences:
            # already sent enough times to this client
            return TaskCheckStatus.NO_BLOCK

        # only allow one pending task. Is there a client pending result?
        pending_client_name = task.props[_KEY_PENDING_CLIENT]
        task_result_timeout = task.props[_KEY_TASK_RESULT_TIMEOUT]
        if pending_client_name is not None:
            # see whether the result has been received
            pending_task = task.last_client_task_map[pending_client_name]
            if pending_task.result_received_time is None:
                # result has not been received
                # Note: in this case, the pending client and the asking client must not be the
                # same, because this would be a resend case already taken care of by the controller.
                if pending_client_name == client_name:
                    raise RuntimeError("Logic Error: must not be here for client {}".format(client_name))

                # should this client timeout?
                if task_result_timeout and time.time() - pending_task.task_sent_time > task_result_timeout:
                    # timeout!
                    # give up on the pending task and move to the next target
                    sent_target_count[pending_client_name] -= 1
                    pass
                else:
                    # continue to wait
                    return TaskCheckStatus.BLOCK

        # can send
        task.props[_KEY_PENDING_CLIENT] = client_name
        sent_target_count[client_name] = send_count + 1
        return TaskCheckStatus.SEND

    def check_task_exit(self, task: Task) -> Tuple[bool, TaskCompletionStatus]:
        """Determine whether the task should exit.

        Args:
            task (Task): an instance of Task

        Returns:
            Tuple[bool, TaskCompletionStatus]:
                first entry in the tuple means whether to exit the task or not.  If it's True, the task should exit.
                second entry in the tuple indicates the TaskCompletionStatus.
        """
        # are we waiting for any client?
        num_targets = 0 if task.targets is None else len(task.targets)
        if num_targets == 0:
            # nothing has been sent
            return False, TaskCompletionStatus.IGNORED

        # see whether all targets are sent
        sent_target_count = task.props[_KEY_SEND_TARGET_COUNTS]

        total_sent = 0
        for v in sent_target_count.values():
            total_sent += v

        if total_sent < num_targets:
            return False, TaskCompletionStatus.IGNORED

        # client_tasks might have not been added to task
        if len(task.client_tasks) < num_targets:
            return False, TaskCompletionStatus.IGNORED

        for c_t in task.client_tasks:
            if c_t.result_received_time is None:
                return False, TaskCompletionStatus.IGNORED

        return True, TaskCompletionStatus.OK

    def check_task_result(self, result: Shareable, client_task: ClientTask, fl_ctx: FLContext):
        """Check the result received from the client.

        See whether the client_task is the last one in the task's list
        If not, then it is a late response and ReservedHeaderKey.REPLY_IS_LATE is
        set to True in result's header.

        Args:
            result (Shareable): an instance of Shareable
            client_task (ClientTask): the task processing state of the client
            fl_ctx (FLContext): fl context that comes with the task request
        """
        task = client_task.task
        if client_task != task.client_tasks[-1]:
            result.set_header(key=ReservedHeaderKey.REPLY_IS_LATE, value=True)
