# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.controller_spec import ClientTask, SendOrder, Task, TaskCompletionStatus
from nvflare.apis.fl_context import FLContext

from .task_manager import TaskCheckStatus, TaskManager

_KEY_ORDER = "__order"
_KEY_TASK_ASSIGN_TIMEOUT = "__task_assignment_timeout"


class SendTaskManager(TaskManager):
    def __init__(self, task: Task, send_order: SendOrder, task_assignment_timeout):
        """Task manager for send controller.

        Args:
            task (Task): an instance of Task
            send_order (SendOrder): the order of clients to receive task
            task_assignment_timeout (int): timeout value on a client requesting its task
        """
        TaskManager.__init__(self)
        if task_assignment_timeout is None or task_assignment_timeout <= 0:
            task_assignment_timeout = 0
        task.props[_KEY_ORDER] = send_order
        task.props[_KEY_TASK_ASSIGN_TIMEOUT] = task_assignment_timeout

    def check_task_send(self, client_task: ClientTask, fl_ctx: FLContext) -> TaskCheckStatus:
        """Determine whether the task should be sent to the client.

        Args:
            client_task (ClientTask): the task processing state of the client
            fl_ctx (FLContext): fl context that comes with the task request

        Returns:
            TaskCheckStatus: NO_BLOCK for not sending the task, BLOCK for waiting, SEND for OK to send
        """
        task = client_task.task
        if len(task.client_tasks) > 0:  # already sent to one client
            if client_task.task_sent_time is not None:  # the task was sent to this client!
                if client_task.result_received_time is not None:
                    # the task result was already received
                    # this task is actually done - waiting to end by the monitor
                    return TaskCheckStatus.NO_BLOCK
                else:
                    return TaskCheckStatus.SEND
            else:  # the task was sent to someone else
                return TaskCheckStatus.NO_BLOCK

        # in SEQUENTIAL mode - targets must be explicitly specified
        # is this client eligible?
        try:
            client_idx = task.targets.index(client_task.client.name)
        except ValueError:
            client_idx = -1

        if client_idx < 0:
            # this client is not a target
            return TaskCheckStatus.NO_BLOCK

        if task.props[_KEY_ORDER] == SendOrder.ANY:
            return TaskCheckStatus.SEND

        task_assignment_timeout = task.props[_KEY_TASK_ASSIGN_TIMEOUT]
        if task_assignment_timeout == 0:
            # no client timeout - can only send to the first target
            eligible_client_idx = 0
        else:
            elapsed = time.time() - task.create_time
            eligible_client_idx = int(elapsed / task_assignment_timeout)

        if client_idx <= eligible_client_idx:
            return TaskCheckStatus.SEND
        else:
            # this client is currently not eligible but could be later
            # since this client is involved in the task, we need to wait until this task is resolved!
            return TaskCheckStatus.BLOCK

    def check_task_exit(self, task: Task) -> Tuple[bool, TaskCompletionStatus]:
        """Determine whether the task should exit.

        Args:
            task (Task): an instance of Task

        Tuple[bool, TaskCompletionStatus]:
            first entry in the tuple means whether to exit the task or not.  If it's True, the task should exit.
            second entry in the tuple indicates the TaskCompletionStatus.
        """
        if len(task.client_tasks) > 0:
            # there should be only a single item in the task's client status list
            # because only a single client is sent the task!
            for s in task.client_tasks:
                if s.result_received_time is not None:
                    # this task is done!
                    return True, TaskCompletionStatus.OK

        # no one is working on this task yet or the task is not done
        return False, TaskCompletionStatus.IGNORED
