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

from .task_manager import TaskCheckStatus, TaskManager

_KEY_MIN_RESPS = "__min_responses"
_KEY_WAIT_TIME_AFTER_MIN_RESPS = "__wait_time_after_min_received"
_KEY_MIN_RESPS_RCV_TIME = "__min_resps_received_time"


class BcastTaskManager(TaskManager):
    def __init__(self, task: Task, min_responses: int = 0, wait_time_after_min_received: int = 0):
        """Task manager for broadcast controller.

        Args:
            task (Task): an instance of Task
            min_responses (int, optional): the minimum number of responses so this task is considered finished. Defaults to 0.
            wait_time_after_min_received (int, optional): additional wait time for late clients to contribute their results. Defaults to 0.
        """
        TaskManager.__init__(self)
        task.props[_KEY_MIN_RESPS] = min_responses
        task.props[_KEY_WAIT_TIME_AFTER_MIN_RESPS] = wait_time_after_min_received
        task.props[_KEY_MIN_RESPS_RCV_TIME] = None

    def check_task_exit(self, task: Task) -> Tuple[bool, TaskCompletionStatus]:
        """Determine if the task should exit.

        Args:
            task (Task): an instance of Task

        Tuple[bool, TaskCompletionStatus]:
            first entry in the tuple means whether to exit the task or not.  If it's True, the task should exit.
            second entry in the tuple indicates the TaskCompletionStatus.
        """
        if len(task.client_tasks) == 0:
            # nothing has been sent - continue to wait
            return False, TaskCompletionStatus.IGNORED

        clients_responded = 0
        clients_not_responded = 0
        for s in task.client_tasks:
            if s.result_received_time is None:
                clients_not_responded += 1
            else:
                clients_responded += 1

        if clients_responded >= len(task.targets):
            # all clients have responded!
            return True, TaskCompletionStatus.OK

        # if min_responses is 0, need to have all client tasks responded
        if task.props[_KEY_MIN_RESPS] == 0 and clients_not_responded > 0:
            return False, TaskCompletionStatus.IGNORED

        # check if minimum responses are received
        if clients_responded == 0 or clients_responded < task.props[_KEY_MIN_RESPS]:
            # continue to wait
            return False, TaskCompletionStatus.IGNORED

        # minimum responses received
        min_resps_received_time = task.props[_KEY_MIN_RESPS_RCV_TIME]
        if min_resps_received_time is None:
            min_resps_received_time = time.time()
            task.props[_KEY_MIN_RESPS_RCV_TIME] = min_resps_received_time

        # see whether we have waited for long enough
        if time.time() - min_resps_received_time >= task.props[_KEY_WAIT_TIME_AFTER_MIN_RESPS]:
            # yes - exit the task
            return True, TaskCompletionStatus.OK
        else:
            # no - continue to wait
            return False, TaskCompletionStatus.IGNORED


class BcastForeverTaskManager(TaskManager):
    def __init__(self):
        """Task manager for broadcast controller with forever waiting time."""
        TaskManager.__init__(self)

    def check_task_send(self, client_task: ClientTask, fl_ctx: FLContext) -> TaskCheckStatus:
        """Determine whether the task should be sent to the client.

        Args:
            client_task (ClientTask): the task processing state of the client
            fl_ctx (FLContext): fl context that comes with the task request


        Returns:
            TaskCheckStatus: NO_BLOCK for not sending the task, SEND for OK to send
        """
        # Note: even if the client may have done the task, we may still send it!
        client_name = client_task.client.name
        if client_task.task.targets is None or client_name in client_task.task.targets:
            return TaskCheckStatus.SEND
        else:
            return TaskCheckStatus.NO_BLOCK

    def check_task_exit(self, task: Task) -> Tuple[bool, TaskCompletionStatus]:
        """Determine whether the task should exit.

        Args:
            task (Task): an instance of Task

        Tuple[bool, TaskCompletionStatus]:
            first entry in the tuple means whether to exit the task or not.  If it's True, the task should exit.
            second entry in the tuple indicates the TaskCompletionStatus.
        """
        # never exit
        return False, TaskCompletionStatus.IGNORED
