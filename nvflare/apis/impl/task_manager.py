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

import logging
from enum import Enum
from typing import Tuple

from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class TaskCheckStatus(Enum):

    SEND = 1  # send the task to the client
    BLOCK = 2  # do not send the task, and block other tasks
    NO_BLOCK = 3  # do not send the task, and continue checking


class TaskManager(object):
    def __init__(self):
        """Manages tasks for clients.

        Programming Conventions:
        A TaskManager should be implemented as a state-free object.
        All task processing state info should be stored in the Task's props dict.
        Name the keys in the props dict with prefix "__" to avoid potential conflict with
        app-defined props.
        """
        self._name = self.__class__.__name__
        self.logger = logging.getLogger(self._name)

    def check_task_send(self, client_task: ClientTask, fl_ctx: FLContext) -> TaskCheckStatus:
        """Determine whether the task should be sent to the client.

        Default logic:
        If the client already did the task, don't send again (BLOCK).
        If the client is in the task's target list or the task's target
        list is None (meaning all clients), then send the task (SEND). Otherwise, do not block the
        task checking (NO_BLOCK), so next task will be checked.

        Args:
            client_task (ClientTask): the task processing state of the client
            fl_ctx (FLContext): fl context that comes with the task request


        Returns:
            TaskCheckStatus: NO_BLOCK for not sending the task, BLOCK for waiting, SEND for OK to send
        """
        if client_task.result_received_time:
            # the task was already sent to the client AND result was already received
            # do not send again
            return TaskCheckStatus.NO_BLOCK

        client_name = client_task.client.name
        if client_task.task.targets is None or client_name in client_task.task.targets:
            return TaskCheckStatus.SEND
        else:
            return TaskCheckStatus.NO_BLOCK

    def check_task_exit(self, task: Task) -> Tuple[bool, TaskCompletionStatus]:
        """Determine whether the task should exit.

        Args:
            task (Task): an instance of Task

        Returns:
            Tuple[bool, TaskCompletionStatus]:
                first entry in the tuple means whether to exit the task or not.  If it's True, the task should exit.
                second entry in the tuple indicates the TaskCompletionStatus.
        """
        pass

    def check_task_result(self, result: Shareable, client_task: ClientTask, fl_ctx: FLContext):
        """Check the result received from the client.

        The manager can set appropriate headers into the result to indicate certain conditions (e.g.
        late response).

        Args:
            result (Shareable): the result to be checked
            client_task (ClientTask): the task processing state of the client
            fl_ctx (FLContext): fl context that comes with the task request
        """
        pass
