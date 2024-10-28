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
_KEY_TASK_ASSIGN_TIMEOUT = "__task_assignment_timeout"
_KEY_TASK_RESULT_TIMEOUT = "__task_result_timeout"
_KEY_LAST_SEND_IDX = "__last_send_idx"
_PENDING_CLIENT_TASK = "__pending_client_task"


class SequentialRelayTaskManager(TaskManager):
    def __init__(self, task: Task, task_assignment_timeout, task_result_timeout, dynamic_targets: bool):
        """Task manager for relay controller on SendOrder.SEQUENTIAL.

        Args:
            task (Task): an instance of Task
            task_assignment_timeout (int): timeout value on a client requesting its task
            task_result_timeout (int): timeout value on reply of one client
            dynamic_targets (bool): allow clients to join after this task starts
        """
        TaskManager.__init__(self)
        if task_assignment_timeout is None:
            task_assignment_timeout = 0

        if task_result_timeout is None:
            task_result_timeout = 0

        task.props[_KEY_DYNAMIC_TARGETS] = dynamic_targets
        task.props[_KEY_TASK_ASSIGN_TIMEOUT] = task_assignment_timeout
        task.props[_KEY_TASK_RESULT_TIMEOUT] = task_result_timeout
        task.props[_KEY_LAST_SEND_IDX] = -1  # client index of last send
        task.props[_PENDING_CLIENT_TASK] = None

    def check_task_send(self, client_task: ClientTask, fl_ctx: FLContext) -> TaskCheckStatus:
        """Determine whether the task should be sent to the client.

        Args:
            client_task (ClientTask): the task processing state of the client
            fl_ctx (FLContext): fl context that comes with the task request

        Returns:
            TaskCheckStatus: NO_BLOCK for not sending the task, BLOCK for waiting, SEND for OK to send
        """
        client_name = client_task.client.name
        task = client_task.task
        if task.props[_KEY_DYNAMIC_TARGETS]:
            if task.targets is None:
                task.targets = []
            if client_name not in task.targets:
                self.logger.debug("client_name: {} added to task.targets".format(client_name))
                task.targets.append(client_name)

        # is this client eligible?
        if client_name not in task.targets:
            # this client is not a target
            return TaskCheckStatus.NO_BLOCK

        # adjust client window
        win_start_idx, win_end_idx = self._determine_window(task)
        self.logger.debug("win_start_idx={}, win_end_idx={}".format(win_start_idx, win_end_idx))
        if win_start_idx < 0:
            # wait for this task to end by the monitor
            return TaskCheckStatus.BLOCK

        # see whether this client is in the window
        for i in range(win_start_idx, win_end_idx):
            if client_name == task.targets[i]:
                # this client is in the window!
                self.logger.debug("last_send_idx={}".format(i))
                task.props[_KEY_LAST_SEND_IDX] = i
                return TaskCheckStatus.SEND

        # this client is not in the window
        return TaskCheckStatus.NO_BLOCK

    def _determine_window(self, task: Task) -> Tuple[int, int]:
        """Returns two indexes (starting/ending) of a window of client candidates.

        When starting is negative and ending is 0, the window is closed and the task should exit
        When both starting and ending are negative, there is no client candidate as current client task has not returned

        Args:
            task (Task): an instance of Task

        Returns:
            Tuple[int, int]: starting and ending indices of a window of client candidates.

        """
        # adjust client window
        task_result_timeout = task.props[_KEY_TASK_RESULT_TIMEOUT]
        last_send_idx = task.props[_KEY_LAST_SEND_IDX]
        last_send_target = task.targets[last_send_idx]

        if last_send_idx >= 0 and last_send_target in task.last_client_task_map:
            # see whether the result has been received
            last_task = task.last_client_task_map[last_send_target]
            self.logger.debug("last_task={}".format(last_task))

            if last_task.result_received_time is None:
                # result has not been received
                # should this client timeout?
                if task_result_timeout and time.time() - last_task.task_sent_time > task_result_timeout:
                    # timeout!
                    # we give up on this client and move to the next target
                    win_start_idx = last_send_idx + 1
                    win_start_time = last_task.task_sent_time + task_result_timeout
                    self.logger.debug(
                        "client task result timed out. win_start_idx={}, win_start_time={}".format(
                            win_start_idx, win_start_time
                        )
                    )
                else:
                    # continue to wait
                    self.logger.debug("keep waiting on task={}".format(task))
                    return -1, -1
            else:
                # result has been received!
                win_start_idx = last_send_idx + 1
                win_start_time = last_task.result_received_time
                self.logger.debug(
                    "result received. win_start_idx={}, win_start_time={}".format(win_start_idx, win_start_time)
                )
        else:
            # nothing has been sent
            win_start_idx = 0
            win_start_time = task.schedule_time
            self.logger.debug(
                "nothing has been sent. win_start_idx={}, win_start_time={}".format(win_start_idx, win_start_time)
            )

        num_targets = 0 if task.targets is None else len(task.targets)
        if num_targets and win_start_idx >= num_targets:
            # we reached the end of targets
            # so task should exit
            return -1, 0

        task_assignment_timeout = task.props[_KEY_TASK_ASSIGN_TIMEOUT]
        if task_assignment_timeout:
            win_size = int((time.time() - win_start_time) / task_assignment_timeout) + 1
        else:
            win_size = 1

        self.logger.debug("win_size={}".format(win_size))
        win_end_idx = win_start_idx + win_size

        # Should exit if win extends past the entire target list + 1
        if task_assignment_timeout and win_end_idx > num_targets + 1:
            return -1, 0
        if win_end_idx > num_targets:
            win_end_idx = num_targets

        self.logger.debug("win_end_idx={}".format(win_end_idx))
        return win_start_idx, win_end_idx

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
        win_start_idx, win_end_idx = self._determine_window(task)

        self.logger.debug("check_task_exit: win_start_idx={}, win_end_idx={}".format(win_start_idx, win_end_idx))
        if win_start_idx < 0 and win_end_idx == 0:
            last_send_idx = task.props[_KEY_LAST_SEND_IDX]
            last_send_target = task.targets[last_send_idx]

            if last_send_idx >= 0 and last_send_target in task.last_client_task_map:
                # see whether the result has been received
                last_client_task = task.last_client_task_map[last_send_target]
                if last_client_task.result_received_time is not None:
                    return True, TaskCompletionStatus.OK
            return True, TaskCompletionStatus.TIMEOUT
        else:
            return False, TaskCompletionStatus.IGNORED

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
        # see whether the client_task is the last one in the task's list
        # If not, then it is a late response
        task = client_task.task
        if client_task != task.client_tasks[-1]:
            result.set_header(key=ReservedHeaderKey.REPLY_IS_LATE, value=True)
