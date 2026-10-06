# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.controller_spec import Task, TaskCompletionStatus
from nvflare.apis.impl.seq_relay_manager import SequentialRelayTaskManager
from nvflare.apis.shareable import Shareable


def _manager_and_task(targets):
    task = Task(name="__test_task", data=Shareable())
    task.targets = targets
    task.schedule_time = time.time()
    manager = SequentialRelayTaskManager(
        task=task,
        task_assignment_timeout=1,
        task_result_timeout=1,
        dynamic_targets=True,
    )
    return manager, task


class TestSequentialRelayTaskManager:
    def test_check_task_exit_with_no_targets_yet(self):
        """The task monitor calls this before any client has joined.

        A relay with targets of "*", or one scheduled before a client has
        registered, leaves task.targets empty, and the monitor thread has no
        exception handling, so indexing the list there took the thread down and
        the job stopped timing anything out.
        """
        manager, task = _manager_and_task([])

        assert manager.check_task_exit(task) == (False, TaskCompletionStatus.IGNORED)

    def test_check_task_exit_keeps_waiting_with_one_target(self):
        manager, task = _manager_and_task(["__test_client0"])

        assert manager.check_task_exit(task) == (False, TaskCompletionStatus.IGNORED)
