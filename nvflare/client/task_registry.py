# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Dict, Optional

from .config import ClientConfig
from .constants import SYS_ATTRS
from .flare_agent import RC, FlareAgent, Task


class TaskRegistry:
    """This class is used to remember attributes that need to be shared for a user code."""

    def __init__(self, config: ClientConfig, rank: Optional[str] = None, flare_agent: Optional[FlareAgent] = None):
        self.flare_agent = flare_agent
        self.config = config

        self.received_task: Optional[Task] = None
        self.task_name: str = ""
        self.cache_loaded = False
        self.sys_info = {}
        for k, v in self.config.config.items():
            if k in SYS_ATTRS:
                self.sys_info[k] = v
        self.rank = rank

    def _receive(self, timeout: Optional[float] = None):
        if not self.flare_agent:
            return

        task = self.flare_agent.get_task(timeout)

        if task is None:
            raise RuntimeError(f"no received task within timeout: {timeout}")

        if task.data is None:
            raise RuntimeError("no received task.data")

        self.received_task = task
        self.task_name = task.task_name
        self.cache_loaded = True

    def set_task_name(self, task_name: str) -> None:
        """Sets the current task name.

        This method is only used in multiprocess scenario in the lightning API.
        For non-rank 0 processes, they are not getting tasks from the FLARE side,
        thus they rely on the rank 0 process to tell them the current task name
        and will use this method to set it.

        Args:
            task_name (str): current task name
        """
        self.task_name = task_name

    def get_task(self, timeout: Optional[float] = None) -> Optional[Task]:
        """Gets the cached received task.

        Args:
            timeout (float, optional): If specified, this call is blocked only for the specified amount of time.
                If not specified, this call is blocked forever until a task has been received or agent has been closed.

        Returns:
            None if flare agent is None; or a Task object if task is available within timeout.
        """
        if not self.cache_loaded:
            self._receive(timeout)
        return self.received_task

    def get_sys_info(self) -> Dict:
        """Gets NVFlare system information.

        Returns:
            A dict of system information.
        """
        return self.sys_info

    def submit_task(self, data: Any, return_code: str = RC.OK) -> bool:
        """Submits result of the current task.

        Args:
           data: task result
           return_code (str): return code of the task execution

        Returns:
            whether the result is submitted successfully
        """
        if not self.flare_agent or not self.task_name or self.received_task is None:
            return False

        return self.flare_agent.submit_result(result=data, rc=return_code)

    def clear(self) -> None:
        """Clears the cached received task."""
        self.received_task = None
        self.cache_loaded = False

    def __str__(self):
        return f"{self.__class__.__name__}(config: {self.config.get_config()})"

    def __del__(self):
        if self.flare_agent:
            self.flare_agent.stop()
