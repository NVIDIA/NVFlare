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

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace


class LauncherCompleteStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class Launcher(ABC):
    def initialize(self, fl_ctx: FLContext) -> None:
        pass

    def finalize(self, fl_ctx: FLContext) -> None:
        pass

    @staticmethod
    def get_app_dir(fl_ctx: FLContext) -> Optional[str]:
        """Gets the deployed application directory."""
        if fl_ctx is not None:
            workspace: Workspace = fl_ctx.get_engine().get_workspace()
            app_dir = workspace.get_app_dir(fl_ctx.get_job_id())
            if app_dir is not None:
                return os.path.abspath(app_dir)
        return None

    @abstractmethod
    def launch_task(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> bool:
        """Launches external system to handle a task.

        Args:
            task_name (str): task name.
            shareable (Shareable): input shareable.
            fl_ctx (FLContext): fl context.
            abort_signal (Signal): signal to check during execution to determine whether this task is aborted.

        Returns:
            Whether launch success or not.
        """
        pass

    @abstractmethod
    def wait_task(self, task_name: str, fl_ctx: FLContext, timeout: Optional[float] = None) -> LauncherCompleteStatus:
        """Waits for external system to end.

        Args:
            task_name (str): task name.
            fl_ctx (FLContext): fl context.
            timeout (optional, float): time to wait for task.

        Returns:
            The completion status of Launcher.
        """
        pass

    @abstractmethod
    def stop_task(self, task_name: str, fl_ctx: FLContext) -> None:
        """Stops external system and free up resources.

        Args:
            task_name (str): task name.
            fl_ctx (FLContext): fl context.
        """
        pass
