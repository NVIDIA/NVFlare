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

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace


class LauncherRunStatus:
    COMPLETE_SUCCESS = "success"
    COMPLETE_FAILED = "failed"
    RUNNING = "running"
    NOT_RUNNING = "not_running"


class Launcher(FLComponent, ABC):
    def initialize(self, fl_ctx: FLContext) -> None:
        pass

    def finalize(self, fl_ctx: FLContext) -> None:
        pass

    @staticmethod
    def get_app_dir(fl_ctx: FLContext) -> str:
        """Gets the deployed application directory."""
        workspace: Workspace = fl_ctx.get_engine().get_workspace()
        app_dir = workspace.get_app_dir(fl_ctx.get_job_id())
        return os.path.abspath(app_dir)

    def needs_deferred_stop(self) -> bool:
        """Returns True if stop_task() should be deferred to a background thread.

        Deferred stop is needed when the launcher terminates the external process on
        each stop_task() call (launch_once=False), so the process can stay alive long
        enough for the server to finish downloading large tensors from it.

        For launch_once=True launchers the subprocess lives for the entire job, so
        deferring would block the next round's launch indefinitely — return False.
        """
        return False

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
    def stop_task(self, task_name: str, fl_ctx: FLContext, abort_signal: Signal) -> None:
        """Stops external system and free up resources.

        Args:
            task_name (str): task name.
            fl_ctx (FLContext): fl context.

        Note:
            Implementations must be idempotent and thread-safe. LauncherExecutor may call
            stop_task() from a deferred background thread and, in extreme timeout scenarios,
            concurrently from the main task thread as a fallback. A second concurrent or
            sequential call must be a safe no-op (e.g. guard on a null process reference
            inside a lock, as SubprocessLauncher does).
        """
        pass

    @abstractmethod
    def check_run_status(self, task_name: str, fl_ctx: FLContext) -> str:
        """Checks the run status of Launcher."""
        pass
