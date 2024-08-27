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

import logging
import os
import shlex
import subprocess
from threading import Thread
from typing import Optional

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.launcher import Launcher, LauncherRunStatus
from nvflare.private.fed.utils.fed_utils import add_custom_dir_to_path


def log_subprocess_output(process, logger):
    for c in iter(process.stdout.readline, b""):
        logger.info(c.decode().rstrip())


class SubprocessLauncher(Launcher):
    def __init__(self, script: str, launch_once: bool = True, clean_up_script: Optional[str] = None):
        """Initializes the SubprocessLauncher.

        Args:
            script (str): Script to be launched using subprocess.
            clean_up_script (Optional[str]): Optional clean up script to be run after the main script execution.
        """
        super().__init__()

        self._app_dir = None
        self._process = None
        self._script = script
        self._launch_once = launch_once
        self._clean_up_script = clean_up_script
        self.logger = logging.getLogger(self.__class__.__name__)

    def initialize(self, fl_ctx: FLContext):
        self._app_dir = self.get_app_dir(fl_ctx)
        if self._launch_once:
            self._start_external_process(fl_ctx)

    def finalize(self, fl_ctx: FLContext) -> None:
        if self._launch_once and self._process:
            self._stop_external_process()

    def launch_task(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> bool:
        if not self._launch_once:
            self._start_external_process(fl_ctx)
        return True

    def stop_task(self, task_name: str, fl_ctx: FLContext, abort_signal: Signal) -> None:
        if not self._launch_once:
            self._stop_external_process()

    def _start_external_process(self, fl_ctx: FLContext):
        if self._process is None:
            command = self._script
            env = os.environ.copy()
            env["CLIENT_API_TYPE"] = "EX_PROCESS_API"

            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
            app_custom_folder = workspace.get_app_custom_dir(job_id)
            add_custom_dir_to_path(app_custom_folder, env)

            command_seq = shlex.split(command)
            self._process = subprocess.Popen(
                command_seq, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=self._app_dir, env=env
            )
            self._log_thread = Thread(target=log_subprocess_output, args=(self._process, self.logger))
            self._log_thread.start()

    def _stop_external_process(self):
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._log_thread.join()
            if self._clean_up_script:
                command_seq = shlex.split(self._clean_up_script)
                process = subprocess.Popen(command_seq, cwd=self._app_dir)
                process.wait()
            self._process = None

    def check_run_status(self, task_name: str, fl_ctx: FLContext) -> str:
        if self._process is None:
            return LauncherRunStatus.NOT_RUNNING
        return_code = self._process.poll()
        if return_code is None:
            return LauncherRunStatus.RUNNING
        if return_code == 0:
            return LauncherRunStatus.COMPLETE_SUCCESS
        return LauncherRunStatus.COMPLETE_FAILED
