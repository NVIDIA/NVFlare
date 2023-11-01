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
import shlex
import subprocess
import sys
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.launcher import Launcher, LauncherCompleteStatus


class SubprocessLauncher(Launcher):
    def __init__(self, script: str, clean_up_script: Optional[str] = None):
        """Initializes the SubprocessLauncher.

        Args:
            script (str): Script to be launched using subprocess.
            clean_up_script (Optional[str]): Optional clean up script to be run after the main script execution.
        """
        super().__init__()

        self._app_dir = None
        self._process = None
        self._script = script
        self._clean_up_script = clean_up_script

    def initialize(self, fl_ctx: FLContext):
        self._app_dir = self.get_app_dir(fl_ctx)

    def launch_task(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> bool:
        if self._process is None:
            command = self._script
            env = os.environ.copy()
            command_seq = shlex.split(command)
            self._process = subprocess.Popen(
                command_seq, stdout=sys.stdout, stderr=subprocess.STDOUT, cwd=self._app_dir, env=env
            )
            return True
        return False

    def wait_task(self, task_name: str, fl_ctx: FLContext, timeout: Optional[float] = None) -> LauncherCompleteStatus:
        if self._process:
            return_code = self._process.wait(timeout)
            if return_code == 0:
                return LauncherCompleteStatus.SUCCESS
            return LauncherCompleteStatus.FAILED
        return LauncherCompleteStatus.SUCCESS

    def stop_task(self, task_name: str, fl_ctx: FLContext) -> None:
        if self._process:
            self._process.terminate()
            self._process.wait()
            if self._clean_up_script:
                command_seq = shlex.split(self._clean_up_script)
                process = subprocess.Popen(command_seq, cwd=self._app_dir)
                process.wait()
            self._process = None
