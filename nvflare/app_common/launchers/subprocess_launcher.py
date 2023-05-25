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
import threading
from typing import Tuple

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.launcher import Launcher, LauncherStatus


class SubprocessLauncher(Launcher):
    def __init__(self, script: str):
        """SubprocessLauncher.

        Args:
            script (str): Script to be launched with subprocess.
        """
        super().__init__()

        self._app_dir = None
        self._process = None
        self._script = script

    def initialize(self, fl_ctx: FLContext):
        super().initialize(fl_ctx)
        self._app_dir = self.get_app_dir()

    def launch(self, task_name: str, dxo: DXO, stop_event: threading.Event):
        command = self._script
        env = os.environ.copy()
        command_seq = shlex.split(command)
        self._process = subprocess.Popen(
            command_seq, stdout=sys.stdout, stderr=subprocess.STDOUT, cwd=self._app_dir, env=env
        )

    def check_status(self, task_name: str) -> Tuple[str, str]:
        if not self._process:
            return LauncherStatus.FAILED, "no active process"
        return_code = self._process.poll()

        if return_code is not None:
            if return_code == 0:
                return LauncherStatus.SUCCESS, ""
            else:
                return LauncherStatus.FAILED, f"External process finished with return code: {return_code}"
        return LauncherStatus.RUNNING, ""

    def stop(self, task_name: str):
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
