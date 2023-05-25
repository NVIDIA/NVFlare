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

import threading
import time

from nvflare.apis.dxo import DXO, DataKind
from nvflare.app_common.abstract.launcher import LauncherStatus
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher


class TestSubprocessLauncher:
    def test_launch(self):
        task_name = "__test_task"
        launcher = SubprocessLauncher("echo 'test'")
        launcher.launch(task_name, DXO(DataKind.WEIGHTS, {}), threading.Event())
        time.sleep(1.0)
        status, _ = launcher.check_status(task_name)
        assert status == LauncherStatus.SUCCESS

    def test_stop(self):
        task_name = "__test_task"
        launcher = SubprocessLauncher("python -c \"for i in range(1000000): print('cool')\"")
        launcher.launch(task_name, DXO(DataKind.WEIGHTS, {}), threading.Event())
        status, _ = launcher.check_status(task_name)
        assert status == LauncherStatus.RUNNING
        launcher.stop(task_name)
        status, msg = launcher.check_status(task_name)
        assert status == LauncherStatus.FAILED
        assert msg == "no active process"

    def test_check_status(self):
        task_name = "__test_task"
        launcher = SubprocessLauncher('python -c "print(not_exist_obj)"')
        launcher.launch(task_name, DXO(DataKind.WEIGHTS, {}), threading.Event())
        time.sleep(1.0)
        status, msg = launcher.check_status(task_name)
        assert status == LauncherStatus.FAILED
        assert msg == "External process finished with return code: 1"
