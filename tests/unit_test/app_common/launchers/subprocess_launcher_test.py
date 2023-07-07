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

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher


class TestSubprocessLauncher:
    def test_launch(self):
        task_name = "__test_task"
        launcher = SubprocessLauncher("echo 'test'")
        dxo = DXO(DataKind.WEIGHTS, {})
        fl_ctx = FLContext()
        signal = Signal()
        status = launcher.launch_task(task_name, dxo.to_shareable(), fl_ctx, signal)
        assert status is True

    def test_stop(self):
        task_name = "__test_task"
        launcher = SubprocessLauncher("python -c \"for i in range(1000000): print('cool')\"")
        dxo = DXO(DataKind.WEIGHTS, {})
        fl_ctx = FLContext()
        signal = Signal()
        status = launcher.launch_task(task_name, dxo.to_shareable(), fl_ctx, signal)
        assert status is True
        launcher.stop_task(task_name, fl_ctx)

        assert launcher._process is None
