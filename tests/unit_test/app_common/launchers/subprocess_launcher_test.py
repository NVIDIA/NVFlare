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

import shutil
import tempfile

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher


class TestSubprocessLauncher:
    def test_launch(self):
        tempdir = tempfile.mkdtemp()
        fl_ctx = FLContext()
        launcher = SubprocessLauncher("echo 'test'")
        launcher._app_dir = tempdir

        signal = Signal()
        task_name = "__test_task"
        dxo = DXO(DataKind.WEIGHTS, {})
        status = launcher.launch_task(task_name, dxo.to_shareable(), fl_ctx, signal)
        assert status is True
        shutil.rmtree(tempdir)

    def test_stop(self):
        tempdir = tempfile.mkdtemp()
        fl_ctx = FLContext()
        launcher = SubprocessLauncher("python -c \"for i in range(1000000): print('cool')\"", shutdown_timeout=0.0)
        launcher._app_dir = tempdir

        signal = Signal()
        task_name = "__test_task"
        dxo = DXO(DataKind.WEIGHTS, {})
        status = launcher.launch_task(task_name, dxo.to_shareable(), fl_ctx, signal)
        assert status is True
        launcher.stop_task(task_name, fl_ctx, signal)

        assert launcher._process is None
        shutil.rmtree(tempdir)

    def test_default_launch_once_is_true(self):
        """Test that launch_once defaults to True."""
        launcher = SubprocessLauncher("echo 'test'")
        assert launcher._launch_once is True

    def test_default_shutdown_timeout_is_zero(self):
        """Test that shutdown_timeout defaults to 0.0."""
        launcher = SubprocessLauncher("echo 'test'")
        assert launcher._shutdown_timeout == 0.0

    @pytest.mark.parametrize(
        "launch_once,shutdown_timeout",
        [
            (True, 0.0),  # Default values
            (False, 0.0),  # launch_once=False with default timeout
            (True, 10.0),  # launch_once=True with custom timeout
            (False, 15.0),  # launch_once=False with custom timeout
            (True, 100.0),  # Large timeout value
        ],
    )
    def test_launch_once_and_shutdown_timeout_initialization(self, launch_once, shutdown_timeout):
        """Test various launch_once and shutdown_timeout configurations."""
        launcher = SubprocessLauncher(script="echo 'test'", launch_once=launch_once, shutdown_timeout=shutdown_timeout)
        assert launcher._launch_once == launch_once
        assert launcher._shutdown_timeout == shutdown_timeout

    def test_launch_once_false_behavior(self):
        """Test that launch_once=False launches process per task."""
        tempdir = tempfile.mkdtemp()
        fl_ctx = FLContext()
        launcher = SubprocessLauncher("echo 'test'", launch_once=False)
        launcher._app_dir = tempdir

        signal = Signal()
        task_name = "__test_task"
        dxo = DXO(DataKind.WEIGHTS, {})

        # First task should launch and complete
        status = launcher.launch_task(task_name, dxo.to_shareable(), fl_ctx, signal)
        assert status is True

        # Stop should terminate the process since launch_once=False
        launcher.stop_task(task_name, fl_ctx, signal)
        assert launcher._process is None

        shutil.rmtree(tempdir)

    def test_launch_once_true_behavior(self):
        """Test that launch_once=True keeps process running across tasks."""
        tempdir = tempfile.mkdtemp()
        fl_ctx = FLContext()

        # Use a long-running process
        launcher = SubprocessLauncher('python3 -c "import time; time.sleep(100)"', launch_once=True)
        launcher._app_dir = tempdir

        # Initialize should launch the process
        launcher.initialize(fl_ctx)
        assert launcher._process is not None
        first_process = launcher._process

        signal = Signal()
        task_name = "__test_task"
        dxo = DXO(DataKind.WEIGHTS, {})

        # Launch task should not create a new process
        status = launcher.launch_task(task_name, dxo.to_shareable(), fl_ctx, signal)
        assert status is True
        assert launcher._process is first_process  # Same process

        # Stop task should NOT terminate the process since launch_once=True
        launcher.stop_task(task_name, fl_ctx, signal)
        assert launcher._process is first_process  # Still the same process
        assert launcher._process is not None

        # Finalize should stop the process
        launcher.finalize(fl_ctx)
        assert launcher._process is None

        shutil.rmtree(tempdir)

    def test_shutdown_timeout_parameter(self):
        """Test that shutdown_timeout is properly stored."""
        launcher = SubprocessLauncher("echo 'test'", shutdown_timeout=30.0)
        assert launcher._shutdown_timeout == 30.0

    def test_clean_up_script_with_launch_once(self):
        """Test that clean_up_script can be used with launch_once=False."""
        tempdir = tempfile.mkdtemp()
        fl_ctx = FLContext()

        launcher = SubprocessLauncher(
            script="echo 'main'",
            launch_once=False,
            clean_up_script="echo 'cleanup'",
            shutdown_timeout=0.0,
        )
        launcher._app_dir = tempdir

        signal = Signal()
        task_name = "__test_task"
        dxo = DXO(DataKind.WEIGHTS, {})

        # Launch and stop task
        status = launcher.launch_task(task_name, dxo.to_shareable(), fl_ctx, signal)
        assert status is True

        launcher.stop_task(task_name, fl_ctx, signal)
        assert launcher._process is None

        shutil.rmtree(tempdir)
