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

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("exit_path", ["sys_exit", "process_monitor"])
def test_server_exit_preserves_first_cancellation_without_waiting_forever(tmp_path, exit_path):
    marker = tmp_path / "cancelled"
    script = textwrap.dedent(
        """
        import sys, threading
        from pathlib import Path
        from unittest.mock import Mock
        from nvflare.private.fed.server.server_engine import ServerEngine
        from nvflare.apis.fl_constant import RunProcessKey

        class Handle:
            def terminate(self):
                Path(sys.argv[1]).write_text("cancel requested")
            def wait(self):
                threading.Event().wait()

        engine = ServerEngine.__new__(ServerEngine)
        engine.lock = threading.Lock()
        engine.logger = Mock()
        engine.engine_info = Mock()
        engine.send_command_to_child_runner_process = Mock(return_value="OK")
        engine.run_processes = {"job-1": {RunProcessKey.JOB_HANDLE: Handle()}}
        remove = engine._remove_run_processes
        def bounded_grace(**kwargs):
            kwargs["max_wait"] = 0.05
            remove(**kwargs)
        engine._remove_run_processes = bounded_grace
        if sys.argv[2] == "process_monitor":
            from nvflare.fuel.f3.mpm import MainProcessMonitor
            engine.executor = Mock()
            engine.shutdown_streamer = Mock()
            def main():
                engine.abort_app_on_server("job-1")
                engine.close()
            rc = MainProcessMonitor.run(main, run_dir=str(Path(sys.argv[1]).parent),
                shutdown_grace_time=0, cleanup_grace_time=0)
            sys.exit(rc)
        else:
            engine.abort_app_on_server("job-1")
            sys.exit(0)
    """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(marker), exit_path], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "cancel requested"
    assert "RuntimeError" not in result.stderr
