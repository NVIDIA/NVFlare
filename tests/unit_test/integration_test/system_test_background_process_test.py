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

import importlib.util
import os
import shlex
import signal
import sys
import textwrap
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INTEGRATION_ROOT = _REPO_ROOT / "tests" / "integration_test"


def _load_system_test_module(monkeypatch):
    monkeypatch.chdir(_INTEGRATION_ROOT)
    monkeypatch.setenv("NVFLARE_TEST_FRAMEWORK", "client_api_attach_ccwf")
    spec = importlib.util.spec_from_file_location(
        "system_test_background_process_module", _INTEGRATION_ROOT / "system_test.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup requires POSIX signals")
def test_background_cleanup_kills_surviving_child_after_group_leader_exits(tmp_path, monkeypatch):
    system_test = _load_system_test_module(monkeypatch)
    child_ready = tmp_path / "child_ready"
    program = textwrap.dedent(
        f"""
        import os
        import signal
        import time
        from pathlib import Path

        child_pid = os.fork()
        if child_pid:
            Path({str(child_ready)!r}).write_text(str(child_pid), encoding="utf-8")
            while True:
                time.sleep(1)

        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(1)
        """
    )
    command = f"{sys.executable} -c {shlex.quote(program)}"
    process = system_test.run_command_in_subprocess(command)
    background_process = (command, process, process.pid)

    deadline = time.monotonic() + 5.0
    while not child_ready.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert child_ready.exists(), "background leader did not start its child"

    try:
        system_test._stop_background_processes([background_process], graceful_timeout=0.1, kill_timeout=2.0)
        assert not system_test._is_process_group_alive(process.pid)
    finally:
        if system_test._is_process_group_alive(process.pid):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)
