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

import socket
import subprocess
import sys
import time

import pytest

from nvflare.fuel.sec import step_ca_admin_cert as step_ca
from tests.integration_test.src.utils import stop_process_group
from tests.process_launcher_test_utils import _assert_pid_not_running
from tests.timing_utils import wait_for


@pytest.mark.parametrize("interrupted", [False, True])
def test_step_ca_stops_login_child_and_releases_port(monkeypatch, tmp_path, interrupted):
    ready = tmp_path / "login.ready"
    child_code = (
        "import os, socket, sys, threading\n"
        "from pathlib import Path\n"
        "listener = socket.socket()\n"
        "listener.bind(('127.0.0.1', 0))\n"
        "listener.listen()\n"
        "ready = Path(sys.argv[1])\n"
        "staging = ready.with_suffix('.tmp')\n"
        "staging.write_text(f'{os.getpid()} {listener.getsockname()[1]}')\n"
        "staging.replace(ready)\n"
        "threading.Event().wait(30)\n"
    )
    parent_code = "import subprocess, sys\nsubprocess.run([sys.executable, '-c', sys.argv[1], sys.argv[2]])"
    process = subprocess.Popen([sys.executable, "-c", parent_code, child_code, str(ready)], start_new_session=True)
    wait = process.wait
    try:
        # Start the timeout only after the real child owns its callback port.
        pid, port = map(int, wait_for(lambda: ready.read_text() if ready.exists() else None).split())
        if interrupted:

            def interrupt_wait(*_args, **_kwargs):
                process.wait = wait
                raise KeyboardInterrupt

            process.wait = interrupt_wait
        monkeypatch.setattr(step_ca.subprocess, "Popen", lambda *_args, **_kwargs: process)
        error = KeyboardInterrupt if interrupted else step_ca.AdminCertProviderError
        started = time.monotonic()
        with pytest.raises(error, match=None if interrupted else "timed out after 0.01 seconds"):
            step_ca._run_step(["step"], timeout=0.01)
        assert time.monotonic() - started < 5
        _assert_pid_not_running(pid)
        assert process.poll() is not None
        with socket.socket() as retry:
            retry.bind(("127.0.0.1", port))
    finally:
        process.wait = wait
        stop_process_group(process)
