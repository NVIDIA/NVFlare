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

import errno
import os
import subprocess
import sys

import psutil
import pytest

from tests import process_test_utils as processes
from tests.timing_utils import expire_deadline_when_ready, wait_for

pytestmark = [pytest.mark.timeout(40), pytest.mark.skipif(os.name != "posix", reason="POSIX detached trainer")]

_WRAPPER = """
import os, pathlib, subprocess, sys, time
child = subprocess.Popen(
    [sys.executable, '-c',
     'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print(\"ready\", flush=True); time.sleep(60)'],
    start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
)
assert child.stdout.readline().strip() == "ready"
child.stdout.close()
print('original job output', flush=True)
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
if sys.argv[2] == 'timeout':
    time.sleep(60)
else:
    sys.exit(int(sys.argv[2]))
"""


@pytest.mark.parametrize("outcome", ["0", "7", "timeout"])
@pytest.mark.parametrize("pidfd_error", [False, True])
def test_runner_stops_detached_trainers_and_retains_outcome(tmp_path, monkeypatch, outcome, pidfd_error):
    ready = tmp_path / "trainer.pid"
    if pidfd_error:
        monkeypatch.setattr(
            processes.psutil, "wait_procs", lambda *_, **__: (_ for _ in ()).throw(OSError(errno.EINVAL, "pidfd"))
        )
    command = [sys.executable, "-c", _WRAPPER, str(ready), outcome]
    if outcome == "timeout":
        expire_deadline_when_ready(monkeypatch, processes, ready.exists)
    child = None
    try:
        if outcome == "timeout":
            with pytest.raises(subprocess.TimeoutExpired) as caught:
                processes.run_owned_process(command, timeout=5, cleanup_grace=0.2)
            assert "original job output" in caught.value.output
            assert caught.value.timeout == 5
        else:
            result = processes.run_owned_process(command, timeout=10, cleanup_grace=0.2)
            assert result.returncode == int(outcome)
            assert "original job output" in result.stdout
        assert ready.exists(), "controlled trainer never reached readiness"
        try:
            child = psutil.Process(int(ready.read_text()))
        except psutil.NoSuchProcess:
            return
        assert not processes.process_running(child), "detached trainer survived runner cleanup"
    finally:
        if child is None and ready.exists():
            try:
                child = psutil.Process(int(ready.read_text()))
            except psutil.NoSuchProcess:
                pass
        if child is not None and processes.process_running(child):
            child.kill()


@pytest.mark.parametrize("failure", ["startup", "interruption"])
def test_partial_startup_and_interruption_close_owned_processes(tmp_path, failure):
    ready = tmp_path / "trainer.pid"
    process = None
    try:
        with pytest.raises((FileNotFoundError, KeyboardInterrupt)):
            with processes.OwnedProcesses(grace=0.2) as owner:
                process = owner.start([sys.executable, "-c", _WRAPPER, str(ready), "timeout"])
                wait_for(ready.exists)
                if failure == "startup":
                    owner.start([str(tmp_path / "missing-command")])
                else:
                    raise KeyboardInterrupt("controlled interruption")
        assert process.poll() is not None
        try:
            child = psutil.Process(int(ready.read_text()))
        except psutil.NoSuchProcess:
            return
        assert not processes.process_running(child)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if ready.exists():
            try:
                child = psutil.Process(int(ready.read_text()))
                if processes.process_running(child):
                    child.kill()
            except psutil.NoSuchProcess:
                pass


def test_nonzero_command_and_cleanup_error_keep_both_diagnostics(monkeypatch):
    close = processes.OwnedProcesses.close

    def close_then_fail(owner):
        close(owner)
        raise RuntimeError("controlled cleanup error")

    monkeypatch.setattr(processes.OwnedProcesses, "close", close_then_fail)
    with pytest.raises(RuntimeError) as caught:
        processes.run_owned_process(
            [sys.executable, "-c", "import sys; print('training traceback', file=sys.stderr); sys.exit(7)"]
        )
    assert caught.value.returncode == 7
    assert "training traceback" in caught.value.stderr
    assert "returncode=7" in str(caught.value)
    assert "controlled cleanup error" in str(caught.value)
