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

"""Exercise launcher shutdown with real processes and a file sentinel for CLOSE.

These tests cover worker exit and reaping, not the complete FLARE rank protocol.
"""

import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.executors import multi_process_executor as mpe
from nvflare.fuel.common.multi_process_executor_constants import MultiProcessCommandNames

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "killpg"),
    reason="requires Unix process groups",
)

_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "multi_process_shutdown_worker.py"


class _SentinelCell:
    def __init__(self, directory):
        self.directory = directory

    def fire_and_forget(self, **kwargs):
        assert kwargs["topic"] == MultiProcessCommandNames.CLOSE
        (self.directory / "close").touch()


class _Executor(mpe.MultiProcessExecutor):
    def get_multi_process_command(self):
        return "unused"


def _executor(worker, directory):
    executor = _Executor()
    executor.targets = ["rank-0", "rank-1"]
    executor.engine = SimpleNamespace(client=SimpleNamespace(cell=_SentinelCell(directory)))
    executor.exe_process = worker
    return executor


@contextmanager
def _start_worker(mode, directory):
    readiness = ["ready-0", "ready-1"] if mode == "launcher" else ["ready"]
    log_path = directory / "worker.log"
    with log_path.open("w") as log:
        worker = subprocess.Popen(
            [sys.executable, str(_WORKER_SCRIPT), mode, str(directory)],
            start_new_session=True,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 10.0
            while not all((directory / name).exists() for name in readiness):
                if worker.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail(f"worker readiness failed: {log_path.read_text()}")
                time.sleep(0.01)
            yield worker
        finally:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait(timeout=5.0)


def _assert_reaped(worker):
    # finalize must have reaped the launcher before the test's cleanup runs.
    with pytest.raises(ChildProcessError):
        os.waitpid(worker.pid, os.WNOHANG)


def test_finalize_allows_rank_exit_and_reaps_launcher(tmp_path):
    with _start_worker("launcher", tmp_path) as worker:
        _executor(worker, tmp_path).finalize(FLContext())

        assert (tmp_path / "close").exists()
        assert (tmp_path / "finished-0").exists()
        assert (tmp_path / "finished-1").exists()
        assert (tmp_path / "launcher-finished").exists()
        assert worker.returncode == 0
        _assert_reaped(worker)


def test_finalize_kills_and_reaps_unresponsive_launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(mpe, "_WORKER_SHUTDOWN_TIMEOUT", 0.1)
    monkeypatch.setattr(mpe, "_WORKER_KILL_TIMEOUT", 2.0)
    with _start_worker("hung", tmp_path) as worker:
        start = time.monotonic()
        _executor(worker, tmp_path).finalize(FLContext())
        elapsed = time.monotonic() - start

        assert (tmp_path / "close").exists()
        assert worker.returncode == -signal.SIGKILL
        _assert_reaped(worker)
        assert elapsed < 5.0
