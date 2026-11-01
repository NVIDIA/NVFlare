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


"""Shared setup for policy unit tests and real process integration scenarios."""


import os
import sys
import time
from unittest.mock import Mock

import psutil
import pytest

from nvflare.apis.task_launcher_spec import TaskLaunchRequest
from tests.local_task_launcher import ProcessTaskHandle, ProcessTaskLauncher

requires_waitid = pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("waitid", "WNOWAIT", "P_PID", "WEXITED", "WNOHANG", "CLD_EXITED")),
    reason="requires waitid/WNOWAIT (Linux, or macOS with Python >= 3.13)",
)


@pytest.fixture(autouse=True)
def cleanup_launched_groups(monkeypatch):
    handles = []
    launch = ProcessTaskLauncher.launch_task

    def tracked_launch(launcher, request):
        handle = launch(launcher, request)
        handles.append(handle)
        return handle

    monkeypatch.setattr(ProcessTaskLauncher, "launch_task", tracked_launch)
    yield
    for handle in handles:
        handle.cancel()


def _request(tmp_path, attempt_id, code, *args, environment=None):
    return TaskLaunchRequest(
        job_id="job-1",
        site_name="site-1",
        task_id="task-1",
        attempt_id=attempt_id,
        argv=(sys.executable, "-c", code, *map(str, args)),
        environment=dict(os.environ) if environment is None else environment,
        cwd=str(tmp_path),
    )


def _wait_for_file(path, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            if path.stat().st_size > 0:
                return
        except FileNotFoundError:
            pass
        if time.monotonic() >= deadline:
            pytest.fail(f"timed out waiting for contents of {path}")
        time.sleep(0.01)


def _assert_pid_not_running(pid):
    try:
        if psutil.Process(pid).status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            return
    except psutil.NoSuchProcess:
        return
    pytest.fail(f"process {pid} is still running after confirmed process-group settlement")


def _process_running(process):
    try:
        # The retained Process checks identity; the process may still disappear
        # between that check and status(), especially after deliberate SIGTERM.
        return process.is_running() and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


def _mock_handle(tmp_path):
    return ProcessTaskHandle(
        _request(tmp_path, "fault", "pass"),
        Mock(pid=1234, poll=lambda: 0),
        stop_grace_period=0.1,
        descendant_settle_timeout=0.1,
        poll_interval=0.01,
    )
