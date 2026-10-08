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

import signal
from unittest.mock import Mock

import pytest

from nvflare.private.fed.app.client import task_worker_process as process
from nvflare.private.fed.task_worker import worker


@pytest.fixture(autouse=True)
def _owned_group(monkeypatch):
    monkeypatch.setattr(process.os, "getpgrp", lambda: 123)


@pytest.mark.parametrize("lost", ["worker", "job"])
def test_guardian_kills_the_owned_group_when_either_parent_is_lost(monkeypatch, lost):
    monkeypatch.setattr(process.os, "getppid", Mock(side_effect=[123, 1] if lost == "worker" else [123]))
    task_worker = Mock()
    task_worker.ppid.return_value = 456 if lost == "worker" else 1
    task_worker.status.return_value = process.psutil.STATUS_SLEEPING
    parent = Mock()
    parent.status.return_value = process.psutil.STATUS_SLEEPING
    monkeypatch.setattr(process.psutil, "Process", lambda pid: task_worker if pid == 123 else parent)
    monkeypatch.setattr(process.time, "sleep", Mock())
    monkeypatch.setattr(process.os, "getpgrp", lambda: 123)
    kill = Mock()
    monkeypatch.setattr(process.os, "killpg", kill)
    monkeypatch.setattr(process.os, "_exit", Mock(side_effect=SystemExit(1)))
    with pytest.raises(SystemExit):
        process._watch_parent(456, 123)
    task_worker.kill.assert_called_once()
    kill.assert_called_once_with(123, signal.SIGKILL)


@pytest.mark.parametrize("lost", ["worker_zombie", "parent_zombie", "worker_identity", "parent_identity"])
def test_guardian_rejects_unreaped_or_reused_processes_without_reparenting(monkeypatch, lost):
    task_worker = Mock()
    task_worker.ppid.return_value = 456
    task_worker.status.return_value = (
        process.psutil.STATUS_ZOMBIE if lost == "worker_zombie" else process.psutil.STATUS_SLEEPING
    )
    task_worker.is_running.return_value = lost != "worker_identity"
    parent = Mock()
    parent.status.return_value = (
        process.psutil.STATUS_ZOMBIE if lost == "parent_zombie" else process.psutil.STATUS_SLEEPING
    )
    parent.is_running.return_value = lost != "parent_identity"
    monkeypatch.setattr(process.os, "getppid", lambda: 123)
    monkeypatch.setattr(process.psutil, "Process", lambda pid: task_worker if pid == 123 else parent)
    sleep = Mock()
    monkeypatch.setattr(process.time, "sleep", sleep)
    monkeypatch.setattr(process.os, "getpgrp", lambda: 123)
    events = []
    task_worker.kill.side_effect = lambda: events.append("worker")
    monkeypatch.setattr(process.os, "killpg", lambda pid, sig: events.append((pid, sig)))
    monkeypatch.setattr(process.os, "_exit", Mock(side_effect=SystemExit(1)))
    with pytest.raises(SystemExit):
        process._watch_parent(456, 123)
    sleep.assert_not_called()
    assert events == ["worker", (123, signal.SIGKILL)]


@pytest.mark.parametrize("missing", ["worker", "parent"])
def test_guardian_still_cleans_owned_group_if_process_lookup_fails(monkeypatch, missing):
    task_worker = Mock()

    def lookup(pid):
        if pid == (123 if missing == "worker" else 456):
            raise process.psutil.NoSuchProcess(pid)
        return task_worker

    monkeypatch.setattr(process.psutil, "Process", lookup)
    kill = Mock()
    monkeypatch.setattr(process.os, "killpg", kill)
    monkeypatch.setattr(process.os, "_exit", Mock(side_effect=SystemExit(1)))
    with pytest.raises(SystemExit):
        process._watch_parent(456, 123)
    kill.assert_called_once_with(123, signal.SIGKILL)
    assert task_worker.kill.call_count == (0 if missing == "worker" else 1)


def test_guardian_still_cleans_group_if_worker_is_already_gone(monkeypatch):
    task_worker = Mock()
    task_worker.is_running.return_value = False
    task_worker.kill.side_effect = process.psutil.NoSuchProcess(123)
    monkeypatch.setattr(process.os, "getppid", lambda: 123)
    monkeypatch.setattr(process.psutil, "Process", lambda _pid: task_worker)
    kill = Mock()
    monkeypatch.setattr(process.os, "killpg", kill)
    monkeypatch.setattr(process.os, "_exit", Mock(side_effect=SystemExit(1)))
    with pytest.raises(SystemExit):
        process._watch_parent(456, 123)
    kill.assert_called_once_with(123, signal.SIGKILL)


@pytest.mark.parametrize("fails", [False, True])
def test_cli_arms_guard_before_compute_and_exits_without_atexit_joins(monkeypatch, fails):
    events = []
    monkeypatch.setattr(
        process.sys, "argv", ["worker", "--bootstrap", "/attempt/bootstrap.json", "--parent_pid", "456"]
    )
    monkeypatch.setattr(process.os, "getpid", lambda: 123)
    monkeypatch.setattr(process.os, "getpgrp", lambda: 123)
    monkeypatch.setattr(process.os, "getppid", lambda: 456)
    monkeypatch.setattr("nvflare.apis.job_launcher_spec.pop_credential_env", lambda: events.append("strip_credentials"))
    monkeypatch.setattr(process, "_start_parent_guard", lambda _parent: events.append("guard"))

    def compute(path):
        assert path == "/attempt/bootstrap.json"
        assert process.sys.argv == ["nvflare-task-worker"]
        events.append("compute")
        if fails:
            raise RuntimeError("compute failure")

    monkeypatch.setattr(worker, "run_worker", compute)
    monkeypatch.setattr(process.logging, "shutdown", lambda: events.append("flush"))
    exit_call = Mock(side_effect=SystemExit())
    monkeypatch.setattr(process.os, "_exit", exit_call)
    with pytest.raises(SystemExit):
        process.main()
    assert events == ["strip_credentials", "guard", "compute", "flush"]
    exit_call.assert_called_once_with(1 if fails else 0)
