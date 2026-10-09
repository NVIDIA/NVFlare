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
    monkeypatch.setattr(process.os, "getpid", lambda: 789)


@pytest.fixture
def parents():
    parent, task_worker = Mock(pid=456), Mock(pid=123)
    for member in (parent, task_worker):
        member.is_running.return_value = True
        member.status.return_value = process.psutil.STATUS_SLEEPING
    return parent, task_worker


@pytest.mark.parametrize("lost", ["zombie", "dead", "reused", "missing"])
def test_guardian_kills_owned_group_on_job_parent_loss(monkeypatch, parents, lost):
    parent, task_worker = parents
    parent.status.return_value = lost
    parent.is_running.return_value = lost != "reused"
    if lost == "missing":
        parent.status.side_effect = process.psutil.NoSuchProcess(parent.pid)
    sleep = Mock()
    monkeypatch.setattr(process.time, "sleep", sleep)
    monkeypatch.setattr(process.os, "getpgrp", lambda: 123)
    events = []
    task_worker.kill.side_effect = lambda: events.append("worker")
    monkeypatch.setattr(process.os, "killpg", lambda pid, sig: events.append((pid, sig)))
    monkeypatch.setattr(process.os, "_exit", Mock(side_effect=SystemExit(1)))
    with pytest.raises(SystemExit):
        process._watch_parent(parent, task_worker)
    sleep.assert_not_called()
    assert events == ["worker", (123, signal.SIGKILL)]


@pytest.mark.parametrize("exited", ["zombie", "dead", "reused", "missing"])
def test_guardian_keeps_watching_job_after_worker_exit(monkeypatch, parents, exited):
    parent, task_worker = parents
    parent.is_running.side_effect = [True, True, False]
    task_worker.status.return_value = exited
    task_worker.is_running.return_value = exited != "reused"
    if exited == "missing":
        task_worker.status.side_effect = process.psutil.NoSuchProcess(task_worker.pid)
    task_worker.kill.side_effect = process.psutil.NoSuchProcess(task_worker.pid)
    snapshot = Mock(return_value=None)  # A descendant is still cleaning up.
    monkeypatch.setattr(process, "_dead_descendant_snapshot", snapshot)
    kill = Mock()
    monkeypatch.setattr(process.os, "killpg", kill)
    sleep = Mock(side_effect=lambda _: kill.assert_not_called())
    monkeypatch.setattr(process.time, "sleep", sleep)
    monkeypatch.setattr(process.os, "_exit", Mock(side_effect=SystemExit(1)))
    with pytest.raises(SystemExit):
        process._watch_parent(parent, task_worker)
    assert snapshot.call_count == sleep.call_count == 2
    kill.assert_called_once_with(123, signal.SIGKILL)


def test_guardian_exits_without_signals_after_two_stable_dead_snapshots(monkeypatch, parents):
    parent, task_worker = parents
    task_worker.status.return_value = process.psutil.STATUS_ZOMBIE
    snapshot = Mock(side_effect=[None, {(222, 1.0)}, set(), set(), set()])
    monkeypatch.setattr(process, "_dead_descendant_snapshot", snapshot)
    kill = Mock()
    monkeypatch.setattr(process.os, "killpg", kill)
    sleep = Mock()
    monkeypatch.setattr(process.time, "sleep", sleep)
    exit_call = Mock(side_effect=SystemExit())
    monkeypatch.setattr(process.os, "_exit", exit_call)
    with pytest.raises(SystemExit):
        process._watch_parent(parent, task_worker)
    assert sleep.call_count == 2
    task_worker.kill.assert_not_called()
    kill.assert_not_called()
    exit_call.assert_called_once_with(0)


@pytest.mark.parametrize("state", ["live", "dead", "vanished", "denied", "no_group_dead", "no_group_live"])
def test_guardian_descendant_snapshot_requires_known_dead_members(monkeypatch, state):
    monkeypatch.setattr(process.psutil, "pids", lambda: [0, 123, 789, 111, 222])

    def group(pid):
        assert pid not in (0, 123, 789)
        if pid == 222 and state.startswith("no_group"):
            raise ProcessLookupError()
        return 123 if pid == 222 else 999

    monkeypatch.setattr(process.os, "getpgid", group)
    member = Mock()
    member.status.return_value = (
        process.psutil.STATUS_ZOMBIE if state.endswith("dead") else process.psutil.STATUS_RUNNING
    )
    member.create_time.return_value = 1.0
    lookup = Mock(return_value=member)
    if state in ("vanished", "denied"):
        lookup.side_effect = (
            process.psutil.NoSuchProcess(222) if state == "vanished" else process.psutil.AccessDenied(222)
        )
    monkeypatch.setattr(process.psutil, "Process", lookup)
    assert process._dead_descendant_snapshot(123) == ({(222, 1.0)} if state.endswith("dead") else None)
    lookup.assert_called_once_with(222)


def test_guardian_treats_inaccessible_process_as_live():
    member = Mock()
    member.status.side_effect = process.psutil.AccessDenied(456)
    assert process._is_live(member)


def test_guard_captures_owner_and_worker_identities_before_fork(monkeypatch, parents):
    parent, task_worker = parents
    task_worker.ppid.return_value = parent.pid
    lookups = []

    def lookup(pid):
        lookups.append(pid)
        return parent if pid == parent.pid else task_worker

    def fork():
        assert lookups == [parent.pid, task_worker.pid]
        return 987

    monkeypatch.setattr(process.os, "getpid", lambda: task_worker.pid)
    monkeypatch.setattr(process.psutil, "Process", lookup)
    monkeypatch.setattr(process.os, "fork", fork)
    monkeypatch.setattr(process.signal, "signal", Mock())
    process._start_parent_guard(parent.pid)


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
