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
    monkeypatch.setattr(process.os, "getpid", lambda: 123)
    monkeypatch.setattr(process.os, "getppid", lambda: 456)


@pytest.mark.parametrize("fails", [False, True])
def test_cli_runs_without_forking_a_guardian_and_exits_without_atexit_joins(monkeypatch, fails):
    events = []
    monkeypatch.setattr(
        process.sys, "argv", ["worker", "--bootstrap", "/attempt/bootstrap.json", "--parent_pid", "456"]
    )
    fork = Mock(side_effect=AssertionError("task entry must not fork a guardian"))
    monkeypatch.setattr(process.os, "fork", fork)
    monkeypatch.setattr("nvflare.apis.job_launcher_spec.pop_credential_env", lambda: events.append("strip_credentials"))

    def compute(path, *, context_binding, protected_context_keys):
        assert path == "/attempt/bootstrap.json"
        assert process.sys.argv == ["nvflare-task-worker"]
        assert callable(context_binding)
        assert protected_context_keys == {"__client_name__", "__process_type__"}
        events.append("compute")
        if fails:
            raise RuntimeError("compute failure")

    monkeypatch.setattr(worker, "run_worker", compute)
    monkeypatch.setattr(process.logging, "shutdown", lambda: events.append("flush"))
    exit_call = Mock(side_effect=SystemExit())
    monkeypatch.setattr(process.os, "_exit", exit_call)
    with pytest.raises(SystemExit):
        process.main()
    assert events == ["strip_credentials", "compute", "flush"]
    fork.assert_not_called()
    exit_call.assert_called_once_with(1 if fails else 0)


@pytest.mark.parametrize("parent, group", [(1, 123), (456, 999)])
def test_cli_requires_own_process_group_and_live_parent(monkeypatch, parent, group):
    monkeypatch.setattr(
        process.sys, "argv", ["worker", "--bootstrap", "/attempt/bootstrap.json", "--parent_pid", str(parent)]
    )
    monkeypatch.setattr(process.os, "getpgrp", lambda: group)
    with pytest.raises(RuntimeError, match="live parent and its own process group"):
        process.main()


def test_cli_stops_before_compute_if_launch_parent_has_changed(monkeypatch):
    monkeypatch.setattr(
        process.sys, "argv", ["worker", "--bootstrap", "/attempt/bootstrap.json", "--parent_pid", "999"]
    )
    kill = Mock(side_effect=SystemExit())
    monkeypatch.setattr(process.os, "killpg", kill)
    compute = Mock()
    monkeypatch.setattr(worker, "run_worker", compute)
    with pytest.raises(SystemExit):
        process.main()
    kill.assert_called_once_with(123, signal.SIGKILL)
    compute.assert_not_called()
