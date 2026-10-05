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
from nvflare.private.fed.client.task_worker_client_api import CLIENT_TASK_CONTEXT_KEYS, bind_client_task_context
from nvflare.private.fed.task_worker import worker


@pytest.mark.parametrize("lost", ["worker", "job"])
def test_guardian_kills_the_owned_group_when_either_parent_is_lost(monkeypatch, lost):
    monkeypatch.setattr(process.os, "getppid", Mock(side_effect=[123, 1] if lost == "worker" else [123]))
    parent = Mock()
    parent.ppid.return_value = 456 if lost == "worker" else 1
    monkeypatch.setattr(process.psutil, "Process", lambda _pid: parent)
    monkeypatch.setattr(process.time, "sleep", Mock())
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

    def compute(path, *, context_binding, protected_context_keys):
        assert path == "/attempt/bootstrap.json"
        assert context_binding is bind_client_task_context
        assert protected_context_keys is CLIENT_TASK_CONTEXT_KEYS
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
