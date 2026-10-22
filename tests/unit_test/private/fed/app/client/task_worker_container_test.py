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

from unittest.mock import Mock

import pytest

from nvflare.private.fed.app.client import task_worker_container as container
from nvflare.private.fed.task_worker import worker


@pytest.mark.parametrize("fails", [False, True])
def test_pid_one_worker_reuses_authorized_worker_and_exits_without_atexit(monkeypatch, fails):
    events = []
    monkeypatch.setattr(container.os, "getpid", lambda: 1)
    monkeypatch.setattr(container.sys, "argv", ["worker", "--bootstrap", "/job/bootstrap.json"])
    monkeypatch.setattr(container, "pop_credential_env", lambda: events.append("strip"))
    monkeypatch.setattr(container, "_flush_logs", lambda: events.append("flush"))

    def compute(path, *, context_binding, protected_context_keys):
        assert path == "/job/bootstrap.json"
        assert callable(context_binding)
        assert protected_context_keys == {"__client_name__", "__process_type__"}
        events.append("compute")
        if fails:
            raise RuntimeError("failed")

    monkeypatch.setattr(worker, "run_worker", compute)
    exit_call = Mock(side_effect=SystemExit())
    monkeypatch.setattr(container.os, "_exit", exit_call)
    with pytest.raises(SystemExit):
        container.main()
    assert events == ["strip", "compute", "flush"]
    exit_call.assert_called_once_with(1 if fails else 0)


def test_worker_requires_owned_container_pid_namespace(monkeypatch):
    monkeypatch.setattr(container.os, "getpid", lambda: 100)
    monkeypatch.setattr(container.sys, "argv", ["worker", "--bootstrap", "/job/bootstrap.json"])
    with pytest.raises(RuntimeError, match="PID 1"):
        container.main()
