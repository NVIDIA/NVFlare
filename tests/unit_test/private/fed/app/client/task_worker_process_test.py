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

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.app_common.np.constants import NPConstants
from nvflare.private.fed.app.client import task_worker_process as process
from nvflare.private.fed.task_worker import worker
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity, WorkerBootstrap, write_bootstrap


@pytest.fixture
def _owned_group(monkeypatch):
    monkeypatch.setattr(process.os, "getpgrp", lambda: 123)
    monkeypatch.setattr(process.os, "getpid", lambda: 123)
    monkeypatch.setattr(process.os, "getppid", lambda: 456)


@pytest.mark.parametrize("fails", [False, True])
def test_cli_runs_without_forking_a_guardian_and_exits_without_atexit_joins(monkeypatch, fails, _owned_group):
    events = []
    argv = ["worker", "--bootstrap", "/attempt/bootstrap.json", "--parent_pid", "456"]
    monkeypatch.setattr(process.sys, "argv", argv)
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


@pytest.mark.parametrize("parent, group", [(0, 123), (-1, 123), (1, 123), (456, 999)])
def test_cli_requires_own_process_group_and_live_parent(monkeypatch, parent, group, _owned_group):
    monkeypatch.setattr(
        process.sys, "argv", ["worker", "--bootstrap", "/attempt/bootstrap.json", "--parent_pid", str(parent)]
    )
    monkeypatch.setattr(process.os, "getpgrp", lambda: group)
    with pytest.raises(RuntimeError, match="live parent and its own process group"):
        process.main()


@pytest.mark.parametrize("scope_args", [[], ["--execution_scope", "process"]])
@pytest.mark.parametrize("actual_parent", [1, 456])
def test_cli_requires_explicit_process_parent(monkeypatch, scope_args, actual_parent, _owned_group):
    monkeypatch.setattr(process.sys, "argv", ["worker", "--bootstrap", "/attempt/bootstrap.json", *scope_args])
    monkeypatch.setattr(process.os, "getppid", lambda: actual_parent)
    compute = Mock()
    monkeypatch.setattr(worker, "run_worker", compute)
    monkeypatch.setattr(process.os, "_exit", Mock(side_effect=SystemExit(0)))
    with pytest.raises(SystemExit) as error:
        process.main()
    assert error.value.code == 2
    compute.assert_not_called()


def test_cli_stops_before_compute_if_launch_parent_has_changed(monkeypatch, _owned_group):
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


@pytest.mark.parametrize("authorized", [True, False])
@pytest.mark.parametrize("scope", ["process", "native"])
def test_real_worker_preserves_authorization_and_immutable_result_contract(tmp_path, isolated_fobs, authorized, scope):
    # Actual CPU interpreter and unmodified NPTrainer. This tests container
    # entry/worker behavior, not Kubernetes pod/runtime/storage qualification.
    root = tmp_path / "workspace"
    (root / "startup").mkdir(parents=True)
    (root / "local").mkdir()
    app = root / "job" / "app_site"
    (app / "config").mkdir(parents=True)
    (app / "custom").mkdir()
    (root / "job" / "meta.json").write_text(json.dumps({"byoc": False}))
    marker = root / "imported"
    (app / "custom" / "untrusted.py").write_text(f"open({str(marker)!r}, 'w').write('imported')\n")
    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    store = FileTaskArtifactStore(str(root / "job" / "artifacts"))
    store.create_attempt(identity)
    store.write_input(identity, DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: np.array([1.0])}).to_shareable())
    bootstrap = WorkerBootstrap(
        identity,
        store.root_dir,
        str(root),
        {"path": "nvflare.app_common.np.np_trainer.NPTrainer" if authorized else "untrusted.Executor"},
    )
    path = store.bootstrap_path(identity)
    write_bootstrap(path, bootstrap)
    repo_root = Path(__file__).resolve().parents[6]
    env = {key: value for key, value in os.environ.items() if key not in JobProcessEnv.ALL}
    env["PYTHONPATH"] = str(repo_root)
    argv = [sys.executable, "-m", "nvflare.private.fed.app.client.task_worker_process", "--bootstrap", path]
    argv.extend(["--parent_pid", str(os.getpid())] if scope == "process" else ["--execution_scope", "native"])
    completed = subprocess.run(
        argv,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        start_new_session=scope == "process",
    )
    assert completed.returncode == (0 if authorized else 1), completed.stdout + completed.stderr
    assert not marker.exists()
    if authorized:
        result, completion = store.read_result(identity)
        assert from_shareable(result).data[NPConstants.NUMPY_KEY].tolist() == [2.0]
        assert completion.identity == identity
    else:
        assert store.read_failure(identity)["error_type"] == "ComponentNotAuthorized"
        assert not Path(store.attempt_dir(identity), "completion.json").exists()


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("pid,parent,group", [(1, 0, 1), (17, 1, 1)])
def test_native_entry_strips_credentials_without_local_parent_or_group_assumptions(
    monkeypatch, fails, pid, parent, group
):
    from nvflare.private.fed.task_worker import worker

    events = []
    monkeypatch.setattr(
        process.sys, "argv", ["worker", "--bootstrap", "/attempt/bootstrap.json", "--execution_scope", "native"]
    )
    monkeypatch.setattr(process.os, "getpid", lambda: pid)
    monkeypatch.setattr(process.os, "getpgrp", lambda: group)
    monkeypatch.setattr(process.os, "getppid", lambda: parent)
    monkeypatch.setattr(process.logging, "shutdown", lambda: None)
    exit_call = Mock(side_effect=SystemExit())
    monkeypatch.setattr(process.os, "_exit", exit_call)
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "GPU-plugin-assigned-uuid")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr("nvflare.apis.job_launcher_spec.pop_credential_env", lambda: events.append("credentials"))

    def run(path, *, context_binding, protected_context_keys):
        assert events == ["credentials"]
        assert sys.argv == ["nvflare-task-worker"]
        assert path == "/attempt/bootstrap.json"
        assert callable(context_binding)
        assert protected_context_keys == {"__client_name__", "__process_type__"}
        assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "GPU-plugin-assigned-uuid"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        if fails:
            raise ValueError("worker failure")

    monkeypatch.setattr(worker, "run_worker", run)
    with pytest.raises(SystemExit):
        process.main()
    exit_call.assert_called_once_with(1 if fails else 0)


def test_native_scope_rejects_a_local_parent_pid_before_compute(monkeypatch):
    monkeypatch.setattr(
        process.sys,
        "argv",
        ["worker", "--bootstrap", "/attempt/bootstrap.json", "--execution_scope", "native", "--parent_pid", "1"],
    )
    compute = Mock()
    monkeypatch.setattr(worker, "run_worker", compute)
    with pytest.raises(SystemExit) as error:
        process.main()
    assert error.value.code == 2
    compute.assert_not_called()
