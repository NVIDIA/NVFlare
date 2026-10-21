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
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_opt.task_launcher import k8s_worker
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity, WorkerBootstrap, write_bootstrap


@pytest.mark.parametrize("authorized", [True, False])
def test_real_worker_preserves_authorization_and_immutable_result_contract(tmp_path, isolated_fobs, authorized):
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
    repo_root = Path(__file__).resolve().parents[4]
    env = {key: value for key, value in os.environ.items() if key not in JobProcessEnv.ALL}
    env["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        [sys.executable, "-m", "nvflare.app_opt.task_launcher.k8s_worker", "--bootstrap", path],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
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
def test_entry_strips_credentials_before_compute_without_a_guardian(monkeypatch, fails):
    from nvflare.private.fed.task_worker import worker

    events = []
    monkeypatch.setattr(k8s_worker.sys, "argv", ["worker", "--bootstrap", "/attempt/bootstrap.json"])
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
    assert k8s_worker.main() == (1 if fails else 0)
