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
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.private.fed.task_worker import (
    ContextProperty,
    FileTaskArtifactStore,
    IncompleteTaskArtifactError,
    TaskAttemptIdentity,
    WorkerBootstrap,
    write_bootstrap,
)
from nvflare.private.fed.task_worker.protocol import WORKER_MODULE
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize

_PROBE_MODULE = """
import json
import os

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.shareable import Shareable


class ValueComponent(FLComponent):
    def __init__(self, value):
        super().__init__()
        self.value = value


class ProbeExecutor(Executor):
    def __init__(self, marker_path):
        super().__init__()
        self.marker_path = marker_path

    def handle_event(self, event_type, fl_ctx):
        if event_type in (EventType.START_RUN, EventType.END_RUN):
            with open(self.marker_path, "a") as stream:
                stream.write(event_type + "\\n")

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        try:
            fl_ctx.get_engine().get_cell()
        except RuntimeError as e:
            unsupported = str(e)
        else:
            unsupported = "missing explicit failure"
        return Shareable({
            "value": shareable["value"] + fl_ctx.get_engine().get_component("value").value,
            "pid": os.getpid(),
            "site": fl_ctx.get_identity_name(),
            "job_id": fl_ctx.get_job_id(),
            "task_id": fl_ctx.get_prop("__task_id__"),
            "context": fl_ctx.get_prop("explicit_value"),
            "unsupported": unsupported,
            "credentials": [name for name in %r if name in os.environ],
        })


class RaiseExecutor(ProbeExecutor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        raise ValueError("deliberate worker failure")
""" % (
    JobProcessEnv.ALL,
)


@pytest.fixture(autouse=True)
def _initialize_fobs():
    nvflare_fobs_initialize()


def _workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "startup").mkdir(parents=True)
    (root / "local").mkdir()
    app_root = root / "job-1" / "app_site-1"
    custom_dir = app_root / "custom"
    custom_dir.mkdir(parents=True)
    (app_root / "config").mkdir()
    (custom_dir / "worker_components.py").write_text(_PROBE_MODULE)
    (root / "job-1" / "meta.json").write_text(json.dumps({"byoc": True}))
    return root


def _identity(attempt_id, task_id="task-1", task_name="train"):
    return TaskAttemptIdentity(
        job_id="job-1",
        site_name="site-1",
        task_id=task_id,
        task_name=task_name,
        attempt_id=attempt_id,
    )


def _stage(store, identity, workspace_root, executor, data, components=(), context_properties=None):
    store.create_attempt(identity)
    store.write_input(identity, data)
    bootstrap = WorkerBootstrap(
        identity=identity,
        artifact_root=store.root_dir,
        workspace_root=str(workspace_root),
        executor=executor,
        components=components,
        context_properties=context_properties or {},
    )
    path = store.bootstrap_path(identity)
    write_bootstrap(path, bootstrap)
    return path


def _run_process(bootstrap_path, extra_env=None):
    repo_root = str(Path(__file__).resolve().parents[5])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([repo_root, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", WORKER_MODULE, "--bootstrap", bootstrap_path],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_real_worker_process_commits_after_finalization_and_exposes_only_supported_services(tmp_path):
    workspace_root = _workspace(tmp_path)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("attempt-1")
    marker = tmp_path / "events.txt"
    bootstrap_path = _stage(
        store,
        identity,
        workspace_root,
        executor={"path": "worker_components.ProbeExecutor", "args": {"marker_path": str(marker)}},
        components=({"id": "value", "path": "worker_components.ValueComponent", "args": {"value": 4}},),
        context_properties={"explicit_value": ContextProperty("transported", private=True, sticky=False)},
        data=Shareable({"value": 3}),
    )

    process = _run_process(bootstrap_path, {name: "must-not-reach-executor" for name in JobProcessEnv.ALL})

    assert process.returncode == 0, process.stderr
    result, completion = store.read_result(identity)
    assert result["value"] == 7
    assert result["pid"] == completion.worker_pid
    assert result["pid"] != os.getpid()
    assert completion.worker_ppid == os.getpid()
    assert result["site"] == identity.site_name
    assert result["job_id"] == identity.job_id
    assert result["task_id"] == identity.task_id
    assert result["context"] == "transported"
    assert "does not provide a federation Cell" in result["unsupported"]
    assert result["credentials"] == []
    assert marker.read_text().splitlines() == ["_start_run", "_end_run"]
    assert completion.completed_at >= completion.started_at
    assert completion.diagnostics["user_cpu_seconds"] >= 0
    assert store.read_input(identity)["value"] == 3


def test_executor_exception_runs_finalization_but_does_not_commit_success(tmp_path):
    workspace_root = _workspace(tmp_path)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("attempt-failure")
    marker = tmp_path / "events.txt"
    bootstrap_path = _stage(
        store,
        identity,
        workspace_root,
        executor={"path": "worker_components.RaiseExecutor", "args": {"marker_path": str(marker)}},
        data=Shareable(),
    )

    process = _run_process(bootstrap_path)

    assert process.returncode != 0
    assert "deliberate worker failure" in process.stderr
    assert marker.read_text().splitlines() == ["_start_run", "_end_run"]
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_result(identity)
    failure = json.loads((Path(store.attempt_dir(identity)) / "failure.json").read_text())
    assert failure["identity"] == identity.to_dict()
    assert failure["error_type"] == "ValueError"


def test_component_authorization_rejects_custom_executor_before_import(tmp_path):
    workspace_root = _workspace(tmp_path)
    (workspace_root / "job-1" / "meta.json").write_text(json.dumps({"byoc": False}))
    import_marker = tmp_path / "imported.txt"
    custom_dir = workspace_root / "job-1" / "app_site-1" / "custom"
    (custom_dir / "forbidden_components.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(import_marker)!r}).write_text('component was imported')\n"
        "from nvflare.apis.executor import Executor\n"
        "class ForbiddenExecutor(Executor):\n"
        "    def execute(self, task_name, shareable, fl_ctx, abort_signal):\n"
        "        return shareable\n"
    )
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("attempt-forbidden")
    _stage(
        store,
        identity,
        workspace_root,
        executor={"path": "forbidden_components.ForbiddenExecutor", "args": {}},
        data=Shareable(),
    )

    process = _run_process(store.bootstrap_path(identity))

    assert process.returncode != 0
    assert "not in allow_list" in process.stderr
    assert not import_marker.exists()
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_result(identity)


def test_nested_component_is_authorized_before_any_component_import(tmp_path):
    workspace_root = _workspace(tmp_path)
    (workspace_root / "job-1" / "meta.json").write_text(json.dumps({"byoc": False}))
    (workspace_root / "local" / "resources.json.default").write_text(
        json.dumps({"class_allow_list": ["allowed_components.AllowedExecutor"]})
    )
    import_marker = tmp_path / "nested-imported.txt"
    custom_dir = workspace_root / "job-1" / "app_site-1" / "custom"
    (custom_dir / "allowed_components.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(import_marker)!r}).write_text('outer imported')\n"
        "from nvflare.apis.executor import Executor\n"
        "class AllowedExecutor(Executor):\n"
        "    def __init__(self, helper):\n"
        "        super().__init__()\n"
        "        self.helper = helper\n"
        "    def execute(self, task_name, shareable, fl_ctx, abort_signal):\n"
        "        return shareable\n"
    )
    (custom_dir / "forbidden_nested.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(import_marker)!r}).write_text('nested imported')\n"
        "class ForbiddenNested:\n"
        "    pass\n"
    )
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("attempt-forbidden-nested")
    executor = {
        "path": "allowed_components.AllowedExecutor",
        "args": {"helper": {"path": "forbidden_nested.ForbiddenNested", "args": {}}},
    }
    _stage(store, identity, workspace_root, executor=executor, data=Shareable())

    process = _run_process(store.bootstrap_path(identity))

    assert process.returncode != 0
    assert "forbidden_nested.ForbiddenNested" in process.stderr
    assert "not in allow_list" in process.stderr
    assert not import_marker.exists()


def test_bootstrap_cannot_override_authoritative_job_metadata(tmp_path):
    workspace_root = _workspace(tmp_path)
    (workspace_root / "job-1" / "meta.json").write_text(json.dumps({"byoc": False}))
    import_marker = tmp_path / "imported.txt"
    custom_dir = workspace_root / "job-1" / "app_site-1" / "custom"
    (custom_dir / "metadata_bypass.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(import_marker)!r}).write_text('imported')\n"
        "from nvflare.apis.executor import Executor\n"
        "class MetadataBypassExecutor(Executor):\n"
        "    def execute(self, task_name, shareable, fl_ctx, abort_signal):\n"
        "        return shareable\n"
    )
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("attempt-meta-override")
    _stage(
        store,
        identity,
        workspace_root,
        executor={"path": "metadata_bypass.MetadataBypassExecutor", "args": {}},
        data=Shareable(),
        context_properties={"__job_meta__": ContextProperty({"byoc": True})},
    )

    process = _run_process(store.bootstrap_path(identity))

    assert process.returncode != 0
    assert "cannot override framework context property '__job_meta__'" in process.stderr
    assert not import_marker.exists()


_CLIENT_SCRIPT = """
import argparse
import json
import os
import threading
from pathlib import Path
import numpy as np
import nvflare.client as flare

parser = argparse.ArgumentParser()
parser.add_argument('--attempt-dir')
parser.add_argument('--fail-after-send', action='store_true')
args = parser.parse_args()
flare.init()
count = 0
while flare.is_running():
    model = flare.receive()
    count += 1
    flare.log('loss', 0.5, flare.AnalyticsDataType.SCALAR)
    if flare.is_evaluate():
        output = flare.FLModel(metrics={'accuracy': float(model.params['weights'].sum())})
    else:
        output = flare.FLModel(params={'weights': model.params['weights'] + 2}, metrics={'accuracy': 1.0})
    flare.send(output)
    attempt = Path(args.attempt_dir)
    assert (attempt / 'result.fobs').is_file()
    assert not (attempt / 'completion.json').exists()
    flare.log('after_send', 1, flare.AnalyticsDataType.SCALAR)
    if args.fail_after_send:
        raise RuntimeError('failure after durable send')
assert count == 1
assert flare.receive() is None
Path(args.attempt_dir, 'script-finished.json').write_text(json.dumps({
    'pid': os.getpid(), 'main_thread': threading.current_thread() is threading.main_thread(),
    'site': flare.get_site_name(), 'job': flare.get_job_id(), 'task': flare.get_task_name()
}))
"""


@pytest.mark.parametrize("task_name", ["train", "validate"])
@pytest.mark.parametrize("transfer_type", ["FULL", "DIFF"])
def test_client_api_script_runs_one_assignment_on_main_thread_with_durable_handoff(tmp_path, task_name, transfer_type):
    workspace_root = _workspace(tmp_path)
    (workspace_root / "job-1/app_site-1/custom/train.py").write_text(_CLIENT_SCRIPT)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("attempt-client-api", task_name=task_name)
    executor = {
        "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
        "args": {
            "execution_mode": "in_process",
            "task_script_path": "train.py",
            "task_script_args": ["--attempt-dir", store.attempt_dir(identity)],
            "params_exchange_format": "numpy",
            "params_transfer_type": transfer_type,
        },
    }
    data = FLModelUtils.to_shareable(FLModel(params={"weights": np.asarray([1.0, 2.0])}, current_round=1))
    _stage(store, identity, workspace_root, executor=executor, data=data)

    process = _run_process(store.bootstrap_path(identity))

    assert process.returncode == 0, process.stderr
    result, completion = store.read_result(identity)
    model = FLModelUtils.from_shareable(result)
    if task_name == "train":
        expected = [2.0, 2.0] if transfer_type == "DIFF" else [3.0, 4.0]
        np.testing.assert_array_equal(model.params["weights"], expected)
    else:
        assert model.params is None
        assert model.metrics == {"accuracy": 3.0}
    finished = json.loads((Path(store.attempt_dir(identity)) / "script-finished.json").read_text())
    assert finished == {
        "pid": completion.worker_pid,
        "main_thread": True,
        "site": "site-1",
        "job": "job-1",
        "task": task_name,
    }
    assert [record["key"] for record in store.read_analytics(identity, completion)] == ["loss", "after_send"]
    assert model.current_round == 1


def test_client_api_failure_after_send_does_not_commit_success(tmp_path):
    workspace_root = _workspace(tmp_path)
    (workspace_root / "job-1/app_site-1/custom/train.py").write_text(_CLIENT_SCRIPT)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("attempt-client-api-failure")
    executor = {
        "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
        "args": {
            "execution_mode": "in_process",
            "task_script_path": "train.py",
            "task_script_args": ["--attempt-dir", store.attempt_dir(identity), "--fail-after-send"],
        },
    }
    data = FLModelUtils.to_shareable(FLModel(params={"weights": np.asarray([1.0, 2.0])}))
    _stage(store, identity, workspace_root, executor, data)
    process = _run_process(store.bootstrap_path(identity))

    assert process.returncode != 0
    assert "failure after durable send" in process.stderr
    assert (Path(store.attempt_dir(identity)) / "result.fobs").is_file()
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_result(identity)


def test_unmodified_np_trainer_keeps_workspace_model_state_across_fresh_workers(tmp_path):
    workspace_root = _workspace(tmp_path)
    (workspace_root / "job-1" / "meta.json").write_text(json.dumps({"byoc": False}))
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    executor = {"path": "nvflare.app_common.np.np_trainer.NPTrainer", "args": {"delta": 2}}
    train_identity = _identity("attempt-train", task_id="task-train", task_name=AppConstants.TASK_TRAIN)
    train_data = DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: np.asarray([1.0, 2.0])}).to_shareable()
    _stage(store, train_identity, workspace_root, executor, train_data)

    train_process = _run_process(store.bootstrap_path(train_identity))

    assert train_process.returncode == 0, train_process.stderr
    trained, train_completion = store.read_result(train_identity)
    np.testing.assert_allclose(from_shareable(trained).data[NPConstants.NUMPY_KEY], [3.0, 4.0])

    submit_identity = _identity("attempt-submit", task_id="task-submit", task_name=AppConstants.TASK_SUBMIT_MODEL)
    _stage(store, submit_identity, workspace_root, executor, Shareable())
    submit_process = _run_process(store.bootstrap_path(submit_identity))

    assert submit_process.returncode == 0, submit_process.stderr
    submitted, submit_completion = store.read_result(submit_identity)
    np.testing.assert_allclose(from_shareable(submitted).data[NPConstants.NUMPY_KEY], [3.0, 4.0])
    assert train_completion.worker_pid != submit_completion.worker_pid


def test_unmodified_np_validator_runs_in_worker(tmp_path):
    workspace_root = _workspace(tmp_path)
    (workspace_root / "job-1" / "meta.json").write_text(json.dumps({"byoc": False}))
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("attempt-validate", task_id="task-validate", task_name=AppConstants.TASK_VALIDATION)
    executor = {"path": "nvflare.app_common.np.np_validator.NPValidator", "args": {}}
    data = DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: np.asarray([1.0, 2.0])}).to_shareable()
    _stage(store, identity, workspace_root, executor, data)

    process = _run_process(store.bootstrap_path(identity))

    assert process.returncode == 0, process.stderr
    result, completion = store.read_result(identity)
    validated = from_shareable(result)
    assert validated.data_kind == DataKind.METRICS
    assert isinstance(validated.data["accuracy"], float)
    assert completion.worker_pid != os.getpid()
