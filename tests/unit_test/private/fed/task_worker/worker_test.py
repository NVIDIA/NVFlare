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
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import EventScope, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.decomposers.flare_decomposers import DXODecomposer
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.decomposers.common_decomposers import FLModelDecomposer
from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer
from nvflare.app_common.executors.client_api.backend_spec import CLIENT_API_BACKEND_FACTORY
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposer import DictDecomposer
from nvflare.private.fed.task_worker import (
    ContextProperty,
    FileTaskArtifactStore,
    IncompleteTaskArtifactError,
    TaskAttemptIdentity,
    WorkerBootstrap,
    worker,
    write_bootstrap,
)
from nvflare.private.fed.task_worker.protocol import WORKER_MODULE
from nvflare.private.fed.task_worker.runtime import TaskRuntime, UnsupportedTaskRuntimeService
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
    # Other suites reset the registry without resetting module-level guards.
    # Register this fixture's exact dependencies on every invocation.
    fobs.register(DictDecomposer(Shareable))
    fobs.register(DictDecomposer(ModelLearnable))
    fobs.register(DXODecomposer)
    fobs.register(FLModelDecomposer)
    fobs.register(NumpyArrayDecomposer)


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


@pytest.mark.parametrize(
    "property_name",
    [
        FLContextKey.JOB_META,
        CLIENT_API_BACKEND_FACTORY,
        FLContextKey.CLIENT_NAME,
        FLContextKey.PROCESS_TYPE,
        FLContextKey.RUN_ABORT_SIGNAL,
    ],
)
def test_bootstrap_cannot_override_authoritative_runtime_properties(tmp_path, property_name):
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
        context_properties={property_name: ContextProperty({"byoc": True})},
    )

    process = _run_process(store.bootstrap_path(identity))

    assert process.returncode != 0
    assert f"cannot override framework context property {property_name!r}" in process.stderr
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
    assert (attempt / 'script_result.fobs').is_file()
    assert not (attempt / 'result.fobs').exists()
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
    assert (Path(store.attempt_dir(identity)) / "script_result.fobs").is_file()
    assert not (Path(store.attempt_dir(identity)) / "result.fobs").exists()
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_result(identity)


@pytest.mark.parametrize("finalizer_fails", [False, True])
def test_client_api_uses_generic_task_hooks_and_commits_only_after_finalization(tmp_path, finalizer_fails):
    workspace = _workspace(tmp_path)
    custom_dir = workspace / "job-1/app_site-1/custom"
    (custom_dir / "train.py").write_text(
        "import nvflare.client as flare\n"
        "flare.init()\nflare.receive()\nflare.send(flare.FLModel(metrics={'accuracy': 1}))\n"
    )
    (custom_dir / "result_hook.py").write_text(
        "from nvflare.apis.event_type import EventType\n"
        "from nvflare.apis.fl_component import FLComponent\n"
        "from nvflare.apis.fl_constant import FLContextKey\n"
        "from nvflare.app_common.utils.fl_model_utils import FLModelUtils\n"
        "class ResultHook(FLComponent):\n"
        "    def handle_event(self, event_type, fl_ctx):\n"
        "        if event_type == EventType.AFTER_TASK_EXECUTION:\n"
        "            result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)\n"
        "            model = FLModelUtils.from_shareable(result)\n"
        "            model.metrics['accuracy'] = 2\n"
        "            result.update(FLModelUtils.to_shareable(model))\n"
        f"        if event_type == EventType.END_RUN and {finalizer_fails!r}:\n"
        "            raise RuntimeError('finalizer failed')\n"
    )
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("task-hooks")
    config = {
        "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
        "args": {"execution_mode": "in_process", "task_script_path": "train.py"},
    }
    path = _stage(
        store,
        identity,
        workspace,
        config,
        FLModelUtils.to_shareable(FLModel(metrics={"seed": 1})),
        components=[{"id": "hook", "path": "result_hook.ResultHook"}],
    )
    process = _run_process(path)
    if finalizer_fails:
        assert process.returncode != 0
        assert "finalizer failed" in process.stderr
        with pytest.raises(IncompleteTaskArtifactError):
            store.read_result(identity)
    else:
        assert process.returncode == 0, process.stderr
        assert FLModelUtils.from_shareable(store.read_result(identity)[0]).metrics == {"accuracy": 2}


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


@pytest.mark.parametrize(
    "service",
    [
        "get_cell",
        "register_aux_message_handler",
        "send_aux_request",
        "multicast_aux_requests",
        "fire_and_forget_aux_request",
        "dispatch",
        "stream_objects",
        "get_task_assignment",
        "send_task_result",
        "validate_targets",
        "get_widget",
        "build_component",
        "abort_app",
    ],
)
def test_task_runtime_explicitly_rejects_job_based_services(service):
    runtime = TaskRuntime(None, "endpoint", "job")
    with pytest.raises(UnsupportedTaskRuntimeService, match="does not provide"):
        getattr(runtime, service)()


def test_task_runtime_owns_local_context_and_component_views():
    workspace, component = object(), object()
    runtime = TaskRuntime(workspace, "endpoint", "job")
    runtime.set_compute_graph({"helper": component}, FLComponent())
    assert runtime.get_workspace() is workspace
    assert runtime.get_component("helper") is component
    view = runtime.get_all_components()
    view.clear()
    assert runtime.get_component("helper") is component
    fl_ctx = runtime.new_context()
    assert fl_ctx.get_identity_name() == "endpoint"
    assert fl_ctx.get_job_id() == "job"
    assert fl_ctx.get_run_abort_signal() is runtime.abort_signal
    assert fl_ctx.get_prop(FLContextKey.CLIENT_NAME) is None
    assert fl_ctx.get_process_type() is None
    runtime.fire_event("local", fl_ctx)
    fl_ctx.set_prop(FLContextKey.EVENT_SCOPE, EventScope.FEDERATION, private=True)
    with pytest.raises(UnsupportedTaskRuntimeService, match="federated events"):
        runtime.fire_event("federated", fl_ctx)


def test_task_runtime_panic_latches_first_failure_and_triggers_abort():
    runtime = TaskRuntime(None, "endpoint", "job")
    component = FLComponent()
    runtime.set_compute_graph({}, component)
    fl_ctx = runtime.new_context()
    component.system_panic("first failure", fl_ctx)
    component.system_panic("later failure", fl_ctx)
    assert runtime.abort_signal.triggered
    assert runtime.new_context().get_run_abort_signal().triggered
    with pytest.raises(RuntimeError, match="FATAL_SYSTEM_ERROR: first failure"):
        runtime.raise_if_failed()


@pytest.mark.parametrize("phase", [EventType.START_RUN, "execute", EventType.AFTER_TASK_EXECUTION, EventType.END_RUN])
def test_worker_panic_never_commits_completion_and_still_finalizes(tmp_path, monkeypatch, phase):
    events = []

    class PanicExecutor(Executor):
        def handle_event(self, event_type, fl_ctx):
            events.append(event_type)
            if event_type == phase:
                self.system_panic("compute panic", fl_ctx)

        def execute(self, task_name, shareable, fl_ctx, abort_signal):
            if phase == "execute":
                self.system_panic("compute panic", fl_ctx)
            return Shareable({"done": True})

    def build_graph(_bootstrap, _workspace, _fl_ctx, runtime):
        executor = PanicExecutor()
        runtime.set_compute_graph({}, executor)
        return executor

    monkeypatch.setattr(worker, "_build_compute_graph", build_graph)
    workspace = _workspace(tmp_path)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("panic")
    path = _stage(store, identity, workspace, {}, Shareable())
    with pytest.raises(RuntimeError, match="FATAL_SYSTEM_ERROR: compute panic"):
        worker.run_worker(path)
    assert events.count(EventType.END_RUN) == 1
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_completion(identity)
    failure = json.loads((Path(store.attempt_dir(identity)) / "failure.json").read_text())
    assert "compute panic" in failure["message"]


def test_job_metadata_must_be_mapping_or_absent(monkeypatch):
    monkeypatch.setattr(worker, "get_job_meta_from_workspace", Mock(side_effect=FileNotFoundError()))
    assert worker._read_job_meta(None, "job") == {}
    monkeypatch.setattr(worker, "get_job_meta_from_workspace", lambda *_args: [])
    with pytest.raises(RuntimeError, match="metadata must be a dict"):
        worker._read_job_meta(None, "job")


def test_checked_event_surfaces_component_failure_and_clears_exception_state():
    fl_ctx = FLContext()
    error = ValueError("initializer failed")
    engine = SimpleNamespace(
        fire_event=lambda *_args: fl_ctx.set_prop(FLContextKey.EXCEPTIONS, {"helper": error}),
        raise_if_failed=lambda: None,
    )
    with pytest.raises(RuntimeError, match="helper.*initializer failed") as raised:
        worker._fire_checked(engine, EventType.START_RUN, fl_ctx)
    assert raised.value.__cause__ is error
    assert fl_ctx.get_prop(FLContextKey.EXCEPTIONS) is None


def test_peer_context_is_restored_from_staged_shareable():
    data, fl_ctx = Shareable(), FLContext()
    data.set_peer_props({"origin": "server"})
    worker._restore_peer_context(data, fl_ctx)
    assert fl_ctx.get_peer_context().get_prop("origin") == "server"


def test_worker_rejects_non_executor_component(tmp_path, monkeypatch):
    workspace = worker.Workspace(str(_workspace(tmp_path)), site_name="site-1")
    bootstrap = WorkerBootstrap(_identity("invalid-executor"), str(tmp_path), workspace.get_root_dir(), {})
    engine, fl_ctx = worker._new_context(bootstrap, workspace)
    builder = Mock(build_component=lambda *_args: object())
    monkeypatch.setattr(worker, "WorkerComponentBuilder", lambda **_kwargs: builder)
    with pytest.raises(TypeError, match="instead of Executor"):
        worker._build_compute_graph(bootstrap, workspace, fl_ctx, engine)


def test_invalid_executor_result_still_finalizes_compute_graph(tmp_path, monkeypatch):
    workspace = worker.Workspace(str(_workspace(tmp_path)), site_name="site-1")
    bootstrap = WorkerBootstrap(_identity("invalid-result"), str(tmp_path), workspace.get_root_dir(), {})
    executor = SimpleNamespace(execute=lambda *_args: {})
    monkeypatch.setattr(worker, "_build_compute_graph", lambda *_args: executor)
    events = []
    monkeypatch.setattr(worker, "_fire_checked", lambda _engine, event, _ctx: events.append(event))
    with pytest.raises(TypeError, match="instead of Shareable"):
        worker._execute(bootstrap, Shareable(), workspace, None)
    assert events == [EventType.START_RUN, EventType.BEFORE_TASK_EXECUTION, EventType.END_RUN]


def test_worker_preserves_original_failure_if_failure_record_cannot_be_written(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    path = _stage(store, _identity("failure-record"), workspace, {}, Shareable())
    monkeypatch.setattr(worker, "pop_credential_env", Mock())
    monkeypatch.setattr(worker, "_execute", Mock(side_effect=RuntimeError("original failure")))
    monkeypatch.setattr(FileTaskArtifactStore, "record_failure", Mock(side_effect=OSError("disk full")))
    previous_path = sys.path.copy()
    with pytest.raises(RuntimeError, match="original failure"):
        worker.run_worker(path)
    assert sys.path == previous_path


@pytest.mark.parametrize("exit_code", [None, 0, 2])
def test_client_api_script_system_exit_preserves_completion_semantics(tmp_path, exit_code):
    workspace = _workspace(tmp_path)
    script = "import nvflare.client as flare\nflare.init()\nflare.receive()\nflare.send(flare.FLModel(metrics={'done': 1}))\n"
    (workspace / "job-1/app_site-1/custom/train.py").write_text(script + f"raise SystemExit({exit_code!r})\n")
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("system-exit")
    config = {
        "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
        "args": {
            "execution_mode": "in_process",
            "task_script_path": "train.py",
        },
    }
    path = _stage(store, identity, workspace, config, FLModelUtils.to_shareable(FLModel(metrics={"seed": 1})))
    process = _run_process(path)
    if exit_code in (None, 0):
        assert process.returncode == 0, process.stderr
        assert FLModelUtils.from_shareable(store.read_result(identity)[0]).metrics == {"done": 1}
    else:
        assert process.returncode != 0
        with pytest.raises(IncompleteTaskArtifactError):
            store.read_completion(identity)
