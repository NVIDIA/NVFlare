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
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import psutil
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.shareable import Shareable
from nvflare.apis.task_launcher_spec import TaskLaunchError, TaskLaunchRequest
from nvflare.app_common.abstract.statistics_spec import StatisticConfig
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.statistics.statisitcs_objects_decomposer import fobs_registration
from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher
from nvflare.fuel.utils import fobs
from nvflare.private.fed.task_worker import (
    ContextProperty,
    FileTaskArtifactStore,
    IncompleteTaskArtifactError,
    TaskAttemptIdentity,
    WorkerBootstrap,
    artifacts,
    protocol,
    read_bootstrap,
    worker,
    write_bootstrap,
)
from nvflare.private.fed.task_worker.runtime import TaskRuntime, UnsupportedTaskRuntimeService

_PROBE_MODULE = """
import json
import os

from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.statistics_spec import Statistics


class ProbeExecutor(Executor):
    def __init__(self, marker_path):
        super().__init__()
        self.marker_path = marker_path

    def initialize(self, fl_ctx):
        with open(self.marker_path, "a") as stream:
            stream.write("initialize\\n")

    def finalize(self, fl_ctx):
        with open(self.marker_path, "a") as stream:
            stream.write("finalize\\n")

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        try:
            fl_ctx.get_engine().get_cell()
        except RuntimeError as e:
            unsupported = str(e)
        else:
            unsupported = "missing explicit failure"
        return Shareable({
            "value": shareable["value"] + 4,
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


class FailLargeValueExecutor(ProbeExecutor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if shareable["value"] > 5:
            raise ValueError("deliberate worker failure")
        return super().execute(task_name, shareable, fl_ctx, abort_signal)


class FailingFinalizeStatistics(Statistics):
    def features(self):
        return {}

    def count(self, dataset_name, feature_name):
        return 8

    def finalize(self, fl_ctx):
        raise ValueError("statistics cleanup failed")


class ReferenceExecutor(Executor):
    def __init__(self, value, options):
        super().__init__()
        self.value = value
        self.options = options

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        return Shareable({"value": self.value + shareable["value"], "options": self.options})
""" % (
    JobProcessEnv.ALL,
)


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


def _stage(store, identity, workspace_root, executor, data, components=(), context_properties=None, session_dir=None):
    if not executor:
        executor = {"path": "nvflare.app_common.np.np_trainer.NPTrainer"}
    store.create_attempt(identity)
    store.write_input(identity, data)
    bootstrap = WorkerBootstrap(
        identity=identity,
        artifact_root=store.root_dir,
        workspace_root=str(workspace_root),
        executor=executor,
        components=components,
        context_properties=context_properties or {},
        max_payload_bytes=store.max_payload_bytes,
        session_dir=session_dir,
    )
    path = store.bootstrap_path(identity)
    write_bootstrap(path, bootstrap)
    return path


def _run_process(bootstrap_path, extra_env=None):
    if not hasattr(os, "waitid"):
        pytest.skip("ProcessTaskLauncher requires waitid/WNOWAIT (macOS Python >= 3.13)")
    repo_root = str(Path(__file__).resolve().parents[5])
    env = {key: value for key, value in os.environ.items() if key not in JobProcessEnv.ALL}
    env["PYTHONPATH"] = repo_root
    env.update(extra_env or {})
    bootstrap = read_bootstrap(bootstrap_path)
    identity = bootstrap.identity
    log_path = Path(bootstrap_path).with_name("worker.log")
    # Capture subprocess diagnostics without taking leader reaping from the launcher.
    code = (
        "import os,sys;"
        f"log=open({str(log_path)!r},'w');"
        "os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);"
        f"sys.argv=['worker','--bootstrap',{bootstrap_path!r}];"
        "from nvflare.private.fed.app.client.task_worker_process import main;main()"
    )
    request = TaskLaunchRequest(
        identity.job_id,
        identity.site_name,
        identity.task_id,
        identity.attempt_id,
        (sys.executable, "-c", code),
        environment=env,
        cwd=repo_root,
    )
    launcher = ProcessTaskLauncher(stop_grace_period=1.0, descendant_settle_timeout=0.5)
    try:
        handle = launcher.launch_task(request)
    except TaskLaunchError as error:
        error.handle.cancel()
        raise
    try:
        status = handle.wait_for_settlement(timeout=30)
        assert status.settled
        assert status.failure_reason is None, status
        return SimpleNamespace(
            returncode=status.exit_code if status.exit_code is not None else -status.termination_signal,
            stderr=log_path.read_text() if log_path.exists() else "",
            status=status,
        )
    finally:
        if not handle.poll().settled:
            handle.cancel()


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
    assert marker.read_text().splitlines() == ["initialize", "finalize"]
    assert completion.completed_at >= completion.started_at
    assert completion.diagnostics["user_cpu_seconds"] >= 0
    assert store.read_input(identity)["value"] == 3


def _stage_session(tmp_path, executor_path, marker):
    workspace_root = _workspace(tmp_path)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    first, second = _identity("attempt-1"), _identity("attempt-2", task_id="task-2", task_name="validate")
    bootstrap_path = _stage(
        store,
        first,
        workspace_root,
        executor={"path": executor_path, "args": {"marker_path": str(marker)}},
        data=Shareable({"value": 3}),
        session_dir=str(session_dir),
    )
    store.create_attempt(second)
    store.write_input(second, Shareable({"value": 10}))
    artifacts._write_json_exclusive(str(session_dir), "000002.json", second.to_dict())
    artifacts._write_json_exclusive(str(session_dir), protocol.SESSION_END_MARKER, {})
    return store, first, second, bootstrap_path


def test_session_worker_serves_staged_attempts_on_one_executor_until_the_end_marker(tmp_path):
    marker = tmp_path / "events.txt"
    store, first, second, bootstrap_path = _stage_session(tmp_path, "worker_components.ProbeExecutor", marker)

    process = _run_process(bootstrap_path)

    assert process.returncode == 0, process.stderr
    result_1, completion_1 = store.read_result(first)
    result_2, completion_2 = store.read_result(second)
    assert (result_1["value"], result_2["value"]) == (7, 14)
    assert result_1["pid"] == result_2["pid"] == completion_1.worker_pid == completion_2.worker_pid
    assert (result_1["task_id"], result_2["task_id"]) == ("task-1", "task-2")
    assert completion_2.started_at >= completion_1.completed_at
    assert marker.read_text().splitlines() == ["initialize", "finalize"]


def test_session_worker_stops_waiting_when_its_client_job_is_gone(tmp_path, monkeypatch):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    bootstrap = WorkerBootstrap(_identity("attempt-1"), str(tmp_path), str(tmp_path), {}, session_dir=str(session_dir))
    parents = iter([4242, 4242, 1])
    monkeypatch.setattr(worker.os, "getppid", lambda: next(parents))
    monkeypatch.setattr(worker.time, "sleep", lambda _interval: None)

    assert list(worker._attempt_identities(bootstrap)) == [bootstrap.identity]


def test_session_worker_failure_keeps_the_earlier_completion_and_ends_the_session(tmp_path):
    marker = tmp_path / "events.txt"
    store, first, second, bootstrap_path = _stage_session(tmp_path, "worker_components.FailLargeValueExecutor", marker)

    process = _run_process(bootstrap_path)

    assert process.returncode != 0
    assert store.read_result(first)[0]["value"] == 7
    assert not Path(store.attempt_dir(first), "failure.json").exists()
    assert store.read_failure(second)["error_type"] == "ValueError"
    assert not Path(store.attempt_dir(second), "completion.json").exists()
    assert marker.read_text().splitlines() == ["initialize", "finalize"]


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
    assert marker.read_text().splitlines() == ["initialize", "finalize"]
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


def test_nested_component_is_rejected_before_any_component_import(tmp_path):
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
    assert "nested components are not supported" in process.stderr
    assert not import_marker.exists()


@pytest.mark.parametrize(
    "property_name",
    [
        ReservedKey.ENGINE,
        ReservedKey.MANAGER,
        ReservedKey.IDENTITY_NAME,
        FLContextKey.JOB_META,
        FLContextKey.CLIENT_NAME,
        FLContextKey.PROCESS_TYPE,
        FLContextKey.RUN_ABORT_SIGNAL,
        FLContextKey.TASK_ATTEMPT_ID,
        FLContextKey.TASK_ATTEMPT_REQUIRED,
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


@pytest.mark.parametrize("wrapped", [False, True])
def test_unmodified_np_trainer_keeps_workspace_model_state_across_fresh_workers(tmp_path, wrapped):
    workspace_root = _workspace(tmp_path)
    (workspace_root / "job-1" / "meta.json").write_text(json.dumps({"byoc": False}))
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    executor = {"path": "nvflare.app_common.np.np_trainer.NPTrainer", "args": {"delta": 2}}
    if wrapped:
        executor["args"] = {"class_attributes": executor["args"]}
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
        "get_component",
        "get_all_components",
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


def test_task_runtime_owns_local_context():
    workspace = object()
    runtime = TaskRuntime(workspace, "endpoint", "job")
    assert runtime.get_workspace() is workspace
    fl_ctx = runtime.new_context()
    assert fl_ctx.get_identity_name() == "endpoint"
    assert fl_ctx.get_job_id() == "job"
    assert fl_ctx.get_run_abort_signal() is runtime.abort_signal
    assert fl_ctx.get_prop(FLContextKey.CLIENT_NAME) is None
    assert fl_ctx.get_process_type() is None


def test_task_runtime_panic_latches_first_failure_and_triggers_abort():
    runtime = TaskRuntime(None, "endpoint", "job")
    component = FLComponent()
    fl_ctx = runtime.new_context()
    component.system_panic("first failure", fl_ctx)
    component.system_panic("later failure", fl_ctx)
    assert runtime.abort_signal.triggered
    assert runtime.new_context().get_run_abort_signal().triggered
    with pytest.raises(RuntimeError, match="FATAL_SYSTEM_ERROR: first failure"):
        runtime.raise_if_failed()


def test_resetting_abort_signal_cannot_restore_success():
    runtime = TaskRuntime(None, "endpoint", "job")
    runtime.abort_signal.trigger(True)
    runtime.abort_signal.reset()
    assert not runtime.abort_signal.triggered
    with pytest.raises(RuntimeError, match="aborted"), runtime.completion_guard():
        pytest.fail("an aborted attempt must not publish completion")


@pytest.mark.parametrize("federated", [False, True])
def test_task_runtime_rejects_general_events_and_cannot_ignore_the_failure(federated):
    runtime = TaskRuntime(None, "endpoint", "job")
    component = FLComponent()
    component.handle_event = Mock()
    ctx = runtime.new_context()
    with pytest.raises(UnsupportedTaskRuntimeService, match="event dispatch"):
        if federated:
            component.fire_fed_event("custom", Shareable(), ctx)
        else:
            component.fire_event("custom", ctx)
    component.handle_event.assert_not_called()
    with pytest.raises(UnsupportedTaskRuntimeService), runtime.completion_guard():
        pytest.fail("catching an unsupported event request must not permit completion")


@pytest.mark.parametrize("first", ["operation", "panic", "abort"])
def test_task_runtime_preserves_first_failure_across_categories(first):
    runtime = TaskRuntime(None, "endpoint", "job")
    ctx = runtime.new_context()
    component = FLComponent()
    if first == "panic":
        component.system_panic("FIRST_PANIC_FAILURE", ctx)
    elif first == "abort":
        runtime.abort_signal.trigger(True)
    with pytest.raises(ValueError, match="FIRST_OPERATION_FAILURE"):
        with runtime.operation():
            raise ValueError("FIRST_OPERATION_FAILURE")
    if first == "operation":
        component.system_panic("LATER_PANIC_FAILURE", ctx)
        runtime.abort_signal.trigger(True)
    expected = {"operation": "FIRST_OPERATION_FAILURE", "panic": "FIRST_PANIC_FAILURE", "abort": "was aborted"}
    with pytest.raises((ValueError, RuntimeError), match=expected[first]), runtime.completion_guard():
        pytest.fail("a later failure must not replace the first failure")


def test_direct_failure_does_not_depend_on_shared_context_diagnostics(tmp_path, monkeypatch):
    class FailingExecutor(Executor):
        def execute(self, task_name, data, fl_ctx, abort_signal):
            raise ValueError("compute failure")

        def finalize(self, fl_ctx):
            fl_ctx.remove_prop(FLContextKey.EXCEPTIONS, force_removal=True)
            fl_ctx.set_prop(FLContextKey.TASK_RESULT, Shareable({"computed": True}), private=True, sticky=False)

    def build_executor(_config):
        executor = FailingExecutor()
        return executor

    monkeypatch.setattr(worker, "_build_executor", build_executor)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("context-diagnostics")
    path = _stage(store, identity, _workspace(tmp_path), {}, Shareable())
    with pytest.raises(ValueError, match="compute failure"):
        worker.run_worker(path)
    failure = json.loads(Path(store.attempt_dir(identity), "failure.json").read_text())
    assert failure["error_type"] == "ValueError"
    assert "compute failure" in failure["message"]
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_result(identity)


def test_completion_rejects_a_direct_operation_still_running(tmp_path):
    runtime = TaskRuntime(None, "endpoint", "job")
    started, release = threading.Event(), threading.Event()

    def compute():
        with pytest.raises(ValueError, match="operation failed"):
            with runtime.operation():
                started.set()
                assert release.wait(timeout=5)
                raise ValueError("operation failed")

    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("active-operation")
    store.create_attempt(identity)
    reference = store.stage_result(identity, Shareable())
    callback = threading.Thread(target=compute, daemon=True)
    callback.start()
    try:
        assert started.wait(timeout=5)
        with pytest.raises(RuntimeError, match="still active"):
            store.commit_staged_result(identity, reference, publication_guard=runtime.completion_guard)
        with pytest.raises(IncompleteTaskArtifactError):
            store.read_completion(identity)
    finally:
        release.set()
        callback.join(timeout=5)
    assert not callback.is_alive()
    with pytest.raises(ValueError, match="operation failed"):
        runtime.raise_if_failed()


@pytest.mark.parametrize("fault", ["panic", "abort"])
def test_completion_winning_the_race_closes_runtime(tmp_path, monkeypatch, fault):
    runtime = TaskRuntime(None, "endpoint", "job")
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("completion-wins")
    store.create_attempt(identity)
    reference = store.stage_result(identity, Shareable({"produced": True}))
    started = threading.Event()
    rejected = threading.Event()

    def fail_after_commit_begins():
        started.set()
        with pytest.raises(RuntimeError, match="already completed"):
            if fault == "panic":
                FLComponent().system_panic("too late", runtime.new_context())
            else:
                runtime.abort_signal.trigger(True)
        assert Path(store.attempt_dir(identity), "completion.json").is_file()
        rejected.set()

    callback = threading.Thread(target=fail_after_commit_begins, daemon=True)
    link = protocol.os.link

    def publish_with_callback(source, destination, **kwargs):
        if destination == "completion.json":
            callback.start()
            assert started.wait(timeout=5)
        return link(source, destination, **kwargs)

    monkeypatch.setattr(protocol.os, "link", publish_with_callback)
    completion = store.commit_staged_result(identity, reference, publication_guard=runtime.completion_guard)
    callback.join(timeout=5)
    assert not callback.is_alive()
    assert rejected.is_set()
    assert store.read_completion(identity) == completion
    assert not runtime.abort_signal.triggered
    runtime.raise_if_failed()


def test_failed_completion_publication_releases_runtime_guard(tmp_path, monkeypatch):
    runtime = TaskRuntime(None, "endpoint", "job")
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("publication-fails")
    store.create_attempt(identity)
    reference = store.stage_result(identity, Shareable())

    def fail_link(*args, **kwargs):
        raise OSError("publication failed")

    monkeypatch.setattr(protocol.os, "link", fail_link)
    with pytest.raises(OSError, match="publication failed"):
        store.commit_staged_result(identity, reference, publication_guard=runtime.completion_guard)
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_completion(identity)
    assert not list(Path(store.attempt_dir(identity)).glob(".completion.json.*"))
    callback = threading.Thread(target=runtime.abort_signal.trigger, args=(True,), daemon=True)
    callback.start()
    callback.join(timeout=5)
    assert not callback.is_alive()
    with pytest.raises(RuntimeError, match="aborted"):
        runtime.raise_if_failed()


def test_direct_operation_can_abort_without_holding_publication_lock():
    runtime = TaskRuntime(None, "endpoint", "job")

    def compute():
        with runtime.operation():
            runtime.abort_signal.trigger(True)

    callback = threading.Thread(target=compute, daemon=True)
    callback.start()
    callback.join(timeout=5)
    assert not callback.is_alive()
    with pytest.raises(RuntimeError, match="aborted"):
        runtime.raise_if_failed()


@pytest.mark.parametrize("phase", ["initialize", "execute", "finalize"])
def test_worker_panic_never_commits_completion_and_still_finalizes(tmp_path, monkeypatch, phase):
    calls = []

    class PanicExecutor(Executor):
        def initialize(self, fl_ctx):
            calls.append("initialize")
            if phase == "initialize":
                self.system_panic("compute panic", fl_ctx)

        def execute(self, task_name, data, fl_ctx, abort_signal):
            calls.append("execute")
            if phase == "execute":
                self.system_panic("compute panic", fl_ctx)
            return Shareable({"done": True})

        def finalize(self, fl_ctx):
            calls.append("finalize")
            if phase == "finalize":
                self.system_panic("compute panic", fl_ctx)

    def build_executor(_config):
        executor = PanicExecutor()
        return executor

    monkeypatch.setattr(worker, "_build_executor", build_executor)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("panic")
    path = _stage(store, identity, _workspace(tmp_path), {}, Shareable())
    with pytest.raises(RuntimeError, match="FATAL_SYSTEM_ERROR: compute panic"):
        worker.run_worker(path)
    assert calls.count("finalize") == 1
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_completion(identity)
    failure = json.loads(Path(store.attempt_dir(identity), "failure.json").read_text())
    assert "compute panic" in failure["message"]


def test_job_metadata_must_be_mapping_or_absent(monkeypatch):
    monkeypatch.setattr(worker, "get_job_meta_from_workspace", Mock(side_effect=FileNotFoundError()))
    assert worker._read_job_meta(None, "job") == {}
    monkeypatch.setattr(worker, "get_job_meta_from_workspace", lambda *_args: [])
    with pytest.raises(RuntimeError, match="metadata must be a dict"):
        worker._read_job_meta(None, "job")


def test_peer_context_is_restored_from_staged_shareable():
    data, fl_ctx = Shareable(), FLContext()
    data.set_peer_props({"origin": "server"})
    worker._restore_peer_context(data, fl_ctx)
    assert fl_ctx.get_peer_context().get_prop("origin") == "server"


def test_worker_rejects_non_executor_component(monkeypatch):
    monkeypatch.setattr(worker, "instantiate_class", lambda *_args: object())
    with pytest.raises(TypeError, match="instead of Executor"):
        worker._build_executor({"path": "nvflare.app_common.np.np_trainer.NPTrainer"})


def test_invalid_executor_result_still_finalizes_executor(tmp_path, monkeypatch):
    workspace = worker.Workspace(str(_workspace(tmp_path)), site_name="site-1")
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("invalid-result")
    store.create_attempt(identity)
    store.write_input(identity, Shareable())
    bootstrap = WorkerBootstrap(identity, store.root_dir, workspace.get_root_dir(), {})
    finalized = Mock()
    executor = SimpleNamespace(execute=lambda *_args: {}, finalize=finalized)

    def build_executor(_config):
        return executor

    monkeypatch.setattr(worker, "_build_executor", build_executor)
    with pytest.raises(TypeError, match="instead of Shareable"):
        worker._execute(bootstrap, workspace, store, worker.AttemptBinding(identity))
    finalized.assert_called_once()


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


@pytest.mark.parametrize("phase", ["initialize", "execute", "finalize"])
@pytest.mark.parametrize("fault", ["exception", "abort"])
def test_lifecycle_failure_or_abort_never_publishes_completion(tmp_path, monkeypatch, phase, fault):
    calls = []

    class FailingExecutor(Executor):
        def check(self, action, fl_ctx):
            calls.append(action)
            if action == phase:
                if fault == "exception":
                    raise RuntimeError("lifecycle failure")
                fl_ctx.get_run_abort_signal().trigger(True)

        def initialize(self, fl_ctx):
            self.check("initialize", fl_ctx)

        def execute(self, task_name, data, fl_ctx, abort_signal):
            self.check("execute", fl_ctx)
            return Shareable({"produced": True})

        def finalize(self, fl_ctx):
            self.check("finalize", fl_ctx)

    def build_executor(_config):
        executor = FailingExecutor()
        return executor

    monkeypatch.setattr(worker, "_build_executor", build_executor)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("lifecycle")
    path = _stage(store, identity, _workspace(tmp_path), {}, Shareable())
    with pytest.raises(RuntimeError):
        worker.run_worker(path)
    assert calls.count("finalize") == 1
    assert not Path(store.attempt_dir(identity), "completion.json").exists()
    assert not Path(store.attempt_dir(identity), "result.fobs").exists()


def test_direct_finalization_precedes_completion_and_attempt_cannot_rerun(tmp_path, monkeypatch):
    calls = []

    class FinalizingExecutor(Executor):
        def execute(self, *args):
            calls.append("execute")
            return Shareable({"original": True})

        def finalize(self, fl_ctx):
            calls.append("finalize")
            assert not Path(store.attempt_dir(identity), "completion.json").exists()
            fl_ctx.get_prop(FLContextKey.TASK_RESULT)["finalized"] = True

    def build_executor(_config):
        executor = FinalizingExecutor()
        return executor

    monkeypatch.setattr(worker, "_build_executor", build_executor)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("finalize")
    path = _stage(store, identity, _workspace(tmp_path), {}, Shareable())
    worker.run_worker(path)
    result, completion = store.read_result(identity)
    assert result["original"] and result["finalized"]
    with pytest.raises(FileExistsError):
        worker.run_worker(path)
    assert calls == ["execute", "finalize"]
    assert store.read_completion(identity) == completion
    assert not Path(store.attempt_dir(identity), "failure.json").exists()


def test_executor_is_authorized_before_import_logging_or_decoding(tmp_path, monkeypatch):
    workspace_root = _workspace(tmp_path)
    (workspace_root / "job-1" / "meta.json").write_text('{"byoc":false}')
    (workspace_root / "local" / "resources.json.default").write_text('{"class_allow_list":["allowed.First"]}')
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("preflight")
    path = _stage(
        store,
        identity,
        workspace_root,
        {"path": "forbidden.Executor"},
        Shareable(),
    )
    imported = Mock(side_effect=AssertionError("import before authorization"))
    monkeypatch.setattr(worker, "configure_logging", imported)
    monkeypatch.setattr(worker, "fobs_initialize", imported)
    monkeypatch.setattr(FileTaskArtifactStore, "read_input", imported)
    monkeypatch.setattr(worker, "_build_executor", imported)
    with pytest.raises(Exception, match="not in allow_list"):
        worker.run_worker(path)
    imported.assert_not_called()


def test_credentials_and_argv_are_cleared_before_application_imports(tmp_path, monkeypatch):
    workspace_root = _workspace(tmp_path)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    path = _stage(store, _identity("credentials"), workspace_root, {}, Shareable())
    for name in JobProcessEnv.ALL:
        monkeypatch.setenv(name, "secret-canary")
    monkeypatch.setattr(sys, "argv", ["worker", "--auth_token", "secret-canary"])

    def importing(*args, **kwargs):
        assert not set(JobProcessEnv.ALL).intersection(os.environ)
        assert sys.argv == ["nvflare-task-worker"]
        raise RuntimeError("import boundary checked")

    monkeypatch.setattr(worker, "fobs_initialize", importing)
    with pytest.raises(RuntimeError, match="import boundary checked"):
        worker.run_worker(path)


def test_failure_during_result_serialization_leaves_only_a_staged_result(tmp_path, monkeypatch):
    runtime_context = []

    class SuccessfulExecutor(Executor):
        def execute(self, task_name, data, ctx, abort_signal):
            runtime_context.append(ctx)
            return Shareable({"produced": True})

    def build_executor(_config):
        executor = SuccessfulExecutor()
        return executor

    monkeypatch.setattr(worker, "_build_executor", build_executor)
    stage = FileTaskArtifactStore.stage_result

    def stage_then_panic(store, identity, data):
        reference = stage(store, identity, data)
        FLComponent().system_panic("serialization panic", runtime_context[0])
        return reference

    monkeypatch.setattr(FileTaskArtifactStore, "stage_result", stage_then_panic)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("serialization")
    path = _stage(store, identity, _workspace(tmp_path), {}, Shareable())
    with pytest.raises(RuntimeError, match="serialization panic"):
        worker.run_worker(path)
    assert Path(store.attempt_dir(identity), "result.fobs").is_file()
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_result(identity)


@pytest.mark.parametrize("boundary", ["result_verification", "completion_write"])
@pytest.mark.parametrize("fault", ["panic", "abort", None])
def test_background_failure_before_completion_publication(tmp_path, monkeypatch, boundary, fault):
    runtime_context = []

    class SuccessfulExecutor(Executor):
        def execute(self, task_name, data, ctx, abort_signal):
            runtime_context.append(ctx)
            return Shareable({"produced": True})

    def build_executor(_config):
        executor = SuccessfulExecutor()
        return executor

    def background_callback():
        ctx = runtime_context[0]
        if fault == "panic":
            FLComponent().system_panic("background panic", ctx)
        elif fault == "abort":
            ctx.get_run_abort_signal().trigger(True)

    callbacks = []

    def inject_callback():
        callback = threading.Thread(target=background_callback, daemon=True)
        callbacks.append(callback)
        callback.start()
        callback.join(timeout=5)
        assert not callback.is_alive(), "completion preparation blocked a background callback"

    fingerprint = artifacts._fingerprint

    def verify_with_callback(stream, max_payload_bytes=None):
        # Input verification precedes execute; inject at result verification.
        if runtime_context and not callbacks:
            inject_callback()
        return fingerprint(stream, max_payload_bytes)

    publish = artifacts._publish_exclusive

    def write_with_callback(directory, name, writer, **kwargs):
        def staged_writer(stream):
            writer(stream)
            inject_callback()

        return publish(directory, name, staged_writer if name == "completion.json" else writer, **kwargs)

    monkeypatch.setattr(worker, "_build_executor", build_executor)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("background-failure")
    path = _stage(store, identity, _workspace(tmp_path), {}, Shareable())
    if boundary == "result_verification":
        monkeypatch.setattr(artifacts, "_fingerprint", verify_with_callback)
    else:
        monkeypatch.setattr(artifacts, "_publish_exclusive", write_with_callback)

    if fault:
        with pytest.raises(RuntimeError, match="background panic" if fault == "panic" else "aborted"):
            worker.run_worker(path)
        with pytest.raises(IncompleteTaskArtifactError):
            store.read_completion(identity)
        assert Path(store.attempt_dir(identity), "failure.json").is_file()
    else:
        worker.run_worker(path)
        assert store.read_result(identity)[0]["produced"]
        assert not Path(store.attempt_dir(identity), "failure.json").exists()
    assert len(callbacks) == 1
    assert Path(store.attempt_dir(identity), "result.fobs").is_file()
    assert not list(Path(store.attempt_dir(identity)).glob(".completion.json.*"))


def _wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("bounded process probe timed out")
        time.sleep(0.05)


def _live(process):
    try:
        return process.is_running() and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


@pytest.mark.parametrize("shutdown", ["cancel", "normal", "stubborn"])
def test_launcher_preserves_descendant_shutdown_windows(tmp_path, shutdown):
    if not hasattr(os, "waitid"):
        pytest.skip("requires ProcessTaskLauncher waitid/WNOWAIT support")
    workspace_root = _workspace(tmp_path)
    ready = tmp_path / "child-ready"
    cleaned = tmp_path / "child-cleaned"
    child_code = (
        "import os,signal,sys,time,psutil\n"
        "from pathlib import Path\n"
        "owner=psutil.Process(os.getppid())\n"
        "def cleanup(*args):\n"
        " time.sleep(0.5)\n"
        f" Path({str(cleaned)!r}).write_text('cleaned')\n"
        " sys.exit(0)\n"
        f"signal.signal(signal.SIGTERM, {'cleanup' if shutdown == 'cancel' else 'signal.SIG_IGN'})\n"
        f"Path({str(ready.with_suffix('.tmp'))!r}).write_text(str(os.getpid()))\n"
        f"os.replace({str(ready.with_suffix('.tmp'))!r}, {str(ready)!r})\n"
        + (
            "while owner.is_running() and owner.status() not in (psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):\n"
            " time.sleep(0.01)\n"
            "cleanup()\n"
            if shutdown == "normal"
            else "time.sleep(30)\n"
        )
    )
    custom = workspace_root / "job-1" / "app_site-1" / "custom" / "shutdown_child.py"
    custom.write_text(
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        "from nvflare.apis.executor import Executor\n"
        "from nvflare.apis.shareable import Shareable\n"
        "class ChildExecutor(Executor):\n"
        " def execute(self,*args):\n"
        f"  subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
        f"  while not Path({str(ready)!r}).exists(): time.sleep(0.01)\n"
        + ("  time.sleep(30)\n" if shutdown == "cancel" else "  return Shareable()\n")
    )
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("descendant-shutdown")
    path = _stage(store, identity, workspace_root, {"path": "shutdown_child.ChildExecutor"}, Shareable())
    repo_root = str(Path(__file__).resolve().parents[5])
    env = {key: value for key, value in os.environ.items() if key not in JobProcessEnv.ALL}
    env["PYTHONPATH"] = repo_root
    request = TaskLaunchRequest(
        identity.job_id,
        identity.site_name,
        identity.task_id,
        identity.attempt_id,
        (sys.executable, "-m", "nvflare.private.fed.app.client.task_worker_process", "--bootstrap", path),
        environment=env,
        cwd=repo_root,
    )
    launcher = ProcessTaskLauncher(stop_grace_period=2, descendant_settle_timeout=2, poll_interval=0.02)
    try:
        handle = launcher.launch_task(request)
    except TaskLaunchError as error:
        error.handle.cancel()
        raise
    try:
        _wait_until(ready.exists)
        try:
            child = psutil.Process(int(ready.read_text()))
        except psutil.NoSuchProcess:
            # A normally exiting child may finish before this test is scheduled.
            child = None
        status = handle.cancel() if shutdown == "cancel" else handle.wait_for_settlement(timeout=15)
        assert status.settled
        assert child is None or not _live(child)
        if shutdown == "stubborn":
            assert not cleaned.exists()
            assert status.failure_reason == "task leader exited while descendants remained alive"
        else:
            assert cleaned.read_text() == "cleaned"
            assert status.failure_reason is None
            if shutdown == "cancel":
                assert status.termination_signal == signal.SIGTERM
                assert status.cancel_requested
            else:
                assert status.succeeded
    finally:
        if not handle.poll().settled:
            handle.cancel()


@pytest.mark.parametrize("phase", ["initialize", "execute", "finalize"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failure_record_keeps_first_exception_through_cleanup(tmp_path, monkeypatch, phase, cleanup_fails):
    calls = []

    class FailingExecutor(Executor):
        def initialize(self, fl_ctx):
            calls.append("initialize")
            if phase == "initialize":
                raise ValueError("ROOT CAUSE from first failure")

        def execute(self, *args):
            if phase == "execute":
                raise ValueError("ROOT CAUSE from first failure")
            return Shareable()

        def finalize(self, fl_ctx):
            calls.append("finalize")
            if phase == "finalize":
                raise ValueError("ROOT CAUSE from first failure")
            if cleanup_fails:
                raise RuntimeError("secondary cleanup failure")

    def build_executor(_config):
        executor = FailingExecutor()
        return executor

    monkeypatch.setattr(worker, "_build_executor", build_executor)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("first-failure")
    path = _stage(store, identity, _workspace(tmp_path), {}, Shareable())
    with pytest.raises(ValueError, match="ROOT CAUSE"):
        worker.run_worker(path)
    failure = json.loads(Path(store.attempt_dir(identity), "failure.json").read_text())
    assert failure["error_type"] == "ValueError"
    assert "ROOT CAUSE" in failure["message"]
    assert "secondary cleanup failure" not in failure["message"]
    assert calls.count("finalize") == 1
    assert not Path(store.attempt_dir(identity), "completion.json").exists()


@pytest.mark.parametrize(
    "policy, audit_action",
    [
        ({"class_allow_list": ["*"]}, "component_authorization.class_allow_list_disabled"),
        (
            {"class_allow_list": ["nvflare."], "class_list_enforcement_mode": "warn"},
            "component_authorization.unlisted_class_allowed",
        ),
    ],
)
def test_fresh_worker_preserves_component_authorization_audit(tmp_path, policy, audit_action):
    from nvflare.apis.workspace import Workspace

    workspace_root = _workspace(tmp_path)
    (workspace_root / "job-1" / "meta.json").write_text(json.dumps({"byoc": False}))
    (workspace_root / "local" / "resources.json.default").write_text(json.dumps(policy))
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("audited")
    path = _stage(
        store,
        identity,
        workspace_root,
        {"path": "worker_components.ReferenceExecutor", "args": {"value": 4, "options": {}}},
        Shareable({"value": 3}),
    )
    process = _run_process(path)
    assert process.returncode == 0, process.stderr
    audit_path = Path(Workspace(str(workspace_root), site_name=identity.site_name).get_audit_file_path(identity.job_id))
    assert audit_path.exists()
    text = audit_path.read_text()
    assert audit_action in text
    assert identity.job_id in text
    if policy.get("class_list_enforcement_mode") == "warn":
        assert "worker_components.ReferenceExecutor" in text


def test_worker_applies_payload_policy_from_bootstrap(tmp_path):
    workspace_root = _workspace(tmp_path)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"), max_payload_bytes=512)
    identity = _identity("payload-policy")
    path = _stage(
        store,
        identity,
        workspace_root,
        {
            "path": "worker_components.ReferenceExecutor",
            "args": {"value": 4, "options": {"large": "x" * 1024}},
        },
        Shareable({"value": 3}),
    )
    assert read_bootstrap(path).max_payload_bytes == 512
    process = _run_process(path)
    assert process.returncode != 0
    failure = json.loads(Path(store.attempt_dir(identity), "failure.json").read_text())
    assert "payload is too large" in failure["message"]
    assert not Path(store.attempt_dir(identity), "result.fobs").exists()
    assert not Path(store.attempt_dir(identity), "completion.json").exists()


def test_completion_remains_readable_after_directory_fsync_failure(tmp_path, monkeypatch):
    class SuccessfulExecutor(Executor):
        def execute(self, *args):
            return Shareable({"produced": True})

    def build_executor(_config):
        executor = SuccessfulExecutor()
        return executor

    monkeypatch.setattr(worker, "_build_executor", build_executor)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("post-publication-error")
    path = _stage(store, identity, _workspace(tmp_path), {}, Shareable())
    completion_path = Path(store.attempt_dir(identity), "completion.json")
    fsync = protocol.os.fsync
    failed = False

    def fail_after_completion(fd):
        import stat

        nonlocal failed
        if not failed and completion_path.exists() and stat.S_ISDIR(os.fstat(fd).st_mode):
            failed = True
            raise OSError("completion directory durability failure")
        fsync(fd)

    monkeypatch.setattr(protocol.os, "fsync", fail_after_completion)
    with pytest.raises(OSError, match="durability failure"):
        worker.run_worker(path)
    assert failed
    assert store.read_result(identity)[0]["produced"]
    failure = json.loads(Path(store.attempt_dir(identity), "failure.json").read_text())
    assert "durability failure" in failure["message"]


def test_backward_wall_clock_does_not_discard_completed_compute(tmp_path, monkeypatch):
    class SuccessfulExecutor(Executor):
        def execute(self, *args):
            return Shareable({"produced": True})

    def build_executor(_config):
        executor = SuccessfulExecutor()
        return executor

    monkeypatch.setattr(worker, "_build_executor", build_executor)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("backward-clock")
    path = _stage(store, identity, _workspace(tmp_path), {}, Shareable())
    monkeypatch.setattr(
        worker,
        "time",
        SimpleNamespace(time=Mock(side_effect=[200.0, 100.0, 100.0]), monotonic=Mock(side_effect=[10.0, 12.0])),
    )
    completion = worker.run_worker(path)
    assert completion.started_at == 200.0
    assert completion.completed_at == 100.0
    assert completion.diagnostics["elapsed_seconds"] == 2.0
    assert store.read_result(identity)[0]["produced"]
    assert not Path(store.attempt_dir(identity), "failure.json").exists()


def test_worker_matches_cj_custom_module_precedence(tmp_path, monkeypatch):
    installed = tmp_path / "installed"
    installed.mkdir()
    dependency_name = "_p3_import_probe_dependency"
    executor_name = "_p3_import_probe_executor"
    (installed / f"{dependency_name}.py").write_text("VALUE = 'installed'\n")
    monkeypatch.syspath_prepend(str(installed))
    workspace_root = _workspace(tmp_path)
    custom = workspace_root / "job-1" / "app_site-1" / "custom"
    (custom / f"{dependency_name}.py").write_text("VALUE = 'custom'\n")
    (custom / f"{executor_name}.py").write_text(
        "from nvflare.apis.executor import Executor\n"
        "from nvflare.apis.shareable import Shareable\n"
        f"import {dependency_name} as dependency\n"
        "class QuietExecutor(Executor):\n"
        "    def execute(self, *args):\n"
        "        return Shareable({'value': dependency.VALUE})\n"
    )
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("import-precedence")
    path = _stage(store, identity, workspace_root, {"path": f"{executor_name}.QuietExecutor"}, Shareable())
    original_path = sys.path.copy()
    try:
        worker.run_worker(path)
        assert store.read_result(identity)[0]["value"] == "installed"
        assert sys.path == original_path
    finally:
        sys.modules.pop(dependency_name, None)
        sys.modules.pop(executor_name, None)


def test_flcomponent_logging_does_not_dispatch_handlers():
    runtime = TaskRuntime(None, "endpoint", "job")
    component = FLComponent()
    component.handle_event = Mock()
    ctx = runtime.new_context()
    component.log_info(ctx, "message already logged")
    component.log_warning(ctx, "message already logged")
    component.log_error(ctx, "message already logged")
    component.handle_event.assert_not_called()
    runtime.raise_if_failed()


@pytest.mark.parametrize("phase", ["initialize", "execute", "finalize"])
@pytest.mark.parametrize("first", ["panic", "abort"])
def test_direct_exception_cannot_replace_an_earlier_panic_or_abort(tmp_path, monkeypatch, phase, first):
    calls = []

    class FailingExecutor(Executor):
        def check(self, action, fl_ctx):
            calls.append(action)
            if action == phase:
                if first == "panic":
                    self.system_panic("FIRST_PANIC", fl_ctx)
                else:
                    fl_ctx.get_run_abort_signal().trigger(True)
                raise ValueError("SECONDARY_DIRECT_ERROR")

        def initialize(self, fl_ctx):
            self.check("initialize", fl_ctx)

        def execute(self, task_name, data, fl_ctx, abort_signal):
            self.check("execute", fl_ctx)
            return Shareable()

        def finalize(self, fl_ctx):
            self.check("finalize", fl_ctx)

    def build_executor(_config):
        executor = FailingExecutor()
        return executor

    monkeypatch.setattr(worker, "_build_executor", build_executor)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("first-direct-failure")
    path = _stage(store, identity, _workspace(tmp_path), {}, Shareable())
    with pytest.raises(RuntimeError, match="FIRST_PANIC" if first == "panic" else "aborted"):
        worker.run_worker(path)
    failure = json.loads(Path(store.attempt_dir(identity), "failure.json").read_text())
    assert "SECONDARY_DIRECT_ERROR" not in failure["message"]
    assert calls.count("finalize") == 1
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_completion(identity)


def test_statistics_finalize_failure_prevents_worker_completion(tmp_path):
    root = _workspace(tmp_path)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("stats-finalize-failure", task_name=StC.FED_STATS_TASK)
    fobs_registration()
    bootstrap_path = _stage(
        store,
        identity,
        root,
        executor={
            "path": "nvflare.app_common.executors.statistics.statistics_executor.StatisticsExecutor",
            "args": {"generator_id": "statistics"},
        },
        components=({"id": "statistics", "path": "worker_components.FailingFinalizeStatistics"},),
        data=Shareable(
            {
                StC.STATISTICS_TASK_KEY: StC.STATS_1st_STATISTICS,
                StC.STATS_TARGET_STATISTICS: fobs.dumps([StatisticConfig("count", {})]),
            }
        ),
    )
    process = _run_process(bootstrap_path)
    assert process.returncode != 0, process.stderr
    failure = store.read_failure(identity)
    assert failure["error_type"] == "ValueError"
    assert "statistics cleanup failed" in failure["message"]
    assert not Path(store.attempt_dir(identity), "completion.json").exists()
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_result(identity)
