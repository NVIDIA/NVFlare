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
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.task_execution import TaskArtifactCleanup
from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskExecutionStatus
from nvflare.apis.utils.decomposers.flare_decomposers import DXODecomposer
from nvflare.apis.workspace import Workspace
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.decomposers.common_decomposers import FLModelDecomposer
from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer
from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposer import DictDecomposer
from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore, IncompleteTaskArtifactError
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize

_PROBE_MODULE = """
import os
import time

from nvflare.apis.executor import Executor
from nvflare.apis.shareable import Shareable


class ProbeExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        return Shareable({"value": shareable["value"] + 1, "worker_pid": os.getpid()})


class AbruptExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        os._exit(0)


class RaiseExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        raise ValueError("deliberate supervisor test failure")


class SlowExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        time.sleep(30)
        return Shareable()
"""


@pytest.fixture(autouse=True)
def _initialize_fobs():
    nvflare_fobs_initialize()
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
    (app_root / "config").mkdir(parents=True)
    custom_dir = app_root / "custom"
    custom_dir.mkdir()
    (custom_dir / "supervisor_probe.py").write_text(_PROBE_MODULE)
    (root / "job-1" / "meta.json").write_text(json.dumps({"byoc": True}))
    return Workspace(str(root), site_name="site-1")


def _context(workspace, task_id="task-1"):
    manager = FLContextManager(identity_name="site-1", job_id="job-1")
    fl_ctx = manager.new_context()
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=True)
    fl_ctx.set_prop(FLContextKey.TASK_ID, task_id, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.TASK_NAME, "train", private=True, sticky=False)
    return fl_ctx


def _executor(class_name="ProbeExecutor", artifact_cleanup=TaskArtifactCleanup.JOB, **kwargs):
    executor = TaskWorkerExecutor(
        executor={"path": f"supervisor_probe.{class_name}", "args": {}},
        components=[],
        poll_interval=0.01,
        **kwargs,
    )
    executor.set_task_launcher(ProcessTaskLauncher(poll_interval=0.01), artifact_cleanup=artifact_cleanup)
    return executor


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1, True, False, "0.5"])
def test_supervisor_rejects_invalid_worker_timeout(value):
    with pytest.raises(ValueError, match="worker_timeout"):
        TaskWorkerExecutor(executor={}, components=[], worker_timeout=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0, -1, True, False, None, "0.5"])
def test_supervisor_rejects_invalid_poll_interval(value):
    with pytest.raises(ValueError, match="poll_interval"):
        TaskWorkerExecutor(executor={}, components=[], poll_interval=value)


@pytest.mark.parametrize("timeout", [None, 0, 0.1, 1])
def test_supervisor_preserves_valid_timeout_boundaries(timeout):
    executor = TaskWorkerExecutor(executor={}, components=[], worker_timeout=timeout)
    assert executor.worker_timeout == timeout


def test_zero_worker_timeout_expires_before_staging_or_launch(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    executor = _executor(worker_timeout=0)
    launch = Mock()
    monkeypatch.setattr(executor._get_task_launcher(), "launch_task", launch)
    with pytest.raises(TimeoutError, match="expired before launch"):
        executor.execute("train", Shareable(), _context(workspace), Signal())
    launch.assert_not_called()
    runtime_root = os.path.join(workspace.get_run_dir("job-1"), ".nvflare", "task-execution")
    assert not os.path.exists(runtime_root)


def test_client_api_analytics_are_replayed_after_settlement_and_released_after_publication(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    script = """
import nvflare.client as flare
flare.init()
while flare.is_running():
    model = flare.receive()
    flare.log('loss', 0.5, flare.AnalyticsDataType.SCALAR)
    flare.send(flare.FLModel(metrics={'accuracy': 1.0}))
"""
    with open(os.path.join(workspace.get_app_custom_dir("job-1"), "train.py"), "w") as stream:
        stream.write(script)
    executor = TaskWorkerExecutor(
        executor={
            "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
            "args": {"execution_mode": "in_process", "task_script_path": "train.py"},
        },
        components=[],
    )
    executor.set_task_launcher(ProcessTaskLauncher(), artifact_cleanup=TaskArtifactCleanup.ACCEPTED)
    emitted = []

    def record(comp, dxo, ctx, event_type, fire_fed_event):
        assert comp is executor
        assert executor._active_handle.poll().settled
        assert event_type == "analytix_log_stats"
        assert fire_fed_event is False
        emitted.append(dxo.data)

    monkeypatch.setattr("nvflare.private.fed.client.task_worker_executor.send_analytic_dxo", record)
    fl_ctx = _context(workspace)
    result = executor.execute("train", FLModelUtils.to_shareable(FLModel(metrics={"seed": 1})), fl_ctx, Signal())
    assert FLModelUtils.from_shareable(result).metrics == {"accuracy": 1.0}
    assert len(emitted) == 1
    assert emitted[0]["track_key"] == "loss"
    analytics_files = list((tmp_path / "workspace").rglob("analytics.fobs"))
    assert len(analytics_files) == 1
    fl_ctx.set_prop(FLContextKey.TASK_RESULT_SEND_SUCCESS, True, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.TASK_RESULT_ACCEPTED, True, private=True, sticky=False)
    executor.handle_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)
    assert not analytics_files[0].exists()


def _diagnostics(workspace):
    path = os.path.join(workspace.get_run_dir("job-1"), ".nvflare", "task-execution", "diagnostics.jsonl")
    with open(path) as stream:
        return [json.loads(line) for line in stream]


def test_supervisor_runs_fresh_worker_settles_and_releases_payloads_after_publication(tmp_path):
    workspace = _workspace(tmp_path)
    fl_ctx = _context(workspace)
    executor = _executor(artifact_cleanup=TaskArtifactCleanup.ACCEPTED)
    executor.handle_event(EventType.START_RUN, fl_ctx)

    result = executor.execute("train", Shareable({"value": 2}), fl_ctx, Signal())

    assert result["value"] == 3
    assert result["worker_pid"] != os.getpid()
    records = _diagnostics(workspace)
    assert [record["event"] for record in records] == ["launched", "settled"]
    assert records[-1]["settled_timestamp"] >= records[0]["launch_timestamp"]
    attempt_id = records[-1]["attempt_id"]
    attempt_dir = os.path.join(workspace.get_run_dir("job-1"), ".nvflare", "task-execution", "attempts", attempt_id)
    assert os.path.isfile(os.path.join(attempt_dir, "result.fobs"))

    fl_ctx.set_prop(FLContextKey.TASK_RESULT_SEND_SUCCESS, True, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.TASK_RESULT_ACCEPTED, True, private=True, sticky=False)
    executor.handle_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)

    publication = _diagnostics(workspace)[-1]
    assert publication["event"] == "publication"
    assert publication["publication_outcome"] == "accepted"
    assert publication["publication_timestamp"] >= publication["settled_timestamp"]
    assert publication["worker_pid"] == result["worker_pid"]
    assert publication["supervisor_pid"] == os.getpid()
    assert os.path.isfile(os.path.join(attempt_dir, "completion.json"))
    assert not os.path.exists(os.path.join(attempt_dir, "result.fobs"))
    assert not os.path.exists(os.path.join(attempt_dir, "input.fobs"))


@pytest.mark.parametrize("sent, accepted", [(False, None), (True, False), (True, None)])
def test_supervisor_retains_payloads_when_publication_is_not_accepted(tmp_path, sent, accepted):
    workspace = _workspace(tmp_path)
    fl_ctx = _context(workspace)
    executor = _executor(artifact_cleanup=TaskArtifactCleanup.ACCEPTED)
    result = executor.execute("train", Shareable({"value": 2}), fl_ctx, Signal())
    assert result["value"] == 3
    attempt_id = _diagnostics(workspace)[-1]["attempt_id"]
    attempt_dir = os.path.join(workspace.get_run_dir("job-1"), ".nvflare", "task-execution", "attempts", attempt_id)

    fl_ctx.set_prop(FLContextKey.TASK_RESULT_SEND_SUCCESS, sent, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.TASK_RESULT_ACCEPTED, accepted, private=True, sticky=False)
    executor.handle_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)

    assert _diagnostics(workspace)[-1]["publication_outcome"] == "not_accepted"
    assert os.path.isfile(os.path.join(attempt_dir, "result.fobs"))
    assert os.path.isfile(os.path.join(attempt_dir, "input.fobs"))


def test_site_approved_secret_reaches_script_without_credentials_or_persisting_values(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    script = """
import argparse, os
import nvflare.client as flare
from nvflare.apis.job_launcher_spec import JobProcessEnv
parser = argparse.ArgumentParser()
parser.add_argument('--key')
args = parser.parse_args()
assert args.key == 'site-only test value'
assert 'UNAPPROVED_SECRET' not in os.environ
assert not set(JobProcessEnv.ALL).intersection(os.environ)
flare.init()
flare.receive()
flare.send(flare.FLModel(metrics={'key_received': True}))
"""
    script_path = os.path.join(workspace.get_app_custom_dir("job-1"), "secret_script.py")
    with open(script_path, "w") as stream:
        stream.write(script)
    monkeypatch.setenv("TEST_TASK_KEY", "site-only test value")
    monkeypatch.setenv("UNAPPROVED_SECRET", "excluded")
    for name in JobProcessEnv.ALL:
        monkeypatch.setenv(name, "excluded-credential")
    executor = TaskWorkerExecutor(
        executor={
            "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
            "args": {
                "execution_mode": "in_process",
                "task_script_path": "secret_script.py",
                "task_script_args": ["--key", "${secret:TEST_TASK_KEY}"],
            },
        },
        components=[],
    )
    executor.set_task_launcher(ProcessTaskLauncher(), environment_variables=["TEST_TASK_KEY"])
    result = executor.execute(
        "train", FLModelUtils.to_shareable(FLModel(metrics={"seed": 1})), _context(workspace), Signal()
    )
    assert FLModelUtils.from_shareable(result).metrics == {"key_received": True}
    for path in (tmp_path / "workspace").rglob("*.json*"):
        if path.name == "meta.json" or path.name.startswith("diagnostics") or path.name == "bootstrap.json":
            assert "site-only test value" not in path.read_text()


@pytest.mark.parametrize("name", [*JobProcessEnv.ALL, "CUDA_VISIBLE_DEVICES", "NVFLARE_CLIENT_API_BOOTSTRAP"])
def test_site_environment_policy_cannot_forward_protected_variables(name):
    with pytest.raises(ValueError, match="protected variable"):
        _executor().set_task_launcher(ProcessTaskLauncher(), environment_variables=[name])


def test_supervisor_rejects_clean_exit_without_committed_result(tmp_path):
    workspace = _workspace(tmp_path)
    fl_ctx = _context(workspace)
    executor = _executor("AbruptExecutor")

    with pytest.raises(IncompleteTaskArtifactError, match="completion.json"):
        executor.execute("train", Shareable(), fl_ctx, Signal())

    records = _diagnostics(workspace)
    assert records[-1]["event"] == "settled"
    assert records[-1]["settled_timestamp"] is not None


def test_supervisor_rejects_stale_completion_after_settlement(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    fl_ctx = _context(workspace)
    executor = _executor()
    original = FileTaskArtifactStore.read_result

    def stale_result(store, identity):
        records = _diagnostics(workspace)
        assert records[-1]["event"] == "settled"
        completion_path = os.path.join(store.attempt_dir(identity), "completion.json")
        completion = json.loads(open(completion_path).read())
        completion["identity"]["attempt_id"] = "stale-attempt"
        with open(completion_path, "w") as stream:
            json.dump(completion, stream)
        return original(store, identity)

    monkeypatch.setattr(FileTaskArtifactStore, "read_result", stale_result)

    with pytest.raises(ValueError, match="stale task completion identity"):
        executor.execute("train", Shareable({"value": 1}), fl_ctx, Signal())


def test_supervisor_reads_success_only_after_settlement(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    fl_ctx = _context(workspace)
    executor = _executor()
    original = FileTaskArtifactStore.read_result
    observed = []

    def read_after_settlement(store, identity):
        records = _diagnostics(workspace)
        observed.append(records[-1]["event"])
        assert records[-1]["settled_timestamp"] is not None
        return original(store, identity)

    monkeypatch.setattr(FileTaskArtifactStore, "read_result", read_after_settlement)

    result = executor.execute("train", Shareable({"value": 1}), fl_ctx, Signal())

    assert result["value"] == 2
    assert observed == ["settled"]


def test_supervisor_reports_nonzero_worker_exit_and_retains_failure_artifacts(tmp_path):
    workspace = _workspace(tmp_path)
    fl_ctx = _context(workspace)
    executor = _executor("RaiseExecutor")

    with pytest.raises(RuntimeError, match="task worker failed: exit code"):
        executor.execute("train", Shareable(), fl_ctx, Signal())

    record = _diagnostics(workspace)[-1]
    attempt_dir = os.path.join(
        workspace.get_run_dir("job-1"), ".nvflare", "task-execution", "attempts", record["attempt_id"]
    )
    assert record["event"] == "settled"
    assert os.path.isfile(os.path.join(attempt_dir, "failure.json"))
    assert os.path.isfile(os.path.join(attempt_dir, "input.fobs"))


def test_supervisor_abort_signal_cancels_worker_and_retains_attempt(tmp_path):
    workspace = _workspace(tmp_path)
    fl_ctx = _context(workspace)
    executor = _executor("SlowExecutor", worker_timeout=30)
    abort_signal = Signal()
    outcome = []

    def run():
        try:
            outcome.append(executor.execute("train", Shareable(), fl_ctx, abort_signal))
        except BaseException as e:
            outcome.append(e)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    records = []
    while time.monotonic() < deadline:
        try:
            records = _diagnostics(workspace)
        except FileNotFoundError:
            pass
        if records and records[-1]["event"] == "launched":
            break
        time.sleep(0.01)
    assert records and records[-1]["event"] == "launched"
    worker_pid = records[-1]["worker_pid"]

    abort_signal.trigger(True)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert outcome[0].get_return_code() == ReturnCode.TASK_ABORTED
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)
    attempt_dir = os.path.join(
        workspace.get_run_dir("job-1"), ".nvflare", "task-execution", "attempts", records[-1]["attempt_id"]
    )
    assert os.path.isfile(os.path.join(attempt_dir, "input.fobs"))


def test_supervisor_timeout_cancels_and_confirms_settlement(tmp_path):
    workspace = _workspace(tmp_path)
    fl_ctx = _context(workspace)
    executor = _executor("SlowExecutor", worker_timeout=0.05)

    with pytest.raises(RuntimeError, match="timed out"):
        executor.execute("train", Shareable(), fl_ctx, Signal())

    record = _diagnostics(workspace)[-1]
    assert record["event"] == "settled"
    worker_pid = record["worker_pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


def test_supervisor_rejects_effective_gpu_job_resources_before_staging(tmp_path):
    workspace = _workspace(tmp_path)
    with open(workspace.get_job_meta_path("job-1"), "w") as stream:
        json.dump({"byoc": True, "resource_spec": {"site-1": {"num_of_gpus": 1}}}, stream)
    executor = _executor()

    with pytest.raises(RuntimeError, match="CPU Process workers only.*num_of_gpus"):
        executor.execute("train", Shareable({"value": 1}), _context(workspace), Signal())

    runtime_root = os.path.join(workspace.get_run_dir("job-1"), ".nvflare", "task-execution")
    assert not os.path.exists(runtime_root)


def test_supervisor_rejects_effective_gpu_launcher_spec_before_staging(tmp_path):
    workspace = _workspace(tmp_path)
    with open(workspace.get_job_meta_path("job-1"), "w") as stream:
        json.dump(
            {
                "byoc": True,
                "launcher_spec": {
                    "default": {"slurm": {"gpus_per_node": 2}},
                    "site-1": {"slurm": {"nodes": 1}},
                },
            },
            stream,
        )
    executor = _executor()

    with pytest.raises(RuntimeError, match="CPU Process workers only.*gpus_per_node"):
        executor.execute("train", Shareable({"value": 1}), _context(workspace), Signal())

    runtime_root = os.path.join(workspace.get_run_dir("job-1"), ".nvflare", "task-execution")
    assert not os.path.exists(runtime_root)


def test_worker_environment_masks_gpu_visibility_and_credentials(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    for name in JobProcessEnv.ALL:
        monkeypatch.setenv(name, "secret")

    environment = TaskWorkerExecutor._worker_environment()

    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["HIP_VISIBLE_DEVICES"] == ""
    assert environment["ROCR_VISIBLE_DEVICES"] == ""
    assert environment["NVIDIA_VISIBLE_DEVICES"] == "none"
    assert not set(JobProcessEnv.ALL).intersection(environment)


def test_supervisor_requires_site_runtime_launcher_before_staging(tmp_path):
    workspace = _workspace(tmp_path)
    executor = TaskWorkerExecutor(
        executor={"path": "supervisor_probe.ProbeExecutor", "args": {}},
        components=[],
    )

    with pytest.raises(RuntimeError, match="site runtime must inject"):
        executor.execute("train", Shareable({"value": 1}), _context(workspace), Signal())

    runtime_root = os.path.join(workspace.get_run_dir("job-1"), ".nvflare", "task-execution")
    assert not os.path.exists(runtime_root)


def test_supervisor_rejects_launcher_replacement():
    executor = TaskWorkerExecutor(
        executor={"path": "supervisor_probe.ProbeExecutor", "args": {}},
        components=[],
    )
    first = ProcessTaskLauncher()
    executor.set_task_launcher(first)
    executor.set_task_launcher(first)

    with pytest.raises(RuntimeError, match="already been configured"):
        executor.set_task_launcher(ProcessTaskLauncher())


def test_diagnostic_write_failure_cancels_launched_worker(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    executor = _executor("SlowExecutor", worker_timeout=None)
    handles = []
    launcher = executor._get_task_launcher()
    launch_task = launcher.launch_task

    def capture_handle(request):
        handle = launch_task(request)
        handles.append(handle)
        return handle

    def fail_diagnostic(*_args, **_kwargs):
        raise OSError("disk")

    monkeypatch.setattr(launcher, "launch_task", capture_handle)
    monkeypatch.setattr(executor, "_append_diagnostic", fail_diagnostic)

    with pytest.raises(OSError, match="disk"):
        executor.execute("train", Shareable(), _context(workspace), Signal())

    assert len(handles) == 1
    assert handles[0].poll().settled
    assert handles[0].poll().cancel_requested
    assert executor._active_handle is None


@pytest.mark.parametrize("executor, components", [(None, []), ({}, {}), ({}, [None])])
def test_supervisor_requires_inert_config_shapes(executor, components):
    with pytest.raises(TypeError):
        TaskWorkerExecutor(executor=executor, components=components)


def test_supervisor_rejects_invalid_launcher_and_halts_after_cancellation_failure():
    executor = _executor()
    with pytest.raises(TypeError, match="TaskLauncherSpec"):
        executor.set_task_launcher(object())
    executor._active_handle = Mock(cancel=Mock(side_effect=RuntimeError("cannot settle")))
    with pytest.raises(RuntimeError, match="cannot settle"):
        executor.handle_event(EventType.ABORT_TASK, None)
    assert executor._stopping
    executor._active_handle = None
    executor.handle_event(EventType.START_RUN, None)
    assert not executor._stopping
    executor.handle_event(EventType.END_RUN, None)
    assert executor._stopping


def test_supervisor_rejects_invalid_or_concurrent_assignment_before_launch(tmp_path):
    executor = _executor()
    fl_ctx = _context(_workspace(tmp_path))
    with pytest.raises(TypeError, match="Shareable"):
        executor.execute("train", {}, fl_ctx, Signal())
    executor._execution_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="active task"):
            executor.execute("train", Shareable(), fl_ctx, Signal())
    finally:
        executor._execution_lock.release()
    fl_ctx.set_prop(FLContextKey.TASK_ID, None, private=True, sticky=False)
    with pytest.raises(RuntimeError, match="current task ID"):
        executor.execute("train", Shareable(), fl_ctx, Signal())
    fl_ctx.set_prop(FLContextKey.TASK_ID, "task-1", private=True, sticky=False)
    executor.handle_event(EventType.END_RUN, fl_ctx)
    with pytest.raises(RuntimeError, match="stopping"):
        executor.execute("train", Shareable(), fl_ctx, Signal())


def test_supervisor_resource_validation_handles_absent_metadata_and_nested_values(tmp_path):
    workspace = _workspace(tmp_path)
    fl_ctx = _context(workspace)
    os.unlink(workspace.get_job_meta_path("job-1"))
    TaskWorkerExecutor._validate_cpu_only_runtime(fl_ctx)
    fl_ctx.set_prop(FLContextKey.JOB_META, [], private=True)
    TaskWorkerExecutor._validate_cpu_only_runtime(fl_ctx)
    assert TaskWorkerExecutor._effective_site_settings(None, "site-1", "default") == {}
    assert TaskWorkerExecutor._effective_site_settings({"default": [], "site-1": None}, "site-1", "default") == {}
    assert (
        TaskWorkerExecutor._find_nonempty_gpu_setting({"nested": [None, {"gpu": "0"}]}) == "resource_spec.nested[1].gpu"
    )
    assert TaskWorkerExecutor._find_nonempty_gpu_setting({"nested": [None, {"gpu": False}]}) is None


def _fake_execution(tmp_path, monkeypatch, status=None, artifact_cleanup=TaskArtifactCleanup.JOB):
    executor = _executor(artifact_cleanup=artifact_cleanup)
    fl_ctx = _context(_workspace(tmp_path))
    status = status or TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0, settled=True)
    handle = SimpleNamespace(
        execution_id="test:attempt",
        poll=Mock(return_value=status),
        wait_for_settlement=Mock(return_value=status),
        cancel=Mock(return_value=status),
    )
    monkeypatch.setattr(executor._get_task_launcher(), "launch_task", lambda _request: handle)
    completion = SimpleNamespace(worker_pid=123, worker_ppid=12, started_at=1, completed_at=2)
    monkeypatch.setattr(FileTaskArtifactStore, "read_result", lambda *_args: (Shareable({"result": 1}), completion))
    monkeypatch.setattr(FileTaskArtifactStore, "read_analytics", lambda *_args: [])
    return executor, fl_ctx, handle


def test_supervisor_cancels_when_terminal_worker_fails_to_settle_before_timeout(tmp_path, monkeypatch):
    executor, fl_ctx, handle = _fake_execution(tmp_path, monkeypatch)
    handle.wait_for_settlement.side_effect = TimeoutError("descendants did not settle")
    with pytest.raises(RuntimeError, match="timed out"):
        executor.execute("train", Shareable(), fl_ctx, Signal())
    handle.cancel.assert_called_once()
    assert executor._active_handle is None


@pytest.mark.parametrize("failure", ["unsettled", "signal", "pid", "duplicate"])
def test_supervisor_rejects_invalid_completion_or_unsettled_execution(tmp_path, monkeypatch, failure):
    status = None
    if failure == "unsettled":
        status = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0, settled=False)
    elif failure == "signal":
        status = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, termination_signal=15, settled=True)
    executor, fl_ctx, handle = _fake_execution(tmp_path, monkeypatch, status)
    if failure == "pid":
        handle.process_group_id = 456
    elif failure == "duplicate":
        executor._pending_publication["task-1"] = "earlier result"
    expected = {
        "unsettled": "did not settle",
        "signal": "termination signal",
        "pid": "PID",
        "duplicate": "pending publication",
    }
    with pytest.raises((RuntimeError, ValueError), match=expected[failure]):
        executor.execute("train", Shareable(), fl_ctx, Signal())
    if failure == "unsettled":
        handle.cancel.assert_called_once()
        assert executor._stopping
        assert executor._active_handle is handle


@pytest.mark.parametrize("failure", ["cancel", "poll"])
def test_supervisor_keeps_runtime_stopped_if_cleanup_cannot_confirm_settlement(tmp_path, monkeypatch, failure):
    executor, fl_ctx, handle = _fake_execution(tmp_path, monkeypatch)
    panic = Mock()
    monkeypatch.setattr(executor, "system_panic", panic)
    monkeypatch.setattr(executor, "_append_diagnostic", Mock(side_effect=OSError("disk failed")))
    if failure == "cancel":
        handle.poll.return_value = TaskExecutionStatus(TaskExecutionPhase.RUNNING)
        handle.cancel.side_effect = RuntimeError("cleanup failed")
    else:
        handle.poll.side_effect = RuntimeError("status unavailable")
    with pytest.raises(RuntimeError, match="cleanup failed|status unavailable"):
        executor.execute("train", Shareable(), fl_ctx, Signal())
    assert executor._stopping
    assert executor._active_handle is handle
    assert panic.call_count >= 1


def test_local_abort_during_launch_is_latched_and_returns_task_aborted(tmp_path, monkeypatch):
    executor, fl_ctx, handle = _fake_execution(tmp_path, monkeypatch)

    def abort_during_launch(_request):
        assert executor._active_handle is None
        executor.handle_event(EventType.ABORT_TASK, fl_ctx)
        return handle

    monkeypatch.setattr(executor._get_task_launcher(), "launch_task", abort_during_launch)
    assert executor.execute("train", Shareable(), fl_ctx, Signal()).get_return_code() == ReturnCode.TASK_ABORTED
    handle.cancel.assert_called_once()
    assert executor._active_handle is None
    assert executor._active_abort_signal is None
    assert not executor._pending_publication


def test_framework_secure_logging_policy_is_forwarded_without_site_opt_in(monkeypatch):
    monkeypatch.setenv("NVFLARE_SECURE_LOGGING", "true")
    monkeypatch.setenv("FL_LOG_LEVEL", "INFO")
    environment = TaskWorkerExecutor._worker_environment()
    assert environment["NVFLARE_SECURE_LOGGING"] == "true"
    assert environment["FL_LOG_LEVEL"] == "INFO"


def test_result_timeout_excludes_worker_startup_and_post_send_finalizers(tmp_path, monkeypatch):
    executor, fl_ctx, handle = _fake_execution(tmp_path, monkeypatch)
    executor.result_wait_timeout = 1
    handle.poll.side_effect = [
        TaskExecutionStatus(TaskExecutionPhase.RUNNING),
        TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0, settled=True),
        TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0, settled=True),
    ]
    monkeypatch.setattr(FileTaskArtifactStore, "result_wait_state", Mock(side_effect=[(None, None), (0, 0.5)]))
    assert executor.execute("train", Shareable(), fl_ctx, Signal())["result"] == 1
    handle.cancel.assert_not_called()


def test_result_timeout_cancels_after_script_receive(tmp_path, monkeypatch):
    executor, fl_ctx, handle = _fake_execution(tmp_path, monkeypatch)
    executor.result_wait_timeout = 1
    monkeypatch.setattr(FileTaskArtifactStore, "result_wait_state", lambda *_args: (0, None))
    with pytest.raises(RuntimeError, match="timed out"):
        executor.execute("train", Shareable(), fl_ctx, Signal())
    handle.cancel.assert_called_once()


def test_malformed_analytics_cannot_discard_successful_task_result(tmp_path, monkeypatch):
    executor, fl_ctx, _handle = _fake_execution(tmp_path, monkeypatch)
    monkeypatch.setattr(FileTaskArtifactStore, "read_analytics", lambda *_args: [{"bad": "record"}])
    log = Mock()
    monkeypatch.setattr(executor, "log_error", log)
    assert executor.execute("train", Shareable(), fl_ctx, Signal())["result"] == 1
    log.assert_called_once()


@pytest.mark.parametrize("failure", ["diagnostics", "payload_cleanup"])
def test_publication_cleanup_failures_preserve_evidence_and_are_logged(tmp_path, monkeypatch, failure):
    executor, fl_ctx, _handle = _fake_execution(tmp_path, monkeypatch, artifact_cleanup=TaskArtifactCleanup.ACCEPTED)
    executor.handle_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)  # No pending result is harmless.
    executor.execute("train", Shareable(), fl_ctx, Signal())
    fl_ctx.set_prop(FLContextKey.TASK_RESULT_SEND_SUCCESS, True, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.TASK_RESULT_ACCEPTED, True, private=True, sticky=False)
    release = Mock(side_effect=OSError("cleanup failed"))
    monkeypatch.setattr(FileTaskArtifactStore, "release_payloads", release)
    log = Mock()
    if failure == "diagnostics":
        monkeypatch.setattr(executor, "_append_diagnostic", Mock(side_effect=OSError("disk failed")))
        monkeypatch.setattr(executor, "log_error", log)
    else:
        monkeypatch.setattr(executor, "log_warning", log)
    executor.handle_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)
    log.assert_called_once()
    if failure == "diagnostics":
        release.assert_not_called()


@pytest.mark.parametrize("accepted", [True, False, None])
@pytest.mark.parametrize("policy", [TaskArtifactCleanup.JOB, TaskArtifactCleanup.RETAIN])
def test_site_cleanup_policy_controls_payloads_until_and_after_job_end(tmp_path, monkeypatch, accepted, policy):
    executor, fl_ctx, _handle = _fake_execution(tmp_path, monkeypatch, artifact_cleanup=policy)
    executor.execute("train", Shareable(), fl_ctx, Signal())
    identity, store = next(iter(executor._retained_attempts.values()))
    completion = store.commit_result(identity, Shareable({"result": 1}))
    result_path = os.path.join(store.attempt_dir(identity), "result.fobs")
    input_path = os.path.join(store.attempt_dir(identity), "input.fobs")
    fl_ctx.set_prop(FLContextKey.TASK_RESULT_SEND_SUCCESS, True)
    fl_ctx.set_prop(FLContextKey.TASK_RESULT_ACCEPTED, accepted)
    executor.handle_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)
    assert os.path.isfile(result_path) and os.path.isfile(input_path)

    executor.handle_event(EventType.END_RUN, fl_ctx)

    assert os.path.exists(result_path) is (policy == TaskArtifactCleanup.RETAIN)
    assert os.path.exists(input_path) is (policy == TaskArtifactCleanup.RETAIN)
    assert store.read_completion(identity) == completion
    executor.handle_event(EventType.END_RUN, fl_ctx)  # Cleanup is idempotent.


def test_job_cleanup_is_deferred_while_execution_gate_is_held_and_preserves_unowned_attempts(tmp_path, monkeypatch):
    executor, fl_ctx, _handle = _fake_execution(tmp_path, monkeypatch)
    executor.execute("train", Shareable(), fl_ctx, Signal())
    identity, store = next(iter(executor._retained_attempts.values()))
    other = type(identity)("job-1", "site-1", "other-task", "train", "other-attempt")
    store.create_attempt(other)
    store.write_input(other, Shareable())
    executor._execution_lock.acquire()
    try:
        executor.handle_event(EventType.END_RUN, fl_ctx)
        assert os.path.exists(os.path.join(store.attempt_dir(identity), "input.fobs"))
    finally:
        executor._execution_lock.release()
    executor._cleanup_job_payloads(fl_ctx)
    assert not os.path.exists(os.path.join(store.attempt_dir(identity), "input.fobs"))
    assert os.path.exists(os.path.join(store.attempt_dir(other), "input.fobs"))


def test_job_cleanup_cannot_delete_payloads_while_worker_settlement_is_unconfirmed(tmp_path, monkeypatch):
    status = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0, settled=False)
    executor, fl_ctx, _handle = _fake_execution(tmp_path, monkeypatch, status)
    with pytest.raises(RuntimeError, match="did not settle"):
        executor.execute("train", Shareable(), fl_ctx, Signal())
    release = Mock()
    monkeypatch.setattr(FileTaskArtifactStore, "release_payloads", release)
    executor.handle_event(EventType.END_RUN, fl_ctx)
    release.assert_not_called()


def test_execution_finally_completes_deferred_job_cleanup_after_cancel_settles(tmp_path, monkeypatch):
    executor, fl_ctx, handle = _fake_execution(tmp_path, monkeypatch)

    def end_job_while_launching(_request):
        executor.handle_event(EventType.END_RUN, fl_ctx)
        assert executor._retained_attempts  # Execution owns the read/cleanup gate.
        return handle

    monkeypatch.setattr(executor._get_task_launcher(), "launch_task", end_job_while_launching)
    assert executor.execute("train", Shareable(), fl_ctx, Signal()).get_return_code() == ReturnCode.TASK_ABORTED
    handle.cancel.assert_called_once()
    assert executor._active_handle is None
    assert not executor._retained_attempts


def test_job_cleanup_failures_are_logged_and_remain_retryable(tmp_path, monkeypatch):
    executor, fl_ctx, _handle = _fake_execution(tmp_path, monkeypatch)
    executor.execute("train", Shareable(), fl_ctx, Signal())
    release = Mock(side_effect=OSError("disk unavailable"))
    log = Mock()
    monkeypatch.setattr(FileTaskArtifactStore, "release_payloads", release)
    monkeypatch.setattr(executor, "log_warning", log)
    executor.handle_event(EventType.END_RUN, fl_ctx)
    log.assert_called_once()
    assert executor._retained_attempts
    release.side_effect = None
    executor.handle_event(EventType.END_RUN, fl_ctx)
    assert not executor._retained_attempts


def test_supervisor_rejects_invalid_cleanup_policy_or_reconfiguration():
    executor = _executor()
    with pytest.raises(ValueError, match="artifact_cleanup"):
        executor.set_task_launcher(executor._get_task_launcher(), artifact_cleanup="task")
    with pytest.raises(RuntimeError, match="already been configured"):
        executor.set_task_launcher(executor._get_task_launcher(), artifact_cleanup=TaskArtifactCleanup.RETAIN)
