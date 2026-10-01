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
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
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


def _executor(class_name="ProbeExecutor", **kwargs):
    executor = TaskWorkerExecutor(
        executor={"path": f"supervisor_probe.{class_name}", "args": {}},
        components=[],
        poll_interval=0.01,
        **kwargs,
    )
    executor.set_task_launcher(ProcessTaskLauncher(poll_interval=0.01))
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
    executor.set_task_launcher(ProcessTaskLauncher())
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
    executor.handle_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)
    assert not analytics_files[0].exists()


def _diagnostics(workspace):
    path = os.path.join(workspace.get_run_dir("job-1"), ".nvflare", "task-execution", "diagnostics.jsonl")
    with open(path) as stream:
        return [json.loads(line) for line in stream]


def test_supervisor_runs_fresh_worker_settles_and_releases_payloads_after_publication(tmp_path):
    workspace = _workspace(tmp_path)
    fl_ctx = _context(workspace)
    executor = _executor()
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


def test_supervisor_retains_payloads_when_publication_is_not_accepted(tmp_path):
    workspace = _workspace(tmp_path)
    fl_ctx = _context(workspace)
    executor = _executor()
    result = executor.execute("train", Shareable({"value": 2}), fl_ctx, Signal())
    assert result["value"] == 3
    attempt_id = _diagnostics(workspace)[-1]["attempt_id"]
    attempt_dir = os.path.join(workspace.get_run_dir("job-1"), ".nvflare", "task-execution", "attempts", attempt_id)

    fl_ctx.set_prop(FLContextKey.TASK_RESULT_SEND_SUCCESS, False, private=True, sticky=False)
    executor.handle_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)

    assert _diagnostics(workspace)[-1]["publication_outcome"] == "not_accepted"
    assert os.path.isfile(os.path.join(attempt_dir, "result.fobs"))
    assert os.path.isfile(os.path.join(attempt_dir, "input.fobs"))


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
            executor.execute("train", Shareable(), fl_ctx, abort_signal)
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
    assert outcome and "aborted" in str(outcome[0])
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
