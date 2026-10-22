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
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
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
    read_bootstrap,
)
from tests.task_worker_test_utils import _identity, _live, _stage, _wait_until, _workspace


def _run_process(bootstrap_path, extra_env=None):
    if not hasattr(os, "waitid"):
        pytest.skip("ProcessTaskLauncher requires waitid/WNOWAIT (macOS Python >= 3.13)")
    repo_root = str(Path(__file__).resolve().parents[3])
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
            "while True:\n"
            " try:\n"
            "  if not owner.is_running() or owner.status() in (psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD): break\n"
            " except psutil.NoSuchProcess: break\n"
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
    repo_root = str(Path(__file__).resolve().parents[3])
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


pytestmark = [pytest.mark.timeout(120), pytest.mark.usefixtures("isolated_fobs")]
