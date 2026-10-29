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

import copy
import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskLaunchError
from nvflare.app_opt.job_launcher.workspace_cell_transfer import ENV_WORKSPACE_OWNER_FQCN
from nvflare.app_opt.task_launcher.k8s_launcher import SETTLEMENT_FINALIZER, K8sTaskLauncher
from nvflare.app_opt.task_launcher.k8s_task_service import K8sTaskClient
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from tests.k8s_task_test_utils import ApiError, wire_client


def eventually(check, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.001)
    pytest.fail("CP owner did not reach the expected state")


@pytest.fixture
def control(setup, monkeypatch):
    backend, api, request, store, bootstrap = setup
    cell, service = wire_client(monkeypatch)
    owner = service.add_job(backend, "job", "site-1", "job-token")
    client = K8sTaskClient(cell, "job", "site-1", 0.05, 0.001, 0.05)
    yield backend, api, request, cell, service, owner, client
    cell.drop_reply = False
    if api.pod is not None:
        api.read_error = None
        api.patch_error = None
        api.pod.metadata.uid = "uid-1"
        api.pod.metadata.annotations = copy.deepcopy(api.manifest["metadata"]["annotations"])
        api.pod.metadata.finalizers = [SETTLEMENT_FINALIZER]
        api.finish(exit_code=143)
    owner.release()
    owner.thread.join(2)
    assert not owner.thread.is_alive(), "test left a CP owner running"


def raw(cell, **overrides):
    data = dict(
        job_id="job",
        token="job-token",
        action="launch",
        task_id="task",
        attempt_id="attempt",
        bootstrap="/ignored",
        cwd="/ignored",
    )
    data.update(overrides)
    return cell.send_request(request=new_cell_message({}, json.dumps(data)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("token", "other-job-token"),
        ("job_id", "other-job"),
        ("token", ["job-token"]),
    ],
)
def test_other_job_capability_cannot_launch_or_observe(control, field, value):
    _, api, _, cell, _, _, _ = control
    response = raw(cell, **{field: value})
    assert response.get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK
    assert "create" not in api.calls


def test_capability_also_requires_the_expected_cj_origin(control):
    _, api, request, cell, _, _, _ = control
    cell.origin = "site-parent.other-job"
    response = raw(cell, bootstrap=request.argv[4], cwd=request.cwd)
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED
    assert not api.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest", {"spec": {"serviceAccountName": "cp-authority", "hostPID": True}}),
        ("namespace", "other-job"),
        ("image", "unapproved/image"),
        ("environment", {"NVFL_WORKSPACE_TRANSFER_TOKEN": "stolen"}),
        ("resources", {"nvidia.com/gpu": "100"}),
        ("site_name", "another-site"),
    ],
)
def test_cj_cannot_choose_pod_authority_or_shape(control, field, value):
    _, api, request, cell, _, _, _ = control
    response = raw(cell, bootstrap=request.argv[4], cwd=request.cwd, **{field: value})
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert not api.calls


@pytest.mark.parametrize("field", ["bootstrap", "cwd"])
def test_cross_job_paths_are_rejected_before_pod_creation(control, field):
    _, api, request, cell, _, _, _ = control
    paths = dict(bootstrap=request.argv[4], cwd=request.cwd)
    paths[field] = request.cwd + "/../another-job/bootstrap.json"
    response = raw(cell, **paths)
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert not api.calls


def test_task_result_waits_for_cp_confirmed_physical_settlement(control):
    _, api, request, _, _, _, client = control
    handle = client.launch(request)
    eventually(lambda: api.pod is not None)
    api.running()
    eventually(lambda: handle.poll().phase == TaskExecutionPhase.RUNNING)
    assert not handle.poll().settled
    api.finish(sandbox_down=False)
    eventually(lambda: "delete" in api.calls)
    assert not handle.poll().settled
    api.finish()
    assert handle.wait_for_settlement(2).settled
    assert api.deleted


def test_lost_launch_reply_is_owned_and_idempotent(control):
    _, api, request, cell, _, _, client = control
    cell.drop_reply = True
    with pytest.raises(TaskLaunchError) as failure:
        client.launch(request)
    cell.drop_reply = False
    handle = client.launch(request)
    eventually(lambda: api.pod is not None)
    assert api.calls.count("create") == 1
    api.finish()
    assert handle.wait_for_settlement(2).settled
    assert failure.value.handle.poll().settled


def test_lost_reply_cancel_fences_delayed_launch_before_acquisition(control):
    _, api, request, _, _, owner, client = control
    # Deterministically pause the owner before it can consume the launch.
    owner.release()
    owner.thread.join(2)
    with owner.lock:
        owner.closed = owner.released = False
        owner.attempts.clear()
    # Reinstall this inert owner to model a launch delayed behind cancellation.
    with client.cell.service.lock:
        client.cell.service.jobs["job"] = owner
    client.call("cancel", task_id=request.task_id, attempt_id=request.attempt_id)
    with pytest.raises(TaskLaunchError):
        client.launch(request)
    assert not api.calls
    assert client.call("poll", task_id=request.task_id, attempt_id=request.attempt_id)["settled"]


def test_cp_retains_and_cancels_a_create_returning_after_cj_exit(control):
    _, api, request, _, service, owner, client = control
    entered, release = threading.Event(), threading.Event()

    def block_create():
        api.running()
        entered.set()
        assert release.wait(2)

    api.on_create = block_create
    handle = client.launch(request)
    assert entered.wait(2)
    owner.release()  # CP may now tear down CJ without losing the in-flight create.
    assert service.jobs["job"] is owner
    release.set()
    eventually(lambda: "delete" in api.calls)
    assert not owner.attempts[(request.task_id, request.attempt_id)].status.settled
    api.finish(exit_code=143)
    owner.thread.join(2)
    assert api.deleted
    assert "job" not in service.jobs
    assert not owner.thread.is_alive()


def test_cj_death_cancels_workers_without_cj_rpc(control):
    backend, api, request, _, _, owner, client = control
    client.launch(request)
    eventually(lambda: api.pod is not None)
    api.running()
    read_worker = api.read_namespaced_pod

    def read(*, name, **kwargs):
        if name == "cj":
            raise ApiError(404)
        return read_worker(name=name, **kwargs)

    api.read_namespaced_pod = read
    with owner.lock:
        owner.cj_name, owner.cj_uid = "cj", "cj-uid"
    owner.wake.set()
    eventually(lambda: "delete" in api.calls)
    assert owner.released
    assert not owner.attempts[(request.task_id, request.attempt_id)].status.settled
    api.finish(exit_code=143)
    owner.thread.join(2)
    assert api.deleted


def test_new_attempt_cannot_overlap_an_unsettled_worker(control):
    _, api, request, _, _, _, client = control
    client.launch(request)
    eventually(lambda: api.pod is not None)
    api.running()
    with pytest.raises(TaskLaunchError):
        client.launch(replace(request, attempt_id="another"))
    assert api.calls.count("create") == 1


def test_public_cj_launcher_has_no_kubernetes_api_method(control):
    _, _, request, cell, _, _, _ = control
    proxy = K8sTaskLauncher("site", "trusted", request.cwd, "pvc", "no-api")
    assert not hasattr(proxy, "_api")
    context = Mock()
    context.get_job_id.return_value, context.get_identity_name.return_value = "job", "site-1"
    context.get_engine.return_value.get_cell.return_value = cell
    context.get_prop.return_value = {}
    proxy.handle_event(EventType.START_RUN, context)
    handle = proxy.launch_task(request)
    assert handle.request == request


def test_cp_rejects_a_bootstrap_parent_symlink_to_another_job(control, tmp_path):
    _, api, request, _, _, _, client = control
    outside = tmp_path / "other-job"
    outside.mkdir()
    # Copy a syntactically valid bootstrap, then reach it through a job-owned symlink.
    from pathlib import Path

    (outside / "bootstrap.json").write_bytes(Path(request.argv[4]).read_bytes())
    alias = tmp_path / "symlink-parent"
    alias.symlink_to(outside, target_is_directory=True)
    handle = client.launch(replace(request, argv=(*request.argv[:4], str(alias / "bootstrap.json"), *request.argv[5:])))
    status = handle.wait_for_settlement(2)
    assert status.failure_reason
    assert "create" not in api.calls


def test_cp_mount_path_is_translated_without_leaking_into_worker_manifest(control, tmp_path):
    from nvflare.private.fed.task_worker.protocol import WorkerBootstrap, write_bootstrap

    backend, api, request, _, _, _, client = control
    cp_root = tmp_path / "cp-pvc" / "job-subtree"
    (cp_root / "artifacts").mkdir(parents=True)
    worker_root = "/worker/durable-workspace"
    backend.workspace_root, backend.storage_root = worker_root, str(cp_root)
    backend._storage_checked = True  # CP's trusted-root preflight ran before CJ.
    from nvflare.private.fed.task_worker.protocol import read_bootstrap

    original = read_bootstrap(request.argv[4])
    staged = cp_root / "staged" / "bootstrap.json"
    write_bootstrap(
        str(staged),
        WorkerBootstrap(original.identity, worker_root + "/artifacts", worker_root, {"path": "unused.Compute"}),
    )
    request = replace(
        request, argv=(*request.argv[:4], worker_root + "/staged/bootstrap.json", *request.argv[5:]), cwd=worker_root
    )
    client.launch(request)
    eventually(lambda: api.pod is not None)
    container = api.manifest["spec"]["containers"][0]
    assert container["workingDir"] == worker_root
    assert worker_root + "/staged/bootstrap.json" in container["args"]
    assert str(cp_root) not in json.dumps(api.manifest)
    assert container["volumeMounts"][0]["mountPath"] == worker_root


def test_cj_end_run_is_bounded_when_cp_retains_unsettleable_execution(control):
    from nvflare.apis.fl_constant import ReservedKey
    from nvflare.apis.fl_context import FLContext
    from nvflare.apis.task_launcher_spec import TaskSettlementError
    from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
    from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor

    backend, api, request, cell, service, owner, client = control
    proxy = K8sTaskLauncher("site", "trusted", request.cwd, "pvc", "no-api")
    proxy._client = client
    adapter = TaskWorkerExecutor({}, (), TaskExecutionPolicy(proxy))
    context = FLContext()
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_cell=lambda: cell))
    handle = client.launch(request)
    eventually(lambda: api.pod is not None)
    api.running()
    api.read_error = ApiError(503)
    adapter.supervisor = Mock()
    adapter.supervisor.take_issues.return_value = []
    adapter.supervisor.settlement_unconfirmed = False
    adapter.supervisor.end_run.side_effect = TaskSettlementError("CP still owns execution", handle.poll())
    adapter._end_native_run = Mock()
    with pytest.raises(TaskSettlementError):
        adapter.handle_event(EventType.END_RUN, context)
    assert adapter._ending
    assert owner.closed
    assert service.jobs["job"] is owner
    assert not owner.released
    adapter._end_native_run.assert_not_called()
    adapter.supervisor.end_run.assert_called_once()
    assert "delete" not in api.calls


def test_unknown_cancel_flood_seals_admission_without_evicting_fences(control, monkeypatch):
    _, api, request, _, _, owner, client = control
    monkeypatch.setattr("nvflare.app_opt.task_launcher.k8s_task_service._MAX_CANCEL_FENCES", 3)
    for index in range(3):
        client.call("cancel", task_id=request.task_id, attempt_id=str(index))
    with pytest.raises(Exception, match="CP task cancel rejected"):
        client.call("cancel", task_id=request.task_id, attempt_id="overflow")
    assert owner.closed
    assert len(owner.attempts) == 3
    for index in range(3):
        assert client.call("cancel", task_id=request.task_id, attempt_id=str(index))["settled"]
        with pytest.raises(TaskLaunchError):
            client.launch(replace(request, attempt_id=str(index)))
    with pytest.raises(TaskLaunchError):
        client.launch(request)
    assert not api.calls


def test_cj_observer_is_a_slow_backstop_not_the_task_poll_loop():
    from nvflare.app_opt.task_launcher.k8s_task_service import _CJ_OBSERVATION_INTERVAL, _Job

    backend = SimpleNamespace(core_v1=Mock(), namespace="site", request_timeout=1)
    backend.core_v1.read_namespaced_pod.return_value = SimpleNamespace(
        metadata=SimpleNamespace(uid="cj-uid", deletion_timestamp=None),
        status=SimpleNamespace(phase="Running"),
    )
    owner = _Job(backend, "job", "site", "token", "origin")
    owner.attach_cj("cj", "cj-uid")
    owner.observe_cj()
    for _ in range(100):
        owner.observe_cj()
    backend.core_v1.read_namespaced_pod.assert_called_once()
    assert _CJ_OBSERVATION_INTERVAL == 10
    owner.attach_cj("cj", "cj-uid")
    owner.observe_cj()
    assert backend.core_v1.read_namespaced_pod.call_count == 2


def test_end_run_is_safe_after_start_run_fails(control, monkeypatch):
    _, _, request, _, _, _, _ = control
    proxy = K8sTaskLauncher("site", "trusted", request.cwd, "pvc", "no-api")
    monkeypatch.delenv(ENV_WORKSPACE_OWNER_FQCN)
    context = Mock()
    context.get_prop.return_value = {}
    with pytest.raises(RuntimeError, match="CP-provisioned"):
        proxy.handle_event(EventType.START_RUN, context)
    proxy.handle_event(EventType.END_RUN, context)
    proxy.handle_event(EventType.END_RUN, context)


def test_rejected_launch_flood_is_bounded_and_seals_admission(control):
    backend, api, request, _, _, owner, client = control
    backend.max_attempts = 3
    invalid = replace(request, argv=(*request.argv[:4], request.cwd + "/missing.json", *request.argv[5:]))
    for index in range(3):
        handle = client.launch(replace(invalid, attempt_id=str(index)))
        assert handle.wait_for_settlement(2).failure_reason
    with pytest.raises(TaskLaunchError):
        client.launch(replace(invalid, attempt_id="overflow"))
    assert owner.closed
    assert len(owner.attempts) == 3
    assert owner.cancel_fences == 0
    assert not api.calls
    assert client.call("cancel", task_id=request.task_id, attempt_id="0")["settled"]
