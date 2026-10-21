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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, TaskResultReceipt
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskLauncherError,
    TaskLaunchError,
    TaskLaunchRequest,
    TaskResourceRequest,
    TaskSettlementError,
)
from nvflare.app_opt.job_launcher.k8s_launcher import task_workspace_subpath
from nvflare.app_opt.task_launcher.k8s_launcher import (
    SETTLEMENT_FINALIZER,
    WORKER_MODULE,
    K8sTaskLauncher,
    check_posix_storage,
    task_pod_resources,
)
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore, IncompleteTaskArtifactError
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity, WorkerBootstrap, write_bootstrap
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor, TaskSupervisorOptions


class ApiError(Exception):
    def __init__(self, status):
        self.status = status


class PodApi:
    """A deterministic API model; no claim of runtime or scheduler qualification."""

    def __init__(self):
        self.pod = None
        self.calls = []
        self.deleted = False
        self.create_error = None
        self.read_error = None
        self.patch_error = None
        self.on_create = None
        self.on_delete = None

    def create_namespaced_pod(self, *, namespace, body, **kwargs):
        self.manifest = copy.deepcopy(body)
        if self.create_error:
            raise self.create_error
        self.pod = NS(
            metadata=NS(**body["metadata"], uid="uid-1", resource_version="1", deletion_timestamp=None),
            spec=NS(node_name=None, containers=[NS(name="nvflare-task")], init_containers=[], ephemeral_containers=[]),
            status=NS(
                phase="Pending",
                reason=None,
                conditions=[],
                container_statuses=[],
                init_container_statuses=[],
                ephemeral_container_statuses=[],
            ),
        )
        self.calls.append("create")
        if self.on_create:
            self.on_create()
        return copy.deepcopy(self.pod)

    def read_namespaced_pod(self, **kwargs):
        self.calls.append("read")
        if self.read_error:
            raise self.read_error
        if self.pod is None:
            raise ApiError(404)
        return copy.deepcopy(self.pod)

    def delete_namespaced_pod(self, *, body, **kwargs):
        assert body["preconditions"]["uid"] == self.pod.metadata.uid
        assert body["gracePeriodSeconds"] > 0
        self.calls.append("delete")
        self.pod.metadata.deletion_timestamp = "deleting"
        if self.on_delete:
            self.on_delete()
        return NS(status="Success")

    def patch_namespaced_pod(self, *, body, **kwargs):
        self.calls.append("patch")
        if self.patch_error:
            raise self.patch_error
        assert body[:3] == [
            {"op": "test", "path": "/metadata/uid", "value": self.pod.metadata.uid},
            {"op": "test", "path": "/metadata/resourceVersion", "value": self.pod.metadata.resource_version},
            {"op": "test", "path": "/metadata/finalizers", "value": self.pod.metadata.finalizers},
        ]
        self.pod.metadata.finalizers = body[3]["value"]
        if not self.pod.metadata.finalizers and self.pod.metadata.deletion_timestamp:
            self.pod = None
            self.deleted = True

    def read_namespaced_pod_log(self, *, limit_bytes, **kwargs):
        assert limit_bytes == 65536
        self.calls.append("logs")
        return "task output\n"

    def running(self):
        self.pod.spec.node_name = "node-1"
        self.pod.status.phase = "Running"
        self.pod.status.container_statuses = [
            NS(name="nvflare-task", state=NS(running=NS(), terminated=None, waiting=None))
        ]

    def finish(self, *, exit_code=0, reason=None, sandbox_down=True):
        self.pod.spec.node_name = "node-1"
        self.pod.status.phase = "Failed" if reason or exit_code else "Succeeded"
        self.pod.status.reason = reason
        self.pod.status.conditions = [NS(type="PodReadyToStartContainers", status="False" if sandbox_down else "True")]
        self.pod.status.container_statuses = [
            NS(name="nvflare-task", state=NS(terminated=NS(exit_code=exit_code), running=None, waiting=None))
        ]


def _bind_to_cj(launcher, request, meta=None):
    context = Mock()
    context.get_job_id.return_value = request.job_id
    context.get_identity_name.return_value = request.site_name
    context.get_prop.side_effect = lambda key, default=None: meta if key == FLContextKey.JOB_META else default
    launcher.handle_event(EventType.START_RUN, context)
    return context


@pytest.fixture
def setup(tmp_path, isolated_fobs):
    identity = TaskAttemptIdentity("job", "site-1", "task", "compute", "attempt")
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    store.create_attempt(identity)
    bootstrap = WorkerBootstrap(identity, store.root_dir, str(tmp_path), {"path": "unused.Compute"})
    path = store.bootstrap_path(identity)
    write_bootstrap(path, bootstrap)
    request = TaskLaunchRequest(
        identity.job_id,
        identity.site_name,
        identity.task_id,
        identity.attempt_id,
        ("/cj/python", "-m", WORKER_MODULE, "--bootstrap", path, "--parent_pid", "456"),
        cwd=str(tmp_path),
    )
    launcher = K8sTaskLauncher(
        namespace="site",
        image="site/image@sha256:trusted",
        workspace_root=str(tmp_path),
        workspace_pvc="existing-pvc",
        worker_service_account="task-no-api",
        cleanup_timeout=5,
        poll_interval=0.001,
        cpu_cores=1.5,
        memory_mb=512,
        gpu_count=1,
        environment={"LANG": "", "MY_SETTING": "explicit"},
    )
    _bind_to_cj(launcher, request)
    api = launcher.core_v1 = PodApi()
    return launcher, api, request, store, bootstrap


def test_manifest_identity_authority_storage_and_native_resources(setup):
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    manifest = api.manifest
    spec, metadata = manifest["spec"], manifest["metadata"]
    container = spec["containers"][0]
    assert handle.execution_id == f"k8s:site/{metadata['name']}@uid-1"
    assert metadata["namespace"] == "site"
    assert len(metadata["labels"]) == 4
    assert json.loads(metadata["annotations"]["nvflare.nvidia.com/task-identity"]) == list(request.identity)
    assert spec["automountServiceAccountToken"] is False
    assert spec["serviceAccountName"] == "task-no-api"
    assert spec["restartPolicy"] == "Never"
    assert container["command"] == ["/usr/local/bin/python"]
    assert container["args"] == [
        "-u",
        "-m",
        WORKER_MODULE,
        "--bootstrap",
        request.argv[4],
        "--execution_scope",
        "native",
    ]
    assert container["resources"] == {
        "requests": {"cpu": "1.5", "memory": "512Mi", "nvidia.com/gpu": "1"},
        "limits": {"cpu": "1.5", "memory": "512Mi", "nvidia.com/gpu": "1"},
    }
    assert {v["name"]: v["value"] for v in container["env"]} == launcher.environment
    assert spec["volumes"][0] == {"name": "workspace", "persistentVolumeClaim": {"claimName": "existing-pvc"}}
    assert "secret" in spec["volumes"][1]
    assert container["volumeMounts"][0] == {
        "name": "workspace",
        "mountPath": launcher.workspace_root,
        "subPath": task_workspace_subpath(request.job_id, request.site_name),
    }
    assert container["volumeMounts"][1]["readOnly"]
    assert container["volumeMounts"][2]["readOnly"]
    assert container["volumeMounts"][2]["mountPath"] == launcher.workspace_root + "/local"
    assert spec["securityContext"]["runAsUser"] == 1000
    assert spec["securityContext"]["fsGroup"] == 1000
    assert handle.poll().phase == TaskExecutionPhase.QUEUED


def test_identity_fenced_and_distinct_across_sites_and_attempts(setup):
    launcher, api, request, store, bootstrap = setup
    first = launcher.launch_task(request)
    with pytest.raises(TaskLauncherError, match="already owned"):
        launcher.launch_task(request)
    names = {first.pod_name}
    for identity in (
        replace(bootstrap.identity, site_name="site-2"),
        replace(bootstrap.identity, attempt_id="another"),
    ):
        other_store = FileTaskArtifactStore(store.root_dir + identity.site_name + identity.attempt_id)
        other_store.create_attempt(identity)
        path = other_store.bootstrap_path(identity)
        write_bootstrap(path, replace(bootstrap, identity=identity, artifact_root=other_store.root_dir))
        other_request = replace(
            request,
            site_name=identity.site_name,
            attempt_id=identity.attempt_id,
            argv=(*request.argv[:4], path, *request.argv[5:]),
        )
        owner = launcher
        if identity.site_name != request.site_name:
            owner = K8sTaskLauncher("site", launcher.image, launcher.workspace_root, "existing-pvc", "task-no-api")
            _bind_to_cj(owner, other_request)
            owner.core_v1 = api
        handle = owner.launch_task(other_request)
        names.add(handle.pod_name)
    assert len(names) == 3


@pytest.mark.parametrize("fault", ["cwd", "bootstrap", "workspace", "identity", "command", "native", "missing-parent"])
def test_invalid_path_or_non_cj_request_fails_before_create(setup, tmp_path, fault):
    launcher, api, request, store, bootstrap = setup
    if fault == "cwd":
        request = replace(request, cwd="/outside")
    elif fault == "bootstrap":
        request = replace(request, argv=(*request.argv[:4], "/outside/bootstrap.json", *request.argv[5:]))
    elif fault == "workspace":
        path = str(tmp_path / "wrong.json")
        write_bootstrap(path, replace(bootstrap, workspace_root="/outside"))
        request = replace(request, argv=(*request.argv[:4], path, *request.argv[5:]))
    elif fault == "identity":
        request = replace(request, task_id="other")
    elif fault == "command":
        request = replace(request, argv=("/bin/sh", "-c", "run-anything"))
    elif fault == "missing-parent":
        request = replace(request, argv=request.argv[:5])
    else:
        request = replace(request, argv=(*request.argv[:5], "--execution_scope", "native"))
    with pytest.raises(ValueError):
        launcher.launch_task(request)
    assert api.calls == []


def test_cj_request_materializes_site_image_environment_and_native_defaults(setup):
    launcher, api, request, _, _ = setup
    launcher.environment = {"SITE_SETTING": "fixed"}
    launcher.resources = TaskResourceRequest(cpu_cores=2, memory_mb=1024, gpu_count=1)
    launcher._job_binding = None
    _bind_to_cj(launcher, request)
    handle = launcher.launch_task(request)
    container = api.manifest["spec"]["containers"][0]
    assert handle.request is request
    assert container["command"] == ["/usr/local/bin/python"]
    assert "--parent_pid" not in container["args"]
    assert container["args"][-2:] == ["--execution_scope", "native"]
    assert container["env"] == [{"name": "SITE_SETTING", "value": "fixed"}]
    assert container["resources"]["requests"] == {"cpu": "2", "memory": "1024Mi", "nvidia.com/gpu": "1"}


@pytest.mark.parametrize("gpu_count", [0, 1, 2])
def test_cj_metadata_routes_cpu_or_gpu_to_task_pod_instead_of_site_default(setup, gpu_count):
    launcher, api, request, _, _ = setup
    launcher.resources = TaskResourceRequest(cpu_cores=1, memory_mb=256, gpu_count=4)
    meta = {
        "resource_spec": {"@default": {"num_of_cpus": 2, "memory": "1Gi"}},
        "launcher_spec": {"default": {"k8s": {"num_of_gpus": gpu_count}}},
    }
    launcher._job_binding = None
    context = _bind_to_cj(launcher, request, meta)
    # Multiple executor adapters share one site launcher and deliver START_RUN.
    launcher.handle_event(EventType.START_RUN, context)
    launcher.launch_task(request)
    container = api.manifest["spec"]["containers"][0]
    assert container["resources"]["requests"]["cpu"] == "2"
    assert container["resources"]["requests"]["memory"] == "1Gi"
    assert container["resources"]["requests"] == container["resources"]["limits"]
    environment = {item["name"]: item["value"] for item in container["env"]}
    if gpu_count:
        assert container["resources"]["limits"]["nvidia.com/gpu"] == str(gpu_count)
        assert "NVIDIA_VISIBLE_DEVICES" not in environment
        assert "CUDA_VISIBLE_DEVICES" not in environment
    else:
        assert "nvidia.com/gpu" not in container["resources"]["limits"]
        assert environment["NVIDIA_VISIBLE_DEVICES"] == "none"
        assert environment["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("meta_key", ["launcher_spec", "resource_spec"])
def test_native_resources_follow_receiving_site_defaults_and_preserve_quantities(meta_key):
    meta = {
        meta_key: {
            "site-1": {"k8s": {"num_of_gpus": 2, "cpu": "1500m", "cpu_request": "500m", "memory": "2Gi"}},
            "site-2": {"k8s": {"num_of_gpus": 0}},
        }
    }
    defaults = TaskResourceRequest(cpu_cores=1, memory_mb=256, gpu_count=4)
    resources = task_pod_resources(meta, "site-1", defaults)
    assert resources["requests"] == {"cpu": "500m", "memory": "2Gi", "nvidia.com/gpu": "2"}
    assert resources["limits"] == {"cpu": "1500m", "memory": "2Gi", "nvidia.com/gpu": "2"}
    assert "nvidia.com/gpu" not in task_pod_resources(meta, "site-2", defaults)["limits"]
    assert task_pod_resources(meta, "unlisted-site", defaults)["limits"]["nvidia.com/gpu"] == "4"


def test_native_default_gpu_can_be_overridden_per_site_without_affecting_other_backends():
    meta = {
        "launcher_spec": {
            "default": {"k8s": {"num_of_gpus": 2}, "docker": {"num_of_gpus": 6}},
            "site-1": {"k8s": {"num_of_gpus": 0}},
        }
    }
    assert "nvidia.com/gpu" not in task_pod_resources(meta, "site-1", TaskResourceRequest())["limits"]
    assert task_pod_resources(meta, "site-2", TaskResourceRequest())["limits"]["nvidia.com/gpu"] == "2"


@pytest.mark.parametrize(
    "native",
    [
        {"num_of_gpus": True},
        {"num_of_gpus": -1},
        {"num_of_gpus": "1"},
        {"cpu": "bad"},
        {"cpu": "NaN"},
        {"memory": -1},
        {"memory": True},
        {"cpu": "1", "cpu_request": "2"},
        {"memory": "1Gi", "memory_request": "2Gi"},
    ],
)
def test_invalid_native_resource_request_fails_before_pod_creation(native):
    with pytest.raises(ValueError):
        task_pod_resources({"launcher_spec": {"site-1": {"k8s": native}}}, "site-1", TaskResourceRequest())


def test_portable_native_conflict_does_not_silently_discard_gpu_admission():
    meta = {
        "resource_spec": {"@default": {"num_of_gpus": 1}},
        "launcher_spec": {"site-1": {"k8s": {"num_of_gpus": 0}}},
    }
    with pytest.raises(ValueError, match="conflicts"):
        task_pod_resources(meta, "site-1", TaskResourceRequest())


def test_resource_binding_cannot_be_reassigned_to_another_cj(setup):
    launcher, api, request, _, _ = setup
    context = Mock()
    context.get_job_id.return_value, context.get_identity_name.return_value = request.job_id, request.site_name
    context.get_prop.return_value = {}
    launcher.handle_event(EventType.START_RUN, context)
    context.get_identity_name.return_value = "other-site"
    with pytest.raises(ValueError, match="already bound"):
        launcher.handle_event(EventType.START_RUN, context)
    with pytest.raises(ValueError, match="does not belong"):
        launcher.launch_task(replace(request, site_name="other-site"))
    assert api.calls == []


def test_cj_request_without_resource_binding_cannot_fall_back_to_wrong_allocation(setup):
    launcher, api, request, _, _ = setup
    launcher._job_binding = None
    with pytest.raises(ValueError, match="START_RUN resource binding"):
        launcher.launch_task(request)
    assert api.calls == []


@pytest.mark.parametrize("name", ["MY_SETTING", "NVFLARE_AUTH_TOKEN", "NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"])
def test_request_cannot_override_site_worker_environment(setup, name):
    from nvflare.apis.job_launcher_spec import JobProcessEnv

    launcher, api, request, _, _ = setup
    name = JobProcessEnv.AUTH_TOKEN if name == "NVFLARE_AUTH_TOKEN" else name
    with pytest.raises(ValueError, match="environment and resources belong to site policy"):
        launcher.launch_task(replace(request, environment={name: "sensitive"}))
    assert api.calls == []


@pytest.mark.parametrize(
    "resources",
    [TaskResourceRequest(cpu_cores=4), TaskResourceRequest(memory_mb=4096), TaskResourceRequest(gpu_count=2)],
)
def test_request_cannot_override_bound_cj_resource_admission(setup, resources):
    launcher, api, request, _, _ = setup
    with pytest.raises(ValueError, match="environment and resources belong to site policy"):
        launcher.launch_task(replace(request, resources=resources))
    assert api.calls == []


def test_failed_storage_primitives_prevent_launch(setup):
    launcher, api, request, _, _ = setup
    with patch("nvflare.app_opt.task_launcher.k8s_launcher.os.link", side_effect=OSError("not supported")):
        with pytest.raises(OSError, match="not supported"):
            launcher.launch_task(request)
    assert api.calls == []
    check_posix_storage(launcher.workspace_root)


@pytest.mark.parametrize("exit_code,reason", [(0, None), (137, "Evicted"), (1, "Error")])
def test_terminal_shutdown_and_removal_required(setup, exit_code, reason):
    launcher, api, request, store, bootstrap = setup
    store.commit_result(bootstrap.identity, Shareable({"value": 2}))
    handle = launcher.launch_task(request)
    api.finish(exit_code=exit_code, reason=reason, sandbox_down=False)
    assert not handle.poll().settled
    assert api.pod is not None
    assert "patch" not in api.calls
    api.finish(exit_code=exit_code, reason=reason)
    assert not handle.poll().settled  # removing finalizer is not confirmed removal
    assert handle.poll().settled
    assert handle.poll().succeeded == (exit_code == 0 and reason is None)
    assert handle.read_logs() == "task output\n"
    assert store.read_result(bootstrap.identity)[0]["value"] == 2


def test_pending_cancellation_requires_fenced_unscheduled_observation(setup):
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    status = handle.cancel()
    assert status.settled and status.cancel_requested and not status.succeeded
    assert api.deleted
    assert api.calls.index("delete") < api.calls.index("patch")
    before = list(api.calls)
    assert handle.cancel() == status
    assert api.calls == before


def test_running_delete_ack_does_not_settle_or_remove_finalizer(setup):
    launcher, api, request, _, _ = setup
    launcher.cleanup_timeout = 0.01
    handle = launcher.launch_task(request)
    api.running()
    assert handle.poll().phase == TaskExecutionPhase.RUNNING
    with pytest.raises(TaskSettlementError) as error:
        handle.cancel()
    assert not error.value.status.settled
    assert api.pod.metadata.finalizers == [SETTLEMENT_FINALIZER]
    assert "patch" not in api.calls
    api.finish(exit_code=143)
    assert handle.wait_for_settlement(1).settled


def test_cancel_after_successful_settlement_does_not_reclassify_result(setup):
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    api.finish()
    completed = handle.wait_for_settlement(1)
    assert completed.succeeded
    calls = api.calls.copy()
    assert handle.cancel() == completed
    assert api.calls == calls


def test_completion_racing_cancellation_settles_and_preserves_output(setup):
    launcher, api, request, store, bootstrap = setup
    handle = launcher.launch_task(request)
    api.running()

    def complete_during_delete():
        store.commit_result(bootstrap.identity, Shareable({"value": 8}))
        api.finish()

    api.on_delete = complete_during_delete
    status = handle.cancel()
    assert status.settled and status.cancel_requested and not status.succeeded
    assert store.read_result(bootstrap.identity)[0]["value"] == 8


def test_pod_removal_preserves_an_open_artifact_reader(setup):
    launcher, api, request, store, bootstrap = setup
    store.commit_result(bootstrap.identity, Shareable({"value": 7}))
    path = store.attempt_dir(bootstrap.identity) + "/result.fobs"
    handle = launcher.launch_task(request)
    with open(path, "rb") as reader:
        expected = reader.read()
        reader.seek(0)
        api.finish()
        assert handle.wait_for_settlement(1).settled
        assert reader.read() == expected
        assert store.read_result(bootstrap.identity)[0]["value"] == 7


@pytest.mark.parametrize("kind", ["sidecar", "init", "ephemeral", "node-lost", "missing-status"])
def test_whole_execution_requires_shutdown_evidence(setup, kind):
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    api.finish()
    if kind == "node-lost":
        api.pod.status.reason = "NodeLost"
    elif kind == "missing-status":
        api.pod.status.container_statuses = []
    else:
        attr = {"sidecar": "containers", "init": "init_containers", "ephemeral": "ephemeral_containers"}[kind]
        getattr(api.pod.spec, attr).append(NS(name="still-running"))
    assert not handle.poll().settled
    assert "patch" not in api.calls
    with pytest.raises(TimeoutError):
        handle.wait_for_settlement(0)


@pytest.mark.parametrize("kind", ["gone", "replaced", "removed-finalizer", "api-outage"])
def test_unexplained_missing_or_changed_pod_remains_unresolved(setup, kind):
    launcher, api, request, _, _ = setup
    launcher.cleanup_timeout = 0.01
    handle = launcher.launch_task(request)
    if kind == "gone":
        api.pod = None
    elif kind == "replaced":
        api.pod.metadata.uid = "foreign-uid"
    elif kind == "removed-finalizer":
        api.pod.metadata.finalizers = []
    else:
        api.read_error = ApiError(403)
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    assert "delete" not in api.calls
    assert "patch" not in api.calls


def test_transient_finalizer_conflict_retries_without_losing_other_finalizers(setup):
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    api.finish()
    api.pod.metadata.finalizers.append("another.example/owner")
    api.patch_error = ApiError(409)
    assert not handle.poll().settled
    api.patch_error = None
    assert not handle.poll().settled
    assert api.pod.metadata.finalizers == ["another.example/owner"]
    assert not handle.poll().settled
    api.pod = None  # other controller completes cleanup
    assert handle.poll().settled


def test_log_access_does_not_read_a_replacement_pod(setup):
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    api.pod.metadata.uid = "replacement"
    assert handle.read_logs() == ""
    assert "logs" not in api.calls


@pytest.mark.parametrize("status", [403, 409, 422, 500])
def test_create_failure_preserves_cleanup_handle_and_identity(setup, status):
    launcher, api, request, _, _ = setup
    api.create_error = ApiError(status)
    with pytest.raises(TaskLaunchError) as error:
        launcher.launch_task(request)
    handle = error.value.handle
    assert handle.request is request
    assert not handle.poll().succeeded
    assert handle.poll().settled == (status != 500)
    with pytest.raises(TaskLauncherError, match="already owned"):
        launcher.launch_task(request)


def test_lost_create_response_recovers_only_nonce_owned_pod(setup):
    launcher, api, request, _, _ = setup
    create = api.create_namespaced_pod

    def accepted_but_response_lost(**kwargs):
        create(**kwargs)
        raise TimeoutError()

    api.create_namespaced_pod = accepted_but_response_lost
    with pytest.raises(TaskLaunchError) as error:
        launcher.launch_task(request)
    handle = error.value.handle
    assert handle.cancel().settled
    assert "@uid-1" in handle.execution_id


@pytest.mark.parametrize(
    "reason", ["ImagePullBackOff", "InvalidImageName", "Unschedulable", "Insufficient nvidia.com/gpu"]
)
def test_pending_failure_and_native_queue_behavior(setup, reason):
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    if reason in ("Unschedulable", "Insufficient nvidia.com/gpu"):
        api.pod.status.conditions = [NS(type="PodScheduled", status="False", reason="Unschedulable", message=reason)]
    else:
        api.pod.status.container_statuses = [
            NS(name="nvflare-task", state=NS(waiting=NS(reason=reason, message="pull failed"), terminated=None))
        ]
    status = handle.poll()
    assert status.phase == TaskExecutionPhase.QUEUED
    assert ("delete" in api.calls) == (reason != "Insufficient nvidia.com/gpu")
    if reason != "Insufficient nvidia.com/gpu":
        assert handle.wait_for_settlement(1).settled
        assert handle.poll().failure_reason


def test_pending_timeout_cleans_up_without_admitting_another_attempt(setup):
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    handle._started -= launcher.pending_timeout + 1
    assert not handle.poll().settled
    assert handle.wait_for_settlement(1).settled
    assert handle.poll().failure_reason == "task pod pending timeout"
    with pytest.raises(TaskLauncherError, match="already owned"):
        launcher.launch_task(request)


def test_unknown_pod_phase_enters_bounded_settlement_with_retained_ownership(setup):
    launcher, api, request, _, _ = setup
    launcher.cleanup_timeout = 0.01
    handle = launcher.launch_task(request)
    api.running()
    api.pod.status.phase = "Unknown"
    status = handle.poll()
    assert status.phase == TaskExecutionPhase.TERMINAL
    assert not status.settled
    assert "no confirmed runtime state" in status.failure_reason
    assert "delete" in api.calls and "patch" not in api.calls
    with pytest.raises(TaskSettlementError):
        handle.cancel()


def test_api_observation_error_is_reported_and_can_recover(setup):
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    api.read_error = ApiError(503)
    assert handle.poll().failure_reason == "pod observation failed: HTTP 503"
    api.read_error = None
    api.finish()
    assert handle.wait_for_settlement(1).succeeded


def _supervised(setup, *, completion=True, corrupt=False, abort=False):
    launcher, api, request, store, bootstrap = setup
    identity = replace(bootstrap.identity, attempt_id="supervised")
    bootstrap = replace(bootstrap, identity=identity)
    signal = Signal()

    def complete():
        if completion:
            store.commit_result(identity, Shareable({"value": 9}))
        if corrupt:
            with open(store.attempt_dir(identity) + "/result.fobs", "ab") as stream:
                stream.write(b"corrupt")
        if abort:
            signal.trigger(True)
        api.finish()

    api.on_create = complete
    supervisor = TaskSupervisor()
    outcome = supervisor.run(
        bootstrap=bootstrap,
        store=store,
        input_payload=Shareable(),
        runtime_root=str(store.root_dir),
        launcher=launcher,
        abort_signal=signal,
        request_factory=lambda actual, path: replace(
            request, attempt_id=actual.attempt_id, argv=(*request.argv[:4], path, *request.argv[5:])
        ),
        options=TaskSupervisorOptions(poll_interval=0.001, settlement_timeout=1),
    )
    return supervisor, outcome


def test_supervisor_validates_after_pod_removal_and_retains_until_receipt(setup):
    _, api, _, store, _ = setup
    supervisor, outcome = _supervised(setup)
    identity = outcome.identity
    assert api.deleted
    assert outcome.result["value"] == 9
    assert supervisor.read_saved_result(identity)["value"] == 9
    supervisor.record_receipt(identity, TaskResultReceipt.RETRY)
    assert store.read_result(identity)[0]["value"] == 9
    supervisor.record_receipt(identity, TaskResultReceipt.RECEIVED)
    with pytest.raises((FileNotFoundError, IncompleteTaskArtifactError)):
        store.read_result(identity)


def test_incomplete_result_after_pod_removal_is_not_published(setup):
    with pytest.raises(IncompleteTaskArtifactError):
        _supervised(setup, completion=False)
    assert setup[1].deleted


def test_corrupt_result_is_retained_and_not_published(setup):
    with pytest.raises(ValueError, match="checksum|size"):
        _supervised(setup, corrupt=True)
    assert setup[1].deleted
    directory = Path(setup[3].attempt_dir(replace(setup[4].identity, attempt_id="supervised")))
    assert (directory / "result.fobs").exists()
    assert (directory / "completion.json").exists()


def test_abort_during_launch_cancels_late_handle_and_retains_output(setup):
    supervisor, outcome = _supervised(setup, abort=True)
    assert outcome.aborted and outcome.result is None
    assert setup[1].deleted
    assert setup[3].read_result(outcome.identity)[0]["value"] == 9
    assert not supervisor.record_receipt(outcome.identity, TaskResultReceipt.RECEIVED)


def test_unconfirmed_resource_settlement_prevents_result_read(setup):
    launcher, api, request, store, bootstrap = setup
    launcher.cleanup_timeout = 0.01
    identity = replace(bootstrap.identity, attempt_id="unsettled")
    bootstrap = replace(bootstrap, identity=identity)
    supervisor = TaskSupervisor()

    def complete_but_keep_runtime():
        store.commit_result(identity, Shareable({"value": 9}))
        api.finish(sandbox_down=False)

    api.on_create = complete_but_keep_runtime
    store.read_result = Mock(side_effect=AssertionError("must not read before settlement"))
    with pytest.raises(TaskSettlementError):
        supervisor.run(
            bootstrap=bootstrap,
            store=store,
            input_payload=Shareable(),
            runtime_root=store.root_dir,
            launcher=launcher,
            abort_signal=Signal(),
            request_factory=lambda actual, path: replace(
                request, attempt_id=actual.attempt_id, argv=(*request.argv[:4], path, *request.argv[5:])
            ),
            options=TaskSupervisorOptions(poll_interval=0.001, settlement_timeout=0.001),
        )
    assert supervisor.settlement_unconfirmed
    store.read_result.assert_not_called()
    assert api.pod is not None


def test_native_checkpoint_survives_pod_removal_and_attempt_payload_release(setup, tmp_path):
    launcher, api, request, store, bootstrap = setup
    identity = bootstrap.identity
    directory = Path(store.attempt_dir(identity))
    (directory / "checkpoint.pending").write_bytes(b"opaque native framework checkpoint")
    checkpoint = store.stage_checkpoint(identity)
    completion = store.commit_result(
        identity, Shareable({"value": 1}), diagnostics={"checkpoint": checkpoint.to_dict()}
    )
    handle = launcher.launch_task(request)
    api.finish()
    assert handle.wait_for_settlement(1).settled
    manifest = tmp_path / "checkpoints" / "slot.checkpoint"
    manifest.parent.mkdir()
    store.commit_checkpoint(completion, str(manifest))
    committed = next(manifest.parent.glob("*.bin"))
    assert committed.stat().st_ino == (directory / "checkpoint.bin").stat().st_ino
    store.release_payloads(identity)  # caller has finished all attempt readers
    assert committed.read_bytes() == b"opaque native framework checkpoint"
    assert manifest.exists()
    later = replace(identity, attempt_id="next")
    store.create_attempt(later)
    store.prepare_checkpoint(later, str(manifest))
    isolated = Path(store.checkpoint_paths(later)["input"])
    assert isolated.read_bytes() == committed.read_bytes()
    assert isolated.stat().st_ino != committed.stat().st_ino
