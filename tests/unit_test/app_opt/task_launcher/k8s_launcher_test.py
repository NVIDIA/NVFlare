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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import TaskResultReceipt
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskLauncherError,
    TaskLaunchError,
    TaskResourceRequest,
    TaskSettlementError,
)
from nvflare.app_opt.job_launcher.k8s_launcher import K8sJobHandle, job_workspace_subpath
from nvflare.app_opt.task_launcher.k8s_launcher import (
    SETTLEMENT_FINALIZER,
    WORKER_MODULE,
    K8sTaskBackend,
    check_posix_storage,
    task_pod_resources,
)
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore, IncompleteTaskArtifactError
from nvflare.private.fed.task_worker.protocol import write_bootstrap
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor, TaskSupervisorOptions
from tests.k8s_task_test_utils import ApiError, _bind_to_cj


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
    assert spec["volumes"][1]["secret"]["secretName"] == "federation-a.site-1"
    assert container["volumeMounts"][0] == {
        "name": "workspace",
        "mountPath": launcher.workspace_root,
        "subPath": job_workspace_subpath(request.job_id, request.site_name),
    }
    assert container["volumeMounts"][1]["readOnly"]
    assert container["volumeMounts"][2]["readOnly"]
    assert container["volumeMounts"][2]["mountPath"] == launcher.workspace_root + "/local"
    assert spec["securityContext"]["runAsUser"] == 1000
    assert spec["securityContext"]["fsGroup"] == 1000
    assert handle.poll().phase == TaskExecutionPhase.QUEUED


@pytest.mark.parametrize("gpu_count", [0, 1])
@pytest.mark.parametrize("startup_failure", [False, True])
def test_task_lifecycle_is_independent_of_job_handles(setup, monkeypatch, gpu_count, startup_failure):
    from nvflare.app_opt.job_launcher.k8s_launcher import K8sJobHandle

    def unexpected_job_handle(*args, **kwargs):
        raise AssertionError("task execution must not construct or use a job handle")

    monkeypatch.setattr(K8sJobHandle, "__init__", unexpected_job_handle)
    monkeypatch.setattr(K8sJobHandle, "_get_container_waiting_failure", unexpected_job_handle)
    launcher, api, request, store, bootstrap = setup
    launcher._job_binding = None
    _bind_to_cj(launcher, request, {"launcher_spec": {"site-1": {"k8s": {"num_of_gpus": gpu_count}}}})
    handle = launcher.launch_task(request)
    if startup_failure:
        api.pod.status.container_statuses = [
            NS(name="nvflare-task", state=NS(waiting=NS(reason="ErrImagePull", message="pull failed"), terminated=None))
        ]
    else:
        store.commit_result(bootstrap.identity, Shareable({"value": 3}))
        api.finish()
    status = handle.wait_for_settlement(1)
    assert api.deleted
    assert status.succeeded == (not startup_failure)
    if startup_failure:
        assert status.failure_reason == "ErrImagePull pull failed"
    else:
        assert store.read_result(bootstrap.identity)[0]["value"] == 3


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
            owner = K8sTaskBackend(
                "site",
                launcher.image,
                launcher.workspace_root,
                "existing-pvc",
                "task-no-api",
                startup_secret_name=launcher.startup_secret_name,
            )
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
    # A failed probe must not be cached or consume the attempt identity.
    launcher.launch_task(request)
    assert api.calls == ["create"]


def test_storage_probe_is_cached_after_success_for_the_bound_cj(setup, monkeypatch):
    launcher, api, request, store, bootstrap = setup
    probe = Mock(wraps=check_posix_storage)
    monkeypatch.setattr("nvflare.app_opt.task_launcher.k8s_launcher.check_posix_storage", probe)
    handle = launcher.launch_task(request)
    api.finish()
    assert handle.wait_for_settlement(1).settled
    identity = replace(bootstrap.identity, attempt_id="second")
    store.create_attempt(identity)
    path = store.bootstrap_path(identity)
    write_bootstrap(path, replace(bootstrap, identity=identity))
    launcher.launch_task(
        replace(request, attempt_id=identity.attempt_id, argv=(*request.argv[:4], path, *request.argv[5:]))
    )
    probe.assert_called_once_with(launcher.workspace_root)


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
    assert handle.poll().retryable == (exit_code != 0)
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
    assert ("delete" in api.calls) == (kind == "removed-finalizer")
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


def test_ambiguous_create_stays_fenced_through_404_until_delayed_admission(setup, monkeypatch):
    from tests.timing_utils import ManualClock

    clock = ManualClock()
    # The deliberately tiny timeout tests policy, not host scheduling latency.
    monkeypatch.setattr(
        "nvflare.app_opt.task_launcher.k8s_launcher.time",
        NS(monotonic=clock.monotonic, sleep=clock.advance),
    )
    launcher, api, request, _, _ = setup
    create = api.create_namespaced_pod
    pending = {}
    launcher.cleanup_timeout = 0.01

    def delayed_create(**kwargs):
        pending.update(kwargs)
        raise TimeoutError("response lost before admission")

    api.create_namespaced_pod = delayed_create
    with pytest.raises(TaskLaunchError) as error:
        launcher.launch_task(request)
    handle = error.value.handle
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    assert not handle.poll().settled
    assert handle.execution_id.endswith("@unresolved")
    assert "delete" not in api.calls
    with pytest.raises(TaskLauncherError, match="already owned"):
        launcher.launch_task(request)
    # A 404 observed before the outstanding create finishes cannot prove that
    # the attempt will never run. Recover the nonce-owned Pod when admitted.
    create(**pending)
    api.running()
    api.on_delete = api.finish
    assert handle.cancel().settled
    assert "@uid-1" in handle.execution_id
    assert api.deleted


@pytest.mark.parametrize("reason", ["Unschedulable", "Insufficient nvidia.com/gpu"])
def test_pending_failure_and_native_queue_behavior(setup, reason):
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    api.pod.status.conditions = [NS(type="PodScheduled", status="False", reason="Unschedulable", message=reason)]
    status = handle.poll()
    assert status.phase == TaskExecutionPhase.QUEUED
    assert ("delete" in api.calls) == (reason != "Insufficient nvidia.com/gpu")
    if reason != "Insufficient nvidia.com/gpu":
        assert handle.wait_for_settlement(1).settled
        assert handle.poll().failure_reason


@pytest.mark.parametrize("reason", ["ImagePullBackOff", "InvalidImageName"])
@pytest.mark.parametrize("missing_evidence", ["terminal-phase", "container-termination", "sandbox-teardown"])
def test_scheduled_image_pull_failure_requires_confirmed_shutdown(setup, reason, missing_evidence):
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    api.pod.spec.node_name = "node-1"
    waiting = [NS(name="nvflare-task", state=NS(waiting=NS(reason=reason, message="pull failed"), terminated=None))]
    api.pod.status.container_statuses = waiting
    status = handle.poll()
    assert status.failure_reason == f"{reason} pull failed"
    assert not status.settled and not status.succeeded
    assert api.calls.count("delete") == 1
    assert api.pod.metadata.deletion_timestamp is not None
    assert api.pod.metadata.finalizers == [SETTLEMENT_FINALIZER]
    assert "patch" not in api.calls

    # Deletion acknowledgement and partial shutdown evidence cannot authorize
    # releasing a scheduled Pod's settlement finalizer.
    api.finish(exit_code=1, reason=reason)
    if missing_evidence == "terminal-phase":
        api.pod.status.phase = "Pending"
    elif missing_evidence == "container-termination":
        api.pod.status.container_statuses = waiting
    else:
        api.pod.status.conditions = [NS(type="PodReadyToStartContainers", status="True")]
    with pytest.raises(TimeoutError, match="pod settlement unconfirmed"):
        handle.wait_for_settlement(0)
    assert api.pod.metadata.finalizers == [SETTLEMENT_FINALIZER]
    assert "patch" not in api.calls

    api.finish(exit_code=1, reason=reason)
    # Removing the finalizer is followed by a separate observation of removal.
    assert not handle.poll().settled
    assert api.deleted
    status = handle.poll()
    assert status.settled and not status.succeeded
    assert status.failure_reason == f"{reason} pull failed"
    assert api.calls.count("delete") == 1
    assert api.calls.index("delete") < api.calls.index("patch")


def test_startup_timeout_cleans_up_without_admitting_another_attempt(setup, monkeypatch):
    from tests.timing_utils import ManualClock

    clock = ManualClock()
    monkeypatch.setattr("nvflare.app_opt.task_launcher.k8s_launcher.time", clock)
    launcher, api, request, _, _ = setup
    api.on_create = lambda: setattr(api.pod.spec, "node_name", "node-1")
    handle = launcher.launch_task(request)
    clock.advance(launcher.startup_timeout + 1)
    assert not handle.poll().settled
    assert handle.poll().failure_reason == "task pod startup timeout"
    api.finish(exit_code=143)
    assert handle.wait_for_settlement(1).settled
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


@pytest.mark.parametrize("running", [False, True])
@pytest.mark.parametrize("error", [ApiError(503), RuntimeError("transport interrupted")])
def test_api_observation_error_is_reported_and_can_recover(setup, running, error):
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    if running:
        api.running()
    phase = handle.poll().phase
    api.read_error = error
    status = handle.poll()
    detail = "HTTP 503" if isinstance(error, ApiError) else "RuntimeError"
    assert status.failure_reason == f"pod observation failed: {detail}"
    assert status.phase == phase
    assert not status.settled
    assert "delete" not in api.calls
    api.read_error = None
    assert handle.poll().failure_reason is None
    api.finish()
    assert handle.wait_for_settlement(1).succeeded


@pytest.mark.parametrize("observation_error", [ApiError(503), RuntimeError("transport interrupted")])
def test_supervisor_keeps_running_after_transient_observation_error(setup, monkeypatch, observation_error):
    launcher, api, request, store, bootstrap = setup
    bootstrap = replace(bootstrap, identity=replace(bootstrap.identity, attempt_id="api-recovery"))
    clock = [0.0]
    reads = [0]
    deleted_at = []
    launcher.cleanup_timeout = 0.1
    monkeypatch.setattr(
        "nvflare.app_opt.task_launcher.k8s_launcher.time",
        NS(monotonic=lambda: clock[0], sleep=lambda delay: clock.__setitem__(0, clock[0] + delay)),
    )
    read = api.read_namespaced_pod

    def observe(**kwargs):
        reads[0] += 1
        clock[0] += 0.02
        if reads[0] == 2:
            raise observation_error
        # Stay Running for ten settlement windows after the API recovers.
        if clock[0] >= 1.0 and api.pod is not None and api.pod.status.phase == "Running":
            store.commit_result(bootstrap.identity, Shareable({"value": 9}))
            api.finish()
        return read(**kwargs)

    api.read_namespaced_pod = observe
    api.on_create = api.running
    api.on_delete = lambda: deleted_at.append(clock[0])
    supervisor = TaskSupervisor()
    outcome = supervisor.run(
        bootstrap=bootstrap,
        store=store,
        input_payload=Shareable(),
        runtime_root=store.root_dir,
        launcher=launcher,
        abort_signal=Signal(),
        request_factory=lambda actual, path: replace(
            request, attempt_id=actual.attempt_id, argv=(*request.argv[:4], path, *request.argv[5:])
        ),
        options=TaskSupervisorOptions(poll_interval=0.001, settlement_timeout=0.1),
    )
    assert outcome.result["value"] == 9
    assert outcome.execution_status.succeeded
    assert deleted_at and min(deleted_at) >= 1.0
    assert api.deleted


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


@pytest.mark.parametrize("state", ["Pending", "Running"])
def test_parent_abort_seals_cp_task_owner_before_deleting_cj(setup, state):
    launcher, api, request, _, _ = setup
    worker = launcher.launch_task(request)
    api.running()
    parent_api = Mock()
    parent_api.read_namespaced_pod.return_value = NS(status=NS(phase=state))
    cj = K8sJobHandle(
        "cj",
        parent_api,
        {
            "name": "cj",
            "image": "site/image",
            "command": "worker",
            "module_args": {},
            "volume_mount_list": [],
            "volume_list": [],
        },
    )
    owner = cj.task_owner = Mock()
    owner.close.side_effect = worker.request_cancel
    owner.release.side_effect = worker.request_cancel
    parent_api.delete_namespaced_pod.side_effect = lambda **_: owner.release.assert_not_called()
    cj.terminate()
    parent_api.delete_namespaced_pod.assert_called_once()
    assert worker.poll().cancel_requested
    assert not worker.poll().settled
    api.finish(exit_code=143)
    assert worker.wait_for_settlement(1).settled


@pytest.mark.parametrize("pending_launch", [False, True], ids=["api-outage", "late-handle"])
def test_cj_end_run_waits_for_native_settlement_after_cleanup_or_readiness_timeout(setup, monkeypatch, pending_launch):
    import threading

    from nvflare.apis.fl_constant import ReservedKey
    from nvflare.apis.fl_context import FLContext
    from nvflare.apis.utils.event import fire_event_to_components
    from nvflare.private.fed.client.client_runner import ClientRunner
    from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
    from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor
    from nvflare.private.fed.task_worker.supervisor import _Attempt
    from nvflare.private.fed.tbi import TBI

    launcher, api, request, store, bootstrap = setup
    launcher.cleanup_timeout = 60
    launcher.poll_interval = 1
    worker = launcher.launch_task(request)
    api.running()
    adapter = TaskWorkerExecutor({}, (), TaskExecutionPolicy(launcher, poll_interval=0.5, settlement_timeout=0.01))
    adapter.system_panic = Mock()
    attempt = _Attempt(bootstrap.identity, store, store.root_dir, handle=None if pending_launch else worker)
    adapter.supervisor._active = attempt
    adapter.supervisor._attempts[bootstrap.identity] = attempt
    if pending_launch:
        adapter._done.clear()
        api.on_delete = lambda: api.finish(exit_code=143)
    else:
        api.read_error = ApiError(503)

    context = FLContext()
    engine = NS(
        new_context=lambda: context,
        get_cell=lambda: NS(decode_pass_through_topics=set()),
        fire_event=lambda event, ctx: fire_event_to_components(event, [adapter], ctx),
    )
    context.set_prop(ReservedKey.ENGINE, engine)
    runner = ClientRunner.__new__(ClientRunner)
    TBI.__init__(runner)
    runner.engine = engine
    runner.task_lock = threading.Lock()
    runner.running_tasks = {}
    runner._run_abort_requested = True
    clock = NS(now=0.0, recovered=False)
    recovery_time = 10 if pending_launch else 240

    def sleep(delay):
        clock.now += delay
        assert clock.now < 400, "CJ shutdown did not complete after API recovery"
        if clock.now >= recovery_time and not clock.recovered:
            assert adapter.supervisor.owns_execution
            assert not api.deleted
            assert api.pod.status.phase == "Running"
            clock.recovered = True
            if pending_launch:
                attempt.handle = worker
                adapter._done.set()
            else:
                assert adapter.supervisor.settlement_unconfirmed
                api.read_error = None
                api.finish(exit_code=143)

    local_time = NS(time=lambda: clock.now, monotonic=lambda: clock.now, sleep=sleep)
    for module in ("nvflare.private.fed.tbi", "nvflare.app_opt.task_launcher.k8s_launcher"):
        monkeypatch.setattr(module + ".time", local_time)
    # The executor did not previously use a clock; raising=False also permits
    # reproducing the regression against that version.
    monkeypatch.setattr("nvflare.private.fed.client.task_worker_executor.time", local_time, raising=False)
    runner.end_run_events_sequence()
    assert clock.recovered
    assert api.deleted
    assert worker.poll().settled
    assert not adapter.supervisor.owns_execution
    assert adapter._done.is_set()
    assert adapter._ending
    if not pending_launch:
        assert adapter.supervisor.settlement_unconfirmed  # Recovery cannot reopen admission or publish a result.
        adapter.system_panic.assert_called_once()
        assert Path(store.root_dir, "diagnostics.jsonl").exists()


def test_gpu_queue_wait_does_not_consume_startup_budget(setup, monkeypatch):
    from tests.timing_utils import ManualClock

    clock = ManualClock()
    monkeypatch.setattr("nvflare.app_opt.task_launcher.k8s_launcher.time", clock)
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    api.pod.status.conditions = [
        NS(type="PodScheduled", status="False", reason="Unschedulable", message="Insufficient nvidia.com/gpu")
    ]
    clock.advance(600)  # Another job legitimately keeps the GPU for ten minutes.
    status = handle.poll()
    assert status.phase == TaskExecutionPhase.QUEUED
    assert status.failure_reason is None
    assert "delete" not in api.calls

    api.pod.spec.node_name = "node-1"
    api.pod.status.conditions = [NS(type="PodScheduled", status="True")]
    assert handle.poll().failure_reason is None
    clock.advance(119)
    assert handle.poll().failure_reason is None
    assert "delete" not in api.calls
    clock.advance(1)
    assert handle.poll().failure_reason == "task pod startup timeout"
    assert "delete" in api.calls
    assert not handle.poll().settled
    api.finish(exit_code=143)
    assert handle.wait_for_settlement(1).settled
    with pytest.raises(TaskLauncherError, match="already owned"):
        launcher.launch_task(request)


@pytest.mark.parametrize("resource", ["cpu", "memory", "nvidia.com/gpu"])
@pytest.mark.parametrize("queue_timeout", [0, 10])
def test_explicit_queue_timeout_settles_unscheduled_task(setup, monkeypatch, resource, queue_timeout):
    from tests.timing_utils import ManualClock

    clock = ManualClock()
    monkeypatch.setattr("nvflare.app_opt.task_launcher.k8s_launcher.time", clock)
    launcher, api, request, _, _ = setup
    launcher.queue_timeout = queue_timeout
    handle = launcher.launch_task(request)
    api.pod.status.conditions = [
        NS(type="PodScheduled", status="False", reason="Unschedulable", message=f"Insufficient {resource}")
    ]
    if queue_timeout:
        clock.advance(queue_timeout - 1)
        assert handle.poll().failure_reason is None
        assert "delete" not in api.calls
        clock.advance(1)
    assert handle.poll().failure_reason == "task pod queue timeout"
    assert handle.wait_for_settlement(1).settled
    assert api.deleted
    with pytest.raises(TaskLauncherError, match="already owned"):
        launcher.launch_task(request)


def test_queue_limit_stops_applying_after_node_assignment(setup, monkeypatch):
    from tests.timing_utils import ManualClock

    clock = ManualClock()
    monkeypatch.setattr("nvflare.app_opt.task_launcher.k8s_launcher.time", clock)
    launcher, api, request, _, _ = setup
    launcher.queue_timeout = 5
    api.on_create = lambda: setattr(api.pod.spec, "node_name", "node-1")
    handle = launcher.launch_task(request)
    clock.advance(10)
    assert handle.poll().failure_reason is None  # Startup has its own 120-second budget.
    assert "delete" not in api.calls
    api.running()
    clock.advance(600)
    assert handle.poll().phase == TaskExecutionPhase.RUNNING
    assert handle.poll().failure_reason is None
    api.on_delete = api.finish
    assert handle.cancel().settled


def test_long_queued_task_remains_cancellable(setup, monkeypatch):
    from tests.timing_utils import ManualClock

    clock = ManualClock()
    monkeypatch.setattr("nvflare.app_opt.task_launcher.k8s_launcher.time", clock)
    launcher, api, request, _, _ = setup
    handle = launcher.launch_task(request)
    clock.advance(3600)
    assert handle.poll().failure_reason is None
    status = handle.cancel()
    assert status.settled and status.cancel_requested
    assert api.deleted


@pytest.mark.parametrize(
    "setting,value",
    [("queue_timeout", value) for value in (-1, True, "5", float("nan"), float("inf"))]
    + [("startup_timeout", value) for value in (None, 0, -1, True, "5", float("nan"), float("inf"))],
)
def test_queue_and_startup_timeouts_reject_invalid_values(tmp_path, setting, value):
    with pytest.raises(ValueError, match=setting):
        K8sTaskBackend(
            "site", "image", str(tmp_path), "pvc", "worker", startup_secret_name="startup-kit", **{setting: value}
        )


@pytest.mark.parametrize("extra", [{"license": 1}, {"mem_per_gpu_in_GiB": 8}])
def test_task_split_rejects_unenforceable_compute_admission_requirements(extra):
    meta = {"resource_spec": {"site-1": {"num_of_gpus": 1, **extra}}, "cj_resource_spec": {"site-1": {}}}
    with pytest.raises(ValueError, match="cannot enforce"):
        task_pod_resources(meta, "site-1", TaskResourceRequest())


def test_legacy_process_gpu_count_still_becomes_worker_admission_after_split():
    meta = {
        "resource_spec": {"site-1": {"process": {"num_of_gpus": 2}, "k8s": {"image": "worker"}}},
        "cj_resource_spec": {"site-1": {}},
    }
    resources = task_pod_resources(meta, "site-1", TaskResourceRequest())
    assert resources["requests"]["nvidia.com/gpu"] == "2"
    assert resources["limits"]["nvidia.com/gpu"] == "2"


def test_legacy_explicit_zero_gpu_overrides_gpu_site_defaults():
    from nvflare.apis.task_launcher_spec import TaskResourceRequest
    from nvflare.app_opt.task_launcher.k8s_launcher import task_pod_resources

    meta = {"resource_spec": {"site-1": {"process": {"num_of_gpus": 0}}}, "cj_resource_spec": {"site-1": {}}}
    resources = task_pod_resources(meta, "site-1", TaskResourceRequest(gpu_count=2))
    assert "nvidia.com/gpu" not in resources["requests"]
    assert "nvidia.com/gpu" not in resources["limits"]
