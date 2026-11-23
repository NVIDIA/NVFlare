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

"""CP host policy and Docker API contracts; no real containers or scheduler."""

import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from docker.models.containers import ContainerCollection
from requests import Request, Response
from requests.exceptions import ConnectionError, ReadTimeout

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessArgs, JobReturnCode
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher import docker_launcher as docker_module
from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher, DockerJobHandle
from nvflare.app_opt.job_launcher.slurm.task_launcher import (
    DOCKER_CJ_LAUNCHER_PATH,
    TASK_LAUNCHER_PATH,
    SlurmTaskLauncher,
    SlurmTaskManager,
)
from nvflare.app_opt.job_launcher.slurm.task_service import ENV_TASK_OWNER, ENV_TASK_TOKEN
from nvflare.private.fed.client.client_executor import launch_job
from nvflare.private.fed.client.task_execution_policy import GPU_ENVIRONMENT_NAMES, TaskExecutionPolicy
from nvflare.private.fed.client.task_runtime import _TaskExecutionState
from nvflare.private.fed.utils.job_cert_utils import JobCertError
from tests.timing_utils import ManualClock


@pytest.fixture
def profile(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    cp_root = root / "cp-view"
    cp_root.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(docker_module.DockerJobLauncher, "WORKSPACE_MOUNT", str(cp_root))
    monkeypatch.setattr(docker_module, "socket", SimpleNamespace(gethostname=lambda: "cp-container"))
    monkeypatch.setenv("NVFL_DOCKER_WORKSPACE", str(root))
    monkeypatch.setenv("NVFL_DOCKER_NETWORK", "site-net")
    for name in ("startup", "local", "job-1/app_site-1/config", "job-1/job_cert", "other-job"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for name in ("fed_client.json", "rootCA.pem", "client.key"):
        (root / "startup" / name).write_text(name)
    for name in ("job.crt", "job.key"):
        (root / "job-1/job_cert" / name).write_text(name)
    workspace = Workspace(str(cp_root), "site-1")
    app_config = root / "job-1/app_site-1/config/config_fed_client.json"
    app_config.write_text(json.dumps({"execution_lifetime": "task"}))
    site = {
        "components": [
            {
                "id": "job_launcher",
                "path": DOCKER_CJ_LAUNCHER_PATH,
                "args": {
                    "workspace": str(root),
                    "default_image": "site/coordinator:tested",
                    "network": "site-net",
                    "default_job_container_kwargs": {"nano_cpus": 1_000_000_000, "mem_limit": 1024**3},
                },
            }
        ],
        "task_execution": {
            "launcher": {
                "path": TASK_LAUNCHER_PATH,
                "args": {
                    "workspace_path": str(root),
                    "shared_filesystem": True,
                    "cgroup_enforced": True,
                    "worker_listener_host": "cp-host.example",
                    "python_path": "/shared/env/bin/python",
                    "poll_interval": 0.01,
                },
            }
        },
    }
    (root / "local/resources.json").write_text(json.dumps(site))
    launcher = ClientDockerJobLauncher(**site["components"][0]["args"])
    dc = Mock()
    dc.api.base_url = "http+docker://localhost"
    dc.api.timeout = 60
    dc.info.return_value = {"OSType": "linux", "MemoryLimit": True, "CpuCfsPeriod": True, "CpuCfsQuota": True}
    parent_container = Mock()
    parent_container.attrs = {
        "Mounts": [{"Source": str(root), "Destination": str(cp_root)}],
        "NetworkSettings": {"Networks": {"site-net": {"IPAddress": "172.20.0.2"}}},
    }
    dc.containers.get.return_value = parent_container
    launcher._docker_client = dc
    native = Mock(spec=docker_module.DockerJobHandle)
    native.poll.return_value = JobReturnCode.UNKNOWN
    native._is_cleanup_complete.side_effect = lambda: native.poll() != JobReturnCode.UNKNOWN
    native.terminate.side_effect = lambda: setattr(native.poll, "return_value", JobReturnCode.ABORTED)
    native.enter_states.return_value = True
    native.can_restart.return_value = True
    monkeypatch.setattr(docker_module, "DockerJobHandle", Mock(return_value=native))
    monkeypatch.setattr(docker_module.os, "getuid", lambda: 1001)
    monkeypatch.setattr(SlurmTaskManager, "initialize", Mock(side_effect=AssertionError("no Slurm CLI in CJ")))
    cell = SimpleNamespace(get_fqcn=lambda: "site-1", register_request_cb=Mock())
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site-1")
    context.set_prop(ReservedKey.RUN_NUM, "job-1")
    engine = SimpleNamespace(
        get_cell=lambda: cell, get_component=lambda _: None, _task_execution_state=_TaskExecutionState()
    )
    context.set_prop(ReservedKey.ENGINE, engine)
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    context.set_prop(
        FLContextKey.JOB_PROCESS_ARGS,
        {
            JobProcessArgs.EXE_MODULE: ("-m", "nvflare.private.fed.app.client.worker_process"),
            JobProcessArgs.WORKSPACE: ("-m", str(cp_root)),
            JobProcessArgs.STARTUP_DIR: ("-w", str(cp_root / "startup")),
            JobProcessArgs.PARENT_URL: ("-p", "tcp://127.0.0.1:8102"),
        },
    )
    meta = {
        "job_id": "job-1",
        "resource_spec": {"@default": {"num_of_cpus": 8, "memory": "8Gi", "num_of_gpus": 2}},
        "cj_resource_spec": {"site-1": {"num_of_cpus": 0.5, "memory": "512Mi", "num_of_gpus": 0}},
    }
    context.set_prop(FLContextKey.JOB_META, meta)
    try:
        yield SimpleNamespace(
            root=root,
            cp_root=cp_root,
            parent_container=parent_container,
            workspace=workspace,
            app_config=app_config,
            site=site,
            launcher=launcher,
            dc=dc,
            native=native,
            cell=cell,
            engine=engine,
            context=context,
            meta=meta,
        )
    finally:
        service = getattr(cell, "_slurm_task_service", None)
        if service:
            assert service.shutdown(timeout=5) == ()


def test_cp_host_container_uses_cj_budget_and_isolated_canonical_paths(profile, monkeypatch):
    p = profile
    original = copy.deepcopy(p.meta)
    for name in ("SLURM_JOB_ID", "SLURM_JWT", "DOCKER_HOST", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "parent-only")
    monkeypatch.setenv("NVFL_DOCKER_NETWORK", "unrelated-bridge")
    handle = launch_job(p.launcher, p.meta, p.context)
    (image,) = p.dc.containers.create.call_args.args
    run = p.dc.containers.create.call_args.kwargs
    assert image == "site/coordinator:tested" and run["network"] == "host"
    assert run["extra_hosts"] == {"site-1": "172.20.0.2"}
    assert (run["nano_cpus"], run["mem_limit"]) == (500_000_000, 512 * 1024**2)
    assert not run.get("device_requests") and not run.get("devices")
    assert run["cap_drop"] == ["ALL"] and run["security_opt"] == ["no-new-privileges:true"]
    env = run["environment"]
    assert not set(env).intersection({"SLURM_JOB_ID", "SLURM_JWT", "DOCKER_HOST"})
    assert all(env[name] == ("void" if name == "NVIDIA_VISIBLE_DEVICES" else "") for name in GPU_ENVIRONMENT_NAMES)
    owner = p.cell._slurm_task_service.jobs["job-1"]
    assert env[ENV_TASK_TOKEN] == owner.token and env[ENV_TASK_OWNER] == "site-1"
    assert owner.backend._binding.resources.gpus_per_node == 2
    assert owner.backend._binding.resources.cpus_per_node == 8
    assert owner.backend._binding.resources.mem_per_node == 8192
    mounts = {m["Target"]: m for m in run["mounts"]}
    assert mounts[str(p.root)]["Type"] == "tmpfs"
    job_mount = mounts[str(p.root / "job-1")]
    assert job_mount["Source"] == str(p.root / "job-1") and not job_mount["ReadOnly"]
    assert mounts[str(p.root / "local")]["ReadOnly"]
    assert all(not m["Target"].endswith(".key") for m in run["mounts"])
    assert not any("docker.sock" in m["Target"] or "other-job" in m["Target"] for m in run["mounts"])
    command = run["command"]
    assert command[command.index("-p") + 1] == "tcp://site-1:8102"
    assert str(p.root) in command and str(p.root / "startup") in command
    assert command[command.index("-m", 3) + 1] == str(p.root)
    assert p.context.get_prop(FLContextKey.JOB_PROCESS_ARGS)[JobProcessArgs.WORKSPACE][1] == str(p.cp_root)
    assert p.meta == original
    assert not handle.can_restart()  # CP host recovery is disabled for this initial profile.
    handle.terminate()
    assert owner.released and owner.closed
    p.native.terminate.assert_called_once_with()


@pytest.mark.parametrize("launch_error", ["api", "timeout", "connection"])
@pytest.mark.parametrize("cleanup_failure", ["lookup", "stop"])
@pytest.mark.parametrize("via_executor", [False, True], ids=["runtime", "executor"])
@pytest.mark.parametrize("task_mode", [False, True], ids=["resident", "task"])
def test_lost_docker_start_response_retains_owner_until_cleanup(
    profile, monkeypatch, launch_error, cleanup_failure, via_executor, task_mode
):
    p = profile
    monkeypatch.setattr(docker_module, "DockerJobHandle", DockerJobHandle)
    p.site["task_execution"]["finalization_timeout"] = 0.03
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    if not task_mode:
        p.app_config.write_text("{}")
        p.meta.pop("cj_resource_spec")
        p.meta["resource_spec"] = {"site-1": {"num_of_gpus": 0}}
    containers = ContainerCollection(p.dc)
    p.dc.containers = containers
    container = Mock()
    container.id = "coordinator-id"
    container.status = "created"
    exists = False
    allow_cleanup, cleanup_attempted = threading.Event(), threading.Event()
    lookups, published = [], []
    p.context.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, published.append, sticky=False)

    def start():
        nonlocal exists
        exists = True
        container.status = "running"
        error = {
            "api": docker_module.docker.errors.APIError,
            "timeout": ReadTimeout,
            "connection": ConnectionError,
        }[launch_error]
        raise error("Docker start response lost")

    def lookup(identity):
        if identity == "cp-container":
            return p.parent_container
        lookups.append(identity)
        if cleanup_failure == "lookup" and not allow_cleanup.is_set():
            cleanup_attempted.set()
            raise docker_module.docker.errors.APIError("Docker query unavailable")
        if not exists:
            raise docker_module.docker.errors.NotFound("auto-removed")
        return container

    def stop(**kwargs):
        nonlocal exists
        if not allow_cleanup.is_set():
            cleanup_attempted.set()
            raise docker_module.docker.errors.APIError("Docker stop unavailable")
        exists = False  # auto_remove after the successful stop

    monkeypatch.setattr(containers, "create", Mock(return_value=container))
    monkeypatch.setattr(containers, "get", Mock(side_effect=lookup))
    container.start.side_effect = start
    container.stop.side_effect = stop
    native_wait = Mock(side_effect=AssertionError("failed start must not enter unbounded Docker wait"))
    monkeypatch.setattr(DockerJobHandle, "wait", native_wait)
    if via_executor:
        from nvflare.private.fed.client.client_executor import JobExecutor, _JobRestart, _PendingJobHandle

        client = SimpleNamespace(token="token", token_signature="signature", ssid="ssid")
        executor = JobExecutor(client, p.workspace.get_startup_kit_dir())
        pending = _PendingJobHandle()
        p.context.set_prop(FLContextKey.ARGS, SimpleNamespace(set=[]), sticky=False)
        args = dict(p.context.get_prop(FLContextKey.JOB_PROCESS_ARGS))
        args.update(
            {
                JobProcessArgs.OPTIONS: ("--set", ""),
                JobProcessArgs.AUTH_TOKEN: ("-t", "token"),
                JobProcessArgs.TOKEN_SIGNATURE: ("-ts", "signature"),
                JobProcessArgs.SSID: ("-d", "ssid"),
            }
        )
        restart = _JobRestart(p.launcher, p.meta, args)
    service = None
    try:
        with pytest.raises((RuntimeError, ReadTimeout, ConnectionError), match="Docker start response lost"):
            if via_executor:
                executor._launch_job("job-1", restart, pending, p.context)
            else:
                launch_job(p.launcher, p.meta, p.context)
        if via_executor:
            if pending._job_handle is not None:
                published.append(pending._job_handle)
        native_wait.assert_not_called()
        container.start.assert_called_once_with()
        assert exists and container.status == "running"
        if not task_mode:
            # Ordinary launches have no task owner that can reconcile an
            # uncertain handle; preserve their original error-return path.
            assert not published
            assert not hasattr(p.cell, "_slurm_task_service")
            container.stop.assert_not_called()
            return
        assert len(published) == 1
        owner = published[0].owner
        service = p.cell._slurm_task_service
        assert owner.cj_handle.container_id == container.id
        assert cleanup_attempted.wait(5)
        assert lookups and set(lookups) == {container.id}
        assert not owner.cj_handle._is_cleanup_complete()
        assert owner.thread.is_alive() and service.jobs["job-1"] is owner
        assert owner.released and not owner.cj_settled

        allow_cleanup.set()
        owner.wake.set()
        owner.thread.join(5)
        assert not owner.thread.is_alive() and not service.jobs
        assert owner.cj_settled and not exists
        container.stop.assert_called()
    finally:
        allow_cleanup.set()
        service = service or getattr(p.cell, "_slurm_task_service", None)
        if service:
            assert service.shutdown(timeout=5) == ()


@pytest.mark.parametrize("appears_before_deadline", [True, False], ids=["late-create", "never-created"])
def test_timed_out_docker_create_reconciles_within_client_timeout(profile, monkeypatch, appears_before_deadline):
    p = profile
    clock = ManualClock()
    monkeypatch.setattr(docker_module, "time", clock)
    p.dc.api.timeout = 12
    p.launcher.timeout = 1
    monkeypatch.setattr(docker_module, "DockerJobHandle", DockerJobHandle)
    p.site["task_execution"]["finalization_timeout"] = 0.03
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    p.dc.containers.create.side_effect = ReadTimeout("Docker create response lost")
    appeared = threading.Event()
    daemon_unavailable = threading.Event()
    container = Mock(id="late-coordinator", status="created")
    removed = threading.Event()
    container.remove.side_effect = lambda **kwargs: removed.set()

    def lookup(identity):
        if identity == "cp-container":
            return p.parent_container
        if daemon_unavailable.is_set():
            raise ConnectionError("daemon unavailable")
        if not appeared.is_set() or removed.is_set():
            raise docker_module.docker.errors.NotFound("not visible")
        return container

    p.dc.containers.get.side_effect = lookup
    try:
        with pytest.raises(ReadTimeout, match="Docker create response lost"):
            launch_job(p.launcher, p.meta, p.context)
        service = p.cell._slurm_task_service
        owner = service.jobs["job-1"]
        handle = owner.cj_handle
        name = p.dc.containers.create.call_args.kwargs["name"]
        assert handle.container_id == name
        clock.advance(p.dc.api.timeout - 1)
        assert handle.poll() == JobReturnCode.UNKNOWN
        handle.terminate()
        assert handle.poll() == JobReturnCode.UNKNOWN
        assert owner.thread.is_alive() and service.jobs["job-1"] is owner
        container.remove.assert_not_called()

        if appears_before_deadline:
            appeared.set()
        else:
            daemon_unavailable.set()
            clock.advance(1)
            assert handle.poll() == JobReturnCode.UNKNOWN
            assert owner.thread.is_alive()
            daemon_unavailable.clear()
        owner.wake.set()
        owner.thread.join(5)
        assert owner.cj_settled and not service.jobs
        if appears_before_deadline:
            assert handle.container_id == container.id
            container.remove.assert_called_once_with(force=True)
        else:
            assert handle.container_id == name
            container.remove.assert_not_called()
        assert handle.poll() == JobReturnCode.ABORTED
        container.start.assert_not_called()
    finally:
        daemon_unavailable.clear()
        appeared.set()
        service = getattr(p.cell, "_slurm_task_service", None)
        if service is not None:
            assert service.shutdown(timeout=5) == ()


def test_not_found_from_lookup_started_before_create_deadline_does_not_settle(monkeypatch):
    clock = ManualClock()
    monkeypatch.setattr(docker_module, "time", clock)
    dc = Mock()
    dc.api.timeout = 10

    def slow_not_found(_):
        clock.advance(dc.api.timeout)
        raise docker_module.docker.errors.NotFound("not visible")

    dc.containers.get.side_effect = slow_not_found
    handle = DockerJobHandle("generated-name", "generated-name", dc, create_pending=True)

    assert handle.poll() == JobReturnCode.UNKNOWN
    assert handle.poll() == JobReturnCode.ABORTED


def test_handle_is_published_before_waiting_for_docker_startup(profile):
    p = profile
    published = []
    p.context.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, published.append)

    def started(_):
        assert len(published) == 1
        assert published[0].owner.cj_handle is p.native
        return True

    p.native.enter_states.side_effect = started
    handle = launch_job(p.launcher, p.meta, p.context)
    assert published == [handle]
    p.native.poll.return_value = JobReturnCode.SUCCESS
    assert handle.poll() == JobReturnCode.SUCCESS and handle.owner.released


def test_publication_failure_stops_cj_and_releases_task_admission(profile):
    p = profile
    p.context.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, Mock(side_effect=RuntimeError("aborted")))
    with pytest.raises(RuntimeError, match="aborted"):
        launch_job(p.launcher, p.meta, p.context)
    p.native.terminate.assert_called_once_with()
    p.native.wait.assert_not_called()  # Launch-failure cleanup is bounded by the owner deadline.
    p.native.enter_states.assert_not_called()


def test_docker_create_rejection_seals_task_owner_without_claiming_a_container(profile):
    p = profile
    owners = []
    published = Mock()
    p.context.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, published)

    def fail(*args, **kwargs):
        owners.append(p.cell._slurm_task_service.jobs["job-1"])
        response = Response()
        response.status_code = 409
        response.reason = "Conflict"
        response.request = Request("POST", "http://docker/containers/create").prepare()
        raise docker_module.docker.errors.APIError("name conflict", response=response)

    p.dc.containers.create.side_effect = fail
    with pytest.raises(RuntimeError, match="409 Client Error"):
        launch_job(p.launcher, p.meta, p.context)
    owner = owners[0]
    owner.thread.join(5)
    assert owner.released and owner.closed and not owner.thread.is_alive()
    assert not p.cell._slurm_task_service.jobs
    published.assert_not_called()
    p.native.terminate.assert_not_called()


def test_docker_proxy_start_needs_cp_capability_but_no_scheduler_environment(profile, monkeypatch):
    p = profile
    for name in ("NVFLARE_SLURM_CHILD_PROCESS", "SLURM_JOB_ID"):
        monkeypatch.delenv(name, raising=False)
    proxy = TaskExecutionPolicy.from_workspace(p.workspace).launcher
    with pytest.raises(RuntimeError, match="CP-provisioned"):
        proxy.handle_event(EventType.START_RUN, p.context)
    monkeypatch.setenv(ENV_TASK_OWNER, "site-1")
    monkeypatch.setenv(ENV_TASK_TOKEN, "job-capability")
    proxy.handle_event(EventType.START_RUN, p.context)
    assert proxy._client.owner == "site-1" and proxy._client.token == "job-capability"
    assert proxy.get_worker_listener_host() == "cp-host.example"
    SlurmTaskManager.initialize.assert_not_called()


@pytest.mark.parametrize("spec", [{"num_of_gpus": 1}, {"num_of_gpus": 1, "mem_per_gpu_in_GiB": 1}])
def test_gpu_cj_budget_is_rejected_before_creation(profile, spec):
    profile.meta["cj_resource_spec"]["site-1"] = spec
    with pytest.raises(ValueError, match="CPU-only"):
        launch_job(profile.launcher, profile.meta, profile.context)
    profile.dc.containers.create.assert_not_called()


@pytest.mark.parametrize("docker_spec", [{"image": "job/image"}, {"device_requests": []}, {"entrypoint": ["sh"]}])
def test_job_cannot_override_cp_host_container(profile, docker_spec):
    profile.meta["launcher_spec"] = {"site-1": {"docker": docker_spec}}
    with pytest.raises(ValueError, match="site-owned"):
        launch_job(profile.launcher, profile.meta, profile.context)
    profile.dc.containers.create.assert_not_called()


def test_resident_job_uses_ordinary_docker_without_task_ownership(profile):
    p = profile
    p.app_config.write_text("{}")
    p.meta.pop("cj_resource_spec")
    p.meta["resource_spec"] = {"site-1": {"num_of_gpus": 0}}
    p.meta["launcher_spec"] = {"site-1": {"docker": {"image": "resident/image"}}}
    p.meta["study"] = "resident-study"
    (p.root / "local/study_runtime.yaml").write_text(
        json.dumps({"format_version": 2, "studies": {"resident-study": {"env": {"RESIDENT_STUDY": "loaded"}}}})
    )
    handle = launch_job(p.launcher, p.meta, p.context)
    assert handle is p.native
    assert not hasattr(p.cell, "_slurm_task_service")
    assert p.dc.containers.run.call_args.args == ("resident/image",)
    run = p.dc.containers.run.call_args.kwargs
    assert run["network"] == "site-net" and "extra_hosts" not in run
    command = run["command"]
    assert command[command.index("-p") + 1] == "tcp://site-1:8102"
    assert command[command.index("-m", 3) + 1] == str(p.cp_root)
    assert str(p.cp_root) in {mount["Target"] for mount in run["mounts"]}
    assert run["environment"]["RESIDENT_STUDY"] == "loaded"


@pytest.mark.parametrize("resident_defaults", [{}, {"shm_size": "4g", "ipc_mode": "host", "runtime": "nvidia"}])
def test_coordinator_resources_are_independent_of_resident_container_defaults(profile, resident_defaults):
    p = profile
    p.launcher.default_job_container_kwargs = dict(resident_defaults)
    p.site["components"][0]["args"]["default_job_container_kwargs"] = dict(resident_defaults)
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    TaskExecutionPolicy.from_workspace(p.workspace)
    p.meta["cj_resource_spec"] = {"site-1": {"num_of_cpus": 64, "memory": "512Gi"}}
    handle = launch_job(p.launcher, p.meta, p.context)
    run = p.dc.containers.create.call_args.kwargs
    assert run["nano_cpus"] == 64_000_000_000
    assert run["mem_limit"] == 512 * 1024**3
    assert not set(resident_defaults).intersection(run)
    assert run["cap_drop"] == ["ALL"]
    assert run["security_opt"] == ["no-new-privileges:true"]
    handle.terminate()
    handle.owner.thread.join(5)
    assert not handle.owner.thread.is_alive()

    p.app_config.write_text("{}")
    p.meta.pop("cj_resource_spec")
    p.meta["resource_spec"] = {}
    launch_job(p.launcher, p.meta, p.context)
    run = p.dc.containers.run.call_args.kwargs
    assert "nano_cpus" not in run and "mem_limit" not in run
    assert {key: run[key] for key in resident_defaults} == resident_defaults
    assert "cap_drop" not in run and "security_opt" not in run
    assert p.launcher.default_job_container_kwargs == resident_defaults


@pytest.mark.parametrize("mutation", ["host-cp", "wrong-mount", "missing-network"])
def test_docker_task_profile_requires_owning_docker_cp(profile, monkeypatch, mutation):
    p = profile
    if mutation == "host-cp":
        monkeypatch.delenv("NVFL_DOCKER_WORKSPACE")
    elif mutation == "wrong-mount":
        p.parent_container.attrs["Mounts"] = []
    else:
        p.parent_container.attrs["NetworkSettings"]["Networks"] = {}
    with pytest.raises(ValueError, match="Docker CP|CP container"):
        launch_job(p.launcher, p.meta, p.context)
    p.dc.containers.create.assert_not_called()


@pytest.mark.parametrize("explicit_null", [False, True])
def test_cj_validates_ordinary_docker_site_without_parent_environment(profile, monkeypatch, explicit_null):
    p = profile
    p.site["components"][0]["args"].pop("workspace")
    if explicit_null:
        p.site["components"][0]["args"]["workspace"] = None
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    monkeypatch.delenv("NVFL_DOCKER_WORKSPACE")
    monkeypatch.delenv("NVFL_DOCKER_NETWORK")
    policy = TaskExecutionPolicy.from_workspace(p.workspace)
    assert policy.launcher.cj_runtime == "docker"
    p.dc.containers.create.assert_not_called()


def test_secure_cj_requires_own_job_certificate_and_preserves_mtls(profile):
    p = profile
    p.context.set_prop(FLContextKey.SECURE_MODE, True)
    args = p.context.get_prop(FLContextKey.JOB_PROCESS_ARGS)
    args[JobProcessArgs.PARENT_URL] = ("-p", "stcp://127.0.0.1:8102")
    args[JobProcessArgs.PARENT_CONN_SEC] = ("--parent_conn_sec", "mtls")
    (p.root / "job-1/job_cert/job.key").unlink()
    with pytest.raises(JobCertError):
        launch_job(p.launcher, p.meta, p.context)
    p.dc.containers.create.assert_not_called()
    (p.root / "job-1/job_cert/job.key").write_text("job.key")
    # Release the failed launch's owner before another explicit launch.
    p.cell._slurm_task_service.shutdown(timeout=5)
    launch_job(p.launcher, p.meta, p.context)
    command = p.dc.containers.create.call_args.kwargs["command"]
    assert command[command.index("-p") + 1] == "stcp://site-1:8102"
    assert command[command.index("--parent_conn_sec") + 1] == "mtls"


def test_live_owner_fences_a_second_cj_launch(profile):
    p = profile
    handle = launch_job(p.launcher, p.meta, p.context)
    with pytest.raises(ValueError, match="already owns"):
        launch_job(p.launcher, p.meta, p.context)
    assert p.dc.containers.create.call_count == 1
    assert not handle.owner.closed
    assert not handle.can_restart()


def test_cp_host_restart_policy_preserves_owner_even_after_physical_exit(profile):
    p = profile
    handle = launch_job(p.launcher, p.meta, p.context)
    assert not handle.can_restart()
    p.native.can_restart.assert_not_called()
    assert not handle.owner.released


def test_heartbeat_cleanup_stops_docker_cj_and_seals_task_owner(profile):
    p = profile
    handle = launch_job(p.launcher, p.meta, p.context)
    assert not hasattr(p.native, "_terminate_for_heartbeat_cleanup")
    handle._terminate_for_heartbeat_cleanup()
    assert handle.owner.closed and handle.owner.released
    p.native.terminate.assert_called_once_with()


def test_cp_host_admission_does_not_reserve_compute_gpus(profile):
    p = profile
    resources = {}
    p.context.set_prop(FLContextKey.CLIENT_RESOURCE_SPECS, resources)
    p.launcher.handle_event(EventType.BEFORE_CHECK_RESOURCE_MANAGER, p.context)
    assert resources == {} and p.context.get_prop(FLContextKey.JOB_BLOCK_REASON) is None
    p.meta["cj_resource_spec"]["site-1"]["num_of_gpus"] = 1
    p.launcher.handle_event(EventType.BEFORE_CHECK_RESOURCE_MANAGER, p.context)
    with pytest.raises(ValueError, match="CPU-only"):
        launch_job(p.launcher, p.meta, p.context)
    p.dc.containers.create.assert_not_called()


def test_root_cp_is_rejected(profile, monkeypatch):
    monkeypatch.setattr(docker_module.os, "getuid", lambda: 0)
    with pytest.raises(ValueError, match="Docker CP with a nonroot"):
        launch_job(profile.launcher, profile.meta, profile.context)
    profile.dc.containers.create.assert_not_called()


@pytest.mark.parametrize(
    "host", [None, "127.0.0.1", "localhost", "test.localhost.", "0.0.0.0", "tcp://CP host:1", "CP host:1"]
)
def test_docker_requires_explicit_worker_reachable_listener(profile, host):
    args = {**profile.site["task_execution"]["launcher"]["args"], "worker_listener_host": host}
    with pytest.raises((ValueError, RuntimeError)):
        SlurmTaskLauncher(**args).validate_site_config(profile.site)


@pytest.mark.parametrize("mutation", ["combined-launcher", "wrong-workspace", "mixed-launchers"])
def test_canonical_site_profile_rejects_incompatible_launchers(profile, mutation):
    site = copy.deepcopy(profile.site)
    if mutation == "combined-launcher":
        site["components"][0][
            "path"
        ] = "nvflare.app_opt.job_launcher.slurm.docker_launcher.ClientDockerSlurmJobLauncher"
    elif mutation == "wrong-workspace":
        site["components"][0]["args"]["workspace"] = str(profile.root / "other-job")
    else:
        site["components"].append({"path": "nvflare.app_common.job_launcher.process_launcher.ProcessJobLauncher"})
    with pytest.raises(ValueError):
        TaskExecutionPolicy.from_config(site["task_execution"]).launcher.validate_site_config(site)


def test_task_policy_selects_the_ordinary_client_docker_launcher(profile):
    server = docker_module.ServerDockerJobLauncher(workspace=str(profile.root))
    with pytest.raises(ValueError, match="unsupported client-job/task launcher pairing"):
        launch_job(server, profile.meta, profile.context)
    profile.dc.containers.create.assert_not_called()
    launch_job(profile.launcher, profile.meta, profile.context)


@pytest.mark.parametrize("unsupported", ["remote", "MemoryLimit", "CpuCfsQuota", "CpuCfsPeriod"])
def test_unsupported_daemon_is_rejected_before_container_creation(profile, unsupported):
    p = profile
    if unsupported == "remote":
        p.dc.api.base_url = "http://remote:2375"
    else:
        p.dc.info.return_value[unsupported] = False
    with pytest.raises(ValueError, match="daemon"):
        launch_job(p.launcher, p.meta, p.context)
    p.dc.containers.create.assert_not_called()


def test_nonstandard_cpu_capability_keys_cannot_satisfy_daemon_check(profile):
    p = profile
    p.dc.info.return_value = {"OSType": "linux", "MemoryLimit": True, "CPUCfsPeriod": True, "CPUCfsQuota": True}
    with pytest.raises(ValueError, match="daemon"):
        launch_job(p.launcher, p.meta, p.context)
    p.dc.containers.create.assert_not_called()


@pytest.mark.parametrize(
    "spec, expected",
    [
        ({}, (None, None)),
        ({"num_of_cpus": 0.75}, (750_000_000, None)),
        ({"memory": "256Mi"}, (None, 256 * 1024**2)),
    ],
)
def test_cj_defaults_are_independent_of_compute(profile, spec, expected):
    p = profile
    p.meta["cj_resource_spec"]["site-1"] = spec
    launch_job(p.launcher, p.meta, p.context)
    run = p.dc.containers.create.call_args.kwargs
    assert (run.get("nano_cpus"), run.get("mem_limit")) == expected


def test_cp_owns_a_physical_backend_that_reaches_native_task_launch(profile, monkeypatch):
    from tests.slurm_task_binding_test_utils import assert_physical_task_dispatch

    p = profile
    handle = launch_job(p.launcher, p.meta, p.context)
    assert_physical_task_dispatch(handle, p.root, monkeypatch)


@pytest.mark.parametrize(
    "budget, key, value",
    [({"num_of_cpus": 64}, "nano_cpus", 64_000_000_000), ({"memory": "512Gi"}, "mem_limit", 512 * 1024**3)],
)
def test_cj_budget_is_not_capped_by_resident_defaults(profile, budget, key, value):
    profile.meta["cj_resource_spec"]["site-1"] = budget
    launch_job(profile.launcher, profile.meta, profile.context)
    assert profile.dc.containers.create.call_args.kwargs[key] == value


@pytest.mark.parametrize(
    "resident_security",
    [["seccomp=unconfined"], ["no-new-privileges:false"], ["seccomp=/site/seccomp.json", "apparmor=site-profile"]],
)
def test_coordinator_security_does_not_inherit_resident_defaults(profile, resident_security):
    p = profile
    p.launcher.default_job_container_kwargs.update(cap_drop=["NET_RAW"], security_opt=resident_security)
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    TaskExecutionPolicy.from_workspace(p.workspace)
    handle = launch_job(p.launcher, p.meta, p.context)
    run = p.dc.containers.create.call_args.kwargs
    assert run["cap_drop"] == ["ALL"]
    assert run["security_opt"] == ["no-new-privileges:true"]
    handle.terminate()
    handle.owner.thread.join(5)
    assert not handle.owner.thread.is_alive()

    p.app_config.write_text("{}")
    p.meta.pop("cj_resource_spec")
    p.meta["resource_spec"] = {}
    launch_job(p.launcher, p.meta, p.context)
    run = p.dc.containers.run.call_args.kwargs
    assert run["cap_drop"] == ["NET_RAW"]
    assert run["security_opt"] == resident_security


def test_docker_memory_string_is_valid_at_policy_load_and_launch(profile):
    p = profile
    p.site["components"][0]["args"]["default_job_container_kwargs"]["mem_limit"] = "1g"
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    TaskExecutionPolicy.from_workspace(p.workspace)
    p.launcher.default_job_container_kwargs["mem_limit"] = "1g"
    p.meta["cj_resource_spec"]["site-1"] = {}
    launch_job(p.launcher, p.meta, p.context)
    assert "mem_limit" not in p.dc.containers.create.call_args.kwargs


def test_federation_docker_default_does_not_replace_site_coordinator(profile):
    p = profile
    p.meta["launcher_spec"] = {"default": {"docker": {"image": "federation/compute"}}, "site-1": {"docker": {}}}
    launch_job(p.launcher, p.meta, p.context)
    assert p.dc.containers.create.call_args.args == ("site/coordinator:tested",)
    assert p.meta["launcher_spec"]["default"]["docker"]["image"] == "federation/compute"


@pytest.mark.parametrize("url", ["tcp://host", "file:///tmp/parent"])
def test_cp_host_rejects_unconnectable_parent_address(profile, url):
    p = profile
    p.context.get_prop(FLContextKey.JOB_PROCESS_ARGS)[JobProcessArgs.PARENT_URL] = ("-p", url)
    with pytest.raises(ValueError, match="parent"):
        launch_job(p.launcher, p.meta, p.context)
    p.dc.containers.create.assert_not_called()


def test_docker_parent_wildcard_listener_uses_cp_container_address(profile):
    p = profile
    p.context.get_prop(FLContextKey.JOB_PROCESS_ARGS)[JobProcessArgs.PARENT_URL] = ("-p", "tcp://0.0.0.0:8102")
    launch_job(p.launcher, p.meta, p.context)
    run = p.dc.containers.create.call_args.kwargs
    assert run["command"][run["command"].index("-p") + 1] == "tcp://site-1:8102"
    assert run["extra_hosts"] == {"site-1": "172.20.0.2"}


@pytest.mark.parametrize("creation_fails", [False, True])
def test_shutdown_retains_inflight_creation_and_cleans_late_container(profile, creation_fails):
    import threading

    from nvflare.private.fed.client.task_runtime import shutdown_task_execution
    from tests.timing_utils import ThreadGroup, join_thread

    p = profile
    entered, finish = threading.Event(), threading.Event()
    published = Mock()
    p.context.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, published)
    p.site["task_execution"]["finalization_timeout"] = 0.01
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    original_create = p.dc.containers.create.return_value

    def create(*args, **kwargs):
        entered.set()
        finish.wait()
        if creation_fails:
            raise RuntimeError("creation failed")
        return original_create

    p.dc.containers.create.side_effect = create

    def launch():
        with pytest.raises(RuntimeError, match="creation failed" if creation_fails else "shut down during"):
            launch_job(p.launcher, p.meta, p.context)

    with ThreadGroup(finish) as threads:
        worker = threads.start(launch)
        assert entered.wait(5), "Docker creation did not enter"
        service = p.cell._slurm_task_service
        owner = service.jobs["job-1"]
        shutdown_task_execution(p.engine)
        assert owner.released and owner.cj_launch_pending
        assert owner.thread.is_alive() and service.jobs["job-1"] is owner
        finish.set()
        join_thread(worker)
        join_thread(owner.thread)
        assert "job-1" not in service.jobs
    published.assert_called_once()
    wrapper = published.call_args.args[0]
    assert wrapper.owner is owner and wrapper.handle is p.native
    p.native.terminate.assert_called_once()
    p.native.wait.assert_not_called()


def test_system_shutdown_terminates_already_attached_coordinator(profile):
    from nvflare.private.fed.client.task_runtime import shutdown_task_execution

    p = profile
    handle = launch_job(p.launcher, p.meta, p.context)
    shutdown_task_execution(p.engine)
    p.native.terminate.assert_called_once()
    assert handle.owner.closed and not handle.owner.thread.is_alive()


@pytest.mark.parametrize("cpus", [0.067, 8.3])
def test_fractional_cj_budget_uses_nanocpu_quotas(profile, cpus):
    p = profile
    p.meta["cj_resource_spec"]["site-1"] = {"num_of_cpus": cpus}
    launch_job(p.launcher, p.meta, p.context)
    assert p.dc.containers.create.call_args.kwargs["nano_cpus"] == round(cpus * 1_000_000_000)


def test_optional_null_launcher_spec_keeps_site_default_coordinator(profile):
    p = profile
    p.meta["launcher_spec"] = None
    launch_job(p.launcher, p.meta, p.context)
    assert p.dc.containers.create.call_args.args == ("site/coordinator:tested",)
