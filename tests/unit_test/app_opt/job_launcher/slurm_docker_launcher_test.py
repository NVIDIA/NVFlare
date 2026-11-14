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

"""Gateway policy and Docker API contracts; no real containers or scheduler."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessArgs, JobReturnCode
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher import docker_launcher as docker_module
from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher
from nvflare.app_opt.job_launcher.slurm import gateway_runtime as gateway_module
from nvflare.app_opt.job_launcher.slurm.task_launcher import SlurmTaskLauncher, SlurmTaskManager
from nvflare.app_opt.job_launcher.slurm.task_policy import DOCKER_CJ_LAUNCHER_PATH, TASK_LAUNCHER_PATH
from nvflare.app_opt.job_launcher.slurm.task_service import ENV_TASK_OWNER, ENV_TASK_TOKEN
from nvflare.private.fed.client.task_execution_policy import GPU_ENVIRONMENT_NAMES, TaskExecutionPolicy
from nvflare.private.fed.client.task_runtime import launch_job
from nvflare.private.fed.utils.job_cert_utils import JobCertError


@pytest.fixture
def profile(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    for name in ("startup", "local", "job-1/app_site-1/config", "job-1/job_cert", "other-job"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for name in ("fed_client.json", "rootCA.pem", "client.key"):
        (root / "startup" / name).write_text(name)
    for name in ("job.crt", "job.key"):
        (root / "job-1/job_cert" / name).write_text(name)
    workspace = Workspace(str(root), "site-1")
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
                    "network": "host",
                    "default_job_container_kwargs": {"nano_cpus": 1_000_000_000, "mem_limit": 1024**3},
                },
            }
        ],
        "task_execution": {
            "launcher": {
                "path": TASK_LAUNCHER_PATH,
                "args": {
                    "workspace_path": str(root),
                    "cj_runtime": "docker",
                    "shared_filesystem": True,
                    "cgroup_enforced": True,
                    "worker_listener_host": "gateway.example",
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
    dc.info.return_value = {"OSType": "linux", "MemoryLimit": True, "CpuCfsPeriod": True, "CpuCfsQuota": True}
    launcher._docker_client = dc
    native = Mock(spec=docker_module.DockerJobHandle)
    native.poll.return_value = JobReturnCode.UNKNOWN
    native.is_cleanup_complete.side_effect = lambda: native.poll() != JobReturnCode.UNKNOWN
    native.terminate.side_effect = lambda: setattr(native.poll, "return_value", JobReturnCode.ABORTED)
    native.enter_states.return_value = True
    native.can_restart.return_value = True
    monkeypatch.setattr(docker_module, "DockerJobHandle", Mock(return_value=native))
    monkeypatch.setattr(gateway_module.sys, "platform", "linux")
    monkeypatch.setattr(gateway_module.os, "getuid", lambda: 1001)
    monkeypatch.setattr(SlurmTaskManager, "initialize", Mock(side_effect=AssertionError("no Slurm CLI in CJ")))
    cell = SimpleNamespace(get_fqcn=lambda: "site-1", register_request_cb=Mock())
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site-1")
    context.set_prop(ReservedKey.RUN_NUM, "job-1")
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_cell=lambda: cell, get_component=lambda _: None))
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    context.set_prop(
        FLContextKey.JOB_PROCESS_ARGS,
        {
            JobProcessArgs.EXE_MODULE: ("-m", "nvflare.private.fed.app.client.worker_process"),
            JobProcessArgs.WORKSPACE: ("-m", str(root)),
            JobProcessArgs.STARTUP_DIR: ("-w", str(root / "startup")),
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
            workspace=workspace,
            app_config=app_config,
            site=site,
            launcher=launcher,
            dc=dc,
            native=native,
            cell=cell,
            context=context,
            meta=meta,
        )
    finally:
        service = getattr(cell, "_slurm_task_service", None)
        if service:
            assert service.shutdown(timeout=5) == ()


def test_gateway_container_uses_cj_budget_and_isolated_canonical_paths(profile, monkeypatch):
    p = profile
    original = copy.deepcopy(p.meta)
    for name in ("SLURM_JOB_ID", "SLURM_JWT", "DOCKER_HOST", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "parent-only")
    monkeypatch.setenv("NVFL_DOCKER_NETWORK", "unrelated-bridge")
    handle = launch_job(p.launcher, p.meta, p.context)
    (image,) = p.dc.containers.run.call_args.args
    run = p.dc.containers.run.call_args.kwargs
    assert image == "site/coordinator:tested" and run["network"] == "host"
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
    assert command[command.index("-p") + 1] == "tcp://127.0.0.1:8102"
    assert str(p.root) in command and str(p.root / "startup") in command
    assert p.meta == original
    assert not handle.can_restart()  # Gateway recovery is disabled for this initial profile.
    handle.terminate()
    assert owner.released and owner.closed
    p.native.terminate.assert_called_once_with()


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


def test_docker_creation_failure_seals_task_owner(profile):
    p = profile
    owners = []

    def fail(*args, **kwargs):
        owners.append(p.cell._slurm_task_service.jobs["job-1"])
        raise RuntimeError("daemon unavailable")

    p.dc.containers.run.side_effect = fail
    with pytest.raises(RuntimeError, match="daemon unavailable"):
        launch_job(p.launcher, p.meta, p.context)
    assert owners[0].released and owners[0].closed


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
    assert proxy.get_worker_listener_host() == "gateway.example"
    SlurmTaskManager.initialize.assert_not_called()


@pytest.mark.parametrize("spec", [{"num_of_gpus": 1}, {"num_of_gpus": 1, "mem_per_gpu_in_GiB": 1}])
def test_gpu_cj_budget_is_rejected_before_creation(profile, spec):
    profile.meta["cj_resource_spec"]["site-1"] = spec
    with pytest.raises(ValueError, match="CPU-only"):
        launch_job(profile.launcher, profile.meta, profile.context)
    profile.dc.containers.run.assert_not_called()


@pytest.mark.parametrize("docker_spec", [{"image": "job/image"}, {"device_requests": []}, {"entrypoint": ["sh"]}])
def test_job_cannot_override_gateway_container(profile, docker_spec):
    profile.meta["launcher_spec"] = {"site-1": {"docker": docker_spec}}
    with pytest.raises(ValueError, match="site-owned"):
        launch_job(profile.launcher, profile.meta, profile.context)
    profile.dc.containers.run.assert_not_called()


def test_resident_job_uses_ordinary_docker_without_task_ownership(profile):
    p = profile
    p.app_config.write_text("{}")
    p.site["task_execution"]["allow_job_lifetime"] = True
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    p.meta.pop("cj_resource_spec")
    p.meta["resource_spec"] = {"site-1": {"num_of_gpus": 0}}
    p.meta["launcher_spec"] = {"site-1": {"docker": {"image": "resident/image"}}}
    handle = launch_job(p.launcher, p.meta, p.context)
    assert handle is p.native
    assert not hasattr(p.cell, "_slurm_task_service")
    assert p.dc.containers.run.call_args.args == ("resident/image",)


def test_secure_cj_requires_own_job_certificate_and_preserves_mtls(profile):
    p = profile
    p.context.set_prop(FLContextKey.SECURE_MODE, True)
    args = p.context.get_prop(FLContextKey.JOB_PROCESS_ARGS)
    args[JobProcessArgs.PARENT_URL] = ("-p", "stcp://127.0.0.1:8102")
    args[JobProcessArgs.PARENT_CONN_SEC] = ("--parent_conn_sec", "mtls")
    (p.root / "job-1/job_cert/job.key").unlink()
    with pytest.raises(JobCertError):
        launch_job(p.launcher, p.meta, p.context)
    p.dc.containers.run.assert_not_called()
    (p.root / "job-1/job_cert/job.key").write_text("job.key")
    # Release the failed launch's owner before another explicit launch.
    p.cell._slurm_task_service.shutdown(timeout=5)
    launch_job(p.launcher, p.meta, p.context)
    command = p.dc.containers.run.call_args.kwargs["command"]
    assert command[command.index("-p") + 1] == "stcp://127.0.0.1:8102"
    assert command[command.index("--parent_conn_sec") + 1] == "mtls"


def test_live_owner_fences_a_second_cj_launch(profile):
    p = profile
    handle = launch_job(p.launcher, p.meta, p.context)
    with pytest.raises(ValueError, match="already owns"):
        launch_job(p.launcher, p.meta, p.context)
    assert p.dc.containers.run.call_count == 1
    assert not handle.owner.closed
    assert not handle.can_restart()


def test_gateway_restart_policy_preserves_owner_even_after_physical_exit(profile):
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


def test_gateway_admission_does_not_reserve_compute_gpus(profile):
    p = profile
    resources = {}
    p.context.set_prop(FLContextKey.CLIENT_RESOURCE_SPECS, resources)
    p.launcher.handle_event(EventType.BEFORE_CHECK_RESOURCE_MANAGER, p.context)
    assert resources == {} and p.context.get_prop(FLContextKey.JOB_BLOCK_REASON) is None
    p.meta["cj_resource_spec"]["site-1"]["num_of_gpus"] = 1
    p.launcher.handle_event(EventType.BEFORE_CHECK_RESOURCE_MANAGER, p.context)
    with pytest.raises(ValueError, match="CPU-only"):
        launch_job(p.launcher, p.meta, p.context)
    p.dc.containers.run.assert_not_called()


@pytest.mark.parametrize("platform, uid", [("darwin", 1001), ("linux", 0)])
def test_unsupported_gateway_host_is_rejected(profile, monkeypatch, platform, uid):
    monkeypatch.setattr(gateway_module.sys, "platform", platform)
    monkeypatch.setattr(gateway_module.os, "getuid", lambda: uid)
    with pytest.raises(ValueError, match="Linux host networking and a nonroot"):
        launch_job(profile.launcher, profile.meta, profile.context)
    profile.dc.containers.run.assert_not_called()


@pytest.mark.parametrize(
    "host", [None, "127.0.0.1", "localhost", "test.localhost.", "0.0.0.0", "tcp://gateway:1", "gateway:1"]
)
def test_docker_requires_explicit_worker_reachable_listener(profile, host):
    args = {**profile.site["task_execution"]["launcher"]["args"], "worker_listener_host": host}
    with pytest.raises((ValueError, RuntimeError)):
        SlurmTaskLauncher(**args)


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
    policy = TaskExecutionPolicy.from_workspace(profile.workspace)
    policy.validate_job_launcher(profile.launcher)
    with pytest.raises(ValueError, match="supported"):
        policy.validate_job_launcher(docker_module.ServerDockerJobLauncher(workspace=str(profile.root)))


@pytest.mark.parametrize("unsupported", ["remote", "windows", "MemoryLimit", "CpuCfsQuota", "CpuCfsPeriod"])
def test_unsupported_daemon_is_rejected_before_container_creation(profile, unsupported):
    p = profile
    if unsupported == "remote":
        p.dc.api.base_url = "http://remote:2375"
    elif unsupported == "windows":
        p.dc.info.return_value["OSType"] = "windows"
    else:
        p.dc.info.return_value[unsupported] = False
    with pytest.raises(ValueError, match="daemon"):
        launch_job(p.launcher, p.meta, p.context)
    p.dc.containers.run.assert_not_called()


def test_nonstandard_cpu_capability_keys_cannot_satisfy_daemon_check(profile):
    p = profile
    p.dc.info.return_value = {"OSType": "linux", "MemoryLimit": True, "CPUCfsPeriod": True, "CPUCfsQuota": True}
    with pytest.raises(ValueError, match="daemon"):
        launch_job(p.launcher, p.meta, p.context)
    p.dc.containers.run.assert_not_called()


@pytest.mark.parametrize(
    "spec, expected",
    [
        ({}, (1_000_000_000, 1024**3)),
        ({"num_of_cpus": 0.75}, (750_000_000, 1024**3)),
        ({"memory": "256Mi"}, (1_000_000_000, 256 * 1024**2)),
    ],
)
def test_cj_defaults_are_independent_of_compute(profile, spec, expected):
    p = profile
    p.meta["cj_resource_spec"]["site-1"] = spec
    launch_job(p.launcher, p.meta, p.context)
    run = p.dc.containers.run.call_args.kwargs
    assert (run["nano_cpus"], run["mem_limit"]) == expected


def test_cp_owns_a_physical_backend_that_reaches_native_task_launch(profile, monkeypatch):
    from tests.slurm_gateway_test_utils import assert_physical_task_dispatch

    p = profile
    handle = launch_job(p.launcher, p.meta, p.context)
    assert_physical_task_dispatch(handle, p.root, monkeypatch)


@pytest.mark.parametrize("budget", [{"num_of_cpus": 64}, {"memory": "512Gi"}])
def test_job_cannot_raise_site_coordinator_ceilings(profile, budget):
    profile.meta["cj_resource_spec"]["site-1"] = budget
    with pytest.raises(ValueError, match="ceiling"):
        launch_job(profile.launcher, profile.meta, profile.context)
    profile.dc.containers.run.assert_not_called()


def test_site_security_profiles_survive_mandatory_restrictions(profile):
    p = profile
    p.launcher.default_job_container_kwargs.update(
        cap_drop=["NET_RAW"], security_opt=["seccomp=/site/seccomp.json", "apparmor=site-profile"]
    )
    launch_job(p.launcher, p.meta, p.context)
    run = p.dc.containers.run.call_args.kwargs
    assert run["cap_drop"] == ["NET_RAW", "ALL"]
    assert run["security_opt"] == ["seccomp=/site/seccomp.json", "apparmor=site-profile", "no-new-privileges:true"]


@pytest.mark.parametrize(
    "settings, error",
    [
        ({"nano_cpus": 0}, "nano_cpus"),
        ({"nano_cpus": True}, "nano_cpus"),
        ({"mem_limit": "bad"}, "mem_limit"),
        ({"mem_limit": False}, "mem_limit"),
        ({"mem_limit": -1}, "mem_limit"),
        ({"privileged": True}, "unsupported"),
        ({"cap_drop": "ALL"}, "cap_drop"),
        ({"security_opt": ["no-new-privileges:false"]}, "cannot disable"),
    ],
)
def test_site_container_mistakes_are_rejected_when_loading_policy(profile, settings, error):
    p = profile
    p.site["components"][0]["args"]["default_job_container_kwargs"].update(settings)
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    with pytest.raises(ValueError, match=error):
        TaskExecutionPolicy.from_workspace(p.workspace)
    p.dc.info.assert_not_called()
    p.dc.containers.run.assert_not_called()


def test_docker_memory_string_is_valid_at_policy_load_and_launch(profile):
    p = profile
    p.site["components"][0]["args"]["default_job_container_kwargs"]["mem_limit"] = "1g"
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    TaskExecutionPolicy.from_workspace(p.workspace)
    p.launcher.default_job_container_kwargs["mem_limit"] = "1g"
    p.meta["cj_resource_spec"]["site-1"] = {}
    launch_job(p.launcher, p.meta, p.context)
    assert p.dc.containers.run.call_args.kwargs["mem_limit"] == 1024**3


def test_federation_docker_default_does_not_replace_site_coordinator(profile):
    p = profile
    p.meta["launcher_spec"] = {"default": {"docker": {"image": "federation/compute"}}, "site-1": {"docker": {}}}
    launch_job(p.launcher, p.meta, p.context)
    assert p.dc.containers.run.call_args.args == ("site/coordinator:tested",)
    assert p.meta["launcher_spec"]["default"]["docker"]["image"] == "federation/compute"


@pytest.mark.parametrize("image", [None, "resident/image"])
def test_gateway_refuses_resident_jobs_unless_site_opts_in(profile, image):
    p = profile
    p.app_config.write_text("{}")
    p.meta.pop("cj_resource_spec")
    if image:
        p.meta["launcher_spec"] = {"site-1": {"docker": {"image": image}}}
    with pytest.raises(ValueError, match="refuses job-lifetime"):
        launch_job(p.launcher, p.meta, p.context)
    p.dc.containers.run.assert_not_called()


@pytest.mark.parametrize("url", ["tcp://0.0.0.0:8102", "tcp://host", "file:///tmp/parent"])
def test_gateway_rejects_unconnectable_parent_address(profile, url):
    p = profile
    p.context.get_prop(FLContextKey.JOB_PROCESS_ARGS)[JobProcessArgs.PARENT_URL] = ("-p", url)
    with pytest.raises(ValueError, match="parent"):
        launch_job(p.launcher, p.meta, p.context)
    p.dc.containers.run.assert_not_called()


@pytest.mark.parametrize("creation_fails", [False, True])
def test_shutdown_retains_inflight_creation_and_cleans_late_container(profile, creation_fails):
    import threading

    from nvflare.private.fed.client.task_runtime import shutdown_task_jobs
    from tests.timing_utils import ThreadGroup, join_thread

    p = profile
    entered, finish = threading.Event(), threading.Event()
    published = Mock()
    p.context.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, published)
    p.site["task_execution"]["finalization_timeout"] = 0.01
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    original_create = p.dc.containers.run.return_value

    def create(*args, **kwargs):
        entered.set()
        finish.wait()
        if creation_fails:
            raise RuntimeError("creation failed")
        return original_create

    p.dc.containers.run.side_effect = create

    def launch():
        with pytest.raises(RuntimeError, match="creation failed" if creation_fails else "shut down during"):
            launch_job(p.launcher, p.meta, p.context)

    with ThreadGroup(finish) as threads:
        worker = threads.start(launch)
        assert entered.wait(5), "Docker creation did not enter"
        service = p.cell._slurm_task_service
        owner = service.jobs["job-1"]
        shutdown_task_jobs(p.cell)
        assert owner.released and owner.cj_launch_pending
        assert owner.thread.is_alive() and service.jobs["job-1"] is owner
        finish.set()
        join_thread(worker)
        join_thread(owner.thread)
        assert "job-1" not in service.jobs
    published.assert_not_called()
    if not creation_fails:
        p.native.terminate.assert_called_once()
        p.native.wait.assert_not_called()
    else:
        p.native.terminate.assert_not_called()


def test_system_shutdown_terminates_already_attached_coordinator(profile):
    from nvflare.private.fed.client.task_runtime import shutdown_task_jobs

    p = profile
    handle = launch_job(p.launcher, p.meta, p.context)
    shutdown_task_jobs(p.cell)
    p.native.terminate.assert_called_once()
    assert handle.owner.closed and not handle.owner.thread.is_alive()


@pytest.mark.parametrize("cpus", [0.067, 8.3])
@pytest.mark.parametrize("request_budget", [False, True])
def test_fractional_site_ceiling_accepts_defaults_and_exactly_equal_budget(profile, cpus, request_budget):
    p = profile
    p.launcher.default_job_container_kwargs["nano_cpus"] = round(cpus * 1_000_000_000)
    spec = {"num_of_cpus": cpus} if request_budget else {}
    p.meta["cj_resource_spec"]["site-1"] = spec
    launch_job(p.launcher, p.meta, p.context)
    assert p.dc.containers.run.call_args.kwargs["nano_cpus"] == round(cpus * 1_000_000_000)


def test_optional_null_launcher_spec_keeps_site_default_coordinator(profile):
    p = profile
    p.meta["launcher_spec"] = None
    launch_job(p.launcher, p.meta, p.context)
    assert p.dc.containers.run.call_args.args == ("site/coordinator:tested",)


def test_client_engine_checks_invalid_site_limits_before_accepting_jobs(profile):
    from nvflare.private.fed.client.client_engine import ClientEngine

    p = profile
    p.site["components"][0]["args"]["default_job_container_kwargs"]["mem_limit"] = "invalid"
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    client = Mock(client_name="site-1", client_args={}, components={}, secure_train=False)
    engine = ClientEngine(client, SimpleNamespace(workspace=str(p.root)), rank=0)
    with pytest.raises(ValueError, match="site mem_limit"):
        engine.fire_event(EventType.SYSTEM_BOOTSTRAP, engine.new_context())
    p.dc.info.assert_not_called()
