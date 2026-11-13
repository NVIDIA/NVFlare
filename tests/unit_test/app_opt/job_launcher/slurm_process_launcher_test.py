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

"""Process CJ ownership and canonical Slurm task policy; no native scheduler calls."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.workspace import Workspace
from nvflare.app_common.job_launcher.client_process_launcher import ClientProcessJobLauncher
from nvflare.app_common.job_launcher.server_process_launcher import ServerProcessJobLauncher
from nvflare.app_opt.job_launcher.slurm.task_launcher import SlurmTaskLauncher
from nvflare.app_opt.job_launcher.slurm.task_policy import PROCESS_CJ_LAUNCHER_PATH, TASK_LAUNCHER_PATH
from nvflare.app_opt.job_launcher.slurm.task_service import ENV_TASK_OWNER, ENV_TASK_TOKEN
from nvflare.private.fed.client.task_execution_policy import GPU_ENVIRONMENT_NAMES, TaskExecutionPolicy
from nvflare.private.fed.task_worker.parent_runtime import launch_job, shutdown_task_jobs


@pytest.fixture
def profile(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    for name in ("startup", "local", "job-1/app_site-1/config"):
        (root / name).mkdir(parents=True, exist_ok=True)
    workspace = Workspace(str(root), "site-1")
    Path(workspace.get_client_app_config_file_path("job-1")).write_text(json.dumps({"execution_lifetime": "task"}))
    site = {
        "components": [{"id": "job_launcher", "path": PROCESS_CJ_LAUNCHER_PATH}],
        "task_execution": {
            "launcher": {
                "path": TASK_LAUNCHER_PATH,
                "args": {
                    "workspace_path": str(root),
                    "cj_runtime": "process",
                    "shared_filesystem": True,
                    "cgroup_enforced": True,
                    "worker_listener_host": "gateway.example",
                    "python_path": "/shared/python",
                },
            }
        },
    }
    (root / "local/resources.json").write_text(json.dumps(site))
    cell = SimpleNamespace(get_fqcn=lambda: "site-1", register_request_cb=Mock())
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site-1")
    context.set_prop(ReservedKey.RUN_NUM, "job-1")
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_cell=lambda: cell))
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    launcher = ClientProcessJobLauncher()
    monkeypatch.setattr(launcher, "get_command", lambda *_: "/shared/python -m actual_cj")
    adapter = Mock()
    adapter.pid = 999999999
    adapter.poll.return_value = None
    spawn = Mock(return_value=adapter)
    monkeypatch.setattr("nvflare.app_common.job_launcher.process_launcher.spawn_process", spawn)
    monkeypatch.setattr(
        "nvflare.app_common.job_launcher.process_launcher.os.killpg", Mock(side_effect=ProcessLookupError)
    )
    meta = {"job_id": "job-1", "launcher_spec": {"site-1": {"slurm": {"gpus_per_node": 4}}}}
    try:
        yield SimpleNamespace(**locals())
    finally:
        service = getattr(cell, "_slurm_task_service", None)
        if service:
            assert service.shutdown(timeout=5) == ()


def test_process_cj_has_cp_owner_and_gpu_free_environment(profile, monkeypatch):
    p = profile
    for name in ("SLURM_JOB_ID", "SLURM_JWT", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "parent-only")
    original = copy.deepcopy(p.meta)
    published = []
    p.context.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, published.append)
    handle = launch_job(p.launcher, p.meta, p.context)
    owner = p.cell._slurm_task_service.jobs["job-1"]
    argv, environment = p.spawn.call_args.args
    assert argv == ["/shared/python", "-m", "actual_cj"]
    assert "SLURM_JOB_ID" not in environment and "SLURM_JWT" not in environment
    assert all(
        environment[name] == ("none" if name == "NVIDIA_VISIBLE_DEVICES" else "") for name in GPU_ENVIRONMENT_NAMES
    )
    assert environment[ENV_TASK_OWNER] == "site-1" and environment[ENV_TASK_TOKEN] == owner.token
    assert owner.backend._binding.resources.gpus_per_node == 4
    assert published == [handle] and p.meta == original
    assert owner.cj_handle is handle.handle
    p.adapter.poll.return_value = 0
    assert handle.poll() == JobReturnCode.SUCCESS
    handle.wait()
    owner.thread.join(5)
    assert not owner.thread.is_alive()


def test_process_cj_recovery_remains_disabled_after_physical_exit(profile):
    p = profile
    handle = launch_job(p.launcher, p.meta, p.context)
    p.adapter.poll.return_value = 0
    assert handle.handle.can_restart() is True
    assert not handle.can_restart()


def test_slurm_study_policy_applies_to_workers_without_process_reparsing(profile, monkeypatch):
    p = profile
    monkeypatch.delenv("GATEWAY_STUDY_WORKER_ONLY", raising=False)
    setup = "export STUDY_WORKER_SETUP=1"
    study = {
        "env": {"GATEWAY_STUDY_WORKER_ONLY": "worker-only"},
        "slurm": {"setup": setup, "partition": "study-gpu"},
    }
    (p.root / "local/study_runtime.yaml").write_text(json.dumps({"format_version": 2, "studies": {"training": study}}))
    p.meta["study"] = "training"
    handle = launch_job(p.launcher, p.meta, p.context)
    p.spawn.assert_called_once()
    _, coordinator_env = p.spawn.call_args.args
    assert "GATEWAY_STUDY_WORKER_ONLY" not in coordinator_env
    binding = handle.owner.backend._binding
    assert dict(binding.environment)["GATEWAY_STUDY_WORKER_ONLY"] == "worker-only"
    assert binding.config.setup == setup
    assert binding.config.sbatch_directives["partition"] == "study-gpu"


@pytest.mark.parametrize("failure", ["spawn", "publish"])
def test_launch_failures_release_cp_ownership(profile, failure):
    p = profile
    if failure == "spawn":
        p.spawn.side_effect = RuntimeError("spawn failed")
    else:
        p.context.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, Mock(side_effect=RuntimeError("publish failed")))
        p.adapter.poll.return_value = 0
    with pytest.raises(RuntimeError, match="failed"):
        launch_job(p.launcher, p.meta, p.context)
    service = p.cell._slurm_task_service
    assert service.shutdown(timeout=5) == ()


@pytest.mark.parametrize("budget", [{"num_of_cpus": 1}, {"memory": "1Gi"}, {"num_of_gpus": 1}])
def test_unenforced_cj_budget_is_rejected_before_spawn(profile, budget):
    p = profile
    p.meta["cj_resource_spec"] = {"site-1": budget}
    with pytest.raises(ValueError, match="cannot enforce"):
        launch_job(p.launcher, p.meta, p.context)
    p.spawn.assert_not_called()


@pytest.mark.parametrize("mutation", ["slurm-runtime", "mixed-launchers", "wrong-launcher", "combined-launcher"])
def test_process_cj_requires_explicit_canonical_profile(profile, mutation):
    p = profile
    if mutation == "slurm-runtime":
        p.site["task_execution"]["launcher"]["args"].pop("cj_runtime")
    elif mutation == "mixed-launchers":
        p.site["components"].append({"path": "other.OtherJobLauncher"})
    elif mutation == "combined-launcher":
        p.site["components"][0][
            "path"
        ] = "nvflare.app_opt.job_launcher.slurm.process_launcher.ClientProcessSlurmJobLauncher"
    else:
        p.site["components"][0]["path"] = "other.OtherJobLauncher"
    (p.root / "local/resources.json").write_text(json.dumps(p.site))
    with pytest.raises(ValueError):
        launch_job(p.launcher, p.meta, p.context)
    p.spawn.assert_not_called()


def test_process_policy_and_system_shutdown(profile):
    p = profile
    policy = TaskExecutionPolicy.from_workspace(p.workspace)
    assert type(policy.launcher) is SlurmTaskLauncher
    policy.validate_job_launcher(p.launcher)
    with pytest.raises(ValueError):
        policy.validate_job_launcher(ServerProcessJobLauncher())
    launch_job(p.launcher, p.meta, p.context)
    p.adapter.poll.return_value = 0
    shutdown_task_jobs(p.cell)
    assert p.cell._slurm_task_service.shutdown(timeout=5) == ()


@pytest.mark.parametrize("lifetime", [None, "job"])
def test_ordinary_process_launcher_keeps_resident_jobs_independent_of_task_policy(profile, lifetime):
    p = profile
    config = {} if lifetime is None else {"execution_lifetime": lifetime}
    Path(p.workspace.get_client_app_config_file_path("job-1")).write_text(json.dumps(config))
    handle = launch_job(p.launcher, p.meta, p.context)
    p.spawn.assert_called_once()
    assert not hasattr(handle, "owner")
    assert not hasattr(p.cell, "_slurm_task_service")


def test_cp_owns_a_physical_backend_that_reaches_native_task_launch(profile, monkeypatch):
    from tests.slurm_gateway_test_utils import assert_physical_task_dispatch

    p = profile
    handle = launch_job(p.launcher, p.meta, p.context)
    assert_physical_task_dispatch(handle, p.root, monkeypatch)
