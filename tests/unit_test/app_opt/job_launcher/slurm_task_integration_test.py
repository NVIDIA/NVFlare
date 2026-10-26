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

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants, ReservedKey, TaskResultReceipt
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import TaskLauncherError, TaskResourceRequest
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.slurm.batch import _submission_argv
from nvflare.app_opt.job_launcher.slurm.config import CommandResult, SubmissionResult
from nvflare.app_opt.job_launcher.slurm.task_job_launcher import ClientSlurmTaskJobLauncher
from nvflare.app_opt.job_launcher.slurm.task_launcher import WORKER_MODULE, SlurmTaskLauncher
from nvflare.app_opt.job_launcher.slurm.task_policy import CJ_LAUNCHER_PATH, TASK_LAUNCHER_PATH, task_resources
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor
from nvflare.utils.job_launcher_utils import get_resource_manager_spec, normalize_cj_resource_spec
from tests.unit_test.app_opt.job_launcher.slurm_launcher_test import _fl_ctx
from tests.unit_test.app_opt.job_launcher.slurm_task_launcher_test import (
    _accounting,
    _backend,
    _query,
    _supervisor_case,
)
from tests.unit_test.private.fed.conftest import isolated_fobs  # noqa: F401


def _site(root):
    executables = {name: "/usr/bin/true" for name in ("sbatch", "squeue", "sacct", "scancel")}
    return {
        "components": [
            {
                "id": "job_launcher",
                "path": CJ_LAUNCHER_PATH,
                "args": {
                    "workspace_path": str(root),
                    "sandbox": "none",
                    "python_path": sys.executable,
                    "executables": executables,
                    "parent_host": "compute.example",
                    "sbatch_directives": {"account": "site-account", "partition": "cpu"},
                    "cj_cpu_cores": 2,
                    "cj_memory_mb": 512,
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
                    "executables": executables,
                    "python_path": sys.executable,
                    "cpu_cores": 4,
                    "memory_mb": 2048,
                    "gpu_count": 2,
                    "environment": {"PYTHONPATH": os.getcwd()},
                    "sbatch_directives": {"account": "site-account", "partition": "gpu"},
                },
            }
        },
    }


@pytest.fixture
def profile(tmp_path, monkeypatch):
    launcher, adapter, request = _backend(tmp_path, monkeypatch, 0)
    root = Path(launcher.manager.config.workspace_path)
    site = _site(root)
    (root / "local" / "resources.json").write_text(json.dumps(site))
    workspace = Workspace(str(root), site_name="site-1")
    config = Path(workspace.get_client_app_config_file_path("job-1"))
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"execution_lifetime": "task"}))
    cj = ClientSlurmTaskJobLauncher(**site["components"][0]["args"])
    task = SlurmTaskLauncher(**site["task_execution"]["launcher"]["args"])
    return root, workspace, site, cj, task


def _context(workspace, meta, job_id="job-1", site_name="site-1"):
    context = FLContext()
    context.set_prop(ReservedKey.RUN_NUM, job_id)
    context.set_prop(ReservedKey.IDENTITY_NAME, site_name)
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    context.set_prop(FLContextKey.JOB_META, meta)
    return context


def test_exact_matching_slurm_site_profile(profile):
    root, workspace, site, cj, task = profile
    task.validate_site_config(site)
    malformed = copy.deepcopy(site)
    malformed["components"] = []
    with pytest.raises(ValueError, match="exactly one"):
        task.validate_site_config(malformed)
    malformed = copy.deepcopy(site)
    malformed["components"][0]["args"]["workspace_path"] = str(root.parent)
    with pytest.raises(ValueError, match="same bare shared workspace"):
        task.validate_site_config(malformed)
    malformed = copy.deepcopy(site)
    malformed["components"].append({"path": "nvflare.app_common.job_launcher.process_launcher.ProcessJobLauncher"})
    with pytest.raises(ValueError, match="different CP/CJ"):
        task.validate_site_config(malformed)


@pytest.mark.parametrize("gpus", [0, 1, 2])
def test_native_site_default_override_and_cpu_memory_mapping(profile, gpus):
    _, _, _, _, task = profile
    meta = {
        "launcher_spec": {
            "default": {"slurm": {"gpus_per_node": 2, "cpus_per_node": 6}},
            "site-1": {
                "slurm": {"gpus_per_node": gpus, "mem_per_node": 4096, "time": "00:03:00", "pending_timeout": 100}
            },
        }
    }
    result = task_resources(task, meta, "site-1")
    assert (result.gpus_per_node, result.cpus_per_node, result.mem_per_node) == (gpus, 6, 4096)
    assert result.time_limit == "00:03:00" and result.pending_timeout == 100
    assert task_resources(task, {}, "site-1").gpus_per_node == 2
    portable = {"resource_spec": {"@default": {"num_of_cpus": 3, "memory": "1Gi", "num_of_gpus": 0}}}
    result = task_resources(task, portable, "site-1")
    assert (result.gpus_per_node, result.cpus_per_node, result.mem_per_node) == (0, 3, 1024)


@pytest.mark.parametrize(
    "meta",
    [
        {"resource_spec": {"@default": {"num_of_gpus": 1}}},
        {"resource_spec": {"site-1": {"process": {"num_of_gpus": 1}}}},
    ],
)
def test_whole_job_cp_gpu_reservation_fails_closed(profile, meta):
    _, _, _, _, task = profile
    with pytest.raises(ValueError, match="whole job"):
        task_resources(task, meta, "site-1")


@pytest.mark.parametrize(
    "native",
    [
        {"nodes": 2},
        {"additional_node_command": "python train.py"},
        {"image": "/shared/image.sif"},
        {"gpus_per_node": -1},
    ],
)
def test_unsupported_task_resource_contract_is_rejected(profile, native):
    _, _, _, _, task = profile
    with pytest.raises((ValueError, RuntimeError)):
        task_resources(task, {"launcher_spec": {"site-1": {"slurm": native}}}, "site-1")


def test_cpu_cj_separate_allocation_preserves_original_task_metadata(profile):
    root, workspace, site, cj, task = profile
    meta = {
        JobConstants.JOB_ID: "job-1",
        "launcher_spec": {
            "site-1": {"slurm": {"gpus_per_node": 2, "cpus_per_node": 8, "mem_per_node": 4096, "time": "00:03:00"}}
        },
    }
    original = copy.deepcopy(meta)
    plan = cj._build_launch_plan(meta, _fl_ctx(root))
    assert plan.resources.gpus_per_node == 0
    assert (plan.resources.cpus_per_node, plan.resources.mem_per_node) == (2, 512)
    assert plan.directives["partition"] == "cpu"
    assert plan.study_env["CUDA_VISIBLE_DEVICES"] == ""
    assert "--gres=none" in _submission_argv(plan, str(root), "cpu-cj", "marker", cj.config)
    assert meta == original


@pytest.mark.parametrize("cj_gpu_count", [0, 1])
def test_cj_site_gpu_default_is_independent_of_compute(profile, cj_gpu_count):
    root, _, site, _, task = profile
    cj = ClientSlurmTaskJobLauncher(**{**site["components"][0]["args"], "cj_gpu_count": cj_gpu_count})
    meta = {
        JobConstants.JOB_ID: "job-1",
        "launcher_spec": {"site-1": {"slurm": {"gpus_per_node": 2, "cpus_per_node": 8, "mem_per_node": 4096}}},
    }
    original = copy.deepcopy(meta)
    plan = cj._build_launch_plan(meta, _fl_ctx(root))
    compute = task_resources(task, meta, "site-1")
    assert (plan.resources.cpus_per_node, plan.resources.mem_per_node, plan.resources.gpus_per_node) == (
        2,
        512,
        cj_gpu_count,
    )
    assert (compute.cpus_per_node, compute.mem_per_node, compute.gpus_per_node) == (8, 4096, 2)
    assert ("--gres=gpu:1" if cj_gpu_count else "--gres=none") in _submission_argv(
        plan, str(root), "cj", "marker", cj.config
    )
    assert plan.study_env.get("CUDA_VISIBLE_DEVICES") == (None if cj_gpu_count else "")
    assert meta == original


@pytest.mark.parametrize(
    "cj_spec, expected",
    [
        ({"@default": {"num_of_cpus": 3, "memory": "2Gi", "num_of_gpus": 1}}, (3, 2048, 1)),
        (
            {
                "@default": {"num_of_cpus": 3, "memory": "2Gi", "num_of_gpus": 1},
                "site-1": {"num_of_cpus": 5, "num_of_gpus": 0},
            },
            (5, 2048, 0),
        ),
        ({"site-2": {"num_of_cpus": 7, "memory": "8Gi", "num_of_gpus": 3}}, (2, 512, 0)),
    ],
)
def test_cj_metadata_overrides_only_cj_resources(profile, cj_spec, expected):
    root, _, _, cj, task = profile
    meta = {
        JobConstants.JOB_ID: "job-1",
        "cj_resource_spec": cj_spec,
        "launcher_spec": {"site-1": {"slurm": {"gpus_per_node": 2, "cpus_per_node": 8, "mem_per_node": 4096}}},
    }
    normalize_cj_resource_spec(meta, {"site-1", "site-2"})
    original = copy.deepcopy(meta)
    plan = cj._build_launch_plan(meta, _fl_ctx(root))
    compute = task_resources(task, meta, "site-1")
    assert (plan.resources.cpus_per_node, plan.resources.mem_per_node, plan.resources.gpus_per_node) == expected
    assert (compute.cpus_per_node, compute.mem_per_node, compute.gpus_per_node) == (8, 4096, 2)
    assert plan.study_env.get("CUDA_VISIBLE_DEVICES") == (None if expected[2] else "")
    assert meta == original


def test_cj_explicit_zero_gpu_overrides_site_default(profile):
    root, _, site, _, _ = profile
    cj = ClientSlurmTaskJobLauncher(**{**site["components"][0]["args"], "cj_gpu_count": 1})
    meta = {
        JobConstants.JOB_ID: "job-1",
        "cj_resource_spec": {"@default": {"num_of_gpus": 0}},
        "launcher_spec": {"site-1": {"slurm": {"gpus_per_node": 2}}},
    }
    normalize_cj_resource_spec(meta, {"site-1"})
    plan = cj._build_launch_plan(meta, _fl_ctx(root))
    assert (plan.resources.cpus_per_node, plan.resources.mem_per_node, plan.resources.gpus_per_node) == (2, 512, 0)
    assert plan.study_env["CUDA_VISIBLE_DEVICES"] == ""
    assert "--gres=none" in _submission_argv(plan, str(root), "cj", "marker", cj.config)


@pytest.mark.parametrize("cj_gpu_count", [0, 1])
def test_portable_compute_uses_per_task_allocation_and_independent_cj_reservation(profile, cj_gpu_count):
    root, _, _, cj, task = profile
    meta = {
        JobConstants.JOB_ID: "job-1",
        "resource_spec": {"@default": {"num_of_cpus": 6, "memory": "3Gi", "num_of_gpus": 2}},
        "cj_resource_spec": {"@default": {"num_of_cpus": 1, "memory": "1Gi", "num_of_gpus": cj_gpu_count}},
    }
    normalize_cj_resource_spec(meta, {"site-1"})
    plan = cj._build_launch_plan(meta, _fl_ctx(root))
    compute = task_resources(task, meta, "site-1")
    assert (plan.resources.cpus_per_node, plan.resources.mem_per_node, plan.resources.gpus_per_node) == (
        1,
        1024,
        cj_gpu_count,
    )
    assert get_resource_manager_spec(meta, "site-1") == ({"num_of_gpus": 1} if cj_gpu_count else {})
    assert (compute.cpus_per_node, compute.mem_per_node, compute.gpus_per_node) == (6, 3072, 2)
    # Shared policy must also accept normalized portable GPU compute admission.
    from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy

    policy = TaskExecutionPolicy.from_config({"launcher": _site(root)["task_execution"]["launcher"]})
    assert policy.task_resources(meta, "site-1").is_empty()


@pytest.mark.parametrize("gpu_count", [0, 1, 2])
def test_legacy_compute_gpu_count_overrides_task_default_without_a_cj_reservation(profile, gpu_count):
    root, _, _, cj, task = profile
    meta = {
        JobConstants.JOB_ID: "job-1",
        "resource_spec": {"site-1": {"process": {"num_of_gpus": gpu_count}}},
    }
    normalize_cj_resource_spec(meta, {"site-1"})
    original = copy.deepcopy(meta)
    assert get_resource_manager_spec(meta, "site-1") == {}
    assert task_resources(task, meta, "site-1").gpus_per_node == gpu_count
    assert cj._build_launch_plan(meta, _fl_ctx(root)).resources.gpus_per_node == 0
    assert meta == original


def test_cj_all_sites_default_and_site_overrides(profile):
    root, _, _, cj, task = profile
    meta = {
        JobConstants.JOB_ID: "job-1",
        "cj_resource_spec": {
            "@default": {"num_of_cpus": 3, "memory": "2Gi", "num_of_gpus": 1},
            "@ALL": {"memory": "3Gi"},
            "site-1": {"num_of_cpus": 5, "num_of_gpus": 0},
        },
    }
    normalize_cj_resource_spec(meta, {"@ALL"})
    plan = cj._build_launch_plan(meta, _fl_ctx(root))
    assert (plan.resources.cpus_per_node, plan.resources.mem_per_node, plan.resources.gpus_per_node) == (5, 3072, 0)


@pytest.mark.parametrize("cpu_share, cores", [(0.25, 1), (1.5, 2), (2, 2)])
def test_cj_cpu_shares_receive_at_least_the_requested_whole_cores(profile, cpu_share, cores):
    root, _, _, cj, _ = profile
    meta = {JobConstants.JOB_ID: "job-1", "cj_resource_spec": {"site-1": {"num_of_cpus": cpu_share}}}
    normalize_cj_resource_spec(meta, {"site-1"})
    plan = cj._build_launch_plan(meta, _fl_ctx(root))
    assert plan.resources.cpus_per_node == cores


def test_cj_resource_role_cannot_replace_resident_compute_admission(profile):
    root, workspace, _, cj, _ = profile
    Path(workspace.get_client_app_config_file_path("job-1")).write_text("{}")
    meta = {JobConstants.JOB_ID: "job-1", "cj_resource_spec": {"site-1": {}}}
    with pytest.raises(ValueError, match="resolved client execution_lifetime"):
        cj._build_launch_plan(meta, _fl_ctx(root))


def test_cj_gpu_memory_requirement_is_rejected(profile):
    root, _, _, cj, _ = profile
    meta = {
        JobConstants.JOB_ID: "job-1",
        "cj_resource_spec": {"site-1": {"num_of_gpus": 1, "mem_per_gpu_in_GiB": 2}},
    }
    with pytest.raises(ValueError, match="per-GPU memory"):
        cj._build_launch_plan(meta, _fl_ctx(root))


@pytest.mark.parametrize("extra", [{"custom_tokens": 1}, {"mem_per_gpu_in_GiB": 2}])
def test_task_cannot_drop_unenforceable_resource_manager_requirements(profile, extra):
    _, _, _, _, task = profile
    meta = {"resource_spec": {"site-1": {"num_of_gpus": 1, **extra}}}
    normalize_cj_resource_spec(meta, {"site-1"})
    with pytest.raises(ValueError, match="cannot enforce custom resource-manager or GPU-memory"):
        task_resources(task, meta, "site-1")


def test_resident_job_keeps_original_slurm_resource_behavior(profile):
    root, workspace, site, cj, task = profile
    Path(workspace.get_client_app_config_file_path("job-1")).write_text("{}")
    plan = cj._build_launch_plan(
        {
            JobConstants.JOB_ID: "job-1",
            "launcher_spec": {"site-1": {"slurm": {"gpus_per_node": 2, "cpus_per_node": 8}}},
        },
        _fl_ctx(root),
    )
    assert plan.resources.gpus_per_node == 2 and plan.resources.cpus_per_node == 8
    cpu = cj._build_launch_plan(
        {JobConstants.JOB_ID: "job-1", "launcher_spec": {"site-1": {"slurm": {"gpus_per_node": 0}}}}, _fl_ctx(root)
    )
    assert cpu.resources.gpus_per_node == 0 and cpu.study_env["CUDA_VISIBLE_DEVICES"] == ""


def test_start_run_binding_is_immutable_and_preserves_study_policy(profile):
    root, workspace, site, cj, task = profile
    (root / "local" / "study_runtime.yaml").write_text(
        "format_version: 2\nstudies:\n  study1:\n    env:\n      DATA_ROOT: /shared/data\n"
        "    slurm:\n      account: study-account\n      partition: study-gpu\n      setup: 'export DATA_ROOT=/shared/override'\n"
    )
    meta = {"study": "study1", "launcher_spec": {"site-1": {"slurm": {"gpus_per_node": 1}}}}
    context = _context(workspace, meta)
    task.handle_event(EventType.START_RUN, context)
    task.handle_event(EventType.START_RUN, context)
    assert task._binding.resources.gpus_per_node == 1
    assert task._binding.config.sbatch_directives["account"] == "study-account"
    assert task._binding.config.sbatch_directives["partition"] == "study-gpu"
    assert dict(task._binding.environment)["DATA_ROOT"] == "/shared/data"
    with pytest.raises(ValueError, match="already bound"):
        task.handle_event(EventType.START_RUN, _context(workspace, meta, job_id="other-job"))
    changed = copy.deepcopy(meta)
    changed["launcher_spec"]["site-1"]["slurm"]["gpus_per_node"] = 0
    with pytest.raises(ValueError, match="already bound"):
        task.handle_event(EventType.START_RUN, _context(workspace, changed))


@pytest.mark.parametrize("gpu_count", [0, 1])
@pytest.mark.usefixtures("isolated_fobs")
def test_staged_cj_request_runs_authorized_worker_with_bound_native_resources(tmp_path, monkeypatch, gpu_count):
    backend = _backend(tmp_path, monkeypatch, gpu_count)
    launcher, adapter, request = backend
    launcher.python_path = sys.executable
    launcher.environment["PYTHONPATH"] = os.getcwd()
    identity, store, arguments = _supervisor_case(backend, write_result=False)
    workspace = Workspace(launcher.manager.config.workspace_path, site_name="site-1")
    meta = {
        "launcher_spec": {"site-1": {"slurm": {"gpus_per_node": gpu_count, "cpus_per_node": 3, "time": "00:03:00"}}}
    }
    launcher.handle_event(EventType.START_RUN, _context(workspace, meta))

    def factory(attempt, path):
        return replace(
            request,
            argv=(
                "/untrusted/cj/python",
                "-m",
                "nvflare.private.fed.app.client.task_worker_process",
                "--bootstrap",
                path,
                "--parent_pid",
                "1234",
            ),
            environment={"CJ_PRIVATE": "not forwarded"},
            resources=TaskResourceRequest(),
        )

    def submit(argv, timeout):
        assert "--cpus-per-task=3" in argv and "--time=00:03:00" in argv
        assert ("--gres=gpu:1" if gpu_count else "--gres=none") in argv
        text = Path(argv[-1]).read_text()
        assert WORKER_MODULE in text and "--parent_pid" not in text and "/untrusted/cj/python" not in text
        assert "CJ_PRIVATE" not in text
        environment = {"SLURM_JOB_ID": "42"}
        if gpu_count:
            environment["CUDA_VISIBLE_DEVICES"] = "0"
        result = subprocess.run(["/bin/bash", argv[-1]], env=environment, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr.decode()
        return SubmissionResult(CommandResult(0, "42", ""), "42")

    adapter.submit.side_effect = submit
    arguments["request_factory"] = factory
    outcome = TaskSupervisor().run(**arguments)
    assert outcome.execution_status.succeeded
    np.testing.assert_array_equal(from_shareable(outcome.result).data["numpy_key"], np.full(3, 3.0))
    assert Path(store.attempt_dir(identity), "result.fobs").exists()


def test_staged_request_requires_start_run_before_submission(tmp_path, monkeypatch):
    launcher, adapter, request = _backend(tmp_path, monkeypatch, 0)
    staged = replace(
        request,
        argv=(
            sys.executable,
            "-m",
            "nvflare.private.fed.app.client.task_worker_process",
            "--bootstrap",
            str(Path(request.cwd, "bootstrap.json")),
            "--parent_pid",
            "1234",
        ),
    )
    with pytest.raises(TaskLauncherError, match="START_RUN"):
        launcher.launch_task(staged)
    adapter.submit.assert_not_called()


@pytest.mark.parametrize("gpu_count", [0, 1])
@pytest.mark.usefixtures("isolated_fobs")
def test_application_flag_selects_slurm_and_runs_through_normal_cj(profile, gpu_count):
    root, workspace, site, cj, task = profile
    config_path = Path(workspace.get_client_app_config_file_path("job-1"))
    config_path.write_text(
        json.dumps(
            {
                "format_version": 3,
                "execution_lifetime": "task",
                "executors": [
                    {
                        "tasks": ["train"],
                        "executor": {
                            "path": "nvflare.app_common.np.np_trainer.NPTrainer",
                            "args": {"delta": 2},
                        },
                    }
                ],
            }
        )
    )
    meta = {"job_id": "job-1", "launcher_spec": {"site-1": {"slurm": {"gpus_per_node": gpu_count}}}}
    (root / "job-1" / "meta.json").write_text(json.dumps(meta))
    args = argparse.Namespace(
        sp_scheme="grpc",
        sp_target="localhost:8002",
        client_name="site-1",
        parent_url=None,
        job_id="job-1",
        workspace=str(root),
    )
    configurator = ClientJsonConfigurator(workspace, str(config_path), args, str(config_path.parent.parent))
    configurator.configure()
    executor = configurator.executors[0].executor
    assert type(executor) is TaskWorkerExecutor
    launcher = executor.policy.launcher
    assert type(launcher) is SlurmTaskLauncher
    assert executor.policy.worker_environment() == {}
    assert executor.policy.task_resources(meta, "site-1").is_empty()
    adapter = Mock()
    adapter.accounting_probe.return_value = CommandResult(0, "", "")
    adapter.active_by_id.return_value = _query()
    adapter.task_accounting.return_value = _accounting()
    launcher.manager.adapter = adapter
    launcher.manager.initialize()

    def submit(argv, timeout):
        assert ("--gres=gpu:1" if gpu_count else "--gres=none") in argv
        assert "--cpus-per-task=4" in argv and "--mem=2048M" in argv
        environment = {"SLURM_JOB_ID": "42"}
        if gpu_count:
            environment["CUDA_VISIBLE_DEVICES"] = "0"
        completed = subprocess.run(["/bin/bash", argv[-1]], env=environment, capture_output=True, timeout=30)
        assert completed.returncode == 0, completed.stderr.decode()
        return SubmissionResult(CommandResult(0, "42", ""), "42")

    adapter.submit.side_effect = submit
    context = _context(workspace, meta)
    context.set_prop(FLContextKey.TASK_ID, "task-1")
    context.set_prop(FLContextKey.TASK_NAME, "train")
    context.set_prop(FLContextKey.TASK_ATTEMPT_ID, "attempt-1")
    executor.handle_event(EventType.START_RUN, context)
    result = executor.execute(
        "train", DXO(DataKind.WEIGHTS, {"numpy_key": np.ones(3)}).to_shareable(), context, Signal()
    )
    np.testing.assert_array_equal(from_shareable(result).data["numpy_key"], np.full(3, 3.0))
    executor.execute("train", DXO(DataKind.WEIGHTS, {"numpy_key": np.ones(3)}).to_shareable(), context, Signal())
    assert adapter.submit.call_count == 1
    context.set_prop(FLContextKey.TASK_RESULT, result)
    context.set_prop(FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.RECEIVED)
    executor.handle_event(EventType.AFTER_SEND_TASK_RESULT, context)
    executor.handle_event(EventType.END_RUN, context)
