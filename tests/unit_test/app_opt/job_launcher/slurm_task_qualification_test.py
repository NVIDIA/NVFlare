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

"""Exercise the exported qualification fixture through real resource mapping."""

import argparse
import json
import runpy
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from nvflare.app_opt.job_launcher.slurm.batch import _submission_argv
from nvflare.app_opt.job_launcher.slurm.launcher import _resolve_resources
from nvflare.app_opt.job_launcher.slurm.task_job_launcher import ClientSlurmTaskJobLauncher
from nvflare.tool.deploy import slurm_deploy
from nvflare.tool.job.job_cli import prepare_job_config
from nvflare.utils.job_launcher_utils import get_cj_resource_spec, get_job_launcher_spec, get_resource_manager_spec
from tests.unit_test.app_opt.job_launcher.slurm_batch_test import _config, _plan
from tests.unit_test.tool.deploy.deploy_commands_test import _make_client_kit, _make_server_kit, _run_prepare

FIXTURE = Path(__file__).resolve().parents[3] / "integration_test" / "data" / "slurm_task_execution"


@pytest.mark.parametrize("device,gpu_count", [("cpu", 0), ("cuda", 1)])
@pytest.mark.parametrize("lifetime", ["job", "task"])
@pytest.mark.parametrize("resource_format", ["native", "portable"])
def test_qualification_exports_bound_server_and_compute_allocations(
    tmp_path, monkeypatch, device, gpu_count, lifetime, resource_format
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(FIXTURE / "export_probe.py"),
            "--device",
            device,
            "--clients",
            "2",
            "--lifetime",
            lifetime,
            "--resource-format",
            resource_format,
            "--output",
            str(tmp_path),
        ],
    )
    runpy.run_path(str(FIXTURE / "export_probe.py"), run_name="__main__")
    suffix = "-portable" if resource_format == "portable" else ""
    job = tmp_path / f"slurm-probe-{device}-{lifetime}{suffix}"
    meta = json.loads((job / "meta.json").read_text())
    server = json.loads((job / "app" / "config" / "config_fed_server.json").read_text())
    client = json.loads((job / "app" / "config" / "config_fed_client.json").read_text())
    assert meta["min_clients"] == 2
    assert server["workflows"][0]["args"]["num_rounds"] == 2
    assert client.get("execution_lifetime", "job") == lifetime
    assert (job / "app" / "custom" / "compute_probe.py").read_bytes() == (FIXTURE / "compute_probe.py").read_bytes()
    # Exercise the same staging path as `nvflare job submit`, even without CLI
    # overrides. Export-only checks miss literal @default/@ALL key corruption.
    staged = tmp_path / "submitted"
    shutil.copytree(job, staged)
    prepare_job_config(argparse.Namespace(job_folder=str(job), config_file=None), ["app"], str(staged))
    submitted = json.loads((staged / "meta.json").read_text())
    for field in ("resource_spec", "cj_resource_spec", "launcher_spec"):
        assert submitted.get(field) == meta.get(field), field
    if lifetime == "task":
        cj = ClientSlurmTaskJobLauncher(
            workspace_path=str(tmp_path), sandbox="none", python_path=sys.executable, executables={}
        )
        for site in ("site-1", "site-2"):
            if resource_format == "portable":
                assert get_cj_resource_spec(submitted, site) == {
                    "num_of_cpus": 2,
                    "memory": "1536Mi",
                    "num_of_gpus": 0,
                }
            resources = cj._cj_resources_for_job(submitted, site)
            plan = replace(_plan(tmp_path, resources=resources), site_name=site, clear_gres=True)
            argv = _submission_argv(plan, str(tmp_path), "cj", "cj", _config(tmp_path))
            assert ("--mem=1536M" if resource_format == "portable" else "--mem=1024M") in argv
            assert ("--cpus-per-task=2" if resource_format == "portable" else "--cpus-per-task=1") in argv
            assert "--gres=none" in argv
    for site in ("server", "site-1", "site-2"):
        spec = get_job_launcher_spec(submitted, site, "slurm")
        resources = _resolve_resources(submitted, site, "none", 120, spec, allow_zero_gpus=site != "server")
        plan = replace(_plan(tmp_path, resources=resources), site_name=site, clear_gres=site != "server")
        argv = _submission_argv(plan, str(tmp_path), "probe", "probe", _config(tmp_path))
        # An omitted --mem inherited the scheduler's all-node default and
        # prevented CJs/parents from running together in the first campaign.
        portable_client = resource_format == "portable" and site != "server"
        assert ("--mem=3072M" if portable_client else "--mem=2048M") in argv
        assert ("--cpus-per-task=2" if portable_client else "--cpus-per-task=1") in argv
        assert "--time=00:20:00" in argv
        expected_gpus = 0 if site == "server" else gpu_count
        if site == "server":
            assert not any(arg.startswith("--gres=") for arg in argv)
        else:
            assert (f"--gres=gpu:{expected_gpus}" if expected_gpus else "--gres=none") in argv
        expected_job_gpus = gpu_count if portable_client and lifetime == "job" else 0
        assert get_resource_manager_spec(submitted, site).get("num_of_gpus", 0) == expected_job_gpus


@pytest.mark.parametrize("role", ["server", "client"])
def test_qualification_deploy_configs_use_supported_security_and_parent_lifetime(tmp_path, capsys, role):
    config = yaml.safe_load((FIXTURE / f"{role}.yaml").read_text())
    slurm_deploy.validate_config(config)
    kit = _make_server_kit(tmp_path) if role == "server" else _make_client_kit(tmp_path)
    output = tmp_path / "prepared"
    _run_prepare(kit, output, config)
    capsys.readouterr()
    comm = json.loads((output / "local" / "comm_config.json").read_text())
    assert comm["internal"]["resources"]["connection_security"] == "mtls"
    if role == "client":
        parent = (output / "startup" / "parent.slurm").read_text()
        assert "#SBATCH --time=1-00:00:00" in parent
        assert "NVIDIA_VISIBLE_DEVICES=none" in parent
    else:
        assert not (output / "startup" / "parent.slurm").exists()
