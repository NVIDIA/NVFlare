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

import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from nvflare.app_opt.job_launcher.slurm.batch import _submission_argv
from nvflare.app_opt.job_launcher.slurm.launcher import _resolve_resources
from nvflare.tool.deploy import slurm_deploy
from nvflare.utils.job_launcher_utils import get_job_launcher_spec, get_resource_manager_spec
from tests.unit_test.app_opt.job_launcher.slurm_batch_test import _config, _plan
from tests.unit_test.tool.deploy.deploy_commands_test import _make_client_kit, _make_server_kit, _run_prepare

FIXTURE = Path(__file__).resolve().parents[3] / "integration_test" / "data" / "slurm_task_execution"


@pytest.mark.parametrize("device,gpu_count", [("cpu", 0), ("cuda", 1)])
@pytest.mark.parametrize("lifetime", ["job", "task"])
def test_qualification_exports_bound_server_and_compute_allocations(tmp_path, monkeypatch, device, gpu_count, lifetime):
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
            "--output",
            str(tmp_path),
        ],
    )
    runpy.run_path(str(FIXTURE / "export_probe.py"), run_name="__main__")
    job = tmp_path / f"slurm-probe-{device}-{lifetime}"
    meta = json.loads((job / "meta.json").read_text())
    server = json.loads((job / "app" / "config" / "config_fed_server.json").read_text())
    client = json.loads((job / "app" / "config" / "config_fed_client.json").read_text())
    assert meta["min_clients"] == 2
    assert server["workflows"][0]["args"]["num_rounds"] == 2
    assert client.get("execution_lifetime", "job") == lifetime
    assert (job / "app" / "custom" / "compute_probe.py").read_bytes() == (FIXTURE / "compute_probe.py").read_bytes()
    for site in ("server", "site-1", "site-2"):
        spec = get_job_launcher_spec(meta, site, "slurm")
        resources = _resolve_resources(meta, site, "none", 120, spec)
        plan = replace(_plan(tmp_path, resources=resources), site_name=site)
        argv = _submission_argv(plan, str(tmp_path), "probe", "probe", _config(tmp_path))
        # An omitted --mem inherited the scheduler's all-node default and
        # prevented CJs/parents from running together in the first campaign.
        assert "--mem=2048M" in argv
        assert "--cpus-per-task=1" in argv
        assert "--time=00:20:00" in argv
        expected_gpus = 0 if site == "server" else gpu_count
        assert (f"--gres=gpu:{expected_gpus}" if expected_gpus else "--gres=none") in argv
        assert not get_resource_manager_spec(meta, site).get("num_of_gpus", 0)


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
