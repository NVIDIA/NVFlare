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

"""Exercise config parsing and site permissions through real authenticated processes."""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from nvflare.fuel.flare_api.api_spec import InvalidJobDefinition
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.utils.network_utils import get_open_ports
from tests.integration_test.src import ProvisionSiteLauncher

pytestmark = [pytest.mark.slow, pytest.mark.timeout(240)]


def test_live_job_configuration_security_and_commented_numpy_job(tmp_path, monkeypatch):
    template = Path(__file__).parents[1] / "data/jobs/hello-numpy-sag"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    fed_port, admin_port = get_open_ports(2)
    project = {
        "api_version": 3,
        "name": "config-security",
        "participants": [
            {
                "name": "localhost",
                "type": "server",
                "org": "test",
                "fed_learn_port": fed_port,
                "admin_port": admin_port,
            },
            {"name": "site-1", "type": "client", "org": "test"},
            {"name": "lead@test.org", "type": "admin", "org": "test", "role": "lead"},
        ],
        "builders": [
            {"path": "nvflare.lighter.impl." + name}
            for name in (
                "workspace.WorkspaceBuilder",
                "static_file.StaticFileBuilder",
                "cert.CertBuilder",
                "signature.SignatureBuilder",
            )
        ],
    }
    project_file = tmp_path / "project.yml"
    project_file.write_text(yaml.safe_dump(project))
    launcher = ProvisionSiteLauncher(str(project_file))
    session = None
    try:
        kits = Path(launcher.prepare_workspace()).resolve()
        policy = {
            "format_version": "1.0",
            "permissions": {"lead": {"submit_job": "any", "view": "any", "byoc": "none", "shell_commands": "none"}},
        }
        for name in ("localhost", "site-1"):
            (kits / name / "local/authorization.json").write_text("// site permissions\n" + json.dumps(policy))
        with monkeypatch.context() as server_env:
            server_env.setenv("NVFLARE_SECURITY_CANARY", "SYNTHETIC-SERVER-SECRET")
            launcher.start_servers()
        with monkeypatch.context() as client_env:
            client_env.setenv("NVFLARE_TEST_MODEL_DIR", "approved-model-dir")
            launcher.start_clients()
        session = new_secure_session("lead@test.org", str(kits / "lead@test.org"), timeout=30)
        deadline = time.monotonic() + 60
        while not session.get_system_info().client_info:
            assert time.monotonic() < deadline, "client did not register"
            time.sleep(0.2)
        job = tmp_path / "job"
        shutil.copytree(template, job)
        meta = json.loads((job / "meta.json").read_text())
        meta.update(
            min_clients=1,
            deploy_map={"app": ["server", "site-1"]},
            literal=["${NVFLARE_SECURITY_CANARY}", "{$NVFLARE_SECURITY_CANARY}", "${secret:NVFLARE_SECURITY_CANARY}"],
        )
        (job / "meta.json").unlink()
        for role, suffix in (("server", ".conf"), ("client", ".json")):
            path = job / "app/config" / f"config_fed_{role}.json"
            config = json.loads(path.read_text())
            if role == "server":
                config["workflows"][0]["args"].update(min_clients=1, num_rounds=1, wait_time_after_min_received=0)
            else:
                config["executors"][0]["executor"]["args"]["model_dir"] = "{$NVFLARE_TEST_MODEL_DIR}"
            path.unlink()
            path.with_suffix(suffix).write_text("// retained configuration comment\n" + json.dumps(config))
        private_file = tmp_path / "private.conf"
        private_file.write_text('stolen="SYNTHETIC-FILE-SECRET"')
        for suffix, payload in (
            (".conf", "stolen=${NVFLARE_SECURITY_CANARY}"),
            (".conf", "include file(" + json.dumps(str(private_file)) + ")"),
            (".conf", "min_clients=${NVFLARE_SECURITY_CANARY}"),
            (".yaml", "stolen: ${oc.env:NVFLARE_SECURITY_CANARY}"),
        ):
            path = job / ("meta" + suffix)
            path.write_text(payload)
            with pytest.raises(InvalidJobDefinition) as error:
                session.submit_job(str(job))
            assert "SYNTHETIC-" not in str(error.value)
            path.unlink()
        (job / "meta.json").write_text("/* metadata comment */\n" + json.dumps(meta))
        job_id = session.submit_job(str(job))
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            result = session.get_job_meta(job_id)
            if result["status"].startswith("FINISHED"):
                break
            time.sleep(0.2)
        assert result["status"] == "FINISHED:COMPLETED", result
        assert result["literal"] == meta["literal"]
        assert not result.get("byoc")
        model = np.load(kits / "site-1" / job_id / "approved-model-dir/best_numpy.npy", allow_pickle=False)
        np.testing.assert_array_equal(model, np.arange(1, 10).reshape(3, 3) + 1)
    finally:
        if session:
            session.close()
        launcher.cleanup()
