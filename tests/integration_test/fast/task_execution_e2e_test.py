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


import importlib
import json
import os
import socket
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
import yaml

from nvflare.recipe import SimEnv
from tests.hello_pt_test_utils import REPO_ROOT, load_hello_pt_module

ADVANCED_DIR = REPO_ROOT / "examples" / "advanced" / "hello-pt-environments"
pytestmark = pytest.mark.usefixtures("single_threaded_torch")


@pytest.mark.timeout(300)
@pytest.mark.parametrize("environment,lifetime", [("sim", "job"), ("poc", "job")])
def test_native_hello_pt_training_and_final_evaluation(tmp_path, monkeypatch, lifetime, environment):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO_ROOT), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.setenv("FL_LOG_LEVEL", "progress")
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    monkeypatch.chdir(ADVANCED_DIR)
    poc_module = importlib.import_module("nvflare.recipe.poc_env")
    monkeypatch.setattr(poc_module, "get_poc_workspace", lambda: str(tmp_path / "poc"))
    with load_hello_pt_module("job.py", example_dir=ADVANCED_DIR) as example:
        args = example.parse_args(
            ["--env", environment, "--execution_lifetime", lifetime, "--num_rounds", "2", "--torch_threads", "1"]
        )
        recipe = example.create_recipe(args)
        env = (
            SimEnv(num_clients=2, workspace_root=str(tmp_path / "simulation"))
            if environment == "sim"
            else example.create_environment(args)
        )
        if environment == "poc":
            project = yaml.safe_load((REPO_ROOT / "tests/integration_test/data/projects/client_api.yml").read_text())
            # Reserve both ports together so xdist workers cannot reuse 8002/8003.
            with ExitStack() as stack:
                sockets = [stack.enter_context(socket.socket()) for _ in range(2)]
                for sock in sockets:
                    sock.bind(("127.0.0.1", 0))
                server = next(p for p in project["participants"] if p["type"] == "server")
                server["fed_learn_port"], server["admin_port"] = [s.getsockname()[1] for s in sockets]
            next(p for p in project["participants"] if p["type"] == "admin")["name"] = env.username
            project_path = tmp_path / "project.yml"
            project_path.write_text(yaml.safe_dump(project))
            env.project_conf_path = str(project_path)
        try:
            run = recipe.execute(env)
            result = run.get_result(timeout=240, clean_up=False)
            assert result is not None
            assert run.get_status() in example.SUCCESS_STATUSES
        finally:
            if environment == "poc":
                env.stop(clean_up=False)
    root = Path(result)
    model_files = list(root.rglob("FL_global_model.pt"))
    assert len(model_files) == 1
    assert isinstance(torch.load(model_files[0], map_location="cpu", weights_only=True)["model"], dict)
    results = json.loads(next(root.rglob("cross_val_results.json")).read_text())
    assert set(results) == {"site-1", "site-2"}
    assert all(30 <= site["SRV_FL_global_model.pt"]["accuracy"] <= 100 for site in results.values())


@pytest.mark.parametrize("gpu", [None, "0"])
def test_local_task_simulation_rejected_before_starting_workers(tmp_path, monkeypatch, gpu):
    from unittest.mock import Mock

    from nvflare.job_config.fed_job_config import FedJobConfig

    monkeypatch.chdir(ADVANCED_DIR)
    with load_hello_pt_module("job.py", example_dir=ADVANCED_DIR) as example:
        # Exercise framework admission independently of the CLI's earlier guard.
        args = example.define_parser().parse_args(["--execution_lifetime", "task"])
        recipe = example.create_recipe(args)
        generate = Mock(side_effect=AssertionError("must reject before export or spawn"))
        monkeypatch.setattr(FedJobConfig, "generate_job_config", generate)
        with pytest.raises(ValueError, match="local Process task execution is deferred"):
            recipe.execute(SimEnv(num_clients=2, gpu_config=gpu, workspace_root=str(tmp_path)))
        generate.assert_not_called()
