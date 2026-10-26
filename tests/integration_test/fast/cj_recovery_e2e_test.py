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

"""Real CP/CJ restart and task redelivery against the unchanged provisioned server."""

import importlib
import json
import os
import socket
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import pytest
import yaml

from nvflare.job_config.api import FedJob
from nvflare.recipe import PocEnv


@pytest.fixture
def recovery_env(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[3]
    fixture = repo / "tests/integration_test/data/cj_recovery"
    monkeypatch.syspath_prepend(str(fixture))
    module = importlib.import_module("recovery_job")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(repo), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.setenv("PATH", os.pathsep.join((str(Path(sys.executable).parent), os.environ["PATH"])))
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    poc_module = importlib.import_module("nvflare.recipe.poc_env")
    monkeypatch.setattr(poc_module, "get_poc_workspace", lambda: str(tmp_path / "poc"))

    project = yaml.safe_load((repo / "tests/integration_test/data/projects/client_api.yml").read_text())
    project["participants"] = [p for p in project["participants"] if p["name"] != "site-2"]
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(2)]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        server = next(p for p in project["participants"] if p["type"] == "server")
        server["fed_learn_port"], server["admin_port"] = [s.getsockname()[1] for s in sockets]
    project_path = tmp_path / "project.yml"
    project_path.write_text(yaml.safe_dump(project))
    env = PocEnv(num_clients=1, username="super@test.org", project_conf_path=str(project_path))
    try:
        yield env, module
    finally:
        env.stop(clean_up=False)


@pytest.mark.timeout(360)
def test_checkpoint_task_survives_repeated_cj_crashes(recovery_env):
    env, module = recovery_env
    job = FedJob(name="cj_recovery", min_clients=1)
    job.to(module.RecoveryController(crashes=4), "server")
    job.to(module.CheckpointExecutor(), "site-1", tasks=["train"])
    job_id = env.deploy(job)
    result = env.get_job_result(job_id, timeout=270)
    assert result is not None
    assert env.get_job_status(job_id) == "FINISHED:COMPLETED"
    results = list(Path(result).rglob("recovery_result.json"))
    assert len(results) == 1
    saved = json.loads(results[0].read_text())
    assert saved["receipts"] == 1
    assert saved["value"] == sum(range(1, 6))
    runs = saved["runs"]
    assert len(runs) == len({r["pid"] for r in runs}) == 5
    for key in ("job", "task", "attempt"):
        assert runs[0][key]
        assert len({r[key] for r in runs}) == 1


@pytest.mark.timeout(180)
def test_detached_trainer_prevents_process_recovery(recovery_env):
    env, module = recovery_env
    job = FedJob(name="cj_detached_trainer", min_clients=1)
    job.to(module.RecoveryController(), "server")
    job.to(module.DetachedTrainerExecutor(), "site-1", tasks=["train"])
    job_id = env.deploy(job)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        records = list(Path(env.poc_workspace).rglob("detached_trainers.json"))
        if records:
            assert len(records) == 1
            launched = json.loads(records[0].read_text())
            assert len(launched) == 1, f"unexpected replacement trainers: {launched}"
            status = env.get_job_status(job_id)
            if status and status.startswith("FINISHED:"):
                assert status == "FINISHED:EXECUTION_EXCEPTION"
                return
        time.sleep(0.1)
    pytest.fail("CJ failure did not finish the job with exactly one detached trainer")
