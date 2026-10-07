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

"""Qualify the unchanged hello-pt application on Linux in all three environments.

The production rows require a running provisioned Process deployment. Set
NVFLARE_E2E_PROD_ADMIN_KIT and NVFLARE_E2E_PROD_WORKSPACE to its admin kit and
participant workspace root. Missing deployment configuration skips those rows;
it is never counted as production acceptance.
"""

import hashlib
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

from nvflare.recipe import PocEnv, ProdEnv, SimEnv
from tests.hello_pt_test_utils import HELLO_PT_DIR, REPO_ROOT, load_hello_pt_module


def _assert_workers(root, final_metrics, num_rounds, job_id=None):
    publications = []
    launches = []
    for path in Path(root).rglob("diagnostics.jsonl"):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if job_id is not None:
            records = [record for record in records if record["job_id"] == job_id]
        publications.extend(record for record in records if record["event"] == "publication")
        launches.extend(record for record in records if record["event"] == "launched")
        for record in records:
            if record["event"] == "publication":
                attempt = path.parent / "attempts" / record["attempt_id"]
                assert (attempt / "completion.json").is_file()
                assert not (attempt / "input.fobs").exists()
                assert not (attempt / "result.fobs").exists()
                assert not (attempt / "analytics.fobs").exists()
    # The existing locator evaluates every persisted server checkpoint, including
    # the best checkpoint when present, not just FL_global_model.pt.
    expected_tasks = {site: {"train": num_rounds, "validate": len(models)} for site, models in final_metrics.items()}
    expected_workers = sum(sum(tasks.values()) for tasks in expected_tasks.values())
    assert len(publications) == expected_workers
    assert len(launches) == expected_workers
    assert len({record["worker_pid"] for record in publications}) == expected_workers
    assert {record["attempt_id"] for record in launches} == {record["attempt_id"] for record in publications}
    assert {record["site_name"] for record in publications} == set(expected_tasks)
    for site, tasks in expected_tasks.items():
        rows = [record for record in publications if record["site_name"] == site]
        assert Counter(record["task_name"] for record in rows) == tasks
        assert len({record["supervisor_pid"] for record in rows}) == 1
        for record in rows:
            assert record["worker_ppid"] == record["supervisor_pid"]
            assert record["worker_pid"] != record["supervisor_pid"]
            assert record["publication_outcome"] == "accepted"
            assert record["worker_completed_timestamp"] <= record["settled_timestamp"]
            assert record["settled_timestamp"] <= record["publication_timestamp"]
    assert all(
        record["launcher_class"] == "nvflare.app_common.task_launcher.process_launcher.ProcessTaskLauncher"
        and record["launcher_mode"] == "process"
        for record in launches
    )


@pytest.mark.skipif(sys.platform != "linux", reason="This matrix qualifies the Colossus Linux Process profile")
@pytest.mark.timeout(360)
@pytest.mark.parametrize("environment", ["sim", "poc", "prod"])
@pytest.mark.parametrize("execution_lifetime", ["job", "task"])
def test_hello_pt_same_client_across_lifetimes(tmp_path, monkeypatch, environment, execution_lifetime):
    kit = os.environ.get("NVFLARE_E2E_PROD_ADMIN_KIT")
    prod_workspace = os.environ.get("NVFLARE_E2E_PROD_WORKSPACE")
    if environment == "prod" and (not kit or not prod_workspace):
        pytest.skip("a provisioned production Process deployment is required")
    monkeypatch.setenv("FL_LOG_LEVEL", "progress")
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO_ROOT), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.chdir(HELLO_PT_DIR)
    source_hash = hashlib.sha256((HELLO_PT_DIR / "client.py").read_bytes()).hexdigest()
    if environment == "sim":
        env = SimEnv(num_clients=2, workspace_root=str(tmp_path / "simulation"))
    elif environment == "poc":
        module = importlib.import_module("nvflare.recipe.poc_env")
        monkeypatch.setattr(module, "get_poc_workspace", lambda: str(tmp_path / "poc"))
        env = PocEnv(num_clients=2)
    else:
        env = ProdEnv(startup_kit_location=kit)
    with load_hello_pt_module("job.py") as module:
        args = module.define_parser().parse_args([])
        recipe = module.create_recipe(args)
        recipe.set_execution_lifetime(execution_lifetime)
        recipe.export(str(tmp_path / "export"))
        exported_client = list((tmp_path / "export").rglob("client.py"))
        assert len(exported_client) == 1
        assert hashlib.sha256(exported_client[0].read_bytes()).hexdigest() == source_hash
        run = recipe.execute(env)
        result = run.get_result(timeout=300, clean_up=False)
        assert run.get_status() == "FINISHED:COMPLETED"
        assert result is not None
        root = Path(result)
        assert list(root.rglob("FL_global_model.pt"))
        evaluation = list(root.rglob("cross_val_results.json"))
        assert len(evaluation) == 1
        final_metrics = json.loads(evaluation[0].read_text())
        assert set(final_metrics) == {"site-1", "site-2"}
        for values in final_metrics.values():
            assert "SRV_FL_global_model.pt" in values
            assert all(name.startswith("SRV_") and "accuracy" in metrics for name, metrics in values.items())
        if environment != "sim":
            participants = env.poc_workspace if environment == "poc" else prod_workspace
            logs = list(Path(participants).rglob("log.txt"))
            assert any("ProcessJobLauncher" in log.read_text(errors="replace") for log in logs)
        if execution_lifetime == "task":
            diagnostic_root = root if environment == "sim" else participants
            _assert_workers(
                diagnostic_root,
                final_metrics=final_metrics,
                num_rounds=args.num_rounds,
                job_id=None if environment == "sim" else run.job_id,
            )
    assert hashlib.sha256((HELLO_PT_DIR / "client.py").read_bytes()).hexdigest() == source_hash
