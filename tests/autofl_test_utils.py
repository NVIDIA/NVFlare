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


"""Shared setup for policy unit tests and real process integration scenarios."""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path


def _load_runner():
    repo_root = Path(__file__).parents[1]
    runner_path = repo_root / "skills" / "nvflare-autofl" / "scripts" / "run_job_campaign.py"
    spec = importlib.util.spec_from_file_location("nvflare_autofl_skill_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_python_result(python, arguments, *, rc=0, output="done\n", runtime=0.1):
    return rc, output, runtime, [python, *arguments]


def _campaign_config():
    return {
        "schema_version": "nvflare.autofl.config.v1",
        "import": {"source_sha256": "a" * 64, "support": {"status": "supported"}},
        "job": {},
        "objective": {
            "metric": "accuracy",
            "requested_metric": "accuracy",
            "optimization_metric": "accuracy",
            "metric_extraction_order": ["accuracy"],
            "mode": "max",
            "mode_contract_source": "core_default",
            "job_key_metric": "accuracy",
            "job_key_metric_source": "core_default",
            "job_key_metric_mode": "max",
            "job_key_metric_mode_source": "core_default",
        },
        "budget": {"fixed_training_budget": {"num_clients": 2, "num_rounds": 1}},
        "environment": {"requested": "sim"},
        "search_space": {"suggested": {"lr": {"default": 0.1}}},
        "trust_contract": {
            "allowed_edit_paths": ["job.py", "client.py"],
            "allowed_create_patterns": ["**/*.py"],
            "unresolved": [],
        },
    }


def _initialize_fake_campaign(runner, tmp_path, monkeypatch, *, target_env="sim", baseline_score=0.5):
    job = tmp_path / "job.py"
    client = tmp_path / "client.py"
    job.write_text("print('job')\n", encoding="utf-8")
    client.write_text("ALGORITHM = 'baseline'\n", encoding="utf-8")
    config = _campaign_config()
    config["environment"]["requested"] = target_env
    monkeypatch.setattr(runner, "import_job_config", lambda *args, **kwargs: deepcopy(config))
    monkeypatch.setattr(runner, "job_help", lambda *args, **kwargs: "")
    monkeypatch.setattr(runner, "write_progress", lambda path, *args: path.write_bytes(b"progress"))

    def fake_run(run_def, **kwargs):
        return runner.RunRecord(
            run_def.status,
            run_def.name,
            baseline_score,
            1.0,
            "none",
            run_def.description,
            "python job.py",
            str(tmp_path / "artifacts" / run_def.name),
        )

    monkeypatch.setattr(runner, "run_job", fake_run)
    argv = ["initialize", str(job), "--env", target_env]
    assert runner.main(argv) == 0
    return job, client, config


def _write_dynamic_metric_job(workspace):
    workspace.mkdir()
    workspace.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job = workspace / "job.py"
    job.write_text(
        """
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def compute_metric():
    return "accuracy"


recipe = FedAvgRecipe(
    name="dynamic_metric",
    min_clients=2,
    num_rounds=1,
    train_script="client.py",
    key_metric=compute_metric(),
    key_metric_mode="min",
)
recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )
    return job


def _mock_successful_baseline(runner, monkeypatch):
    monkeypatch.setattr(runner, "write_progress", lambda path, *args: path.write_bytes(b"progress"))
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            "baseline", run_def.name, 0.5, 1.0, "none", "baseline", "python job.py", "/tmp/baseline"
        ),
    )
