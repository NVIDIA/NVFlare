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

import json
import os
from pathlib import Path

import torch

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_cross_site_evaluation
from tests.hello_pt_test_utils import load_hello_pt_module

INTEGRATION_TEST_ROOT = os.path.dirname(os.path.dirname(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(INTEGRATION_TEST_ROOT))
EXAMPLE_DIR = os.path.join(REPO_ROOT, "examples", "hello-world", "hello-pt")


def test_zero_flag_hello_pt_produces_learned_loadable_final_model(tmp_path, monkeypatch):
    with load_hello_pt_module("job.py") as job_module:
        monkeypatch.chdir(EXAMPLE_DIR)
        existing_pythonpath = os.environ.get("PYTHONPATH")
        source_pythonpath = REPO_ROOT if not existing_pythonpath else os.pathsep.join((REPO_ROOT, existing_pythonpath))
        monkeypatch.setenv("PYTHONPATH", source_pythonpath)
        args = job_module.define_parser().parse_args([])
        expected_final_round = args.num_rounds - 1
        recipe = job_module.create_recipe(args)
        env = SimEnv(num_clients=2, workspace_root=str(tmp_path / "simulation"))

        run = recipe.execute(env)
        result_path = run.get_result()

    server_run_dir = os.path.join(result_path, "server", "simulate_job")
    with open(os.path.join(server_run_dir, "metrics", "metrics_summary.json")) as summary_file:
        metrics_summary = json.load(summary_file)
    summary_metric_names = {metric["name"] for metric in metrics_summary["final_aggregated_metrics"]}

    assert metrics_summary["status"] == "metrics_reported"
    assert metrics_summary["final_round"] == expected_final_round
    assert "accuracy" in summary_metric_names

    with open(os.path.join(server_run_dir, "metrics", "round_metrics.jsonl")) as metrics_file:
        first_round = json.loads(next(metrics_file))
    first_round_metrics = {metric["name"]: metric["value"] for metric in first_round["aggregated_metrics"]}
    initial_accuracy = first_round_metrics["accuracy"]

    with open(os.path.join(server_run_dir, "cross_site_val", "cross_val_results.json")) as results_file:
        final_results = json.load(results_file)
    final_accuracies = [site_results["SRV_FL_global_model.pt"]["accuracy"] for site_results in final_results.values()]

    assert set(final_results) == {"site-1", "site-2"}
    # These functional thresholds are calibrated to the quickstart's fixed
    # model/data seeds and three-round default, not to arbitrary initialization
    # or hyperparameters. The seeded run has margin above both boundaries.
    assert initial_accuracy <= 20.0
    assert min(final_accuracies) >= 60.0
    assert min(final_accuracies) >= initial_accuracy + 40.0

    artifact_path = os.path.join(server_run_dir, "app_server", "FL_global_model.pt")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    with load_hello_pt_module("job.py") as job_module:
        job_module.create_model().load_state_dict(artifact["model"])


def test_hello_pt_submits_and_cross_evaluates_client_models(tmp_path, monkeypatch):
    # Exercise submit_model and the complete client/server evaluation matrix.
    # The beginner entry point deliberately exposes only final-global evaluation.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((REPO_ROOT, os.environ.get("PYTHONPATH", ""))))
    with load_hello_pt_module("job.py") as job_module:
        recipe = FedAvgRecipe(
            name="hello-pt-cross-site",
            min_clients=2,
            num_rounds=1,
            model=job_module.create_model(),
            train_script=os.path.join(EXAMPLE_DIR, "client.py"),
            train_args=["--dataset", "synthetic"],
        )
        add_cross_site_evaluation(recipe)
        run = recipe.execute(SimEnv(num_clients=2, workspace_root=str(tmp_path / "simulation")))
        result_path = run.get_result()

    evaluation_dir = Path(result_path) / "server" / "simulate_job" / "cross_site_val"
    results = json.loads((evaluation_dir / "cross_val_results.json").read_text())
    sites = {"site-1", "site-2"}
    expected_models = sites | {"SRV_FL_global_model.pt"}
    assert set(results) == sites
    for site in sites:
        assert expected_models <= results[site].keys()
        for model in expected_models:
            assert 0.0 <= results[site][model]["accuracy"] <= 100.0
            assert (evaluation_dir / "result_shareables" / f"{site}_{model}").is_file()
    for model in expected_models:
        assert (evaluation_dir / "model_shareables" / model).is_file()
