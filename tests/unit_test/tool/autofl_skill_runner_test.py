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

import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_runner():
    repo_root = Path(__file__).parents[3]
    runner_path = repo_root / "skills" / "nvflare-autofl" / "scripts" / "run_job_campaign.py"
    spec = importlib.util.spec_from_file_location("nvflare_autofl_skill_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _campaign_config():
    return {
        "schema_version": "nvflare.autofl.config.v1",
        "import": {"source_sha256": "a" * 64},
        "job": {"allowed_edit_paths": ["job.py", "client.py"]},
        "objective": {
            "metric": "accuracy",
            "requested_metric": "accuracy",
            "optimization_metric": "accuracy",
            "metric_extraction_order": ["accuracy"],
        },
        "budget": {"fixed_training_budget": {"num_clients": 2, "num_rounds": 1}},
        "environment": {"requested": "sim"},
        "search_space": {"suggested": {"lr": {"default": 0.1}}},
        "trust_contract": {"allowed_edit_paths": ["job.py", "client.py"], "unresolved": []},
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
    argv = ["initialize", str(job), "--env", target_env, "--no-prefer-synthetic"]
    assert runner.main(argv) == 0
    return job, client, config


def test_candidate_args_respect_mutation_schema_bounds():
    runner = _load_runner()
    schema = {"mutable_args": {"lr": {"type": "float", "min": 0.0001, "max": 0.1}}}

    assert runner.candidate_args_allowed(["--lr", "0.1"], schema) == (True, "")

    allowed, reason = runner.candidate_args_allowed(["--lr", "0.2"], schema)
    assert not allowed
    assert "above schema max" in reason


def test_candidate_run_args_cannot_override_fixed_budget():
    runner = _load_runner()
    config = {"budget": {"fixed_training_budget": {"num_clients": 8, "num_rounds": 20}}}

    assert runner.candidate_preserves_fixed_args(["--lr", "0.01"], config, {}) == (True, "")
    assert runner.candidate_preserves_fixed_args(["--num_rounds=2"], config, {})[0] is False
    assert runner.candidate_preserves_fixed_args(["--n-clients", "4"], config, {})[0] is False


def test_candidate_plan_skips_out_of_bounds_learning_rate():
    runner = _load_runner()
    config = {"search_space": {"suggested": {"lr": {"default": 0.05}}}}
    schema = {"mutable_args": {"lr": {"type": "float", "min": 0.0001, "max": 0.1}}}
    help_text = "--lr"

    candidates = list(runner.candidate_plan(config, help_text, max_candidates=10, schema=schema))
    candidate_commands = [" ".join(candidate.args) for candidate in candidates]

    assert "--lr 0.1" in candidate_commands
    assert "--lr 0.2" not in candidate_commands


def test_runner_prefers_explicit_test_accuracy_alias(tmp_path):
    runner = _load_runner()
    result_path = tmp_path / "cross_val_results.json"
    result_path.write_text(
        json.dumps(
            {
                "site-1": {
                    "SRV_FL_global_model.pt": {
                        "accuracy": 0.5,
                        "test_accuracy": 0.8,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert runner.extract_score(tmp_path, ["test_accuracy", "accuracy"]) == 0.8


def test_runner_applies_schema_metric_contract():
    runner = _load_runner()
    config = {
        "objective": {
            "metric": "accuracy",
            "requested_metric": "accuracy",
            "optimization_metric": "accuracy",
            "metric_extraction_order": ["accuracy"],
        }
    }
    schema = {
        "objective": {
            "requested_metric": "accuracy",
            "optimization_metric": "test_accuracy",
            "metric_extraction_order": ["test_accuracy", "accuracy"],
            "metric_source": "held-out CIFAR-10 test set",
        }
    }

    updated = runner.apply_metric_contract(config, "accuracy", schema)

    assert updated["objective"]["metric"] == "accuracy"
    assert updated["objective"]["requested_metric"] == "accuracy"
    assert updated["objective"]["optimization_metric"] == "test_accuracy"
    assert updated["objective"]["metric_extraction_order"] == ["test_accuracy", "accuracy"]
    assert updated["objective"]["metric_source"] == "held-out CIFAR-10 test set"


def test_comparison_budget_suppresses_duplicate_imported_fixed_budget_args():
    runner = _load_runner()
    config = {"budget": {"fixed_training_budget": {"num_clients": 8, "num_rounds": 10}}}
    schema = {
        "comparison_budget_args": {
            "default_candidate_budget": {
                "n_clients": 8,
                "num_rounds": 20,
            }
        }
    }
    help_text = "--n_clients --num_rounds"

    assert runner.build_fixed_args(config, help_text, schema) == []

    args = SimpleNamespace(base_args="", prefer_synthetic=False, synthetic_train_size=1, synthetic_test_size=1)
    assert runner.build_base_args(args, help_text, schema) == ["--n_clients", "8", "--num_rounds", "20"]


def test_run_streams_output_before_timeout(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "run.log"

    rc, output, _ = runner.run(
        [
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(2); print('finished', flush=True)",
        ],
        tmp_path,
        timeout=1,
        log_path=log_path,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert rc == 124
    assert "started" in output
    assert "started" in log_text
    assert "TIMEOUT after 1s" in log_text


def test_run_stops_on_nvflare_simulator_stall_log(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "run.log"
    sim_root = tmp_path / "simulation" / "autofl_candidate"
    server_log = sim_root / "server" / "log_fl.txt"
    server_log.parent.mkdir(parents=True)
    server_log.write_text(
        "SimulatorClientRunner - ERROR - run_client_thread error: RuntimeError: "
        "Failed to create connection to the child process in SimulatorClientRunner, timeout: 60.0\n",
        encoding="utf-8",
    )

    rc, output, runtime = runner.run(
        [
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(30)",
        ],
        tmp_path,
        timeout=30,
        log_path=log_path,
        simulator_stall_roots=[sim_root],
        stall_check_interval=0.01,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert rc == runner.SIMULATOR_STALL_EXIT_CODE
    assert runtime < 5
    assert "SIMULATOR_STALL:" in output
    assert "SIMULATOR_STALL:" in log_text


def test_run_stops_on_nvflare_simulator_no_progress_log(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "run.log"
    sim_root = tmp_path / "simulation" / "autofl_candidate"
    server_log = sim_root / "server" / "log_fl.txt"
    server_log.parent.mkdir(parents=True)
    server_log.write_text("Round 0 started\n", encoding="utf-8")

    rc, output, runtime = runner.run(
        [
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(30)",
        ],
        tmp_path,
        timeout=30,
        log_path=log_path,
        simulator_stall_roots=[sim_root],
        stall_check_interval=0.01,
        simulator_no_progress_timeout=1,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert rc == runner.SIMULATOR_STALL_EXIT_CODE
    assert runtime < 5
    assert "SIMULATOR_STALL: no simulator progress markers changed" in output
    assert "SIMULATOR_STALL: no simulator progress markers changed" in log_text


def test_run_stops_on_stale_partial_simulator_aggregation(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "run.log"
    sim_root = tmp_path / "simulation" / "autofl_candidate"
    server_log = sim_root / "server" / "log_fl.txt"
    site_log = sim_root / "site-1" / "log_fl.txt"
    server_log.parent.mkdir(parents=True)
    site_log.parent.mkdir(parents=True)
    server_log.write_text(
        "Round 0 started\n" "2026-06-25 06:32:33 - FedAvg - INFO - Aggregated 1/8 results\n",
        encoding="utf-8",
    )
    site_log.write_text("[site=site-1] round=0\n", encoding="utf-8")

    rc, output, runtime = runner.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, time; "
                f"path = pathlib.Path({str(site_log)!r}); "
                "time.sleep(0.2); "
                "path.write_text('[site=site-1] round=0\\n[site=site-2] round=0\\n'); "
                "time.sleep(30)"
            ),
        ],
        tmp_path,
        timeout=30,
        log_path=log_path,
        simulator_stall_roots=[sim_root],
        stall_check_interval=0.01,
        simulator_no_progress_timeout=1,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert rc == runner.SIMULATOR_STALL_EXIT_CODE
    assert runtime < 5
    assert "SIMULATOR_STALL: partial simulator aggregation made no server-side progress" in output
    assert "Aggregated 1/8 results" in log_text


def test_runner_state_routes_plateau_to_literature_checkpoint(tmp_path):
    runner = _load_runner()
    records = [
        runner.RunRecord("baseline", "baseline", 0.85, 1.0, "none", "baseline", "python job.py", "/tmp/baseline"),
        runner.RunRecord("discard", "candidate_1", 0.84, 1.0, "none", "candidate", "python job.py", "/tmp/c1"),
        runner.RunRecord("discard", "candidate_2", 0.84, 1.0, "none", "candidate", "python job.py", "/tmp/c2"),
    ]
    results_path = tmp_path / "results.tsv"
    state_path = tmp_path / "state.json"
    runner.write_results(results_path, records)

    state = runner.write_state(
        state_path,
        results_path,
        records,
        None,
        plateau_threshold=2,
    )

    assert state["schema_version"] == "nvflare.autofl.campaign_state.v1"
    assert state["reason"] == "plateau_literature"
    assert state["next_action"] == "run_literature_loop"
    assert state["final_response_allowed"] is False
    assert state == json.loads(state_path.read_text(encoding="utf-8"))


def test_runner_state_finalizes_after_explicit_candidate_cap(tmp_path):
    runner = _load_runner()
    records = [
        runner.RunRecord("baseline", "baseline", 0.85, 1.0, "none", "baseline", "python job.py", "/tmp/baseline"),
        runner.RunRecord("discard", "candidate_1", 0.84, 1.0, "none", "candidate", "python job.py", "/tmp/c1"),
    ]
    results_path = tmp_path / "results.tsv"
    state_path = tmp_path / "state.json"
    runner.write_results(results_path, records)

    state = runner.write_state(state_path, results_path, records, 1)

    assert state["decision"] == "stop"
    assert state["reason"] == "candidate_cap_exhausted"
    assert state["next_action"] == "final_report"
    assert state["final_response_allowed"] is True
    assert state["candidate_cap_source"] == "explicit"


def test_runner_state_ignores_ambient_candidate_cap(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setenv("AUTOFL_MAX_CANDIDATES", "1")
    records = [
        runner.RunRecord("baseline", "baseline", 0.85, 1.0, "none", "baseline", "python job.py", "/tmp/baseline"),
        runner.RunRecord("discard", "candidate_1", 0.84, 1.0, "none", "candidate", "python job.py", "/tmp/c1"),
    ]
    results_path = tmp_path / "results.tsv"
    state_path = tmp_path / "state.json"
    runner.write_results(results_path, records)

    state = runner.write_state(state_path, results_path, records, None)

    assert state["decision"] == "continue"
    assert state["reason"] == "continue"
    assert state["candidate_cap"] is None
    assert state["candidate_cap_source"] == "uncapped"
    assert state["final_response_allowed"] is False


def test_runner_state_marks_infrastructure_retry_non_final(tmp_path):
    runner = _load_runner()
    records = [
        runner.RunRecord(
            runner.INFRASTRUCTURE_RETRY,
            "baseline",
            None,
            1.0,
            "none",
            "baseline",
            "python job.py",
            "/tmp/baseline",
        )
    ]
    results_path = tmp_path / "results.tsv"
    state_path = tmp_path / "state.json"
    runner.write_results(results_path, records)

    state = runner.write_state(state_path, results_path, records, None)

    assert state["decision"] == "retry_infrastructure"
    assert state["reason"] == "infrastructure_retry"
    assert state["next_action"] == "rerun_with_escalated_execution"
    assert state["final_response_allowed"] is False


def test_code_candidate_keeps_improvement_and_restores_discard_without_git(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)

    assert runner.main(["prepare", str(job), "--name", "new_algo", "--hypothesis", "add a new algorithm"]) == 0
    draft = tmp_path / ".nvflare" / "autofl" / "candidates" / "new_algo" / "source"
    draft.joinpath("client.py").write_text("from new_algorithm import VALUE\n", encoding="utf-8")
    draft.joinpath("new_algorithm.py").write_text("VALUE = 'improved'\n", encoding="utf-8")

    def improved_run(run_def, **kwargs):
        return runner.RunRecord(
            "candidate", run_def.name, 0.7, 2.0, "none", run_def.description, "python job.py", "/tmp/new_algo"
        )

    monkeypatch.setattr(runner, "run_job", improved_run)
    assert runner.main(["evaluate", str(job)]) == 0
    assert client.read_text(encoding="utf-8") == "from new_algorithm import VALUE\n"
    assert tmp_path.joinpath("new_algorithm.py").read_text(encoding="utf-8") == "VALUE = 'improved'\n"

    kept_manifest = json.loads(
        tmp_path.joinpath(".nvflare/autofl/candidates/new_algo/candidate_manifest.json").read_text(encoding="utf-8")
    )
    assert kept_manifest["status"] == "keep"
    assert kept_manifest["changed_files"] == ["client.py", "new_algorithm.py"]
    assert kept_manifest["patch_sha256"]

    assert runner.main(["prepare", str(job), "--name", "bad_algo", "--hypothesis", "try a regression"]) == 0
    bad_draft = tmp_path / ".nvflare" / "autofl" / "candidates" / "bad_algo" / "source"
    bad_draft.joinpath("client.py").write_text("ALGORITHM = 'regression'\n", encoding="utf-8")

    def regressed_run(run_def, **kwargs):
        return runner.RunRecord(
            "candidate", run_def.name, 0.3, 2.0, "none", run_def.description, "python job.py", "/tmp/bad_algo"
        )

    monkeypatch.setattr(runner, "run_job", regressed_run)
    assert runner.main(["evaluate", str(job)]) == 0
    assert client.read_text(encoding="utf-8") == "from new_algorithm import VALUE\n"
    records = runner.load_results(tmp_path / "results.tsv")
    assert [record.status for record in records] == ["baseline", "keep", "discard"]
    assert records[1].changed_files == "client.py,new_algorithm.py"
    assert records[1].candidate_manifest.endswith("candidate_manifest.json")


def test_candidate_rejects_unauthorized_existing_source_and_symlink(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    tmp_path.joinpath("secret.py").write_text("SECRET = True\n", encoding="utf-8")
    assert runner.main(["prepare", str(job), "--name", "unsafe", "--hypothesis", "touch secret"]) == 0
    draft = tmp_path / ".nvflare" / "autofl" / "candidates" / "unsafe" / "source"
    draft.joinpath("secret.py").write_text("SECRET = False\n", encoding="utf-8")
    assert runner.main(["evaluate", str(job)]) == 2

    with pytest.raises(ValueError, match="escapes"):
        runner.safe_relative_path(tmp_path, "../outside.py")
    assert tmp_path.joinpath("secret.py").read_text(encoding="utf-8") == "SECRET = True\n"

    draft.joinpath("secret.py").unlink()
    link = draft / "linked.py"
    try:
        link.symlink_to(tmp_path / "secret.py")
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    assert runner.main(["evaluate", str(job)]) == 2


def test_candidate_rejects_stale_manifest_and_budget_drift(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, config = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "stale", "--hypothesis", "change code"]) == 0
    candidate_dir = tmp_path / ".nvflare" / "autofl" / "candidates" / "stale"
    candidate_dir.joinpath("source/client.py").write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")
    manifest_path = candidate_dir / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base_source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert runner.main(["evaluate", str(job)]) == 2

    manifest["base_source_sha256"] = json.loads(
        tmp_path.joinpath(".nvflare/autofl/campaign.json").read_text(encoding="utf-8")
    )["best_source_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    drifted = deepcopy(config)
    drifted["budget"]["fixed_training_budget"]["num_rounds"] = 2
    monkeypatch.setattr(runner, "import_job_config", lambda *args, **kwargs: deepcopy(drifted))
    assert runner.main(["evaluate", str(job)]) == 2
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'baseline'\n"


def test_abandon_candidate_clears_pending_draft_without_touching_best(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "abandoned", "--hypothesis", "temporary idea"]) == 0
    draft = tmp_path / ".nvflare" / "autofl" / "candidates" / "abandoned" / "source" / "client.py"
    draft.write_text("ALGORITHM = 'temporary'\n", encoding="utf-8")

    assert runner.main(["abandon", str(job)]) == 0
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'baseline'\n"
    manifest = json.loads(
        tmp_path.joinpath(".nvflare/autofl/candidates/abandoned/candidate_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "abandoned"
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert state["next_action"] == "propose_candidate"
    assert state["pending_candidate_manifest"] is None


def test_initialize_retries_an_unscored_baseline(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    tmp_path.joinpath("client.py").write_text("ALGORITHM = 'baseline'\n", encoding="utf-8")
    monkeypatch.setattr(runner, "import_job_config", lambda *args, **kwargs: deepcopy(_campaign_config()))
    monkeypatch.setattr(runner, "job_help", lambda *args, **kwargs: "")
    scores = iter([None, 0.5])

    def fake_run(run_def, **kwargs):
        return runner.RunRecord(
            "baseline",
            "baseline",
            next(scores),
            1.0,
            "none",
            "baseline",
            "python job.py",
            "/tmp/baseline",
        )

    monkeypatch.setattr(runner, "run_job", fake_run)
    command = ["initialize", str(job), "--no-prefer-synthetic"]
    assert runner.main(command) == 1
    assert runner.main(command) == 0
    records = runner.load_results(tmp_path / "results.tsv")
    assert [(record.status, record.score) for record in records] == [("baseline", 0.5)]


def test_record_literature_checkpoint_returns_to_agent_proposal(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)

    assert (
        runner.main(
            [
                "record",
                str(job),
                "--literature",
                "--hypothesis",
                "reviewed adaptive federated optimization",
            ]
        )
        == 0
    )
    records = runner.load_results(tmp_path / "results.tsv")
    assert records[-1].status == "literature"
    assert records[-1].diff_summary == "reviewed adaptive federated optimization"
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert state["next_action"] == "propose_candidate"


def test_external_candidate_uses_standard_job_result_recording(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, target_env="prod")
    assert runner.main(["record", str(job), "--baseline", "--score", "0.5", "--job-id", "job-baseline"]) == 0
    assert runner.main(["prepare", str(job), "--name", "prod_algo", "--hypothesis", "production algorithm"]) == 0
    draft = tmp_path / ".nvflare" / "autofl" / "candidates" / "prod_algo" / "source"
    draft.joinpath("client.py").write_text("ALGORITHM = 'production'\n", encoding="utf-8")
    assert runner.main(["evaluate", str(job)]) == 0
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'production'\n"

    manifest_path = tmp_path / ".nvflare" / "autofl" / "candidates" / "prod_algo" / "candidate_manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "ready_for_external_execution"
    assert (
        runner.main(
            [
                "record",
                str(job),
                "--manifest",
                str(manifest_path),
                "--score",
                "0.8",
                "--job-id",
                "job-candidate",
            ]
        )
        == 0
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "keep"
    assert manifest["artifacts"]["job_id"] == "job-candidate"


def test_suggest_returns_fallbacks_without_executing_them(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    capsys.readouterr()
    monkeypatch.setattr(runner, "job_help", lambda *args, **kwargs: "--lr")
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda *args, **kwargs: pytest.fail("suggest must not execute a candidate"),
    )

    assert runner.main(["suggest", str(job), "--limit", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["suggestions"]) == 2
    assert all(item["run_args"] for item in payload["suggestions"])


def test_import_job_config_forwards_minimization_mode(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    output = tmp_path / "autofl.yaml"
    captured = {}

    def fake_run(command, cwd, timeout, log_path):
        captured["command"] = command
        runner.write_yaml(output, _campaign_config())
        return 0, "", 0.0

    monkeypatch.setattr(runner, "run", fake_run)
    args = runner.parse_args(["initialize", str(job), "--mode", "min"])
    runner.import_job_config(args, job, output, tmp_path / "import.log", 10)

    mode_index = captured["command"].index("--mode")
    assert captured["command"][mode_index + 1] == "min"


def test_cli_lifecycle_runs_agent_code_candidate_end_to_end(tmp_path):
    repo_root = Path(__file__).parents[3]
    runner_path = repo_root / "skills" / "nvflare-autofl" / "scripts" / "run_job_campaign.py"
    job = tmp_path / "job.py"
    job.write_text(
        """
import argparse
import json
from pathlib import Path

SCORE = 0.5

class FedAvgRecipe:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class SimEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="run")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--n_clients", type=int, default=2)
    args = parser.parse_args()
    FedAvgRecipe(model=object(), num_rounds=args.num_rounds, min_clients=args.n_clients)
    SimEnv(num_clients=args.n_clients)
    result = Path(f"result-{args.name}")
    result.mkdir(exist_ok=True)
    result.joinpath("metrics_summary.json").write_text(json.dumps({"accuracy": SCORE}))
    print(f"Result can be found in : {result.resolve()}")

if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root), env.get("PYTHONPATH")]))

    subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "initialize",
            str(job),
            "--metric",
            "accuracy",
            "--no-prefer-synthetic",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "prepare",
            str(job),
            "--name",
            "code_candidate",
            "--hypothesis",
            "raise the reported score through a source change",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    draft_job = tmp_path / ".nvflare" / "autofl" / "candidates" / "code_candidate" / "source" / "job.py"
    draft_job.write_text(draft_job.read_text(encoding="utf-8").replace("SCORE = 0.5", "SCORE = 0.8"), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(runner_path), "evaluate", str(job)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    runner = _load_runner()
    records = runner.load_results(tmp_path / "results.tsv")
    assert [(record.status, record.score) for record in records] == [("baseline", 0.5), ("keep", 0.8)]
    assert "SCORE = 0.8" in job.read_text(encoding="utf-8")
    manifest = json.loads(
        tmp_path.joinpath(".nvflare/autofl/candidates/code_candidate/candidate_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "keep"
    assert manifest["changed_files"] == ["job.py"]
