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
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_runner():
    repo_root = Path(__file__).parents[3]
    runner_path = repo_root / "skills" / "nvflare-autofl" / "scripts" / "run_job_campaign.py"
    spec = importlib.util.spec_from_file_location("nvflare_autofl_skill_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_args_respect_mutation_schema_bounds():
    runner = _load_runner()
    schema = {"mutable_args": {"lr": {"type": "float", "min": 0.0001, "max": 0.1}}}

    assert runner.candidate_args_allowed(["--lr", "0.1"], schema) == (True, "")

    allowed, reason = runner.candidate_args_allowed(["--lr", "0.2"], schema)
    assert not allowed
    assert "above schema max" in reason


def test_candidate_plan_skips_out_of_bounds_learning_rate():
    runner = _load_runner()
    config = {"search_space": {"suggested": {"lr": {"default": 0.05}}}}
    schema = {"mutable_args": {"lr": {"type": "float", "min": 0.0001, "max": 0.1}}}
    help_text = "--lr"

    candidates = list(runner.candidate_plan(config, help_text, max_candidates=10, schema=schema))
    candidate_commands = [" ".join(candidate.args) for candidate in candidates]

    assert "--lr 0.1" in candidate_commands
    assert "--lr 0.2" not in candidate_commands


def test_profile_budget_suppresses_duplicate_imported_fixed_budget_args():
    runner = _load_runner()
    config = {"budget": {"fixed_training_budget": {"num_clients": 8, "num_rounds": 10}}}
    schema = {
        "comparison_budget_args": {
            "h100_default_candidate_budget": {
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
