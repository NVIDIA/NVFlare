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
        "import": {"source_sha256": "a" * 64, "support": {"status": "supported"}},
        "job": {},
        "objective": {
            "metric": "accuracy",
            "requested_metric": "accuracy",
            "optimization_metric": "accuracy",
            "metric_extraction_order": ["accuracy"],
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


def _initialize_fake_campaign(runner, tmp_path, monkeypatch, *, target_env="sim", baseline_score=0.5, mode=None):
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
    argv = ["initialize", str(job), "--env", target_env, "--no-prefer-synthetic"]
    if mode is not None:
        argv.extend(["--mode", mode])
    assert runner.main(argv) == 0
    return job, client, config


@pytest.mark.parametrize(
    ("writer_name", "file_name", "replacement"),
    [
        ("write_yaml", "autofl.yaml", {"replacement": "value"}),
        ("write_json", "campaign.json", {"replacement": "value"}),
        ("atomic_write_text", "autofl_report.md", "replacement"),
    ],
)
def test_structured_write_preserves_existing_file_when_temporary_write_fails(
    tmp_path, monkeypatch, writer_name, file_name, replacement
):
    runner = _load_runner()
    config_path = tmp_path / file_name
    original = b"existing: valid\n"
    config_path.write_bytes(original)
    original_write_bytes = Path.write_bytes

    def fail_temporary_write(path, data):
        if path.name.startswith(f".{file_name}.tmp-"):
            original_write_bytes(path, b"partial")
            raise OSError("simulated temporary write failure")
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_temporary_write)

    with pytest.raises(OSError, match="simulated temporary write failure"):
        getattr(runner, writer_name)(config_path, replacement)

    assert config_path.read_bytes() == original
    assert not list(tmp_path.glob(f".{file_name}.tmp-*"))


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
    assert runner.candidate_preserves_fixed_args(["--num_round", "2"], config, {})[0] is False
    with pytest.raises(ValueError, match="canonical long options"):
        runner.candidate_preserves_fixed_args(["-r", "2"], config, {})
    assert runner.candidate_preserves_fixed_args(["--temperature", "-1.5"], config, {}) == (True, "")
    assert runner.candidate_preserves_fixed_args(["--new-algorithm", "fednova"], config, {}) == (True, "")


def test_supported_flags_are_exact_help_tokens():
    runner = _load_runner()
    help_text = "usage: job.py [--num_rounds NUM_ROUNDS] [--name NAME]"

    assert runner.supports_flag(help_text, "--num_rounds")
    assert runner.supports_flag(help_text, "--name")
    assert not runner.supports_flag(help_text, "--num_round")
    assert not runner.supports_flag(help_text, "--round")


def test_candidate_plan_skips_out_of_bounds_learning_rate():
    runner = _load_runner()
    config = {"search_space": {"suggested": {"lr": {"default": 0.05}}}}
    schema = {"mutable_args": {"lr": {"type": "float", "min": 0.0001, "max": 0.1}}}
    help_text = "--lr"

    candidates = list(runner.candidate_plan(config, help_text, max_candidates=10, schema=schema))
    candidate_commands = [" ".join(candidate.args) for candidate in candidates]

    assert "--lr 0.1" in candidate_commands
    assert "--lr 0.2" not in candidate_commands


def test_initialize_merges_existing_mutation_schema_preferred_targets_into_autofl(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    tmp_path.joinpath("client.py").write_text("print('client')\n", encoding="utf-8")
    tmp_path.joinpath("custom_aggregators.py").write_text("class CustomAggregator:\n    pass\n", encoding="utf-8")
    tmp_path.joinpath("mutation_schema.yaml").write_text(
        "preferred_targets:\n  - client.py\n  - custom_aggregators.py\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "import_job_config", lambda *args, **kwargs: deepcopy(_campaign_config()))
    monkeypatch.setattr(runner, "job_help", lambda *args, **kwargs: "")
    monkeypatch.setattr(runner, "write_progress", lambda path, *args: path.write_bytes(b"progress"))
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            run_def.status,
            run_def.name,
            0.5,
            1.0,
            "none",
            run_def.description,
            "python job.py",
            "/tmp/baseline",
        ),
    )

    assert runner.main(["initialize", str(job), "--no-prefer-synthetic"]) == 0

    config = runner.read_yaml(tmp_path / "autofl.yaml")
    assert "custom_aggregators.py" in config["trust_contract"]["allowed_edit_paths"]
    assert config["trust_contract"]["preferred_targets"] == ["client.py", "custom_aggregators.py"]

    assert (
        runner.main(
            [
                "prepare",
                str(job),
                "--name",
                "new_server_aggregator",
                "--hypothesis",
                "implement source-backed server aggregation",
            ]
        )
        == 0
    )
    draft = tmp_path / ".nvflare/autofl/candidates/new_server_aggregator/source/custom_aggregators.py"
    draft.write_text("class CustomAggregator:\n    improved = True\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            "candidate",
            run_def.name,
            0.8,
            1.0,
            "none",
            run_def.description,
            "python job.py",
            "/tmp/candidate",
        ),
    )

    assert runner.main(["evaluate", str(job)]) == 0
    assert "improved = True" in tmp_path.joinpath("custom_aggregators.py").read_text(encoding="utf-8")


def test_missing_or_escaping_preferred_targets_remain_unresolved(tmp_path):
    runner = _load_runner()
    config = deepcopy(_campaign_config())
    schema = {"preferred_targets": ["missing_aggregator.py", "../shared/custom_aggregators.py"]}

    updated = runner.apply_mutation_schema_contract(config, schema, tmp_path)

    assert "missing_aggregator.py" not in updated["trust_contract"]["allowed_edit_paths"]
    assert "../shared/custom_aggregators.py" not in updated["trust_contract"]["allowed_edit_paths"]
    unresolved_reasons = [
        item["reason"] for item in updated["unresolved"] if item["field"] == "mutation_schema.preferred_targets"
    ]
    assert any(reason.startswith("missing_aggregator.py:") for reason in unresolved_reasons)
    assert any(reason.startswith("../shared/custom_aggregators.py:") for reason in unresolved_reasons)


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


def test_metric_order_precedes_artifact_file_order(tmp_path):
    runner = _load_runner()
    tmp_path.joinpath("metrics_summary.json").write_text(json.dumps({"accuracy": 0.7}), encoding="utf-8")
    tmp_path.joinpath("cross_val_results.json").write_text(json.dumps({"test_accuracy": 0.8}), encoding="utf-8")

    assert runner.extract_score(tmp_path, ["test_accuracy", "accuracy"]) == 0.8


def test_structured_metric_extraction_rejects_boolean_values():
    runner = _load_runner()

    assert runner.find_metric_value({"accuracy": True}, ["accuracy"]) is None
    assert runner.find_metric_value({"metrics": [{"name": "accuracy", "value": False}]}, ["accuracy"]) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_metric_paths_reject_non_finite_values(tmp_path, value):
    runner = _load_runner()

    assert runner.find_metric_value({"accuracy": value}, ["accuracy"]) is None
    assert runner.find_metric_value({"metrics": [{"name": "accuracy", "value": value}]}, ["accuracy"]) is None
    assert runner.better(value, 0.5, "max") is False
    assert runner.better(0.6, value, "max") is True
    with pytest.raises(ValueError, match="non-finite score"):
        runner.write_results(
            tmp_path / "results.tsv",
            [runner.RunRecord("keep", "candidate", value, 1.0, "none", "candidate", "run", "/tmp/run")],
        )


def test_text_metric_extraction_supports_scientific_notation(tmp_path):
    runner = _load_runner()
    tmp_path.joinpath("log_fl.txt").write_text("validation_loss: 1.25e-4\n", encoding="utf-8")

    assert runner.extract_score(tmp_path, ["validation_loss"]) == pytest.approx(0.000125)


def test_text_metric_extraction_requires_exact_metric_token_and_records_provenance(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "log.txt"
    log_path.write_text("val_accuracy: 0.9\naccuracy: 0.7\n", encoding="utf-8")

    evidence = runner.extract_metric_evidence(tmp_path, ["accuracy"])

    assert evidence.score == 0.7
    assert evidence.metric_name == "accuracy"
    assert evidence.source == "text:log.txt:line=2"
    assert evidence.artifact == str(log_path.resolve())


def test_text_metric_extraction_ignores_trailing_contextual_mentions(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "log.txt"
    log_path.write_text("final cross-site accuracy: 0.7\ntarget accuracy: 0.5\n", encoding="utf-8")

    evidence = runner.extract_metric_evidence(tmp_path, ["accuracy"])

    assert evidence.score == pytest.approx(0.7)
    assert evidence.source == "text:log.txt:line=1"


def test_text_metric_extraction_supports_keras_progress_output(tmp_path):
    runner = _load_runner()
    tmp_path.joinpath("log.txt").write_text(
        "938/938 - 3s - 3ms/step - accuracy: 0.9687 - loss: 0.0991 - val_accuracy: 0.9816\n",
        encoding="utf-8",
    )

    assert runner.extract_score(tmp_path, ["accuracy"]) == pytest.approx(0.9687)


def test_text_metric_extraction_prefers_nvflare_received_model_evaluation(tmp_path):
    runner = _load_runner()
    tmp_path.joinpath("log.txt").write_text(
        """
2026-07-08 16:25:18,029 - INFO - Accuracy of the received model on round 1 on the test images: 94.3 %
938/938 - 3s - 3ms/step - accuracy: 0.9687 - loss: 0.0991
2026-07-08 16:25:22,029 - INFO - Accuracy of the received model on round 2 on the test images: 98.2 %
""".lstrip(),
        encoding="utf-8",
    )

    evidence = runner.extract_metric_evidence(tmp_path, ["accuracy"])

    assert evidence.score == pytest.approx(98.2)
    assert evidence.source == "text:log.txt:line=3"


def test_text_metric_extraction_supports_embedded_metrics_mapping(tmp_path):
    runner = _load_runner()
    tmp_path.joinpath("log.txt").write_text(
        "validation metric scores on client: site-2 = {'accuracy': 0.764, 'precision': 0.64}\n",
        encoding="utf-8",
    )

    assert runner.extract_score(tmp_path, ["accuracy"]) == pytest.approx(0.764)


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


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_timeout_seconds", [120]),
        ("run_timeout_seconds", {"seconds": 120}),
        ("simulator_no_progress_timeout_seconds", [120]),
        ("simulator_no_progress_timeout_seconds", {"seconds": 120}),
        ("run_timeout_seconds", -1),
        ("run_timeout_seconds", True),
        ("run_timeout_seconds", 1.5),
    ],
)
def test_campaign_timeout_rejects_malformed_schema_values(field, value):
    runner = _load_runner()
    args = SimpleNamespace(timeout=900, simulator_no_progress_timeout=240)
    schema = {"comparison_budget_args": {"default_candidate_budget": {field: value}}}

    with pytest.raises(ValueError, match=f"{field} must be a non-negative integer"):
        runner.campaign_timeout(args, schema)


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


def test_run_streams_partial_line_before_timeout(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "run.log"

    rc, output, runtime = runner.run(
        [
            sys.executable,
            "-c",
            "import sys, time; sys.stdout.write('partial output'); sys.stdout.flush(); time.sleep(30)",
        ],
        tmp_path,
        timeout=1,
        log_path=log_path,
    )

    assert rc == 124
    assert runtime < 5
    assert "partial output" in output
    assert "partial output" in log_path.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="process-group cleanup uses POSIX process groups")
def test_run_terminates_inherited_stdout_descendant_after_leader_exits(tmp_path):
    runner = _load_runner()
    child_pid_path = tmp_path / "descendant.pid"
    child_code = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
        "print('descendant started', flush=True); time.sleep(30)"
    )
    parent_code = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "print('leader exits', flush=True)"
    )

    rc, output, runtime = runner.run(
        [sys.executable, "-c", parent_code],
        tmp_path,
        timeout=1,
        log_path=tmp_path / "run.log",
    )

    assert rc == 124
    assert runtime < 5
    assert "leader exits" in output
    assert "descendant started" in tmp_path.joinpath("run.log").read_text(encoding="utf-8")
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = runner.time.monotonic() + 5
    while runner.time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        proc_stat = Path(f"/proc/{child_pid}/stat")
        if proc_stat.is_file() and proc_stat.read_text(encoding="utf-8").split()[2] == "Z":
            break
        runner.time.sleep(0.05)
    else:
        pytest.fail(f"descendant process {child_pid} survived process-group cleanup")


@pytest.mark.skipif(os.name == "nt", reason="process-group cleanup uses POSIX process groups")
@pytest.mark.parametrize("monitor_error", [KeyboardInterrupt(), RuntimeError("monitor failed")])
def test_run_terminates_child_process_group_when_monitor_raises(tmp_path, monkeypatch, monitor_error):
    runner = _load_runner()
    pid_path = tmp_path / "child.pid"

    def fail_monitor(_roots):
        deadline = runner.time.monotonic() + 5
        while runner.time.monotonic() < deadline:
            try:
                int(pid_path.read_text(encoding="utf-8"))
                break
            except (FileNotFoundError, ValueError):
                pass
            runner.time.sleep(0.01)
        else:
            pytest.fail("child PID was not published before the monitor deadline")
        raise monitor_error

    monkeypatch.setattr(runner, "simulator_stall_message", fail_monitor)
    with pytest.raises(type(monitor_error), match=str(monitor_error) if str(monitor_error) else None):
        runner.run(
            [
                sys.executable,
                "-c",
                f"import os, pathlib, time; pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(30)",
            ],
            tmp_path,
            timeout=0,
            log_path=tmp_path / "run.log",
            simulator_stall_roots=[tmp_path / "simulation"],
            stall_check_interval=0,
        )

    child_pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_run_keeps_complete_log_but_only_bounded_output_tail(tmp_path):
    runner = _load_runner()
    result_dir = tmp_path / "early-result"
    log_path = tmp_path / "run.log"
    payload_size = runner.MAX_CAPTURED_PROCESS_OUTPUT + 65536
    script = (
        f"print('Result can be found in : {result_dir}'); "
        "print('PermissionError'); print('[Errno 1] Operation not permitted'); print('socket.bind'); "
        f"print('x' * {payload_size})"
    )

    rc, output, _ = runner.run(
        [sys.executable, "-c", script],
        tmp_path,
        timeout=10,
        log_path=log_path,
    )

    printed_result, socket_failure = runner.scan_run_log(log_path, tmp_path)
    assert rc == 0
    assert len(output.encode("utf-8")) <= runner.MAX_CAPTURED_PROCESS_OUTPUT
    assert log_path.stat().st_size > runner.MAX_CAPTURED_PROCESS_OUTPUT
    assert "Result can be found" not in output
    assert printed_result == result_dir.resolve()
    assert socket_failure is True


def test_socket_failure_marker_only_requests_human_runner_approval(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")

    def fake_run(_argv, _cwd, _timeout, log_path, **_kwargs):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "PermissionError\n[Errno 1] Operation not permitted\nsocket.bind\n",
            encoding="utf-8",
        )
        return 1, "socket.bind failed\n", 0.1

    monkeypatch.setattr(runner, "run", fake_run)
    record = runner.run_job(
        runner.JobRun("candidate", [], "candidate"),
        python=sys.executable,
        job=job,
        cwd=tmp_path,
        help_text="",
        fixed_args=[],
        base_args=[],
        output_root=tmp_path / "runs",
        timeout=10,
        simulator_no_progress_timeout=0,
        metrics=["accuracy"],
        config={"job": {}},
    )

    assert record.status == runner.INFRASTRUCTURE_RETRY
    assert "human-approved runner execution scope" in record.failure_reason
    assert "escalated execution" not in record.failure_reason

    results_path = tmp_path / "results.tsv"
    runner.write_results(results_path, [record])
    state = runner.write_state(tmp_path / "state.json", results_path, [record], None)

    assert state["next_action"] == runner.SIMULATION_APPROVAL_ACTION
    assert "Pause for human approval" in state["agent_instruction"]
    assert "Never request broader permission" in state["agent_instruction"]


def test_job_help_discovers_flags_without_executing_job(tmp_path):
    runner = _load_runner()
    job = tmp_path / "job.py"
    marker = tmp_path / "executed"
    job.write_text(
        f"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--n_clients", type=int, default=2)
parser.add_argument("-r", "--num_rounds", type=int, default=1)
Path({str(marker)!r}).write_text("executed")
""".lstrip(),
        encoding="utf-8",
    )

    help_text = runner.job_help(sys.executable, job, tmp_path, timeout=1)

    assert runner.supported_long_flags(help_text) == {"--n_clients", "--num_rounds"}
    assert not marker.exists()


def test_job_help_returns_no_flags_for_non_argparse_job(tmp_path):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")

    assert runner.job_help(sys.executable, job, tmp_path, timeout=1) == ""


def test_job_help_ignores_unrelated_builders_and_unreachable_parsers(tmp_path):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text(
        """
import argparse


class Builder:
    def add_argument(self, *args):
        pass


def unused_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unused")


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=1)
    Builder().add_argument("--not-argparse")
    return parser.parse_args()


def main():
    define_parser()
""".lstrip(),
        encoding="utf-8",
    )

    assert runner.job_help(sys.executable, job, tmp_path) == "--num_rounds"


def test_metric_extraction_prefers_structured_artifacts_then_known_text_logs(tmp_path):
    runner = _load_runner()
    nested = tmp_path / "server"
    nested.mkdir()
    nested.joinpath("metrics_summary.json").write_text(json.dumps({"accuracy": 0.7}), encoding="utf-8")
    nested.joinpath("cross_val_results.json").write_text(json.dumps({"accuracy": 0.8}), encoding="utf-8")
    nested.joinpath("log_fl.txt").write_text("accuracy: 0.9\n", encoding="utf-8")
    tmp_path.joinpath("run.log").write_text("accuracy: 0.95\n", encoding="utf-8")

    assert runner.extract_score(tmp_path, ["accuracy"]) == 0.7
    nested.joinpath("metrics_summary.json").unlink()
    assert runner.extract_score(tmp_path, ["accuracy"]) == 0.8
    nested.joinpath("cross_val_results.json").unlink()
    assert runner.extract_score(tmp_path, ["accuracy"]) == 0.95


def test_result_paths_and_static_job_names_are_resolved_deterministically(tmp_path, monkeypatch):
    runner = _load_runner()
    default_root = tmp_path / "system-tmp" / "nvflare" / "simulation"
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(tmp_path / "system-tmp"))
    relative = tmp_path / "relative-result"
    relative.mkdir()
    config = {
        "job": {
            "recipe_args": {"name": {"value": "recipe-name", "confidence": "high"}},
            "fed_job_args": {"name": {"value": "fed-job-name", "confidence": "high"}},
        }
    }

    assert runner.extract_result_dir("Result can be found in : relative-result", tmp_path) == relative.resolve()
    assert runner.extract_result_dir("result location = relative-result", tmp_path) == relative.resolve()
    assert (
        runner.extract_result_dir("The simulation logs can be found at relative-result", tmp_path) == relative.resolve()
    )
    assert runner.imported_job_names(config) == ["recipe-name", "fed-job-name"]
    assert runner.expected_simulator_roots(config, "injected", tmp_path) == [
        (default_root / "injected").resolve(),
        (default_root / "recipe-name").resolve(),
        (default_root / "fed-job-name").resolve(),
    ]
    assert runner.simulator_run_name("baseline", tmp_path) != runner.simulator_run_name(
        "baseline", tmp_path / "other-campaign"
    )
    config["job"]["recipe_args"]["name"]["confidence"] = "low"
    assert runner.imported_job_names(config) == ["fed-job-name"]
    with pytest.raises(ValueError, match="unsafe simulator job name"):
        runner.expected_simulator_roots({}, "../other-run", tmp_path)


def test_run_without_deterministic_result_root_records_actionable_failure(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('done')\n", encoding="utf-8")
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: (0, "done\n", 0.1))

    def probe_must_not_run(*args, **kwargs):
        raise AssertionError("probe must not run when no SimEnv environment was discovered")

    monkeypatch.setattr(runner, "probe_simulator_workspace_override_support", probe_must_not_run)

    record = runner.run_job(
        runner.JobRun("candidate", [], "candidate"),
        python=sys.executable,
        job=job,
        cwd=tmp_path,
        help_text="",
        fixed_args=[],
        base_args=[],
        output_root=tmp_path / "runs",
        timeout=10,
        simulator_no_progress_timeout=0,
        metrics=["accuracy"],
        config={"job": {}},
    )

    assert record.status == "crash"
    assert "print the direct simulator result directory" in record.failure_reason


def test_run_without_result_root_blames_nvflare_without_workspace_override(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('done')\n", encoding="utf-8")
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: (0, "done\n", 0.1))
    monkeypatch.setattr(
        runner,
        "probe_simulator_workspace_override_support",
        lambda *args, **kwargs: {"version": "2.8.0", "supported": False},
    )

    record = runner.run_job(
        runner.JobRun("candidate", [], "candidate"),
        python=sys.executable,
        job=job,
        cwd=tmp_path,
        help_text="",
        fixed_args=[],
        base_args=[],
        output_root=tmp_path / "runs",
        timeout=10,
        simulator_no_progress_timeout=0,
        metrics=["accuracy"],
        config={"job": {}, "environment": {"discovered": {"name": "SimEnv", "args": {}}}},
    )

    assert record.status == "crash"
    assert "installed nvflare (2.8.0) does not honor NVFLARE_SIMULATOR_WORKSPACE_ROOT" in record.failure_reason
    assert f"nvflare>={runner.SIMULATOR_WORKSPACE_OVERRIDE_MIN_NVFLARE_VERSION}" in record.failure_reason
    assert "print the direct simulator result directory" not in record.failure_reason


def test_run_without_result_root_keeps_generic_diagnosis_for_fed_job(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('done')\n", encoding="utf-8")
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: (0, "done\n", 0.1))

    def probe_must_not_run(*args, **kwargs):
        raise AssertionError("probe must not run when the job does not use SimEnv")

    monkeypatch.setattr(runner, "probe_simulator_workspace_override_support", probe_must_not_run)

    record = runner.run_job(
        runner.JobRun("candidate", [], "candidate"),
        python=sys.executable,
        job=job,
        cwd=tmp_path,
        help_text="",
        fixed_args=[],
        base_args=[],
        output_root=tmp_path / "runs",
        timeout=10,
        simulator_no_progress_timeout=0,
        metrics=["accuracy"],
        config={"job": {"fed_job_args": {"name": {"value": "candidate_job", "confidence": "high"}}}},
    )

    assert record.status == "crash"
    assert "no deterministic NVFlare result directory" in record.failure_reason
    assert "upgrade to nvflare" not in record.failure_reason


def test_unresolved_result_dir_failure_reason_uses_version_when_probe_is_inconclusive(tmp_path, monkeypatch):
    runner = _load_runner()
    probes = {}
    sim_env_config = {"environment": {"discovered": {"name": "SimEnv", "args": {}}}}

    def fake_probe(*args, **kwargs):
        return probes["result"]

    monkeypatch.setattr(runner, "probe_simulator_workspace_override_support", fake_probe)

    probes["result"] = {"version": "2.6.2", "supported": None}
    reason = runner.unresolved_result_dir_failure_reason(sys.executable, tmp_path, sim_env_config)
    assert "installed nvflare (2.6.2) does not honor" in reason
    assert f"nvflare>={runner.SIMULATOR_WORKSPACE_OVERRIDE_MIN_NVFLARE_VERSION}" in reason

    probes["result"] = {"version": "", "supported": None}
    assert "print the direct simulator result directory" in runner.unresolved_result_dir_failure_reason(
        sys.executable, tmp_path, sim_env_config
    )

    probes["result"] = {"version": "2.8.0", "supported": True}
    assert "print the direct simulator result directory" in runner.unresolved_result_dir_failure_reason(
        sys.executable, tmp_path, sim_env_config
    )


def test_unresolved_result_dir_failure_reason_stays_generic_without_sim_env(tmp_path, monkeypatch):
    runner = _load_runner()

    def probe_must_not_run(*args, **kwargs):
        raise AssertionError("probe must not run when the discovered environment is not SimEnv")

    monkeypatch.setattr(runner, "probe_simulator_workspace_override_support", probe_must_not_run)

    for config in (
        {"job": {}},
        {"job": {"fed_job_args": {"name": {"value": "fraud_job", "confidence": "high"}}}},
        {"environment": {"discovered": {"name": "PocEnv", "args": {}}}},
        {"environment": {}},
        {"environment": None},
    ):
        reason = runner.unresolved_result_dir_failure_reason(sys.executable, tmp_path, config)
        assert "print the direct simulator result directory" in reason
        assert "upgrade to nvflare" not in reason


def test_probe_simulator_workspace_override_support_inspects_installed_sim_env(tmp_path):
    runner = _load_runner()
    sim_env = tmp_path / "nvflare" / "recipe" / "sim_env.py"
    sim_env.parent.mkdir(parents=True)
    tmp_path.joinpath("nvflare", "__init__.py").write_text("__version__ = '2.7.0'\n", encoding="utf-8")
    sim_env.parent.joinpath("__init__.py").write_text("", encoding="utf-8")

    sim_env.write_text("WORKSPACE_ROOT = '/tmp/nvflare/simulation'\n", encoding="utf-8")
    probe = runner.probe_simulator_workspace_override_support(sys.executable, tmp_path)
    assert probe["supported"] is False
    # version must describe the imported (shadowing) package, not any installed distribution's metadata
    assert probe["version"] == "2.7.0"

    sim_env.write_text(
        "SIMULATOR_WORKSPACE_ROOT_ENV_VAR = 'NVFLARE_SIMULATOR_WORKSPACE_ROOT'\n",
        encoding="utf-8",
    )
    probe = runner.probe_simulator_workspace_override_support(sys.executable, tmp_path)
    assert probe["supported"] is True

    probe = runner.probe_simulator_workspace_override_support(str(tmp_path / "missing-python"), tmp_path)
    assert probe == {"version": "", "supported": None}


def test_probe_simulator_workspace_override_support_uses_sanitized_env(tmp_path, monkeypatch):
    runner = _load_runner()
    captured = {}

    monkeypatch.setenv("PATH", "/safe/bin")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "pythonpath"))
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AUTOFL_TEST_TOKEN", "secret")

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return 0, 'native warning\n{"version": "2.9.0", "supported": true}\nlate stderr warning\n', 0.1

    monkeypatch.setattr(runner, "run", fake_run)

    probe = runner.probe_simulator_workspace_override_support(sys.executable, tmp_path)

    assert probe == {"version": "2.9.0", "supported": True}
    assert captured["env"]["PATH"] == "/safe/bin"
    assert captured["env"]["PYTHONPATH"] == str(tmp_path / "pythonpath")
    assert runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR in captured["env"]
    assert set(captured["env"]) <= set(runner.SIMULATOR_ENV_ALLOWLIST) | {runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR}
    assert "AWS_SECRET_ACCESS_KEY" not in captured["env"]
    assert "AUTOFL_TEST_TOKEN" not in captured["env"]


def test_nvflare_version_predates_workspace_override():
    runner = _load_runner()

    assert runner.nvflare_version_predates_workspace_override("2.8.0")
    assert runner.nvflare_version_predates_workspace_override("2.7.1+160.g67022752b")
    assert not runner.nvflare_version_predates_workspace_override("2.9.0")
    assert not runner.nvflare_version_predates_workspace_override("2.10.0rc1")
    assert not runner.nvflare_version_predates_workspace_override("3.0.0")
    assert not runner.nvflare_version_predates_workspace_override("")
    assert not runner.nvflare_version_predates_workspace_override("unknown")


def test_simulator_child_env_uses_allowlisted_runtime_context(tmp_path, monkeypatch):
    runner = _load_runner()
    simulator_base = tmp_path / "simulation"
    venv = tmp_path / "venv"
    pythonpath = tmp_path / "pythonpath"

    monkeypatch.setenv("PATH", "/safe/bin")
    monkeypatch.setenv("PYTHONPATH", str(pythonpath))
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example")
    monkeypatch.setenv("no_proxy", "localhost")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "ca.pem"))
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "cert.pem"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "Users" / "tester"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AUTOFL_TEST_TOKEN", "secret")

    env = runner.simulator_child_env(simulator_base)

    assert env["PATH"] == "/safe/bin"
    assert env["PYTHONPATH"] == str(pythonpath)
    assert env["VIRTUAL_ENV"] == str(venv)
    assert env["HTTPS_PROXY"] == "http://proxy.example"
    assert env["no_proxy"] == "localhost"
    assert env["REQUESTS_CA_BUNDLE"] == str(tmp_path / "ca.pem")
    assert env["SSL_CERT_FILE"] == str(tmp_path / "cert.pem")
    assert env["USERPROFILE"] == str(tmp_path / "Users" / "tester")
    assert env["APPDATA"] == str(tmp_path / "AppData" / "Roaming")
    assert env["LOCALAPPDATA"] == str(tmp_path / "AppData" / "Local")
    assert env[runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR] == str(simulator_base)
    assert set(env) <= set(runner.SIMULATOR_ENV_ALLOWLIST) | {runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR}
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "AUTOFL_TEST_TOKEN" not in env


def test_simulator_child_env_uses_configured_passthrough(tmp_path, monkeypatch):
    runner = _load_runner()
    simulator_base = tmp_path / "simulation"
    custom_path = tmp_path / "dataset"
    config = {"environment": {runner.SIMULATOR_ENV_PASSTHROUGH_CONFIG_KEY: ["DATASET_DIR", "OMP_NUM_THREADS"]}}

    monkeypatch.setenv("DATASET_DIR", str(custom_path))
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    extra_names = runner.simulator_env_passthrough_names(config)
    env = runner.simulator_child_env(simulator_base, extra_names)

    assert extra_names == ["DATASET_DIR", "OMP_NUM_THREADS"]
    assert env["DATASET_DIR"] == str(custom_path)
    assert env["OMP_NUM_THREADS"] == "4"
    assert set(env) <= set(runner.SIMULATOR_ENV_ALLOWLIST) | set(extra_names) | {
        runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR
    }


@pytest.mark.parametrize(
    "values",
    [
        "DATASET_DIR",
        ["BAD-NAME"],
        [3],
    ],
)
def test_simulator_env_passthrough_names_rejects_invalid_values(values):
    runner = _load_runner()

    with pytest.raises(ValueError, match="simulator_env_passthrough"):
        runner.simulator_env_passthrough_names({"environment": {runner.SIMULATOR_ENV_PASSTHROUGH_CONFIG_KEY: values}})


def test_run_discovers_and_persists_printed_unnamed_simulator_root(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    simulator_base = tmp_path / "simulation"
    result = simulator_base / "recipe-default"

    def fake_run(*args, **kwargs):
        result = Path(kwargs["env"][runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR]) / "recipe-default"
        result.mkdir(parents=True)
        result.joinpath("metrics_summary.json").write_text(json.dumps({"accuracy": 0.81}), encoding="utf-8")
        return 0, f"The simulation logs can be found at {result}\n", 0.1

    monkeypatch.setattr(runner, "run", fake_run)
    config = {
        "job": {},
        "artifacts": {},
        "environment": {
            "discovered": {"args": {"workspace_root": {"value": str(simulator_base), "confidence": "high"}}}
        },
    }

    record = runner.run_job(
        runner.JobRun("baseline", [], "baseline", status="baseline"),
        python=sys.executable,
        job=job,
        cwd=tmp_path,
        help_text="",
        fixed_args=[],
        base_args=[],
        output_root=tmp_path / "runs",
        timeout=10,
        simulator_no_progress_timeout=0,
        metrics=["accuracy"],
        config=config,
    )

    assert record.status == "baseline"
    assert record.score == pytest.approx(0.81)
    assert config["artifacts"]["simulator_result_name"] == "recipe-default"
    assert runner.expected_simulator_roots(config, None, tmp_path) == [result.resolve()]


def test_run_discovers_single_changed_unnamed_simulator_root(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    simulator_base = tmp_path / "simulation"
    result = simulator_base / "recipe-default"

    def fake_run(*args, **kwargs):
        result = Path(kwargs["env"][runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR]) / "recipe-default"
        result.mkdir(parents=True)
        result.joinpath("metrics_summary.json").write_text(json.dumps({"val_acc": 0.73}), encoding="utf-8")
        return 0, "job complete without a result message\n", 0.1

    monkeypatch.setattr(runner, "run", fake_run)
    config = {
        "job": {},
        "artifacts": {},
        "environment": {
            "discovered": {"args": {"workspace_root": {"value": str(simulator_base), "confidence": "high"}}}
        },
    }

    record = runner.run_job(
        runner.JobRun("baseline", [], "baseline", status="baseline"),
        python=sys.executable,
        job=job,
        cwd=tmp_path,
        help_text="",
        fixed_args=[],
        base_args=[],
        output_root=tmp_path / "runs",
        timeout=10,
        simulator_no_progress_timeout=0,
        metrics=["val_acc"],
        config=config,
    )

    assert record.status == "baseline"
    assert record.score == pytest.approx(0.73)
    assert config["artifacts"]["simulator_result_name"] == "recipe-default"


def test_run_rejects_ambiguous_changed_unnamed_simulator_roots(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    simulator_base = tmp_path / "simulation"

    def fake_run(*args, **kwargs):
        simulator_base = Path(kwargs["env"][runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR])
        for name in ("first", "second"):
            result = simulator_base / name
            result.mkdir(parents=True)
            result.joinpath("metrics_summary.json").write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")
        return 0, "ambiguous job complete\n", 0.1

    monkeypatch.setattr(runner, "run", fake_run)
    record = runner.run_job(
        runner.JobRun("candidate", [], "candidate"),
        python=sys.executable,
        job=job,
        cwd=tmp_path,
        help_text="",
        fixed_args=[],
        base_args=[],
        output_root=tmp_path / "runs",
        timeout=10,
        simulator_no_progress_timeout=0,
        metrics=["accuracy"],
        config={
            "job": {},
            "environment": {
                "discovered": {"args": {"workspace_root": {"value": str(simulator_base), "confidence": "high"}}}
            },
        },
    )

    assert record.status == "crash"
    assert record.score is None


def test_run_rejects_printed_result_outside_simulator_workspace(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    outside = tmp_path / "outside"

    def fake_run(*args, **kwargs):
        outside.mkdir()
        return 0, f"Result can be found in : {outside}\n", 0.1

    monkeypatch.setattr(runner, "run", fake_run)
    record = runner.run_job(
        runner.JobRun("candidate", [], "candidate"),
        python=sys.executable,
        job=job,
        cwd=tmp_path,
        help_text="",
        fixed_args=[],
        base_args=[],
        output_root=tmp_path / "runs",
        timeout=10,
        simulator_no_progress_timeout=0,
        metrics=["accuracy"],
        config={
            "job": {},
            "environment": {
                "discovered": {
                    "args": {"workspace_root": {"value": str(tmp_path / "simulation"), "confidence": "high"}}
                }
            },
        },
    )

    assert record.status == "crash"
    assert record.score is None


def test_campaign_workspace_lock_rejects_concurrent_same_job_lifecycle(tmp_path):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")

    with runner.locked_campaign_workspace(tmp_path, "evaluate"):
        result = subprocess.run(
            [sys.executable, runner.__file__, "status", str(job)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    assert result.returncode == 2
    assert "campaign workspace is already in use" in result.stderr


def test_campaign_workspace_locks_are_independent_and_release_after_exception(tmp_path):
    runner = _load_runner()
    first = tmp_path / "first"
    second = tmp_path / "second"

    with pytest.raises(RuntimeError, match="injected lifecycle failure"):
        with runner.locked_campaign_workspace(first, "evaluate"):
            with runner.locked_campaign_workspace(second, "evaluate"):
                raise RuntimeError("injected lifecycle failure")

    with runner.locked_campaign_workspace(first, "status"):
        with runner.locked_campaign_workspace(second, "status"):
            pass


def test_run_job_collects_configured_sim_result_and_standard_nvflare_text_metric(tmp_path):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text(
        f"""
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="run")
args = parser.parse_args()
result = Path(os.environ["{runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR}"]) / args.name
server = result / "server"
server.mkdir(parents=True, exist_ok=True)
server.joinpath("log.txt").write_text("accuracy: 0.76\\n")
print(f"Result can be found in : {{result}}")
""".lstrip(),
        encoding="utf-8",
    )

    record = runner.run_job(
        runner.JobRun("candidate", [], "candidate"),
        python=sys.executable,
        job=job,
        cwd=tmp_path,
        help_text="--name NAME",
        fixed_args=[],
        base_args=[],
        output_root=tmp_path / "runs",
        timeout=10,
        simulator_no_progress_timeout=0,
        metrics=["accuracy"],
        config={
            "job": {},
            "environment": {
                "discovered": {
                    "name": "SimEnv",
                    "args": {
                        "workspace_root": {
                            "value": str(tmp_path / "simulation"),
                            "confidence": "high",
                        }
                    },
                }
            },
        },
    )

    assert record.status == "candidate"
    assert record.score == pytest.approx(0.76)
    assert record.metric_source == "text:log.txt:line=1"
    assert tmp_path.joinpath("runs/candidate/simulation/server/log.txt").is_file()
    assert tmp_path.joinpath("runs/candidate/run.log").is_file()
    assert not tmp_path.joinpath("simulation").exists()


def test_run_job_uses_a_fresh_simulator_workspace_for_each_trial(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    workspaces = []

    def fake_run(*args, **kwargs):
        workspace = Path(kwargs["env"][runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR])
        workspaces.append(workspace)
        result = workspace / "fixed-job"
        result.mkdir(parents=True)
        result.joinpath("metrics_summary.json").write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")
        return 0, f"Result can be found in : {result}\n", 0.1

    monkeypatch.setattr(runner, "run", fake_run)
    config = {"job": {"recipe_args": {"name": {"value": "fixed-job", "confidence": "high"}}}}
    for name in ("candidate-one", "candidate-two"):
        record = runner.run_job(
            runner.JobRun(name, [], name),
            python=sys.executable,
            job=job,
            cwd=tmp_path,
            help_text="",
            fixed_args=[],
            base_args=[],
            output_root=tmp_path / "runs",
            timeout=10,
            simulator_no_progress_timeout=0,
            metrics=["accuracy"],
            config=config,
        )
        assert record.score == pytest.approx(0.5)

    assert workspaces[0] != workspaces[1]
    assert all(not workspace.exists() for workspace in workspaces)


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


def test_runner_uses_campaign_guard_threshold_defaults(tmp_path, monkeypatch):
    runner = _load_runner()
    guard = runner.load_campaign_guard()
    monkeypatch.setattr(guard, "DEFAULT_PLATEAU_THRESHOLD", 3)
    monkeypatch.setattr(guard, "DEFAULT_MIN_DELTA", 0.25)
    monkeypatch.setattr(guard, "DEFAULT_HARD_CRASH_THRESHOLD", 2)
    monkeypatch.delenv("AUTOFL_PLATEAU_THRESHOLD", raising=False)
    monkeypatch.delenv("AUTOFL_PLATEAU_MIN_DELTA", raising=False)
    monkeypatch.delenv("AUTOFL_HARD_CRASH_THRESHOLD", raising=False)

    args = runner.parse_args(["status", "job.py"])

    assert args.plateau_threshold == 3
    assert args.plateau_min_delta == pytest.approx(0.25)
    assert args.hard_crash_threshold == 2

    records = [
        runner.RunRecord("baseline", "baseline", 0.5, 1.0, "none", "baseline", "run", "/tmp/baseline"),
        runner.RunRecord("crash", "candidate_1", None, 1.0, "none", "candidate", "run", "/tmp/c1"),
        runner.RunRecord("crash", "candidate_2", None, 1.0, "none", "candidate", "run", "/tmp/c2"),
    ]
    results_path = tmp_path / "results.tsv"
    runner.write_results(results_path, records)
    state = runner.write_state(tmp_path / "state.json", results_path, records, None)

    assert state["reason"] == "hard_repeated_crash_blocker"
    assert state["plateau"]["threshold"] == 3
    assert state["plateau"]["min_delta"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "option",
    ["--hard-crash-threshold", "--exploration-batch-size", "--family-repeat-limit"],
)
def test_runner_rejects_negative_campaign_thresholds(option):
    runner = _load_runner()
    args = runner.parse_args(["status", "job.py", option, "-1"])

    with pytest.raises(ValueError, match="must be non-negative"):
        runner.validate_args(args)


def test_small_retained_improvement_does_not_reset_plateau_clock(tmp_path):
    runner = _load_runner()
    records = [
        runner.RunRecord("baseline", "baseline", 0.85, 1.0, "none", "baseline", "python job.py", "/tmp/baseline"),
        runner.RunRecord("keep", "candidate_1", 0.8503, 1.0, "client.py", "candidate", "python job.py", "/tmp/c1"),
    ]
    results_path = tmp_path / "results.tsv"
    runner.write_results(results_path, records)

    state = runner.write_state(
        tmp_path / "state.json",
        results_path,
        records,
        None,
        plateau_threshold=1,
        plateau_min_delta=0.0005,
    )

    assert runner.best_retained_record(records, "max").name == "candidate_1"
    assert state["best_score"] == pytest.approx(0.8503)
    assert state["plateau"]["best_score"] == pytest.approx(0.85)
    assert state["reason"] == "plateau_literature"
    assert state["next_action"] == "run_literature_loop"


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
    assert state["remaining_candidates"] == 0
    assert state["abandoned_candidates"] == 0
    for deliverable in ("autofl_report.md", "results.tsv", "progress.png", "baseline vs best"):
        assert deliverable in state["agent_instruction"]


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
    assert state["next_action"] == "await_simulation_runner_approval"
    assert state["final_response_allowed"] is False
    assert "Pause for human approval" in state["agent_instruction"]
    assert "log output" in state["agent_instruction"]


def test_runner_state_infrastructure_retry_keeps_capped_budget_consistent_with_guard(tmp_path):
    runner = _load_runner()
    records = [
        runner.RunRecord("baseline", "baseline", 0.5, 1.0, "none", "baseline", "run", "/tmp/baseline"),
        runner.RunRecord("keep", "candidate_1", 0.6, 1.0, "none", "candidate", "run", "/tmp/c1"),
        runner.RunRecord("discard", "candidate_2", 0.4, 1.0, "none", "candidate", "run", "/tmp/c2"),
        runner.RunRecord("crash", "candidate_3", None, 1.0, "none", "candidate", "run", "/tmp/c3"),
        runner.RunRecord(
            runner.INFRASTRUCTURE_RETRY,
            "candidate_4",
            None,
            1.0,
            "none",
            "candidate",
            "python job.py",
            "/tmp/c4",
        ),
    ]
    results_path = tmp_path / "results.tsv"
    state_path = tmp_path / "state.json"
    runner.write_results(results_path, records)

    state = runner.write_state(state_path, results_path, records, 5)
    guard_state = runner.load_campaign_guard().guard_state(results_path, max_candidates=5)

    assert state["decision"] == "retry_infrastructure"
    assert state["final_response_allowed"] is False
    assert state["candidate_cap"] == 5
    assert state["candidate_attempts"] == 3
    assert state["remaining_candidates"] == 2
    assert state["remaining_candidates"] == state["candidate_cap"] - state["candidate_attempts"]
    assert state["candidate_attempts"] == guard_state["candidate_attempts"]
    assert state["remaining_candidates"] == guard_state["remaining_candidates"]
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["candidate_attempts"] == 3
    assert persisted["remaining_candidates"] == 2


def test_remaining_candidates_clamped_to_zero_when_cap_lowered_below_attempts(tmp_path):
    runner = _load_runner()
    records = [
        runner.RunRecord("baseline", "baseline", 0.5, 1.0, "none", "baseline", "run", "/tmp/baseline"),
        runner.RunRecord("keep", "candidate_1", 0.6, 1.0, "none", "candidate", "run", "/tmp/c1"),
        runner.RunRecord("discard", "candidate_2", 0.4, 1.0, "none", "candidate", "run", "/tmp/c2"),
        runner.RunRecord("crash", "candidate_3", None, 1.0, "none", "candidate", "run", "/tmp/c3"),
        runner.RunRecord(
            runner.INFRASTRUCTURE_RETRY,
            "candidate_4",
            None,
            1.0,
            "none",
            "candidate",
            "python job.py",
            "/tmp/c4",
        ),
    ]
    results_path = tmp_path / "results.tsv"
    state_path = tmp_path / "state.json"
    runner.write_results(results_path, records)

    # A cap lowered below the recorded attempts must not report budget debt:
    # remaining_candidates means "candidates still available", never negative.
    state = runner.write_state(state_path, results_path, records, 2)
    guard_state = runner.load_campaign_guard().guard_state(results_path, max_candidates=2)

    assert state["candidate_attempts"] == 3
    assert state["remaining_candidates"] == 0
    assert guard_state["candidate_attempts"] == 3
    assert guard_state["remaining_candidates"] == 0
    assert state["remaining_candidates"] == guard_state["remaining_candidates"]
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["remaining_candidates"] == 0


def test_initialize_socket_failure_returns_75_without_counting_candidate(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    tmp_path.joinpath("client.py").write_text("print('client')\n", encoding="utf-8")
    monkeypatch.setattr(runner, "import_job_config", lambda *args, **kwargs: deepcopy(_campaign_config()))
    monkeypatch.setattr(runner, "job_help", lambda *args, **kwargs: "")
    monkeypatch.setattr(runner, "write_progress", lambda path, *args: path.write_bytes(b"progress"))
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            runner.INFRASTRUCTURE_RETRY,
            run_def.name,
            None,
            0.1,
            "none",
            run_def.description,
            "python job.py",
            str(tmp_path / "artifacts"),
            "sandbox/socket permission failure",
        ),
    )

    assert runner.main(["initialize", str(job), "--env", "sim", "--no-prefer-synthetic"]) == 75

    records = runner.load_results(tmp_path / "results.tsv")
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert runner.candidate_attempts(records) == 0
    assert state["next_action"] == runner.SIMULATION_APPROVAL_ACTION
    assert state["candidate_attempts"] == 0
    assert state["final_response_allowed"] is False


def test_evaluate_socket_failure_preserves_candidate_for_approved_retry(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "retry_candidate", "--hypothesis", "try update"]) == 0
    manifest_path = tmp_path / ".nvflare/autofl/candidates/retry_candidate/candidate_manifest.json"
    draft_client = manifest_path.parent / "source/client.py"
    draft_client.write_text("ALGORITHM = 'retry'\n", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            runner.INFRASTRUCTURE_RETRY,
            run_def.name,
            None,
            0.1,
            "none",
            run_def.description,
            "python job.py",
            str(tmp_path / "artifacts"),
            "sandbox/socket permission failure",
        ),
    )
    assert runner.main(["evaluate", str(job), "--manifest", str(manifest_path)]) == 75

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared"
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'baseline'\n"
    assert runner.candidate_attempts(runner.load_results(tmp_path / "results.tsv")) == 0
    assert state["next_action"] == runner.SIMULATION_APPROVAL_ACTION

    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            "candidate",
            run_def.name,
            0.8,
            0.1,
            "none",
            run_def.description,
            "python job.py",
            str(tmp_path / "artifacts"),
        ),
    )
    assert runner.main(["evaluate", str(job), "--manifest", str(manifest_path)]) == 0
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "keep"
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'retry'\n"


def test_successful_retry_clears_historical_infrastructure_decision(tmp_path):
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
            "/tmp/failed",
        ),
        runner.RunRecord(
            "baseline",
            "baseline_retry_2",
            0.5,
            1.0,
            "none",
            "baseline",
            "python job.py",
            "/tmp/success",
        ),
    ]
    results_path = tmp_path / "results.tsv"
    runner.write_results(results_path, records)

    state = runner.write_state(tmp_path / "state.json", results_path, records, max_candidates=None)

    assert state["decision"] == "continue"
    assert state["next_action"] == "propose_candidate"


def test_baseline_crash_is_not_counted_as_candidate_attempt():
    runner = _load_runner()
    records = [
        runner.RunRecord("baseline", "baseline", None, 1.0, "none", "baseline", "python job.py", "/tmp/baseline")
    ]

    assert runner.candidate_attempts(records) == 0
    assert runner.is_sandbox_socket_failure(
        "PermissionError: [Errno 1] Operation not permitted in get_open_ports while calling s.bind(('', 0))"
    )


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
    bad_draft.joinpath("discarded_algorithm.py").write_text("VALUE = 'discarded'\n", encoding="utf-8")

    def regressed_run(run_def, **kwargs):
        return runner.RunRecord(
            "candidate", run_def.name, 0.3, 2.0, "none", run_def.description, "python job.py", "/tmp/bad_algo"
        )

    monkeypatch.setattr(runner, "run_job", regressed_run)
    assert runner.main(["evaluate", str(job)]) == 0
    assert client.read_text(encoding="utf-8") == "from new_algorithm import VALUE\n"
    assert not tmp_path.joinpath("discarded_algorithm.py").exists()
    records = runner.load_results(tmp_path / "results.tsv")
    assert [record.status for record in records] == ["baseline", "keep", "discard"]
    assert records[1].changed_files == "client.py,new_algorithm.py"
    assert records[1].candidate_manifest.endswith("candidate_manifest.json")


def test_candidate_runtime_source_drift_is_rejected_and_fully_restored(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, baseline_score=0.8)
    baseline_job = job.read_text(encoding="utf-8")
    baseline_client = client.read_text(encoding="utf-8")
    assert runner.main(["prepare", str(job), "--name", "runtime_drift", "--hypothesis", "improve code"]) == 0
    candidate_dir = tmp_path / ".nvflare" / "autofl" / "candidates" / "runtime_drift"
    candidate_dir.joinpath("source/client.py").write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")

    def mutate_during_run(run_def, **kwargs):
        job.write_text("print('runtime mutation')\n", encoding="utf-8")
        client.unlink()
        tmp_path.joinpath("runtime_generated.py").write_text("VALUE = 'runtime'\n", encoding="utf-8")
        return runner.RunRecord(
            "candidate", run_def.name, 0.9, 1.0, "none", run_def.description, "python job.py", "/tmp/candidate"
        )

    monkeypatch.setattr(runner, "run_job", mutate_during_run)
    assert runner.main(["evaluate", str(job)]) == 0

    assert job.read_text(encoding="utf-8") == baseline_job
    assert client.read_text(encoding="utf-8") == baseline_client
    assert not tmp_path.joinpath("runtime_generated.py").exists()
    manifest = json.loads(candidate_dir.joinpath("candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "crash"
    assert manifest["result"]["score"] is None
    assert "client.py" in manifest["result"]["failure_reason"]
    assert "job.py" in manifest["result"]["failure_reason"]
    assert "runtime_generated.py" in manifest["result"]["failure_reason"]
    patch = candidate_dir.joinpath("candidate.patch").read_text(encoding="utf-8")
    assert "client.py" in patch
    assert "runtime mutation" not in patch
    assert "runtime_generated.py" not in patch
    records = runner.load_results(tmp_path / "results.tsv")
    assert [(record.status, record.score) for record in records] == [("baseline", 0.8), ("crash", None)]


@pytest.mark.skipif(os.name == "nt", reason="runtime symlink rollback uses POSIX symlinks")
def test_candidate_runtime_parent_symlink_drift_is_rejected_and_restored_without_following(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, baseline_score=0.8)
    baseline_client = client.read_text(encoding="utf-8")
    managed_parent = tmp_path / "src"
    managed_parent.mkdir()
    managed_source = managed_parent / "client.py"
    managed_source.write_text("NESTED = 'baseline'\n", encoding="utf-8")
    external_parent = tmp_path.parent / f"{tmp_path.name}-external"
    external_parent.mkdir()
    external_source = external_parent / "client.py"
    external_source.write_text("NESTED = 'external'\n", encoding="utf-8")

    assert runner.main(["prepare", str(job), "--name", "runtime_symlink", "--hypothesis", "improve code"]) == 0
    candidate_dir = tmp_path / ".nvflare" / "autofl" / "candidates" / "runtime_symlink"
    candidate_dir.joinpath("source/client.py").write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")

    def replace_parent_with_symlink(run_def, **kwargs):
        managed_source.unlink()
        managed_parent.rmdir()
        managed_parent.symlink_to(external_parent, target_is_directory=True)
        return runner.RunRecord(
            "candidate", run_def.name, 0.9, 1.0, "none", run_def.description, "python job.py", "/tmp/candidate"
        )

    monkeypatch.setattr(runner, "run_job", replace_parent_with_symlink)
    assert runner.main(["evaluate", str(job)]) == 0

    assert not managed_parent.is_symlink()
    assert managed_source.read_text(encoding="utf-8") == "NESTED = 'baseline'\n"
    assert external_source.read_text(encoding="utf-8") == "NESTED = 'external'\n"
    assert client.read_text(encoding="utf-8") == baseline_client
    manifest = json.loads(candidate_dir.joinpath("candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "crash"
    assert manifest["result"]["score"] is None
    assert "src/client.py" in manifest["result"]["failure_reason"]
    records = runner.load_results(tmp_path / "results.tsv")
    assert [(record.status, record.score) for record in records] == [("baseline", 0.8), ("crash", None)]


@pytest.mark.skipif(os.name == "nt", reason="virtual-environment regression uses a POSIX symlink")
def test_candidate_evaluation_excludes_in_workspace_environment_and_dependency_trees(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, config = _initialize_fake_campaign(runner, tmp_path, monkeypatch, baseline_score=0.5)
    environment_files = [
        ".venv/lib/python3.12/site-packages/pkg/real.py",
        "venv/lib/python3.12/site-packages/pkg/module.py",
        ".tox/py/lib/python3.12/site-packages/pkg/module.py",
        "vendor/site-packages/pkg/module.py",
        "node_modules/tool/module.py",
    ]
    for relative in environment_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 'dependency'\n", encoding="utf-8")
    linked_module = tmp_path / ".venv/lib/python3.12/site-packages/pkg/linked.py"
    linked_module.symlink_to("real.py")

    managed_paths = runner.managed_source_paths(tmp_path, config)
    assert not any(
        path in managed_paths for path in [*environment_files, linked_module.relative_to(tmp_path).as_posix()]
    )

    assert runner.main(["prepare", str(job), "--name", "venv_safe", "--hypothesis", "improve code"]) == 0
    candidate_dir = tmp_path / ".nvflare" / "autofl" / "candidates" / "venv_safe"
    candidate_dir.joinpath("source/client.py").write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            "candidate", run_def.name, 0.7, 1.0, "none", run_def.description, "python job.py", "/tmp/candidate"
        ),
    )

    assert runner.main(["evaluate", str(job)]) == 0
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'candidate'\n"
    assert linked_module.is_symlink()
    manifest = json.loads(candidate_dir.joinpath("candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "keep"


def test_candidate_runtime_source_restore_failure_remains_pending(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, baseline_score=0.8)
    assert runner.main(["prepare", str(job), "--name", "restore_failure", "--hypothesis", "improve code"]) == 0
    candidate_dir = tmp_path / ".nvflare" / "autofl" / "candidates" / "restore_failure"
    candidate_dir.joinpath("source/client.py").write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")

    def mutate_during_run(run_def, **kwargs):
        job.write_text("print('runtime mutation')\n", encoding="utf-8")
        return runner.RunRecord(
            "candidate", run_def.name, 0.9, 1.0, "none", run_def.description, "python job.py", "/tmp/candidate"
        )

    monkeypatch.setattr(runner, "run_job", mutate_during_run)
    monkeypatch.setattr(
        runner,
        "restore_managed_source_versions",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("simulated restore failure")),
    )

    assert runner.main(["evaluate", str(job)]) == 2
    assert "candidate remains pending for recovery" in capsys.readouterr().err
    manifest = json.loads(candidate_dir.joinpath("candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared"
    assert [record.status for record in runner.load_results(tmp_path / "results.tsv")] == ["baseline"]


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


def test_candidate_creation_uses_only_trust_contract_patterns(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    config_path = tmp_path / "autofl.yaml"
    config = runner.read_yaml(config_path)
    config["trust_contract"]["allowed_create_patterns"] = ["algorithms/*.py"]
    runner.write_yaml(config_path, config)

    assert runner.main(["prepare", str(job), "--name", "new_module", "--hypothesis", "add algorithm"]) == 0
    draft = tmp_path / ".nvflare/autofl/candidates/new_module/source"
    draft.joinpath("new_module.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert runner.main(["evaluate", str(job)]) == 2

    draft.joinpath("new_module.py").unlink()
    draft.joinpath("algorithms").mkdir()
    draft.joinpath("algorithms/new_module.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            "candidate", run_def.name, 0.7, 1.0, "none", run_def.description, "python job.py", "/tmp/candidate"
        ),
    )
    assert runner.main(["evaluate", str(job)]) == 0
    assert tmp_path.joinpath("algorithms/new_module.py").is_file()


def test_missing_create_patterns_deny_new_source(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    config_path = tmp_path / "autofl.yaml"
    config = runner.read_yaml(config_path)
    config["trust_contract"].pop("allowed_create_patterns")
    runner.write_yaml(config_path, config)

    assert runner.main(["prepare", str(job), "--name", "denied_module", "--hypothesis", "add algorithm"]) == 0
    draft = tmp_path / ".nvflare/autofl/candidates/denied_module/source"
    draft.joinpath("new_module.py").write_text("VALUE = 1\n", encoding="utf-8")

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


def test_candidate_schema_failure_does_not_modify_workspace(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "bad_schema", "--hypothesis", "change code"]) == 0
    candidate_dir = tmp_path / ".nvflare" / "autofl" / "candidates" / "bad_schema"
    candidate_dir.joinpath("source/client.py").write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")
    tmp_path.joinpath("mutation_schema.yaml").write_text(
        "comparison_budget_args:\n  default_candidate_budget:\n    run_timeout_seconds: fast\n",
        encoding="utf-8",
    )

    assert runner.main(["evaluate", str(job)]) == 2
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'baseline'\n"


def test_candidate_partial_apply_failure_restores_workspace(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    baseline_job = job.read_text(encoding="utf-8")
    baseline_client = client.read_text(encoding="utf-8")
    assert runner.main(["prepare", str(job), "--name", "partial", "--hypothesis", "change two files"]) == 0
    draft = tmp_path / ".nvflare" / "autofl" / "candidates" / "partial" / "source"
    draft.joinpath("client.py").write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")
    draft.joinpath("job.py").write_text("print('candidate')\n", encoding="utf-8")
    original_copy = runner.copy_relative_file

    def fail_second_candidate_copy(source_root, destination_root, relative):
        if source_root == draft and relative == "job.py":
            raise OSError("simulated candidate copy failure")
        original_copy(source_root, destination_root, relative)

    monkeypatch.setattr(runner, "copy_relative_file", fail_second_candidate_copy)

    assert runner.main(["evaluate", str(job)]) == 2
    assert job.read_text(encoding="utf-8") == baseline_job
    assert client.read_text(encoding="utf-8") == baseline_client


def test_candidate_job_help_failure_restores_workspace(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "help_failure", "--hypothesis", "change code"]) == 0
    draft = tmp_path / ".nvflare" / "autofl" / "candidates" / "help_failure" / "source"
    draft.joinpath("client.py").write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")
    draft.joinpath("temporary_algorithm.py").write_text("VALUE = 'candidate'\n", encoding="utf-8")

    def fail_job_help(*args, **kwargs):
        raise OSError("simulated missing Python executable")

    monkeypatch.setattr(runner, "job_help", fail_job_help)

    assert runner.main(["evaluate", str(job)]) == 2
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'baseline'\n"
    assert not tmp_path.joinpath("temporary_algorithm.py").exists()


def test_restore_best_source_removes_explicit_created_file(tmp_path):
    runner = _load_runner()
    workspace = tmp_path / "workspace"
    best_source = tmp_path / "best"
    workspace.mkdir()
    best_source.mkdir()
    workspace.joinpath("created_algorithm.py").write_text("VALUE = 'candidate'\n", encoding="utf-8")

    runner.restore_best_source(workspace, best_source, {}, [], ["created_algorithm.py"])

    assert not workspace.joinpath("created_algorithm.py").exists()


def test_keyboard_interrupt_during_candidate_import_restores_workspace(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "interrupt", "--hypothesis", "change code"]) == 0
    draft = tmp_path / ".nvflare" / "autofl" / "candidates" / "interrupt" / "source"
    draft.joinpath("client.py").write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")

    def interrupt_import(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "import_job_config", interrupt_import)
    args = runner.parse_args(["evaluate", str(job)])
    with pytest.raises(KeyboardInterrupt):
        runner.evaluate_candidate(args, job)

    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'baseline'\n"


def test_candidate_finalization_failure_rolls_back_workspace_and_campaign_files(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "late_failure", "--hypothesis", "improve code"]) == 0
    candidate_dir = tmp_path / ".nvflare" / "autofl" / "candidates" / "late_failure"
    candidate_dir.joinpath("source/client.py").write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")
    original_autofl = tmp_path.joinpath("autofl.yaml").read_bytes()
    original_results = tmp_path.joinpath("results.tsv").read_bytes()
    original_state = tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_bytes()
    original_progress = tmp_path.joinpath("progress.png").read_bytes()
    original_report = tmp_path.joinpath("autofl_report.md").read_bytes()

    def improved_run(run_def, **kwargs):
        return runner.RunRecord(
            "candidate", run_def.name, 0.7, 2.0, "none", run_def.description, "python job.py", "/tmp/late_failure"
        )

    original_refresh = runner.refresh_campaign_artifacts

    def fail_artifact_refresh(*args, **kwargs):
        original_refresh(*args, **kwargs)
        raise OSError("simulated report write failure")

    monkeypatch.setattr(runner, "run_job", improved_run)
    monkeypatch.setattr(runner, "refresh_campaign_artifacts", fail_artifact_refresh)

    assert runner.main(["evaluate", str(job)]) == 2
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'baseline'\n"
    assert tmp_path.joinpath("autofl.yaml").read_bytes() == original_autofl
    assert tmp_path.joinpath("results.tsv").read_bytes() == original_results
    assert tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_bytes() == original_state
    assert tmp_path.joinpath("progress.png").read_bytes() == original_progress
    assert tmp_path.joinpath("autofl_report.md").read_bytes() == original_report
    best_source, best_files = runner.load_best_snapshot(tmp_path / ".nvflare" / "autofl" / "snapshots" / "best")
    assert runner.workspace_matches_snapshot(tmp_path, best_source, best_files)
    metadata = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign.json").read_text(encoding="utf-8"))
    assert metadata["best_candidate"] == "baseline"
    manifest = json.loads(candidate_dir.joinpath("candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared"


def test_candidate_snapshot_stage_failure_preserves_previous_best(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "snapshot_failure", "--hypothesis", "improve code"]) == 0
    draft = tmp_path / ".nvflare" / "autofl" / "candidates" / "snapshot_failure" / "source"
    draft.joinpath("client.py").write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")

    def improved_run(run_def, **kwargs):
        return runner.RunRecord(
            "candidate", run_def.name, 0.7, 2.0, "none", run_def.description, "python job.py", "/tmp/snapshot"
        )

    def fail_snapshot_stage(*args, **kwargs):
        raise OSError("simulated snapshot copy failure")

    monkeypatch.setattr(runner, "run_job", improved_run)
    monkeypatch.setattr(runner, "stage_best_snapshot", fail_snapshot_stage)

    assert runner.main(["evaluate", str(job)]) == 2
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'baseline'\n"
    best_source, best_files = runner.load_best_snapshot(tmp_path / ".nvflare" / "autofl" / "snapshots" / "best")
    assert runner.workspace_matches_snapshot(tmp_path, best_source, best_files)


def test_candidate_discard_restore_failure_retries_rollback(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "discard_failure", "--hypothesis", "regress code"]) == 0
    draft = tmp_path / ".nvflare" / "autofl" / "candidates" / "discard_failure" / "source"
    draft.joinpath("client.py").write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")
    original_restore = runner.restore_best_source
    restore_calls = 0

    def fail_first_restore(*args, **kwargs):
        nonlocal restore_calls
        restore_calls += 1
        if restore_calls == 1:
            raise OSError("simulated restore failure")
        original_restore(*args, **kwargs)

    def regressed_run(run_def, **kwargs):
        return runner.RunRecord(
            "candidate", run_def.name, 0.3, 2.0, "none", run_def.description, "python job.py", "/tmp/discard_failure"
        )

    monkeypatch.setattr(runner, "restore_best_source", fail_first_restore)
    monkeypatch.setattr(runner, "run_job", regressed_run)

    assert runner.main(["evaluate", str(job)]) == 2
    assert restore_calls == 2
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'baseline'\n"


def test_malformed_yaml_returns_clean_cli_errors(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, config = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    capsys.readouterr()
    autofl_yaml = tmp_path / "autofl.yaml"
    autofl_yaml.write_text("job: [\n", encoding="utf-8")

    assert runner.main(["suggest", str(job)]) == 2
    stderr = capsys.readouterr().err
    assert f"Auto-FL suggest failed: invalid YAML in {autofl_yaml}" in stderr
    assert "Traceback" not in stderr

    runner.write_yaml(autofl_yaml, config)
    mutation_schema = tmp_path / "mutation_schema.yaml"
    mutation_schema.write_text("comparison_budget_args: [\n", encoding="utf-8")

    assert runner.main(["suggest", str(job)]) == 2
    stderr = capsys.readouterr().err
    assert f"Auto-FL suggest failed: invalid YAML in {mutation_schema}" in stderr
    assert "Traceback" not in stderr


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


def test_abandoned_candidate_counts_in_state_but_never_as_attempt(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "abandoned", "--hypothesis", "temporary idea"]) == 0
    draft = tmp_path / ".nvflare/autofl/candidates/abandoned/source/client.py"
    draft.write_text("ALGORITHM = 'temporary'\n", encoding="utf-8")
    capsys.readouterr()

    assert runner.main(["abandon", str(job)]) == 0

    payload = json.loads(capsys.readouterr().out)
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert state["abandoned_candidates"] == 1
    assert state["candidate_attempts"] == 0
    assert payload["abandoned_candidates"] == 1
    assert payload["candidate_attempts"] == 0


def test_abandon_rejects_agent_modified_manifest_paths(tmp_path, monkeypatch):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "tampered", "--hypothesis", "temporary idea"]) == 0
    draft = tmp_path / ".nvflare/autofl/candidates/tampered/source/client.py"
    draft.write_text("ALGORITHM = 'temporary'\n", encoding="utf-8")
    manifest_path = tmp_path / ".nvflare/autofl/candidates/tampered/candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["changed_files"] = ["../client.py"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert runner.main(["abandon", str(job)]) == 2
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'baseline'\n"


def test_status_rescans_pending_manifests_before_writing_state(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "pending", "--hypothesis", "change code"]) == 0
    state_path = tmp_path / ".nvflare" / "autofl" / "campaign_state.json"
    state_path.write_text('{"final_response_allowed": true}\n', encoding="utf-8")

    assert runner.main(["status", str(job)]) == 0

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["reason"] == "pending_candidates"
    assert state["next_action"] == "edit_candidate"
    assert state["final_response_allowed"] is False
    assert state["pending_candidate_manifest"].endswith("pending/candidate_manifest.json")


def test_unchanged_status_does_not_regenerate_campaign_artifacts(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    artifact_paths = [tmp_path / "results.tsv", tmp_path / "progress.png", tmp_path / "autofl_report.md"]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in artifact_paths}

    assert runner.main(["status", str(job)]) == 0
    first_state = tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_bytes()
    first_state_mtime = tmp_path.joinpath(".nvflare/autofl/campaign_state.json").stat().st_mtime_ns
    assert runner.main(["status", str(job)]) == 0

    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in artifact_paths} == before
    assert tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_bytes() == first_state
    assert tmp_path.joinpath(".nvflare/autofl/campaign_state.json").stat().st_mtime_ns == first_state_mtime


def test_status_refuses_malformed_candidate_manifest(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "malformed", "--hypothesis", "change code"]) == 0
    manifest = tmp_path / ".nvflare" / "autofl" / "candidates" / "malformed" / "candidate_manifest.json"
    manifest.write_text("not json\n", encoding="utf-8")

    assert runner.main(["status", str(job)]) == 2


def test_status_uses_persisted_custom_stop_file_in_job_directory(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    metadata_path = tmp_path / ".nvflare" / "autofl" / "campaign.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["settings"]["stop_file"] = ["CUSTOM_STOP"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    tmp_path.joinpath("STOP_AUTOFL").touch()

    assert runner.main(["status", str(job)]) == 0
    default_state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert default_state["reason"] == "continue"

    tmp_path.joinpath("CUSTOM_STOP").touch()
    assert runner.main(["status", str(job)]) == 0
    custom_state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert custom_state["reason"] == "manual_stop_file"
    assert custom_state["final_response_allowed"] is True


def test_stop_file_blocks_prepare_before_creating_candidate(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    capsys.readouterr()
    tmp_path.joinpath("STOP_AUTOFL").touch()

    assert runner.main(["prepare", str(job), "--name", "blocked", "--hypothesis", "must not be materialized"]) == 2

    assert "campaign is manually stopped" in capsys.readouterr().err
    assert not tmp_path.joinpath(".nvflare/autofl/candidates/blocked").exists()


def test_stop_file_blocks_evaluate_before_workspace_mutation(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, client, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "blocked", "--hypothesis", "candidate code"]) == 0
    draft = tmp_path / ".nvflare/autofl/candidates/blocked/source/client.py"
    draft.write_text("ALGORITHM = 'candidate'\n", encoding="utf-8")
    manifest = tmp_path / ".nvflare/autofl/candidates/blocked/candidate_manifest.json"
    before_manifest = manifest.read_bytes()
    capsys.readouterr()
    tmp_path.joinpath("STOP_AUTOFL").touch()

    assert runner.main(["evaluate", str(job)]) == 2

    assert "campaign is manually stopped" in capsys.readouterr().err
    assert client.read_text(encoding="utf-8") == "ALGORITHM = 'baseline'\n"
    assert manifest.read_bytes() == before_manifest


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
            run_def.name,
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
    assert [(record.status, record.score) for record in records] == [("baseline", None), ("baseline", 0.5)]
    assert [record.name for record in records] == ["baseline", "baseline_retry_2"]
    metadata = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign.json").read_text(encoding="utf-8"))
    assert metadata["best_candidate"] == "baseline_retry_2"


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
    assert records[-1].literature_event_id == "lit-0001"
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert state["next_action"] == "develop_literature_batch"
    assert state["required_exploration"] == "source_backed_exploration"
    assert state["exploration_batch"]["literature_event_id"] == "lit-0001"
    assert state["exploration_batch"]["completed"] == 0
    assert "--literature-event" in state["agent_instruction"]


def test_results_roundtrip_preserves_candidate_provenance(tmp_path):
    runner = _load_runner()
    records = [
        runner.RunRecord("baseline", "baseline", 0.5, 1.0, "none", "baseline", "python job.py", "/tmp/baseline"),
        runner.RunRecord(
            "keep",
            "fedyogi_faithful",
            0.6,
            1.0,
            "client.py",
            "faithful FedYogi implementation",
            "python job.py",
            "/tmp/run",
            candidate_kind="source_edit",
            algorithm_family="fedyogi",
            literature_event_id="lit-0001",
        ),
    ]
    results_path = tmp_path / "results.tsv"

    runner.write_results(results_path, records)
    loaded = runner.load_results(results_path)

    assert loaded[-1].candidate_kind == "source_edit"
    assert loaded[-1].algorithm_family == "fedyogi"
    assert loaded[-1].literature_event_id == "lit-0001"
    assert loaded[0].candidate_kind == ""


def test_prepare_rejects_unknown_literature_event(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)

    rc = runner.main(
        [
            "prepare",
            str(job),
            "--name",
            "fedyogi_faithful",
            "--hypothesis",
            "faithful FedYogi",
            "--family",
            "fedyogi",
            "--literature-event",
            "lit-9999",
        ]
    )

    assert rc == 2
    assert "unknown literature event id" in capsys.readouterr().err


def test_prepare_persists_family_and_literature_event_in_manifest(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["record", str(job), "--literature", "--hypothesis", "reviewed FedYogi"]) == 0

    assert (
        runner.main(
            [
                "prepare",
                str(job),
                "--name",
                "fedyogi_faithful",
                "--hypothesis",
                "faithful FedYogi",
                "--family",
                "FedYogi",
                "--literature-event",
                "lit-0001",
            ]
        )
        == 0
    )

    manifest = json.loads(
        tmp_path.joinpath(".nvflare/autofl/candidates/fedyogi_faithful/candidate_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["algorithm_family"] == "fedyogi"
    assert manifest["literature_event_id"] == "lit-0001"


def test_external_baseline_may_follow_literature_event(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, target_env="prod")

    assert runner.main(["record", str(job), "--literature", "--hypothesis", "reviewed FedOpt"]) == 0
    assert runner.main(["record", str(job), "--baseline", "--score", "0.5", "--job-id", "baseline-job"]) == 0

    records = runner.load_results(tmp_path / "results.tsv")
    assert [record.status for record in records] == ["literature", "baseline"]


def test_production_initialize_never_requests_simulation_runner_approval(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, target_env="prod")
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))

    assert state["next_action"] == "submit_baseline"
    assert state["next_action"] != runner.SIMULATION_APPROVAL_ACTION


def test_prepare_and_status_never_request_simulation_runner_approval(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)

    assert runner.main(["prepare", str(job), "--name", "draft", "--hypothesis", "draft candidate"]) == 0
    for action in ("prepare", "status"):
        state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
        assert state["next_action"] == "edit_candidate", action
        assert state["next_action"] != runner.SIMULATION_APPROVAL_ACTION
        if action == "prepare":
            assert runner.main(["status", str(job)]) == 0


def test_omitted_metric_uses_imported_job_metric(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    tmp_path.joinpath("client.py").write_text("ALGORITHM = 'baseline'\n", encoding="utf-8")
    config = _campaign_config()
    config["objective"] = {
        "metric": "auc",
        "requested_metric": "auc",
        "optimization_metric": "auc",
        "metric_extraction_order": ["auc"],
        "metric_contract_source": "arg:key_metric",
    }
    monkeypatch.setattr(runner, "import_job_config", lambda *args, **kwargs: deepcopy(config))
    monkeypatch.setattr(runner, "job_help", lambda *args, **kwargs: "")
    monkeypatch.setattr(runner, "write_progress", lambda path, *args: path.write_bytes(b"progress"))

    def fake_run(run_def, **kwargs):
        assert kwargs["metrics"] == ["auc"]
        return runner.RunRecord(
            "baseline", run_def.name, 0.6, 1.0, "none", "baseline", "python job.py", "/tmp/baseline"
        )

    monkeypatch.setattr(runner, "run_job", fake_run)

    assert runner.main(["initialize", str(job), "--no-prefer-synthetic"]) == 0
    metadata = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign.json").read_text(encoding="utf-8"))
    assert metadata["settings"]["metric"] == "auc"


def test_explicit_mutable_campaign_settings_persist_and_uncapped_removes_cap(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)

    assert runner.main(["status", str(job), "--max-candidates", "7", "--timeout", "123"]) == 0
    metadata_path = tmp_path / ".nvflare/autofl/campaign.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["settings"]["max_candidates"] == 7
    assert metadata["settings"]["timeout"] == 123

    assert runner.main(["status", str(job), "--uncapped"]) == 0
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["settings"]["max_candidates"] is None


def test_effective_cap_changes_append_audit_records_to_campaign_metadata(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    metadata_path = tmp_path / ".nvflare/autofl/campaign.json"

    assert runner.main(["status", str(job), "--timeout", "123"]) == 0
    assert "cap_changes" not in json.loads(metadata_path.read_text(encoding="utf-8"))

    assert runner.main(["status", str(job), "--max-candidates", "7"]) == 0
    assert runner.main(["status", str(job), "--max-candidates", "7"]) == 0
    assert runner.main(["status", str(job), "--uncapped"]) == 0

    cap_changes = json.loads(metadata_path.read_text(encoding="utf-8"))["cap_changes"]
    assert [(entry["old"], entry["new"], entry["source"]) for entry in cap_changes] == [
        (None, 7, "explicit"),
        (7, None, "uncapped"),
    ]
    assert all(entry["changed_at"] for entry in cap_changes)


def test_runner_state_reports_budget_and_baseline_accounting(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, baseline_score=1.0, mode="min")
    state_path = tmp_path / ".nvflare/autofl/campaign_state.json"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["remaining_candidates"] is None
    assert state["baseline_status"] == "complete"
    assert state["baseline_score"] == pytest.approx(1.0)
    assert state["improvement"] == pytest.approx(0.0)
    assert state["abandoned_candidates"] == 0

    results_path = tmp_path / "results.tsv"
    records = runner.load_results(results_path)
    records.append(runner.RunRecord("keep", "lower_loss", 0.8, 1.0, "none", "lower loss", "python job.py", ""))
    runner.write_results(results_path, records)

    assert runner.main(["status", str(job), "--max-candidates", "3"]) == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["remaining_candidates"] == 2
    # mode=min: positive improvement means the loss went down against the baseline.
    assert state["improvement"] == pytest.approx(0.2)


def test_status_reuses_persisted_guard_settings(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)

    assert (
        runner.main(
            [
                "status",
                str(job),
                "--exploration-batch-size",
                "5",
                "--family-repeat-limit",
                "9",
            ]
        )
        == 0
    )
    observed = {}
    original_write_state = runner.write_state

    def capture_write_state(*args, **kwargs):
        observed.update(kwargs)
        return original_write_state(*args, **kwargs)

    monkeypatch.setattr(runner, "write_state", capture_write_state)

    assert runner.main(["status", str(job)]) == 0
    assert observed["exploration_batch_size"] == 5
    assert observed["family_repeat_limit"] == 9


def test_status_restores_persisted_minimization_mode(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, baseline_score=1.0, mode="min")
    results_path = tmp_path / "results.tsv"
    records = runner.load_results(results_path)
    records.append(runner.RunRecord("keep", "lower_loss", 0.8, 1.0, "none", "lower loss", "python job.py", ""))
    runner.write_results(results_path, records)

    assert runner.main(["status", str(job)]) == 0

    metadata = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign.json").read_text(encoding="utf-8"))
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert metadata["settings"]["mode"] == "min"
    assert state["best_score"] == pytest.approx(0.8)
    assert runner.main(["status", str(job), "--mode", "max"]) == 2


def test_explicit_immutable_campaign_setting_change_is_rejected(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)

    assert runner.main(["status", str(job), "--metric", "loss"]) == 2


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
        json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))["next_action"]
        == "submit_candidate"
    )
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


def test_external_candidate_record_reimports_fixed_budget(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, config = _initialize_fake_campaign(runner, tmp_path, monkeypatch, target_env="prod")
    assert runner.main(["record", str(job), "--baseline", "--score", "0.5"]) == 0
    assert runner.main(["prepare", str(job), "--name", "prod_algo", "--hypothesis", "production algorithm"]) == 0
    draft = tmp_path / ".nvflare" / "autofl" / "candidates" / "prod_algo" / "source"
    draft.joinpath("client.py").write_text("ALGORITHM = 'production'\n", encoding="utf-8")
    assert runner.main(["evaluate", str(job)]) == 0

    drifted = deepcopy(config)
    drifted["budget"]["fixed_training_budget"]["num_rounds"] = 2
    monkeypatch.setattr(runner, "import_job_config", lambda *args, **kwargs: deepcopy(drifted))
    manifest_path = tmp_path / ".nvflare" / "autofl" / "candidates" / "prod_algo" / "candidate_manifest.json"

    assert runner.main(["record", str(job), "--manifest", str(manifest_path), "--score", "0.8"]) == 2
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "ready_for_external_execution"
    assert [record.status for record in runner.load_results(tmp_path / "results.tsv")] == ["baseline"]


@pytest.mark.parametrize("score", ["nan", "inf", "-inf"])
def test_external_record_rejects_non_finite_explicit_score(tmp_path, monkeypatch, score):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, target_env="prod")

    assert runner.main(["record", str(job), "--baseline", f"--score={score}"]) == 2
    assert runner.load_results(tmp_path / "results.tsv") == []


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

    class FakeImporter:
        __file__ = __file__

        @staticmethod
        def import_job_to_autofl_config(*args, **kwargs):
            captured.update(kwargs)
            return _campaign_config()

        @staticmethod
        def dump_autofl_yaml(config):
            return runner.yaml.safe_dump(config)

    monkeypatch.setattr(runner, "load_job_importer", lambda: FakeImporter)
    args = runner.parse_args(["initialize", str(job), "--mode", "min", "--base-args", "--mode training"])
    runner.import_job_config(args, job, output, tmp_path / "import.log", 10)

    assert captured["mode"] == "min"
    assert captured["job_args"] == ["--mode", "training"]


@pytest.mark.parametrize(
    "config,expected",
    [
        (
            {
                "import": {"support": {"status": "partial"}},
                "budget": {"fixed_training_budget": {"num_rounds": 1}},
            },
            "job surface",
        ),
        (
            {"import": {"support": {"status": "supported"}}, "budget": {}},
            "fixed comparison budget",
        ),
        (
            {
                "import": {"support": {"status": "supported"}},
                "budget": {"fixed_training_budget": {"num_rounds": 1}},
                "unresolved": [{"field": "budget.fixed_training_budget.num_clients", "reason": "dynamic"}],
            },
            "safety-critical fields",
        ),
        (
            {
                "import": {"support": {"status": "supported"}},
                "budget": {"fixed_training_budget": {"num_rounds": 1}},
                "unresolved": [{"field": "objective.metric", "reason": "ambiguous argparse definitions"}],
            },
            "objective.metric",
        ),
    ],
)
def test_campaign_admission_rejects_unresolved_safety_contract(config, expected):
    runner = _load_runner()

    assert expected in "; ".join(runner.campaign_admission_errors(config))


def test_cli_lifecycle_runs_agent_code_candidate_end_to_end(tmp_path):
    repo_root = Path(__file__).parents[3]
    runner_path = repo_root / "skills" / "nvflare-autofl" / "scripts" / "run_job_campaign.py"
    job = tmp_path / "job.py"
    simulation_root = tmp_path / "simulation"
    job.write_text(
        f"""
import argparse
import json
import os
from pathlib import Path
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe as ImportedFedAvgRecipe
from nvflare.recipe import SimEnv as ImportedSimEnv

SCORE = 0.5

class FakeFedAvgRecipe:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class FakeSimEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

ImportedFedAvgRecipe = FakeFedAvgRecipe
ImportedSimEnv = FakeSimEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="run")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--n_clients", type=int, default=2)
    args = parser.parse_args()
    ImportedFedAvgRecipe(model=object(), num_rounds=args.num_rounds, min_clients=args.n_clients)
    ImportedSimEnv(num_clients=args.n_clients, workspace_root={str(simulation_root)!r})
    result = Path(os.environ["NVFLARE_SIMULATOR_WORKSPACE_ROOT"]) / args.name
    result.mkdir(parents=True, exist_ok=True)
    result.joinpath("metrics_summary.json").write_text(json.dumps({{"accuracy": SCORE}}))
    print(f"Result can be found in : {{result.resolve()}}")

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


def test_cross_val_extraction_prefers_best_server_final_global_model_entry(tmp_path):
    runner = _load_runner()
    result_path = tmp_path / "cross_val_results.json"
    result_path.write_text(
        json.dumps(
            {
                "site-1": {
                    "site-1": {"accuracy": 0.99},
                    "SRV_FL_global_model.pt": {"accuracy": 0.71},
                },
                "site-2": {
                    "site-2": {"accuracy": 0.95},
                    "SRV_FL_global_model.pt": {"accuracy": 0.74},
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = runner.extract_metric_evidence(tmp_path, ["accuracy"])

    assert evidence.score == pytest.approx(0.74)
    assert evidence.metric_name == "accuracy"
    assert evidence.source == "structured:cross_val_results.json#server_final"
    assert evidence.artifact == str(result_path.resolve())


def test_cross_val_extraction_resolves_modern_unprefixed_global_model_entries(tmp_path):
    runner = _load_runner()
    tmp_path.joinpath("cross_val_results.json").write_text(
        json.dumps(
            {
                "site-1": {
                    "site-1": {"accuracy": 0.99},
                    "FL_global_model.pt": {"accuracy": 0.71},
                },
                "site-2": {
                    "site-2": {"accuracy": 0.95},
                    "FL_global_model.pt": {"accuracy": 0.74},
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = runner.extract_metric_evidence(tmp_path, ["accuracy"])

    assert evidence.score == pytest.approx(0.74)
    assert evidence.source == "structured:cross_val_results.json#server_final"


def test_cross_val_extraction_resolves_srv_best_only_global_model_entries(tmp_path):
    runner = _load_runner()
    tmp_path.joinpath("cross_val_results.json").write_text(
        json.dumps(
            {
                "site-1": {
                    "site-1": {"accuracy": 0.99},
                    "SRV_best_FL_global_model.pt": {"accuracy": 0.66},
                }
            }
        ),
        encoding="utf-8",
    )

    evidence = runner.extract_metric_evidence(tmp_path, ["accuracy"])

    assert evidence.score == pytest.approx(0.66)
    assert evidence.source == "structured:cross_val_results.json#server_final"


def test_cross_val_extraction_uses_min_over_server_final_entries_in_min_mode(tmp_path):
    runner = _load_runner()
    tmp_path.joinpath("cross_val_results.json").write_text(
        json.dumps(
            {
                "site-1": {
                    "site-1": {"loss": 0.10},
                    "SRV_FL_global_model.pt": {"loss": 0.42},
                },
                "site-2": {"SRV_FL_global_model.pt": {"loss": 0.37}},
            }
        ),
        encoding="utf-8",
    )

    assert runner.extract_score(tmp_path, ["loss"], mode="min") == pytest.approx(0.37)


def test_cross_val_extraction_falls_back_to_first_match_without_server_final_entries(tmp_path):
    runner = _load_runner()
    tmp_path.joinpath("cross_val_results.json").write_text(
        json.dumps({"site-1": {"site-1": {"accuracy": 0.9}, "site-2": {"accuracy": 0.6}}}),
        encoding="utf-8",
    )

    evidence = runner.extract_metric_evidence(tmp_path, ["accuracy"])
    assert evidence.score == pytest.approx(0.9)
    assert evidence.source == "structured:cross_val_results.json"

    tmp_path.joinpath("cross_val_results.json").write_text(
        json.dumps(
            {
                "site-1": {
                    "site-1": {"accuracy": 0.9},
                    "SRV_FL_global_model.pt": {"loss": 0.4},
                }
            }
        ),
        encoding="utf-8",
    )

    evidence = runner.extract_metric_evidence(tmp_path, ["accuracy"])
    assert evidence.score == pytest.approx(0.9)
    assert evidence.source == "structured:cross_val_results.json"
