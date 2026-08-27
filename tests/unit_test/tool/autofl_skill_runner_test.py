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

    assert runner.main(["initialize", str(job)]) == 0

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


def test_runner_uses_campaign_guard_score_comparison():
    runner = _load_runner()

    assert runner.better is runner.load_campaign_guard().better


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_metric_paths_reject_non_finite_values(tmp_path, value):
    runner = _load_runner()

    assert runner.find_metric_value({"accuracy": value}, ["accuracy"]) is None
    assert runner.find_metric_value({"metrics": [{"name": "accuracy", "value": value}]}, ["accuracy"]) is None
    assert runner.better(value, 0.5) is False
    assert runner.better(0.6, value) is True
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
            "metric_invariants": ["definition", "evaluation_timing_and_checkpoint"],
            "metric_change_policy": "restart_campaign_with_repaired_baseline",
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
    assert updated["objective"]["metric_invariants"] == ["definition", "evaluation_timing_and_checkpoint"]
    assert updated["objective"]["metric_change_policy"] == "restart_campaign_with_repaired_baseline"


def test_schema_metric_contract_accepts_minimization_objective():
    runner = _load_runner()
    config = {
        "objective": {
            "metric": "val_loss",
            "requested_metric": "val_loss",
            "optimization_metric": "val_loss",
            "mode": "max",
            "mode_contract_source": "core_default",
            "job_key_metric": "accuracy",
        }
    }
    schema = {
        "objective": {
            "requested_metric": "val_loss",
            "optimization_metric": "val_loss",
            "mode": "min",
        }
    }

    updated = runner.apply_metric_contract(config, "val_loss", schema)

    assert updated["objective"]["mode"] == "min"
    assert updated["objective"]["mode_contract_source"] == "mutation_schema"


def test_schema_bridged_metric_does_not_inherit_job_direction():
    runner = _load_runner()
    config = {
        "objective": {
            "metric": "accuracy",
            "requested_metric": "accuracy",
            "optimization_metric": "accuracy",
            "mode": "min",
            "mode_contract_source": "job:key_metric_mode",
            "job_key_metric": "val_loss",
            "job_key_metric_mode": "min",
            "job_key_metric_mode_source": "job:key_metric_mode",
        }
    }
    schema = {"objective": {"requested_metric": "accuracy", "optimization_metric": "accuracy"}}

    updated = runner.apply_metric_contract(config, "accuracy", schema)

    assert updated["objective"]["mode"] == "max"
    assert updated["objective"]["mode_contract_source"] == "core_default"
    assert updated["objective"]["job_key_metric_mode"] == "min"
    assert updated["objective"]["job_key_metric_mode_source"] == "job:key_metric_mode"


def test_schema_cannot_override_direction_for_job_key_metric():
    runner = _load_runner()
    config = {
        "objective": {
            "metric": "loss",
            "requested_metric": "loss",
            "optimization_metric": "loss",
            "mode": "min",
            "mode_contract_source": "job:key_metric_mode",
            "job_key_metric": "loss",
            "job_key_metric_source": "literal",
        }
    }
    schema = {"objective": {"metric": "loss", "mode": "max"}}

    with pytest.raises(ValueError, match="job.py is authoritative"):
        runner.apply_metric_contract(config, "loss", schema)


def test_schema_mode_conflict_with_implicit_job_default_has_actionable_error():
    runner = _load_runner()
    config = {
        "objective": {
            "metric": "loss",
            "requested_metric": "loss",
            "optimization_metric": "loss",
            "mode": "max",
            "mode_contract_source": "core_default",
            "job_key_metric": "loss",
            "job_key_metric_source": "core_default",
        }
    }

    with pytest.raises(ValueError, match="implicit 'max'.*set key_metric_mode='min' in job.py"):
        runner.apply_metric_contract(config, "loss", {"objective": {"metric": "loss", "mode": "min"}})


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

    args = SimpleNamespace(base_args="")
    assert runner.build_campaign_args(config, args, help_text, schema) == (
        [],
        ["--n_clients", "8", "--num_rounds", "20"],
    )


def test_campaign_args_never_inject_dataset_flags_for_synthetic_capable_jobs():
    runner = _load_runner()
    help_text = "usage: job.py [--synthetic_data] [--train_size TRAIN_SIZE] [--test_size TEST_SIZE]"

    assert runner.build_campaign_args({"budget": {}}, SimpleNamespace(base_args=""), help_text, {}) == ([], [])

    schema = {"comparison_budget_args": {"default_candidate_budget": {"num_rounds": 5}}}
    help_text_with_budget = f"{help_text} [--num_rounds NUM_ROUNDS]"
    assert runner.build_campaign_args({"budget": {}}, SimpleNamespace(base_args=""), help_text_with_budget, schema) == (
        [],
        ["--num_rounds", "5"],
    )


def test_explicit_base_args_pass_through_verbatim():
    runner = _load_runner()
    help_text = "usage: job.py [--synthetic_data] [--train_size TRAIN_SIZE] [--test_size TEST_SIZE]"

    args = SimpleNamespace(base_args="--synthetic_data --train_size 64")
    assert runner.build_campaign_args({"budget": {}}, args, help_text, {}) == (
        [],
        ["--synthetic_data", "--train_size", "64"],
    )


def test_base_args_duplicates_of_fixed_budget_are_emitted_once():
    runner = _load_runner()
    config = {"budget": {"fixed_training_budget": {"num_clients": 2, "num_rounds": 3}}}
    help_text = "--n_clients --num_rounds --update_type"
    args = SimpleNamespace(base_args="--n_clients 2 --num_rounds=3 --update_type full")

    fixed_args, base_args = runner.build_campaign_args(config, args, help_text, {})

    assert fixed_args == ["--n_clients", "2", "--num_rounds", "3"]
    assert base_args == ["--update_type", "full"]


def test_base_args_alias_spelling_matches_only_parser_defined_flags():
    runner = _load_runner()
    config = {"budget": {"fixed_training_budget": {"num_rounds": 3}}}
    args = SimpleNamespace(base_args="--num-rounds 3")

    # The parser does not define --num-rounds: pass it through so argparse still reports the typo.
    assert runner.build_campaign_args(config, args, "--num_rounds", {}) == (
        ["--num_rounds", "3"],
        ["--num-rounds", "3"],
    )

    # The parser defines both spellings as one option: deduplicate and conflict-check across them.
    alias_groups = [["--num-rounds", "--num_rounds"]]
    assert runner.build_campaign_args(config, args, "--num_rounds", {}, alias_groups=alias_groups) == (
        ["--num_rounds", "3"],
        [],
    )
    with pytest.raises(ValueError, match="AUTOFL_BUDGET_ARGUMENT_CONFLICT"):
        runner.build_campaign_args(
            config, SimpleNamespace(base_args="--num-rounds 5"), "--num_rounds", {}, alias_groups=alias_groups
        )


def test_base_args_short_alias_of_pinned_flag_is_deduplicated_and_conflicts_rejected():
    runner = _load_runner()
    config = {"budget": {"fixed_training_budget": {"num_rounds": 5}}}
    alias_groups = [["-r", "--num_rounds"]]

    fixed_args, base_args = runner.build_campaign_args(
        config, SimpleNamespace(base_args="-r 5 --lr 0.1"), "--num_rounds --lr", {}, alias_groups=alias_groups
    )
    assert fixed_args == ["--num_rounds", "5"]
    assert base_args == ["--lr", "0.1"]

    for conflicting in ("-r 3", "-r=3", "-r3"):
        with pytest.raises(ValueError, match=r"AUTOFL_BUDGET_ARGUMENT_CONFLICT.*--num_rounds.*'5'.*'3'"):
            runner.build_campaign_args(
                config, SimpleNamespace(base_args=conflicting), "--num_rounds", {}, alias_groups=alias_groups
            )


def test_base_args_conflicting_fixed_budget_value_is_rejected():
    runner = _load_runner()
    config = {"budget": {"fixed_training_budget": {"num_rounds": 5}}}
    args = SimpleNamespace(base_args="--num_rounds 3")

    with pytest.raises(
        ValueError, match=r"AUTOFL_BUDGET_ARGUMENT_CONFLICT.*--num_rounds.*'5'.*fixed training budget.*'3'"
    ):
        runner.build_campaign_args(config, args, "--num_rounds", {})


def test_base_args_conflicting_comparison_budget_value_is_rejected():
    runner = _load_runner()
    schema = {"comparison_budget_args": {"default_candidate_budget": {"num_rounds": 20}}}
    args = SimpleNamespace(base_args="--num_rounds=3")

    with pytest.raises(
        ValueError, match=r"AUTOFL_BUDGET_ARGUMENT_CONFLICT.*--num_rounds.*'20'.*comparison budget.*'3'"
    ):
        runner.build_campaign_args({"budget": {}}, args, "--num_rounds", schema)


def test_base_args_identical_comparison_budget_value_is_emitted_once():
    runner = _load_runner()
    schema = {"comparison_budget_args": {"default_candidate_budget": {"num_rounds": 5}}}
    args = SimpleNamespace(base_args="--num_rounds 5 --synthetic_data")

    fixed_args, base_args = runner.build_campaign_args({"budget": {}}, args, "--num_rounds --synthetic_data", schema)

    assert fixed_args == []
    assert base_args == ["--synthetic_data", "--num_rounds", "5"]
    assert base_args.count("--num_rounds") == 1


def test_base_args_conflicting_duplicates_of_pinned_flag_are_rejected():
    runner = _load_runner()
    config = {"budget": {"fixed_training_budget": {"num_rounds": 5}}}
    args = SimpleNamespace(base_args="--num_rounds 5 --num-rounds=3")

    with pytest.raises(ValueError, match="AUTOFL_BUDGET_ARGUMENT_CONFLICT"):
        runner.build_campaign_args(config, args, "--num_rounds", {}, alias_groups=[["--num-rounds", "--num_rounds"]])


def test_base_args_ambiguous_abbreviation_of_pinned_flag_is_rejected():
    runner = _load_runner()
    config = {"budget": {"fixed_training_budget": {"num_rounds": 5}}}
    args = SimpleNamespace(base_args="--num_round 3")

    with pytest.raises(ValueError, match=r"AUTOFL_BUDGET_ARGUMENT_CONFLICT.*ambiguous abbreviation.*--num_rounds"):
        runner.build_campaign_args(config, args, "--num_rounds", {})


def test_base_args_boolean_comparison_flag_deduplicates_and_rejects_values():
    runner = _load_runner()
    schema = {"comparison_budget_args": {"default_candidate_budget": {"cross_site_eval": True}}}
    help_text = "--num_rounds --cross_site_eval"

    fixed_args, base_args = runner.build_campaign_args(
        {"budget": {}}, SimpleNamespace(base_args="--cross_site_eval"), help_text, schema
    )
    assert fixed_args == []
    assert base_args == ["--cross_site_eval"]

    # A zero-argument flag never consumes the next token: argparse would bind it to a positional.
    fixed_args, base_args = runner.build_campaign_args(
        {"budget": {}}, SimpleNamespace(base_args="--cross_site_eval dataset.csv"), help_text, schema
    )
    assert fixed_args == []
    assert base_args == ["dataset.csv", "--cross_site_eval"]

    # A clustered short alias (-xv meaning -x -v) sets the pinned flag itself; the runner omits its copy
    # so the option still appears exactly once.
    alias_groups = [["-x", "--cross_site_eval"]]
    fixed_args, base_args = runner.build_campaign_args(
        {"budget": {}}, SimpleNamespace(base_args="-xv"), help_text, schema, alias_groups=alias_groups
    )
    assert fixed_args == []
    assert base_args == ["-xv"]

    # A bare pinned short alias still deduplicates.
    fixed_args, base_args = runner.build_campaign_args(
        {"budget": {}}, SimpleNamespace(base_args="-x"), help_text, schema, alias_groups=alias_groups
    )
    assert fixed_args == []
    assert base_args == ["--cross_site_eval"]

    with pytest.raises(ValueError, match="AUTOFL_BUDGET_ARGUMENT_CONFLICT"):
        runner.build_campaign_args(
            {"budget": {}}, SimpleNamespace(base_args="--cross_site_eval=true"), help_text, schema
        )


def test_base_args_budget_named_flags_stay_verbatim_when_not_pinned():
    runner = _load_runner()
    args = SimpleNamespace(base_args="--seed 1 --seed 2 --model_arch resnet18 resnet50")

    fixed_args, base_args = runner.build_campaign_args({"budget": {}}, args, "--seed --model_arch", {})

    assert fixed_args == []
    assert base_args == ["--seed", "1", "--seed", "2", "--model_arch", "resnet18", "resnet50"]


def test_base_args_exact_job_flag_matching_pinned_prefix_is_not_ambiguous():
    runner = _load_runner()
    schema = {"comparison_budget_args": {"default_candidate_budget": {"batch_size": 16}}}
    args = SimpleNamespace(base_args="--batch 32")

    fixed_args, base_args = runner.build_campaign_args({"budget": {}}, args, "--batch --batch_size", schema)

    assert fixed_args == []
    assert base_args == ["--batch", "32", "--batch_size", "16"]


def test_restore_campaign_settings_ignores_legacy_synthetic_settings(tmp_path, capsys):
    runner = _load_runner()
    args = runner.parse_args(["status", "job.py"])
    settings = runner.campaign_settings(args)
    settings.update({"prefer_synthetic": True, "synthetic_train_size": 2048, "synthetic_test_size": 256})
    metadata = {"settings": settings, "workspace_root": str(tmp_path)}

    assert runner.restore_campaign_settings(args, metadata) is False

    assert not hasattr(args, "prefer_synthetic")
    help_text = "usage: job.py [--synthetic_data] [--train_size TRAIN_SIZE] [--test_size TEST_SIZE]"
    assert runner.build_campaign_args({"budget": {}}, args, help_text, {}) == ([], [])
    # The prior baseline/candidates were scored on injected synthetic data; new runs use
    # real data, so the ledger crosses a data regime — warn instead of staying silent.
    stderr = capsys.readouterr().err
    assert "computed on synthetic data" in stderr
    assert "re-initializing" in stderr


def test_restore_campaign_settings_does_not_warn_without_legacy_synthetic_setting(tmp_path, capsys):
    runner = _load_runner()
    args = runner.parse_args(["status", "job.py"])
    metadata = {"settings": runner.campaign_settings(args), "workspace_root": str(tmp_path)}

    assert runner.restore_campaign_settings(args, metadata) is False
    assert "synthetic" not in capsys.readouterr().err


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


def test_python_command_keeps_configured_interpreter_and_literal_arguments(tmp_path):
    runner = _load_runner()
    marker = tmp_path / "must-not-exist"
    arguments = ["-c", "print('ok')", ";", f"$(touch {marker})", "/bin/sh"]

    command = runner.python_command(sys.executable, arguments, {}, cwd=tmp_path)

    assert command[0] == os.path.abspath(sys.executable)
    assert command[1:] == arguments


def test_run_python_does_not_interpret_shell_metacharacters(tmp_path):
    runner = _load_runner()
    marker = tmp_path / "must-not-exist"
    literal = f"$(touch {marker})"

    rc, output, _, command = runner.run_python(
        sys.executable,
        ["-c", "import json, sys; print(json.dumps(sys.argv[1:]))", literal, "; touch ignored"],
        cwd=tmp_path,
        timeout=10,
        log_path=tmp_path / "run.log",
        env={},
    )

    assert rc == 0
    assert command[0] == os.path.abspath(sys.executable)
    assert command[1:] == ["-c", "import json, sys; print(json.dumps(sys.argv[1:]))", literal, "; touch ignored"]
    assert json.loads(output.strip()) == [literal, "; touch ignored"]
    assert not marker.exists()


def test_run_python_uses_only_the_explicit_child_environment(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "host-secret")

    rc, output, _, _command = runner.run_python(
        sys.executable,
        [
            "-c",
            (
                "import json, os; "
                "print(json.dumps({'safe': os.environ.get('SAFE_VALUE'), "
                "'secret': os.environ.get('AWS_SECRET_ACCESS_KEY')}))"
            ),
        ],
        cwd=tmp_path,
        timeout=10,
        log_path=tmp_path / "run.log",
        env={"SAFE_VALUE": "forwarded"},
    )

    assert rc == 0
    assert json.loads(output.strip()) == {"safe": "forwarded", "secret": None}


@pytest.mark.parametrize(
    ("python", "arguments", "message"),
    [
        ("/missing/python", ["-c", "print('no')"], "does not exist"),
        (None, ["-c", "print('no')"], "non-empty string"),
        (sys.executable, [], "at least one argument"),
        (sys.executable, ["-c", 3], "must be strings"),
        (sys.executable, ["bad\x00argument"], "NUL"),
    ],
)
def test_python_command_rejects_invalid_executables_and_arguments(tmp_path, python, arguments, message):
    runner = _load_runner()

    with pytest.raises(ValueError, match=message):
        runner.python_command(python, arguments, {}, cwd=tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="symlinked virtual-environment interpreter test is POSIX-specific")
def test_resolve_python_interpreter_preserves_virtualenv_symlink(tmp_path):
    runner = _load_runner()
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "campaign-python"
    venv_python.symlink_to(sys.executable)

    resolved = runner.resolve_python_interpreter(venv_python.name, {"PATH": str(venv_bin)}, cwd=tmp_path)

    assert resolved == str(venv_python)
    assert resolved != str(venv_python.resolve())


@pytest.mark.skipif(os.name == "nt", reason="symlinked virtual-environment interpreter test is POSIX-specific")
def test_resolve_python_interpreter_uses_child_cwd_for_relative_path(tmp_path):
    runner = _load_runner()
    job_dir = tmp_path / "job"
    venv_python = job_dir / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)

    resolved = runner.resolve_python_interpreter(".venv/bin/python", {}, cwd=job_dir)

    assert resolved == str(venv_python)


@pytest.mark.skipif(os.name == "nt", reason="relative PATH interpreter test is POSIX-specific")
def test_resolve_python_interpreter_normalizes_relative_path_entry_against_child_cwd(tmp_path):
    runner = _load_runner()
    job_dir = tmp_path / "job"
    venv_python = job_dir / "tools" / "campaign-python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)

    resolved = runner.resolve_python_interpreter("campaign-python", {"PATH": "tools"}, cwd=job_dir)

    assert resolved == str(venv_python)
    assert os.path.isabs(resolved)


def test_run_streams_output_before_timeout(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "run.log"

    rc, output, _, _command = runner.run_python(
        sys.executable,
        [
            "-c",
            "import time; print('started', flush=True); time.sleep(2); print('finished', flush=True)",
        ],
        cwd=tmp_path,
        timeout=1,
        log_path=log_path,
        env={},
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert rc == 124
    assert "started" in output
    assert "started" in log_text
    assert "TIMEOUT after 1s" in log_text


def test_run_streams_partial_line_before_timeout(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "run.log"

    rc, output, runtime, _command = runner.run_python(
        sys.executable,
        [
            "-c",
            "import sys, time; sys.stdout.write('partial output'); sys.stdout.flush(); time.sleep(30)",
        ],
        cwd=tmp_path,
        timeout=1,
        log_path=log_path,
        env={},
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

    rc, output, runtime, _command = runner.run_python(
        sys.executable,
        ["-c", parent_code],
        cwd=tmp_path,
        timeout=1,
        log_path=tmp_path / "run.log",
        env={},
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


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="escaped-process cleanup uses Linux /proc ownership")
@pytest.mark.parametrize("ignore_sigterm", [False, True], ids=["sigterm_honored", "sigterm_ignored"])
def test_run_terminates_detached_descendant_that_escapes_process_group(tmp_path, ignore_sigterm):
    runner = _load_runner()
    child_pid_path = tmp_path / "detached.pid"
    signal_setup = "signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_sigterm else ""
    child_code = (
        "import os, pathlib, signal, time; "
        f"{signal_setup}"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
        "print('detached descendant started', flush=True); time.sleep(90)"
    )
    parent_code = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], start_new_session=True); "
        "print('leader exits', flush=True)"
    )

    rc, output, runtime, _command = runner.run_python(
        sys.executable,
        ["-c", parent_code],
        cwd=tmp_path,
        timeout=2 if ignore_sigterm else 1,
        log_path=tmp_path / "run.log",
        env={},
    )

    assert rc == 124
    assert runtime < (25 if ignore_sigterm else 5)
    assert "leader exits" in output
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
        pytest.fail(f"detached descendant process {child_pid} survived trial-token cleanup")


def test_signal_trial_processes_uses_pidfds_and_revalidates_identity(monkeypatch):
    runner = _load_runner()
    identities = {10: [123, 123], 11: [456, -1]}
    sent = []
    closed = []

    monkeypatch.setattr(runner, "trial_process_ids", lambda _token: [123, 456])
    monkeypatch.setattr(runner, "process_has_trial_token", lambda _process_id, _marker: True)
    monkeypatch.setattr(runner, "pidfd_process_id", lambda pidfd: identities[pidfd].pop(0))
    monkeypatch.setattr(
        runner.os, "pidfd_open", lambda process_id, _flags: {123: 10, 456: 11}[process_id], raising=False
    )
    monkeypatch.setattr(runner.os, "close", closed.append)
    monkeypatch.setattr(runner.signal, "pidfd_send_signal", lambda pidfd, sig: sent.append((pidfd, sig)), raising=False)

    runner.signal_trial_processes("trial-token", runner.signal.SIGTERM)

    assert sent == [(10, runner.signal.SIGTERM)]
    assert closed == [10, 11]


def test_signal_trial_processes_reports_pidfd_open_failure(monkeypatch):
    runner = _load_runner()

    monkeypatch.setattr(runner, "trial_process_ids", lambda _token: [123])
    monkeypatch.setattr(
        runner.os,
        "pidfd_open",
        lambda _process_id, _flags: (_ for _ in ()).throw(PermissionError("pidfd denied")),
        raising=False,
    )
    monkeypatch.setattr(runner.signal, "pidfd_send_signal", lambda _pidfd, _sig: None, raising=False)

    with pytest.raises(runner.TrialProcessCleanupError, match="cannot open pidfd for trial process 123"):
        runner.signal_trial_processes("trial-token", runner.signal.SIGTERM)


def test_terminate_process_retries_pidfd_failure_with_sigkill(monkeypatch):
    runner = _load_runner()
    signals = []
    waits = iter([False, True])

    class ExitedProcess:
        @staticmethod
        def poll():
            return 0

    def signal_trial_processes(_trial_token, sig):
        signals.append(sig)
        if sig == runner.signal.SIGTERM:
            raise runner.TrialProcessCleanupError("temporary pidfd failure")

    monkeypatch.setattr(runner, "signal_trial_processes", signal_trial_processes)
    monkeypatch.setattr(runner, "wait_for_process_tree", lambda *_args, **_kwargs: next(waits))
    monkeypatch.setattr(runner, "trial_process_ids", lambda _trial_token: [])

    runner.terminate_process(ExitedProcess(), trial_token="trial-token")

    assert signals == [runner.signal.SIGTERM, runner.signal.SIGKILL]


def test_run_requires_pidfds_before_starting_linux_process(tmp_path, monkeypatch):
    runner = _load_runner()
    started = False

    def fail_start(*_args, **_kwargs):
        nonlocal started
        started = True
        raise AssertionError("process must not start without race-safe cleanup support")

    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(
        runner,
        "ensure_trial_process_pidfd_support",
        lambda: (_ for _ in ()).throw(runner.TrialProcessCleanupError("pidfds unavailable")),
    )
    monkeypatch.setattr(runner.subprocess, "Popen", fail_start)

    with pytest.raises(runner.TrialProcessCleanupError, match="pidfds unavailable"):
        runner.run_python(
            sys.executable,
            ["-c", "print('must not run')"],
            cwd=tmp_path,
            timeout=10,
            log_path=tmp_path / "run.log",
            env={},
        )

    assert not started


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
        runner.run_python(
            sys.executable,
            [
                "-c",
                f"import os, pathlib, time; pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(30)",
            ],
            cwd=tmp_path,
            timeout=0,
            log_path=tmp_path / "run.log",
            env={},
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

    rc, output, _, _command = runner.run_python(
        sys.executable,
        ["-c", script],
        cwd=tmp_path,
        timeout=10,
        log_path=log_path,
        env={},
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

    def fake_run(_python, _arguments, *, log_path, **_kwargs):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "PermissionError\n[Errno 1] Operation not permitted\nsocket.bind\n",
            encoding="utf-8",
        )
        return _fake_python_result(_python, _arguments, rc=1, output="socket.bind failed\n")

    monkeypatch.setattr(runner, "run_python", fake_run)
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


def test_run_job_argv_contains_each_budget_flag_exactly_once(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    captured = {}

    def fake_run(_python, _arguments, *, log_path, **_kwargs):
        captured["arguments"] = list(_arguments)
        return _fake_python_result(_python, _arguments)

    monkeypatch.setattr(runner, "run_python", fake_run)
    config = {"budget": {"fixed_training_budget": {"num_clients": 2, "num_rounds": 3}}, "job": {}}
    help_text = "--n_clients --num_rounds --update_type"
    args = SimpleNamespace(base_args="--n_clients 2 --num_rounds=3 --update_type full")
    fixed_args, base_args = runner.build_campaign_args(config, args, help_text, {})
    runner.run_job(
        runner.JobRun("baseline", [], "baseline"),
        python=sys.executable,
        job=job,
        cwd=tmp_path,
        help_text=help_text,
        fixed_args=fixed_args,
        base_args=base_args,
        output_root=tmp_path / "runs",
        timeout=10,
        simulator_no_progress_timeout=0,
        metrics=["accuracy"],
        config=config,
    )

    arguments = captured["arguments"]
    for flag in ("--n_clients", "--num_rounds", "--update_type"):
        assert arguments.count(flag) == 1
    assert arguments.index("--num_rounds") < arguments.index("--update_type")


def test_run_job_uses_run_python_prepared_command_without_revalidating(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    prepared_command = ["/resolved/campaign-python", str(job), "--candidate-flag"]

    def fake_run(_python, _arguments, *, log_path, **_kwargs):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("candidate failed\n", encoding="utf-8")
        return 1, "candidate failed\n", 0.1, prepared_command

    monkeypatch.setattr(runner, "run_python", fake_run)
    monkeypatch.setattr(
        runner,
        "python_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("run_job must not revalidate argv")),
    )

    record = runner.run_job(
        runner.JobRun("candidate", ["--candidate-flag"], "candidate"),
        python="./.venv/bin/python",
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

    assert record.run_command == runner.shlex.join(prepared_command)


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
    monkeypatch.setattr(
        runner, "run_python", lambda python, arguments, **_kwargs: _fake_python_result(python, arguments)
    )

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
    monkeypatch.setattr(
        runner, "run_python", lambda python, arguments, **_kwargs: _fake_python_result(python, arguments)
    )
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
    monkeypatch.setattr(
        runner, "run_python", lambda python, arguments, **_kwargs: _fake_python_result(python, arguments)
    )

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

    def fake_run(python, arguments, **kwargs):
        captured["env"] = kwargs["env"]
        return _fake_python_result(
            python,
            arguments,
            output='native warning\n{"version": "2.9.0", "supported": true}\nlate stderr warning\n',
        )

    monkeypatch.setattr(runner, "run_python", fake_run)

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

    def fake_run(python, arguments, **kwargs):
        result = Path(kwargs["env"][runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR]) / "recipe-default"
        result.mkdir(parents=True)
        result.joinpath("metrics_summary.json").write_text(json.dumps({"accuracy": 0.81}), encoding="utf-8")
        return _fake_python_result(python, arguments, output=f"The simulation logs can be found at {result}\n")

    monkeypatch.setattr(runner, "run_python", fake_run)
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

    def fake_run(python, arguments, **kwargs):
        result = Path(kwargs["env"][runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR]) / "recipe-default"
        result.mkdir(parents=True)
        result.joinpath("metrics_summary.json").write_text(json.dumps({"val_acc": 0.73}), encoding="utf-8")
        return _fake_python_result(python, arguments, output="job complete without a result message\n")

    monkeypatch.setattr(runner, "run_python", fake_run)
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

    def fake_run(python, arguments, **kwargs):
        simulator_base = Path(kwargs["env"][runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR])
        for name in ("first", "second"):
            result = simulator_base / name
            result.mkdir(parents=True)
            result.joinpath("metrics_summary.json").write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")
        return _fake_python_result(python, arguments, output="ambiguous job complete\n")

    monkeypatch.setattr(runner, "run_python", fake_run)
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

    def fake_run(python, arguments, **kwargs):
        outside.mkdir()
        return _fake_python_result(python, arguments, output=f"Result can be found in : {outside}\n")

    monkeypatch.setattr(runner, "run_python", fake_run)
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

    def fake_run(python, arguments, **kwargs):
        workspace = Path(kwargs["env"][runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR])
        workspaces.append(workspace)
        result = workspace / "fixed-job"
        result.mkdir(parents=True)
        result.joinpath("metrics_summary.json").write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")
        return _fake_python_result(python, arguments, output=f"Result can be found in : {result}\n")

    monkeypatch.setattr(runner, "run_python", fake_run)
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

    rc, output, runtime, _command = runner.run_python(
        sys.executable,
        [
            "-c",
            "import time; print('started', flush=True); time.sleep(30)",
        ],
        cwd=tmp_path,
        timeout=30,
        log_path=log_path,
        env={},
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

    rc, output, runtime, _command = runner.run_python(
        sys.executable,
        [
            "-c",
            "import time; print('started', flush=True); time.sleep(30)",
        ],
        cwd=tmp_path,
        timeout=30,
        log_path=log_path,
        env={},
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

    rc, output, runtime, _command = runner.run_python(
        sys.executable,
        [
            "-c",
            (
                "import pathlib, time; "
                f"path = pathlib.Path({str(site_log)!r}); "
                "time.sleep(0.2); "
                "path.write_text('[site=site-1] round=0\\n[site=site-2] round=0\\n'); "
                "time.sleep(30)"
            ),
        ],
        cwd=tmp_path,
        timeout=30,
        log_path=log_path,
        env={},
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

    assert runner.best_retained_record(records).name == "candidate_1"
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

    assert runner.main(["initialize", str(job), "--env", "sim"]) == 75

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


def test_candidate_execution_fingerprint_uses_existing_comparison_provenance():
    runner = _load_runner()
    manifest = {
        "base_source_sha256": "a" * 64,
        "fixed_budget_sha256": "b" * 64,
        "run_args": ["--lr", "0.1"],
    }
    fingerprint = runner.candidate_execution_fingerprint(manifest, "c" * 64)

    assert runner.candidate_execution_fingerprint(deepcopy(manifest), "c" * 64) == fingerprint
    for field, value in (
        ("base_source_sha256", "d" * 64),
        ("fixed_budget_sha256", "e" * 64),
        ("run_args", ["--lr", "0.2"]),
    ):
        changed = deepcopy(manifest)
        changed[field] = value
        assert runner.candidate_execution_fingerprint(changed, "c" * 64) != fingerprint
    assert runner.candidate_execution_fingerprint(manifest, "f" * 64) != fingerprint


def test_identical_crashed_candidate_replay_is_counted_and_records_outcome(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, baseline_score=0.8)
    assert runner.main(["status", str(job), "--max-candidates", "2"]) == 0
    calls = []

    outcomes = iter([("crash", None), ("candidate", 0.9)])

    def replay_run(run_def, **kwargs):
        calls.append(run_def.name)
        status, score = next(outcomes)
        return runner.RunRecord(
            status, run_def.name, score, 1.0, "none", run_def.description, "python job.py", "/tmp/run"
        )

    monkeypatch.setattr(runner, "run_job", replay_run)
    for candidate in ("first_crash", "same_after_crash"):
        assert runner.main(["prepare", str(job), "--name", candidate, "--hypothesis", "same candidate"]) == 0
        source = tmp_path / ".nvflare" / "autofl" / "candidates" / candidate / "source" / "client.py"
        source.write_text("ALGORITHM = 'crashing_candidate'\n", encoding="utf-8")
        if candidate == "first_crash":
            assert runner.main(["evaluate", str(job)]) == 0

    second_manifest_path = tmp_path / ".nvflare/autofl/candidates/same_after_crash/candidate_manifest.json"
    assert runner.main(["evaluate", str(job), "--manifest", str(second_manifest_path)]) == 0
    first_manifest = json.loads(
        tmp_path.joinpath(".nvflare/autofl/candidates/first_crash/candidate_manifest.json").read_text(encoding="utf-8")
    )
    second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    replay = second_manifest["crash_replay"]
    assert second_manifest["status"] == "keep"
    assert second_manifest["execution_fingerprint"] == first_manifest["execution_fingerprint"]
    assert replay["execution_fingerprint"] == first_manifest["execution_fingerprint"]
    assert replay["prior_candidate"] == "first_crash"
    assert replay["outcome_status"] == "keep"
    assert replay["recorded_at"]
    assert "crash_repeat_approval" not in second_manifest
    assert calls == ["first_crash", "same_after_crash"]
    records = runner.load_results(tmp_path / "results.tsv")
    assert runner.candidate_attempts(records) == 2
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert state["accounting_instruction"] == runner.ACCOUNTING_INSTRUCTION
    assert state["candidate_cap"] == 2
    assert state["remaining_candidates"] == 0
    assert state["final_response_allowed"] is True
    assert state["reason"] == "candidate_cap_exhausted"


def test_crash_replay_provenance_is_not_stamped_for_infrastructure_retry_or_changed_rerun(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, baseline_score=0.8)
    assert runner.main(["status", str(job), "--max-candidates", "2"]) == 0

    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            "crash", run_def.name, None, 1.0, "none", run_def.description, "python job.py", "/tmp/crash"
        ),
    )
    assert runner.main(["prepare", str(job), "--name", "first_crash", "--hypothesis", "same candidate"]) == 0
    tmp_path.joinpath(".nvflare/autofl/candidates/first_crash/source/client.py").write_text(
        "ALGORITHM = 'crashing_candidate'\n", encoding="utf-8"
    )
    assert runner.main(["evaluate", str(job)]) == 0

    assert runner.main(["prepare", str(job), "--name", "retry", "--hypothesis", "same candidate"]) == 0
    retry_manifest = tmp_path / ".nvflare/autofl/candidates/retry/candidate_manifest.json"
    retry_source = retry_manifest.parent / "source/client.py"
    retry_source.write_text("ALGORITHM = 'crashing_candidate'\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            runner.INFRASTRUCTURE_RETRY,
            run_def.name,
            None,
            1.0,
            "none",
            run_def.description,
            "python job.py",
            "/tmp/infrastructure",
            failure_reason="socket unavailable",
        ),
    )
    assert runner.main(["evaluate", str(job), "--manifest", str(retry_manifest)]) == 75
    manifest = json.loads(retry_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared"
    assert "crash_replay" not in manifest

    retry_source.write_text("ALGORITHM = 'fixed_candidate'\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            "candidate", run_def.name, 0.9, 1.0, "none", run_def.description, "python job.py", "/tmp/success"
        ),
    )
    assert runner.main(["evaluate", str(job), "--manifest", str(retry_manifest)]) == 0
    manifest = json.loads(retry_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "keep"
    assert "crash_replay" not in manifest
    assert runner.candidate_attempts(runner.load_results(tmp_path / "results.tsv")) == 2


def test_invalid_crashed_sibling_does_not_block_candidate_evaluation(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, baseline_score=0.8)
    assert runner.main(["prepare", str(job), "--name", "valid", "--hypothesis", "valid candidate"]) == 0
    valid_manifest = tmp_path / ".nvflare/autofl/candidates/valid/candidate_manifest.json"
    valid_manifest.parent.joinpath("source/client.py").write_text("ALGORITHM = 'valid'\n", encoding="utf-8")

    broken_manifest = tmp_path / ".nvflare/autofl/candidates/broken_crash/candidate_manifest.json"
    broken_manifest.parent.mkdir(parents=True)
    broken_manifest.write_text(
        json.dumps(
            {
                "schema_version": runner.CANDIDATE_MANIFEST_SCHEMA_VERSION,
                "candidate_id": "broken_crash",
                "workspace_root": str(tmp_path),
                "status": "crash",
                "patch_sha256": "a" * 64,
                "run_args": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            "candidate", run_def.name, 0.9, 1.0, "none", run_def.description, "python job.py", "/tmp/success"
        ),
    )

    assert runner.main(["evaluate", str(job), "--manifest", str(valid_manifest)]) == 0
    warning = capsys.readouterr().err
    assert "ignoring invalid sibling candidate manifest" in warning
    assert str(broken_manifest) in warning
    assert json.loads(valid_manifest.read_text(encoding="utf-8"))["status"] == "keep"


def test_changed_candidate_after_crash_is_allowed(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, baseline_score=0.8)

    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            "crash", run_def.name, None, 1.0, "none", run_def.description, "python job.py", "/tmp/crash"
        ),
    )
    assert runner.main(["prepare", str(job), "--name", "crash", "--hypothesis", "first candidate"]) == 0
    tmp_path.joinpath(".nvflare/autofl/candidates/crash/source/client.py").write_text(
        "ALGORITHM = 'first'\n", encoding="utf-8"
    )
    assert runner.main(["evaluate", str(job)]) == 0

    assert runner.main(["prepare", str(job), "--name", "changed", "--hypothesis", "changed candidate"]) == 0
    tmp_path.joinpath(".nvflare/autofl/candidates/changed/source/client.py").write_text(
        "ALGORITHM = 'second'\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda run_def, **kwargs: runner.RunRecord(
            "candidate", run_def.name, 0.9, 1.0, "none", run_def.description, "python job.py", "/tmp/success"
        ),
    )

    assert runner.main(["evaluate", str(job)]) == 0
    assert (
        json.loads(
            tmp_path.joinpath(".nvflare/autofl/candidates/changed/candidate_manifest.json").read_text(encoding="utf-8")
        )["status"]
        == "keep"
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
    command = ["initialize", str(job)]
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
        "job_key_metric": "auc",
        "job_key_metric_source": "arg:key_metric",
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

    assert runner.main(["initialize", str(job)]) == 0
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

    assert runner.main(["status", str(job), "--uncapped", "--confirm-user-approved-cap-change"]) == 0
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
    assert runner.main(["status", str(job), "--uncapped", "--confirm-user-approved-cap-change"]) == 0
    after_approval = metadata_path.read_bytes()
    assert runner.main(["status", str(job), "--uncapped", "--confirm-user-approved-cap-change"]) == 0
    assert metadata_path.read_bytes() == after_approval

    cap_changes = json.loads(metadata_path.read_text(encoding="utf-8"))["cap_changes"]
    assert [(entry["old"], entry["new"], entry["source"], entry["user_approved"]) for entry in cap_changes] == [
        (None, 7, "explicit", False),
        (7, None, "uncapped", True),
    ]
    assert all(entry["changed_at"] for entry in cap_changes)


def test_cap_expansion_requires_specific_user_approval_and_is_write_free_when_rejected(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    metadata_path = tmp_path / ".nvflare/autofl/campaign.json"

    assert runner.main(["status", str(job), "--max-candidates", "2"]) == 0
    before = metadata_path.read_bytes()

    assert runner.main(["status", str(job), "--max-candidates", "3"]) == 2
    assert "requires explicit user approval" in capsys.readouterr().err
    assert metadata_path.read_bytes() == before

    assert runner.main(["status", str(job), "--uncapped"]) == 2
    assert "requires explicit user approval" in capsys.readouterr().err
    assert metadata_path.read_bytes() == before

    assert (
        runner.main(
            [
                "status",
                str(job),
                "--max-candidates",
                "3",
                "--confirm-user-approved-cap-change",
            ]
        )
        == 0
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["settings"]["max_candidates"] == 3
    assert [(entry["old"], entry["new"], entry["user_approved"]) for entry in metadata["cap_changes"]] == [
        (None, 2, False),
        (2, 3, True),
    ]
    after_approval = metadata_path.read_bytes()
    assert (
        runner.main(
            [
                "status",
                str(job),
                "--max-candidates",
                "3",
                "--confirm-user-approved-cap-change",
            ]
        )
        == 0
    )
    assert metadata_path.read_bytes() == after_approval


def test_runner_rejects_abbreviated_lifecycle_options(capsys):
    runner = _load_runner()

    with pytest.raises(SystemExit) as error:
        runner.parse_args(["status", "job.py", "--max-cand", "5"])

    assert error.value.code == 2
    assert "unrecognized arguments: --max-cand 5" in capsys.readouterr().err


def test_cap_confirmation_is_rejected_when_no_expansion_requires_it(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)

    assert runner.main(["status", str(job), "--max-candidates", "3"]) == 0
    assert (
        runner.main(
            [
                "status",
                str(job),
                "--max-candidates",
                "2",
                "--confirm-user-approved-cap-change",
            ]
        )
        == 2
    )
    assert "only valid for a candidate-cap increase" in capsys.readouterr().err
    metadata = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign.json").read_text(encoding="utf-8"))
    assert metadata["settings"]["max_candidates"] == 3


def test_raising_cap_reopens_exhausted_campaign_with_consistent_state(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    results_path = tmp_path / "results.tsv"
    records = runner.load_results(results_path)
    records.append(runner.RunRecord("discard", "candidate_1", 0.4, 1.0, "none", "candidate", "python job.py", ""))
    runner.write_results(results_path, records)
    assert runner.main(["status", str(job), "--max-candidates", "1"]) == 0

    state_path = tmp_path / ".nvflare/autofl/campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["candidate_cap"] == 1
    assert state["final_response_allowed"] is True
    assert state["reason"] == "candidate_cap_exhausted"

    assert (
        runner.main(
            [
                "prepare",
                str(job),
                "--name",
                "reopened",
                "--hypothesis",
                "resume search",
                "--max-candidates",
                "2",
                "--confirm-user-approved-cap-change",
            ]
        )
        == 0
    )

    metadata = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign.json").read_text(encoding="utf-8"))
    assert metadata["settings"]["max_candidates"] == 2
    assert [(entry["old"], entry["new"]) for entry in metadata["cap_changes"]] == [(None, 1), (1, 2)]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["candidate_cap"] == 2
    assert state["final_response_allowed"] is False
    assert state["remaining_candidates"] == 1
    assert tmp_path.joinpath(".nvflare/autofl/candidates/reopened/candidate_manifest.json").exists()


def test_cap_change_with_unrelated_preflight_rejection_keeps_files_consistent(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "draft", "--hypothesis", "draft candidate"]) == 0

    rc = runner.main(
        ["prepare", str(job), "--name", "second", "--hypothesis", "second candidate", "--max-candidates", "5"]
    )

    assert rc == 2
    assert "pending candidate" in capsys.readouterr().err
    metadata = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign.json").read_text(encoding="utf-8"))
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert metadata["settings"]["max_candidates"] == 5
    assert [(entry["old"], entry["new"]) for entry in metadata["cap_changes"]] == [(None, 5)]
    assert state["candidate_cap"] == 5
    assert state["final_response_allowed"] is False
    assert not tmp_path.joinpath(".nvflare/autofl/candidates/second").exists()


def test_preflight_rejection_without_settings_change_is_write_free(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    assert runner.main(["prepare", str(job), "--name", "draft", "--hypothesis", "draft candidate"]) == 0
    watched = [
        tmp_path / ".nvflare/autofl/campaign.json",
        tmp_path / ".nvflare/autofl/campaign_state.json",
        tmp_path / "results.tsv",
    ]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in watched}

    rc = runner.main(["prepare", str(job), "--name", "second", "--hypothesis", "second candidate"])

    assert rc == 2
    assert "pending candidate" in capsys.readouterr().err
    for path in watched:
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before[path]


def test_lowering_cap_below_attempts_clamps_remaining_and_keeps_state_consistent(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    results_path = tmp_path / "results.tsv"
    records = runner.load_results(results_path)
    records.append(runner.RunRecord("discard", "candidate_1", 0.4, 1.0, "none", "candidate", "python job.py", ""))
    records.append(runner.RunRecord("discard", "candidate_2", 0.3, 1.0, "none", "candidate", "python job.py", ""))
    runner.write_results(results_path, records)
    assert runner.main(["status", str(job), "--max-candidates", "3"]) == 0

    rc = runner.main(["prepare", str(job), "--name", "late", "--hypothesis", "late candidate", "--max-candidates", "1"])

    assert rc == 2
    assert "campaign is already final: candidate_cap_exhausted" in capsys.readouterr().err
    metadata = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign.json").read_text(encoding="utf-8"))
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert metadata["settings"]["max_candidates"] == 1
    assert [(entry["old"], entry["new"]) for entry in metadata["cap_changes"]] == [(None, 3), (3, 1)]
    assert state["candidate_cap"] == 1
    assert state["candidate_attempts"] == 2
    assert state["remaining_candidates"] == 0
    assert state["final_response_allowed"] is True
    assert state["reason"] == "candidate_cap_exhausted"
    assert not tmp_path.joinpath(".nvflare/autofl/candidates/late").exists()


def test_runner_state_reports_budget_and_baseline_accounting(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch, baseline_score=0.5)
    state_path = tmp_path / ".nvflare/autofl/campaign_state.json"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["remaining_candidates"] is None
    assert state["baseline_status"] == "complete"
    assert state["baseline_score"] == pytest.approx(0.5)
    assert state["improvement"] == pytest.approx(0.0)
    assert state["abandoned_candidates"] == 0

    results_path = tmp_path / "results.tsv"
    records = runner.load_results(results_path)
    records.append(
        runner.RunRecord("keep", "higher_accuracy", 0.8, 1.0, "none", "higher accuracy", "python job.py", "")
    )
    runner.write_results(results_path, records)

    assert runner.main(["status", str(job), "--max-candidates", "3"]) == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["remaining_candidates"] == 2
    # This campaign uses the default max direction, so improvement is best minus baseline.
    assert state["improvement"] == pytest.approx(0.3)


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


def test_initialize_has_no_mode_flag(tmp_path, capsys):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")

    # Direction comes from the imported job contract, not a runner override.
    with pytest.raises(SystemExit) as excinfo:
        runner.main(["initialize", str(job), "--mode", "min"])

    assert excinfo.value.code == 2
    assert "unrecognized arguments: --mode" in capsys.readouterr().err
    assert not tmp_path.joinpath(".nvflare").exists()


def test_resuming_legacy_minimization_campaign_requires_fresh_initialization(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    metadata_path = tmp_path / ".nvflare/autofl/campaign.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["settings"]["mode"] == "max"
    metadata["settings"]["mode"] = "min"
    metadata.pop("metric_direction_contract_version")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    state_before = tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_bytes()
    capsys.readouterr()

    # Every lifecycle action routes through the same restore_campaign_settings gate.
    for action in ("status", "prepare", "evaluate", "abandon", "record", "suggest"):
        assert runner.main([action, str(job)]) == 2
        stderr = capsys.readouterr().err
        assert "mode='min'" in stderr
        assert "native metric-direction provenance" in stderr

    # The recommended recovery must not be circular: initialize on the legacy campaign
    # is rejected too, but its message names the concrete escape hatch.
    assert runner.main(["initialize", str(job)]) == 2
    stderr = capsys.readouterr().err
    assert "native metric-direction provenance" in stderr
    assert ".nvflare/autofl" in stderr
    assert "fresh workspace" in stderr

    # The campaign never silently flips to maximization: no state was rewritten.
    assert tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_bytes() == state_before


def test_native_minimization_campaign_resumes_and_reports_direction(tmp_path, monkeypatch):
    runner = _load_runner()
    config = _campaign_config()
    config["objective"].update(
        {
            "metric": "val_loss",
            "requested_metric": "val_loss",
            "optimization_metric": "val_loss",
            "metric_extraction_order": ["val_loss"],
            "mode": "min",
            "mode_contract_source": "job:key_metric_mode",
            "job_key_metric": "val_loss",
        }
    )
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    tmp_path.joinpath("client.py").write_text("ALGORITHM = 'baseline'\n", encoding="utf-8")
    monkeypatch.setattr(runner, "import_job_config", lambda *args, **kwargs: deepcopy(config))
    monkeypatch.setattr(runner, "job_help", lambda *args, **kwargs: "")
    monkeypatch.setattr(runner, "write_progress", lambda path, *args: path.write_bytes(b"progress"))
    scores = {"baseline": 0.5, "lower_loss": 0.4, "higher_loss": 0.45}

    def run_job(run_def, **kwargs):
        return runner.RunRecord(
            run_def.status,
            run_def.name,
            scores[run_def.name],
            1.0,
            "none",
            run_def.description,
            "python job.py",
            f"/tmp/{run_def.name}",
        )

    monkeypatch.setattr(runner, "run_job", run_job)

    assert runner.main(["initialize", str(job)]) == 0
    assert runner.main(["prepare", str(job), "--name", "lower_loss", "--hypothesis", "reduce loss"]) == 0
    lower_draft = tmp_path / ".nvflare/autofl/candidates/lower_loss/source/client.py"
    lower_draft.write_text("ALGORITHM = 'lower-loss'\n", encoding="utf-8")
    assert runner.main(["evaluate", str(job)]) == 0
    assert runner.main(["prepare", str(job), "--name", "higher_loss", "--hypothesis", "test regression"]) == 0
    higher_draft = tmp_path / ".nvflare/autofl/candidates/higher_loss/source/client.py"
    higher_draft.write_text("ALGORITHM = 'higher-loss'\n", encoding="utf-8")
    assert runner.main(["evaluate", str(job)]) == 0
    assert runner.main(["status", str(job)]) == 0
    metadata = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign.json").read_text(encoding="utf-8"))
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    records = runner.load_results(tmp_path / "results.tsv")
    assert metadata["settings"]["mode"] == "min"
    assert metadata["metric_direction_contract_version"] == runner.METRIC_DIRECTION_CONTRACT_VERSION
    assert state["mode"] == "min"
    assert state["best_score"] == pytest.approx(0.4)
    assert state["improvement"] == pytest.approx(0.1)
    assert [record.status for record in records] == ["baseline", "keep", "discard"]
    assert tmp_path.joinpath("client.py").read_text(encoding="utf-8") == "ALGORITHM = 'lower-loss'\n"
    raw_status_args = runner.parse_args(["status", str(job)])
    assert raw_status_args.mode == "max"
    _, refreshed = runner.refresh_campaign_state(
        raw_status_args,
        job,
        runner.load_campaign_metadata(tmp_path, job),
        runner.campaign_paths(raw_status_args, job),
    )
    assert refreshed["mode"] == "min"


def test_initialize_revalidates_source_after_in_memory_admission(tmp_path, monkeypatch):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    monkeypatch.setattr(runner, "import_job_config", lambda *args, **kwargs: deepcopy(_campaign_config()))
    args = runner.parse_args(["initialize", str(job)])

    runner.prepare_initial_campaign(args, job)
    job.write_text("print('changed')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after metric-contract admission"):
        runner.admitted_initial_campaign(args, job)


def test_initialize_rejects_conflicting_base_args_before_baseline(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text(
        """
import argparse

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.parse_args()
    recipe = FedAvgRecipe(
        name="literal-budget",
        min_clients=2,
        num_rounds=5,
        train_script="client.py",
        key_metric="accuracy",
    )
    recipe.execute(SimEnv(num_clients=2))


if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("baseline must not execute")),
    )

    assert runner.main(["initialize", str(job), "--base-args", "--num_rounds 3"]) == 2

    stderr = capsys.readouterr().err
    assert "AUTOFL_BUDGET_ARGUMENT_CONFLICT" in stderr
    assert not tmp_path.joinpath(".nvflare").exists()
    assert not tmp_path.joinpath("autofl.yaml").exists()


def test_initialize_rejects_conflicting_short_alias_base_args_before_baseline(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text(
        """
import argparse

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--num_rounds", type=int, default=3)
    parser.parse_args()
    recipe = FedAvgRecipe(
        name="literal-budget",
        min_clients=2,
        num_rounds=5,
        train_script="client.py",
        key_metric="accuracy",
    )
    recipe.execute(SimEnv(num_clients=2))


if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("baseline must not execute")),
    )

    assert runner.main(["initialize", str(job), "--base-args", "-r 3"]) == 2

    stderr = capsys.readouterr().err
    assert "AUTOFL_BUDGET_ARGUMENT_CONFLICT" in stderr
    assert "--num_rounds" in stderr
    assert not tmp_path.joinpath(".nvflare").exists()
    assert not tmp_path.joinpath("autofl.yaml").exists()


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


def test_import_job_config_forwards_job_args_without_direction_plumbing(tmp_path, monkeypatch):
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
    # A job-owned "--mode" flag in --base-args is unrelated to the removed objective direction.
    args = runner.parse_args(["initialize", str(job), "--base-args", "--mode training"])
    runner.import_job_config(args, job, output, tmp_path / "import.log", 10)

    assert "mode" not in captured
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
        (
            {
                "import": {"support": {"status": "supported"}},
                "budget": {"fixed_training_budget": {"num_rounds": 1}},
                "unresolved": [{"field": "objective.job_key_metric", "reason": "dynamic job metric"}],
            },
            "objective.job_key_metric",
        ),
    ],
)
def test_campaign_admission_rejects_unresolved_safety_contract(config, expected):
    runner = _load_runner()

    assert expected in "; ".join(runner.campaign_admission_errors(config))


def test_initialize_rejects_implicit_max_loss_without_writing_campaign_files(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")
    config = _campaign_config()
    config["objective"].update(
        {
            "metric": "val_loss",
            "requested_metric": "val_loss",
            "optimization_metric": "val_loss",
            "metric_extraction_order": ["val_loss"],
            "mode": "max",
            "mode_contract_source": "core_default",
            "job_key_metric": "val_loss",
        }
    )
    monkeypatch.setattr(runner, "import_job_config", lambda *args, **kwargs: deepcopy(config))

    assert runner.main(["initialize", str(job)]) == 2
    assert "AUTOFL_METRIC_DIRECTION_CONFLICT" in capsys.readouterr().err
    assert not tmp_path.joinpath(".nvflare").exists()
    assert not tmp_path.joinpath("autofl.yaml").exists()
    assert not tmp_path.joinpath("autofl_runs").exists()


def test_initialize_rejects_splatted_job_contract_without_writing_campaign_files(tmp_path, capsys):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job = tmp_path / "job.py"
    job.write_text(
        """
import argparse
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


parser = argparse.ArgumentParser()
parser.add_argument("--key_metric", default="val_loss")
args = parser.parse_args()
tuning = {"key_metric": args.key_metric, "key_metric_mode": "min"}
recipe = FedAvgRecipe(
    name="splatted-contract", min_clients=2, num_rounds=1, train_script="client.py", **tuning
)
recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )
    runner = _load_runner()

    assert runner.main(["initialize", str(job)]) == 2

    error = capsys.readouterr().err
    assert "objective.metric" in error
    assert "objective.mode" in error
    assert "job call passes **kwargs" in error
    assert not tmp_path.joinpath(".nvflare").exists()
    assert not tmp_path.joinpath("autofl.yaml").exists()
    assert not tmp_path.joinpath("autofl_runs").exists()


@pytest.mark.parametrize(
    "sim_env_setup",
    [
        'sim_args = {"clients": ["site-1", "site-2"]}\nrecipe.execute(SimEnv(num_clients=0, **sim_args))',
        'def get_clients():\n    return ["site-1", "site-2"]\n\n\nrecipe.execute(SimEnv(clients=get_clients()))',
    ],
)
def test_initialize_rejects_unresolved_sim_env_clients_without_writing_campaign_files(tmp_path, capsys, sim_env_setup):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job = tmp_path / "job.py"
    job.write_text(
        f"""
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


recipe = FedAvgRecipe(
    name="sim-clients", min_clients=2, num_rounds=1, train_script="client.py", key_metric_mode="max"
)
{sim_env_setup}
""".lstrip(),
        encoding="utf-8",
    )
    runner = _load_runner()

    assert runner.main(["initialize", str(job)]) == 2

    error = capsys.readouterr().err
    assert "budget.fixed_training_budget.num_clients" in error
    assert not tmp_path.joinpath(".nvflare").exists()
    assert not tmp_path.joinpath("autofl.yaml").exists()
    assert not tmp_path.joinpath("autofl_runs").exists()


def test_initialize_rejects_splatted_metric_even_with_declared_bridge_and_explicit_direction(tmp_path, capsys):
    tmp_path.joinpath("train.py").write_text("print('train')\n", encoding="utf-8")
    job = tmp_path / "job.py"
    job.write_text(
        """
from nvflare.app_common.executors.script_runner import ScriptRunner
from nvflare.job_config.base_fed_job import BaseFedJob


metric = {"key_metric": "accuracy"}
job = BaseFedJob(
    name="splatted-bridge",
    min_clients=2,
    key_metric_mode="max",
    model_selector=None,
    **metric,
)
runner = ScriptRunner(script="train.py")
""".lstrip(),
        encoding="utf-8",
    )
    tmp_path.joinpath("mutation_schema.yaml").write_text(
        """
objective:
  requested_metric: f1
  optimization_metric: f1
  mode: max
""".lstrip(),
        encoding="utf-8",
    )
    runner = _load_runner()

    assert runner.main(["initialize", str(job), "--metric", "f1"]) == 2

    error = capsys.readouterr().err
    assert "objective.metric" in error
    assert "objective.mode" in error
    assert "job call passes **kwargs" in error
    assert not tmp_path.joinpath(".nvflare").exists()
    assert not tmp_path.joinpath("autofl.yaml").exists()
    assert not tmp_path.joinpath("autofl_runs").exists()


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


def test_initialize_rejects_explicit_metric_matching_unresolved_job_metric_placeholder(tmp_path, capsys):
    runner = _load_runner()
    workspace = tmp_path / "explicit-unbridged"
    job = _write_dynamic_metric_job(workspace)

    assert runner.main(["initialize", str(job), "--metric", "accuracy"]) == 2

    error = capsys.readouterr().err
    assert "AUTOFL_METRIC_NOT_DECLARED" in error
    assert not workspace.joinpath(".nvflare").exists()


def test_initialize_rejects_explicit_metric_bridge_for_unresolved_job_metric(tmp_path, capsys):
    runner = _load_runner()
    workspace = tmp_path / "explicit-bridged"
    job = _write_dynamic_metric_job(workspace)
    workspace.joinpath("mutation_schema.yaml").write_text(
        """
objective:
  requested_metric: accuracy
  optimization_metric: accuracy
  mode: max
""".lstrip(),
        encoding="utf-8",
    )

    assert runner.main(["initialize", str(job), "--metric", "accuracy"]) == 2

    assert "objective.job_key_metric" in capsys.readouterr().err
    assert not workspace.joinpath(".nvflare").exists()


def test_initialize_rejects_unresolved_job_metric_without_user_metric(tmp_path, capsys):
    runner = _load_runner()
    workspace = tmp_path / "implicit"
    job = _write_dynamic_metric_job(workspace)

    assert runner.main(["initialize", str(job)]) == 2

    assert "objective.metric" in capsys.readouterr().err
    assert not workspace.joinpath(".nvflare").exists()


def test_initialize_treats_core_default_accuracy_as_resolved_job_metric(tmp_path, monkeypatch):
    runner = _load_runner()
    workspace = tmp_path / "core-default"
    workspace.mkdir()
    workspace.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job = workspace / "job.py"
    job.write_text(
        """
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


recipe = FedAvgRecipe(name="default_metric", min_clients=2, num_rounds=1, train_script="client.py")
recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )
    _mock_successful_baseline(runner, monkeypatch)

    assert runner.main(["initialize", str(job), "--metric", "accuracy"]) == 0

    config = runner.read_yaml(workspace / "autofl.yaml")
    assert config["objective"]["job_key_metric_source"] == "core_default"
    assert config["objective"]["mode_contract_source"] == "core_default"


def test_initialize_accepts_resolved_job_metric_from_noncanonical_arg_name(tmp_path, monkeypatch):
    runner = _load_runner()
    workspace = tmp_path / "noncanonical-arg"
    workspace.mkdir()
    workspace.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job = workspace / "job.py"
    job.write_text(
        """
import argparse
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


parser = argparse.ArgumentParser()
parser.add_argument("--model_metric", default="accuracy")
args = parser.parse_args()
recipe = FedAvgRecipe(
    name="arg_metric", min_clients=2, num_rounds=1, train_script="client.py", key_metric=args.model_metric
)
recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )
    _mock_successful_baseline(runner, monkeypatch)

    assert runner.main(["initialize", str(job)]) == 0

    config = runner.read_yaml(workspace / "autofl.yaml")
    assert config["objective"]["job_key_metric_source"] == "arg:model_metric"


def test_campaign_admission_allows_unknown_metric_with_core_default_max():
    runner = _load_runner()
    config = _campaign_config()
    config["objective"].update(
        {
            "metric": "custom_quality",
            "requested_metric": "custom_quality",
            "optimization_metric": "custom_quality",
            "mode": "max",
            "mode_contract_source": "core_default",
            "job_key_metric": "custom_quality",
        }
    )

    assert runner.campaign_admission_errors(config) == []


@pytest.mark.parametrize(
    ("source", "requested_metric", "job_metric", "expected"),
    [
        ("literal", "accuracy", "accuracy", False),
        ("arg:key_metric", "accuracy", "accuracy", False),
        ("arg:model_metric", "accuracy", "accuracy", False),
        ("core_default", "accuracy", "accuracy", False),
        ("default", "accuracy", "accuracy", True),
        (None, "accuracy", "accuracy", False),
        ("literal", "accuracy", "val_accuracy", True),
    ],
)
def test_requested_metric_identity_requires_resolved_job_metric(source, requested_metric, job_metric, expected):
    runner = _load_runner()
    objective = _campaign_config()["objective"]
    objective.update(
        {
            "requested_metric": requested_metric,
            "job_key_metric": job_metric,
            "job_key_metric_source": source,
        }
    )

    assert runner.requested_metric_differs_from_job(objective) is expected


def test_requested_metric_identity_handles_incomplete_raw_objective():
    runner = _load_runner()

    assert runner.requested_metric_differs_from_job({}) is True
    assert runner.requested_metric_differs_from_job({"requested_metric": "accuracy"}) is True
    assert (
        runner.assumed_job_key_metric_mode(
            {"mode_contract_source": "job:key_metric_mode"},
            {"mode": "min"},
        )
        == "max"
    )


def test_campaign_admission_requires_schema_for_metric_bridge():
    runner = _load_runner()
    config = _campaign_config()
    config["objective"].update(
        {
            "metric": "accuracy",
            "requested_metric": "accuracy",
            "optimization_metric": "accuracy",
            "job_key_metric": "val_accuracy",
        }
    )

    errors = runner.campaign_admission_errors(config)
    assert any("AUTOFL_METRIC_NOT_DECLARED" in error for error in errors)

    schema = {"objective": {"requested_metric": "accuracy", "optimization_metric": "test_accuracy"}}
    config = runner.apply_metric_contract(config, "accuracy", schema)
    assert runner.campaign_admission_errors(config, schema) == []


def test_lower_is_better_metric_bridge_requires_explicit_schema_mode():
    runner = _load_runner()
    config = _campaign_config()
    config["objective"].update(
        {
            "metric": "val_loss",
            "requested_metric": "val_loss",
            "optimization_metric": "val_loss",
            "metric_extraction_order": ["val_loss"],
            "mode": "min",
            "mode_contract_source": "job:key_metric_mode",
            "job_key_metric": "accuracy",
        }
    )
    schema = {"objective": {"requested_metric": "val_loss", "optimization_metric": "val_loss"}}

    updated = runner.apply_metric_contract(config, "val_loss", schema)
    errors = runner.campaign_admission_errors(updated, schema)

    assert updated["objective"]["mode"] == "max"
    assert updated["objective"]["mode_contract_source"] == "core_default"
    assert any("AUTOFL_METRIC_DIRECTION_CONFLICT" in error for error in errors)
    assert any("mutation_schema.yaml metric bridge" in error for error in errors)


def test_candidate_metric_direction_drift_is_rejected():
    runner = _load_runner()
    current = _campaign_config()
    candidate = deepcopy(current)
    candidate["objective"]["mode"] = "min"
    candidate["objective"]["mode_contract_source"] = "job:key_metric_mode"

    with pytest.raises(ValueError, match="objective metric invariants: mode"):
        runner.candidate_campaign_config(candidate, current, SimpleNamespace(metric="accuracy"), {})


def test_candidate_job_metric_direction_drift_is_rejected_for_bridged_metric():
    runner = _load_runner()
    schema = {
        "objective": {
            "requested_metric": "val_loss",
            "optimization_metric": "val_loss",
            "mode": "min",
        }
    }
    current = _campaign_config()
    current["objective"].update(
        {
            "metric": "val_loss",
            "requested_metric": "val_loss",
            "optimization_metric": "val_loss",
            "metric_extraction_order": ["val_loss"],
            "mode": "min",
            "mode_contract_source": "mutation_schema",
            "job_key_metric": "accuracy",
            "job_key_metric_mode": "max",
            "job_key_metric_mode_source": "core_default",
        }
    )
    candidate = deepcopy(current)
    candidate["objective"].update(
        {
            "mode": "min",
            "mode_contract_source": "job:key_metric_mode",
            "job_key_metric_mode": "min",
            "job_key_metric_mode_source": "job:key_metric_mode",
        }
    )

    with pytest.raises(ValueError, match="objective metric invariants: job_key_metric_mode"):
        runner.candidate_campaign_config(candidate, current, SimpleNamespace(metric="val_loss"), schema)


def test_candidate_import_tolerates_job_metric_absent_from_legacy_campaign_contract():
    runner = _load_runner()
    current = _campaign_config()
    current["objective"].pop("job_key_metric")
    candidate = _campaign_config()

    updated = runner.candidate_campaign_config(candidate, current, SimpleNamespace(metric="accuracy"), {})

    assert "job_key_metric" not in updated["objective"]


def test_candidate_import_tolerates_job_metric_mode_absent_from_legacy_campaign_contract():
    runner = _load_runner()
    current = _campaign_config()
    current["objective"].pop("job_key_metric_mode")
    current["objective"].pop("job_key_metric_mode_source")
    candidate = _campaign_config()

    updated = runner.candidate_campaign_config(candidate, current, SimpleNamespace(metric="accuracy"), {})

    assert "job_key_metric_mode" not in updated["objective"]


def test_candidate_import_backfills_missing_job_metric_mode_from_native_min_contract():
    runner = _load_runner()
    current = _campaign_config()
    current["objective"].update(
        {
            "mode": "min",
            "mode_contract_source": "job:key_metric_mode",
            "job_key_metric": "accuracy",
            "job_key_metric_source": "arg:model_metric",
        }
    )
    current["objective"].pop("job_key_metric_mode")
    current["objective"].pop("job_key_metric_mode_source")
    unchanged_candidate = deepcopy(current)
    unchanged_candidate["objective"].update(
        {
            "job_key_metric_mode": "min",
            "job_key_metric_mode_source": "job:key_metric_mode",
        }
    )

    updated = runner.candidate_campaign_config(unchanged_candidate, current, SimpleNamespace(metric="accuracy"), {})

    assert "job_key_metric_mode" not in updated["objective"]

    changed_candidate = deepcopy(unchanged_candidate)
    changed_candidate["objective"].update(
        {
            "mode": "max",
            "job_key_metric_mode": "max",
        }
    )
    with pytest.raises(ValueError, match="objective metric invariants: mode, job_key_metric_mode"):
        runner.candidate_campaign_config(changed_candidate, current, SimpleNamespace(metric="accuracy"), {})


def test_candidate_import_rejects_job_metric_mode_drift_from_legacy_max_campaign():
    runner = _load_runner()
    schema = {
        "objective": {
            "requested_metric": "val_loss",
            "optimization_metric": "val_loss",
            "mode": "min",
        }
    }
    current = _campaign_config()
    current["objective"].update(
        {
            "metric": "val_loss",
            "requested_metric": "val_loss",
            "optimization_metric": "val_loss",
            "metric_extraction_order": ["val_loss"],
            "mode": "min",
            "mode_contract_source": "mutation_schema",
        }
    )
    current["objective"].pop("job_key_metric")
    current["objective"].pop("job_key_metric_mode")
    current["objective"].pop("job_key_metric_mode_source")
    candidate = deepcopy(current)
    candidate["objective"].update(
        {
            "job_key_metric": "accuracy",
            "job_key_metric_mode": "min",
            "job_key_metric_mode_source": "job:key_metric_mode",
        }
    )

    with pytest.raises(ValueError, match="objective metric invariants: job_key_metric_mode"):
        runner.candidate_campaign_config(candidate, current, SimpleNamespace(metric="val_loss"), schema)


def test_legacy_campaign_without_mode_still_rejects_candidate_direction_change():
    runner = _load_runner()
    current = _campaign_config()
    current["objective"].pop("mode")
    candidate = _campaign_config()
    candidate["objective"].update({"mode": "min", "mode_contract_source": "job:key_metric_mode"})

    with pytest.raises(ValueError, match="objective metric invariants: mode") as exc:
        runner.candidate_campaign_config(candidate, current, SimpleNamespace(metric="accuracy"), {})

    assert "delete the campaign's .nvflare/autofl directory" in str(exc.value)


def test_evaluate_resumes_legacy_max_campaign_without_job_key_metric(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    config_path = tmp_path / "autofl.yaml"
    config = runner.read_yaml(config_path)
    config["objective"].pop("job_key_metric")
    runner.write_yaml(config_path, config)

    assert runner.main(["prepare", str(job), "--name", "legacy", "--hypothesis", "legacy candidate"]) == 0
    draft = tmp_path / ".nvflare/autofl/candidates/legacy/source/client.py"
    draft.write_text("ALGORITHM = 'legacy-candidate'\n", encoding="utf-8")

    assert runner.main(["evaluate", str(job)]) == 0


def test_status_rejects_campaign_direction_artifact_disagreement(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    job, _, _ = _initialize_fake_campaign(runner, tmp_path, monkeypatch)
    metadata_path = tmp_path / ".nvflare/autofl/campaign.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["settings"]["mode"] = "min"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert runner.main(["status", str(job)]) == 2
    assert "campaign metric direction disagrees" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("metric", "mode", "baseline_score", "candidate_score"),
    [
        pytest.param("accuracy", "max", 0.5, 0.8, id="accuracy-max"),
        pytest.param("loss", "min", 0.5, 0.2, id="loss-min"),
    ],
)
def test_cli_lifecycle_runs_agent_code_candidate_end_to_end(tmp_path, metric, mode, baseline_score, candidate_score):
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

SCORE = {baseline_score}

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
    ImportedFedAvgRecipe(
        model=object(),
        num_rounds=args.num_rounds,
        min_clients=args.n_clients,
        key_metric={metric!r},
        key_metric_mode={mode!r},
    )
    ImportedSimEnv(num_clients=args.n_clients, workspace_root={str(simulation_root)!r})
    result = Path(os.environ["NVFLARE_SIMULATOR_WORKSPACE_ROOT"]) / args.name
    result.mkdir(parents=True, exist_ok=True)
    result.joinpath("metrics_summary.json").write_text(json.dumps({{{metric!r}: SCORE}}))
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
    draft_job.write_text(
        draft_job.read_text(encoding="utf-8").replace(f"SCORE = {baseline_score}", f"SCORE = {candidate_score}"),
        encoding="utf-8",
    )
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
    config = runner.read_yaml(tmp_path / "autofl.yaml")
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert [(record.status, record.score) for record in records] == [
        ("baseline", baseline_score),
        ("keep", candidate_score),
    ]
    assert f"SCORE = {candidate_score}" in job.read_text(encoding="utf-8")
    assert config["objective"]["mode"] == mode
    assert state["mode"] == mode
    assert state["improvement"] == pytest.approx(abs(candidate_score - baseline_score))
    manifest = json.loads(
        tmp_path.joinpath(".nvflare/autofl/candidates/code_candidate/candidate_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "keep"
    assert manifest["changed_files"] == ["job.py"]


def test_cross_val_extraction_averages_server_final_global_model_entries(tmp_path):
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

    # Unweighted mean over the global model's per-site scores; site-local entries never count.
    assert evidence.score == pytest.approx((0.71 + 0.74) / 2)
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

    assert evidence.score == pytest.approx((0.71 + 0.74) / 2)
    assert evidence.source == "structured:cross_val_results.json#server_final"


def test_cross_val_extraction_mean_penalizes_easiest_site_bias(tmp_path):
    # Reviewer counterexample: a max reduction would score [0.90, 0.50] as 0.90 and rank it above
    # a uniformly better [0.80, 0.80]; the unweighted mean ranks the uniform candidate higher.
    runner = _load_runner()
    skewed = tmp_path / "skewed"
    uniform = tmp_path / "uniform"
    skewed.mkdir()
    uniform.mkdir()
    skewed.joinpath("cross_val_results.json").write_text(
        json.dumps(
            {
                "site-1": {"SRV_FL_global_model.pt": {"accuracy": 0.90}},
                "site-2": {"SRV_FL_global_model.pt": {"accuracy": 0.50}},
            }
        ),
        encoding="utf-8",
    )
    uniform.joinpath("cross_val_results.json").write_text(
        json.dumps(
            {
                "site-1": {"SRV_FL_global_model.pt": {"accuracy": 0.80}},
                "site-2": {"SRV_FL_global_model.pt": {"accuracy": 0.80}},
            }
        ),
        encoding="utf-8",
    )

    skewed_score = runner.extract_score(skewed, ["accuracy"])
    uniform_score = runner.extract_score(uniform, ["accuracy"])

    assert skewed_score == pytest.approx(0.70)
    assert uniform_score == pytest.approx(0.80)
    assert runner.better(uniform_score, skewed_score)


def test_cross_val_extraction_prefers_final_checkpoint_entries_over_best(tmp_path):
    runner = _load_runner()
    tmp_path.joinpath("cross_val_results.json").write_text(
        json.dumps(
            {
                "site-1": {
                    "SRV_FL_global_model.pt": {"accuracy": 0.60},
                    "SRV_best_FL_global_model.pt": {"accuracy": 0.90},
                },
                "site-2": {
                    "SRV_FL_global_model.pt": {"accuracy": 0.80},
                    "SRV_best_FL_global_model.pt": {"accuracy": 0.95},
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = runner.extract_metric_evidence(tmp_path, ["accuracy"])

    # Only the final-checkpoint class is averaged; best_-checkpoint entries are excluded.
    assert evidence.score == pytest.approx((0.60 + 0.80) / 2)
    assert evidence.source == "structured:cross_val_results.json#server_final"


def test_cross_val_extraction_resolves_srv_best_only_global_model_entries(tmp_path):
    runner = _load_runner()
    tmp_path.joinpath("cross_val_results.json").write_text(
        json.dumps(
            {
                "site-1": {
                    "site-1": {"accuracy": 0.99},
                    "SRV_best_FL_global_model.pt": {"accuracy": 0.66},
                },
                "site-2": {"SRV_best_FL_global_model.pt": {"accuracy": 0.70}},
            }
        ),
        encoding="utf-8",
    )

    evidence = runner.extract_metric_evidence(tmp_path, ["accuracy"])

    # Without any final-checkpoint entries the best_-checkpoint class is averaged instead.
    assert evidence.score == pytest.approx((0.66 + 0.70) / 2)
    assert evidence.source == "structured:cross_val_results.json#server_final"


def test_cross_val_extraction_single_site_mean_is_the_value_itself(tmp_path):
    runner = _load_runner()
    tmp_path.joinpath("cross_val_results.json").write_text(
        json.dumps(
            {
                "site-1": {
                    "site-1": {"loss": 0.10},
                    "SRV_FL_global_model.pt": {"loss": 0.42},
                }
            }
        ),
        encoding="utf-8",
    )

    assert runner.extract_score(tmp_path, ["loss"]) == pytest.approx(0.42)


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
