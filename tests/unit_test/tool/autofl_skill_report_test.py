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

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

RESULT_FIELDS = [
    "status",
    "name",
    "score",
    "runtime_seconds",
    "changed_files",
    "diff_summary",
    "run_command",
    "artifacts",
    "failure_reason",
    "candidate_manifest",
    "base_candidate",
    "patch_sha256",
]


def _load_reporter():
    repo_root = Path(__file__).parents[3]
    script_path = repo_root / "skills" / "nvflare-autofl-report" / "scripts" / "generate_report.py"
    spec = importlib.util.spec_from_file_location("nvflare_autofl_skill_report", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(status, name, score="", **kwargs):
    row = {field: "" for field in RESULT_FIELDS}
    row.update(
        {
            "status": status,
            "name": name,
            "score": score,
            "runtime_seconds": kwargs.pop("runtime_seconds", "10"),
            "changed_files": kwargs.pop("changed_files", "none"),
            "diff_summary": kwargs.pop("diff_summary", name),
            "run_command": kwargs.pop("run_command", "python job.py"),
            "artifacts": kwargs.pop("artifacts", f"runs/{name}"),
        }
    )
    row.update(kwargs)
    return row


def _write_campaign(tmp_path, *, active=False, mode="max"):
    rows = [
        _row(
            "baseline",
            "baseline",
            "0.500000",
            run_command=("python job.py --n_clients 8 --num_rounds 10 --aggregation_epochs 1 --name autofl_baseline"),
        ),
        _row(
            "literature",
            "literature_review_1",
            diff_summary="Review FedProx [src: Li18 arXiv:1812.06127] and SCAFFOLD [src: Karimireddy19].",
            run_command="agent literature review",
        ),
        _row(
            "keep" if mode == "min" else "discard",
            "weak_prox",
            "0.490000",
            base_candidate="baseline",
            diff_summary="Small FedProx term did not help.",
        ),
        _row(
            "discard" if mode == "min" else "keep",
            "algorithm_code",
            "0.600000",
            changed_files="client.py,job.py",
            base_candidate="baseline",
            candidate_manifest=".nvflare/autofl/candidates/algorithm_code/candidate_manifest.json",
            patch_sha256="a" * 64,
            diff_summary="Implement an agent-authored drift correction algorithm.",
            run_command=(
                "python job.py --n_clients 8 --num_rounds 10 --aggregation_epochs 2 --name autofl_algorithm_code"
            ),
        ),
        _row(
            "discard" if mode == "min" else "keep",
            "inherited_tuning",
            "0.650000",
            changed_files="none",
            base_candidate="algorithm_code",
            candidate_manifest=".nvflare/autofl/candidates/inherited_tuning/candidate_manifest.json",
            patch_sha256="b" * 64,
            diff_summary="Tune the retained algorithm without another source patch.",
            run_command=(
                "python job.py --n_clients 8 --num_rounds 10 --aggregation_epochs 2 --lr 0.01 "
                "--name autofl_inherited_tuning"
            ),
        ),
        _row(
            "crash",
            "unstable_branch",
            failure_reason="exit_code=1",
            base_candidate="inherited_tuning",
            diff_summary="A larger update was unstable.",
        ),
    ]
    with tmp_path.joinpath("results.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    tmp_path.joinpath("autofl.yaml").write_text(
        "schema_version: nvflare.autofl.config.v1\n"
        "objective:\n"
        "  requested_metric: accuracy\n"
        "  optimization_metric: test_accuracy\n"
        "  metric_source: held-out test set\n"
        f"  mode: {mode}\n"
        "environment:\n"
        "  requested: sim\n"
        "budget:\n"
        "  fixed_training_budget:\n"
        "    num_clients: 8\n"
        "    num_rounds: 20\n",
        encoding="utf-8",
    )
    state = {
        "reason": "running" if active else "manual_interrupt",
        "final_response_allowed": not active,
        "candidate_cap": None,
        "candidate_cap_source": "uncapped",
        "mode": mode,
    }
    tmp_path.joinpath(".nvflare/autofl").mkdir(parents=True)
    tmp_path.joinpath(".nvflare/autofl/campaign_state.json").write_text(json.dumps(state), encoding="utf-8")
    tmp_path.joinpath("progress.png").write_bytes(b"\x89PNG\r\n\x1a\nexisting plot")
    return rows


def _generate(reporter, tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(reporter, "refresh_plot", lambda *args, **kwargs: None)
    args = reporter.parse_args([str(tmp_path), *extra])
    return reporter.generate(args)


def test_report_refuses_active_campaign_without_confirmation(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path, active=True)
    monkeypatch.setattr(reporter, "refresh_plot", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="final_response_allowed=false"):
        reporter.generate(reporter.parse_args([str(tmp_path)]))

    assert not tmp_path.joinpath("autofl_final_report.md").exists()


def test_confirm_interrupted_reports_without_mutating_state(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path, active=True)
    state_path = tmp_path / ".nvflare/autofl/campaign_state.json"
    original_state = state_path.read_bytes()

    summary = _generate(reporter, tmp_path, monkeypatch, "--confirm-interrupted")

    assert summary["termination"] == {
        "reason": "user_confirmed_interruption",
        "state_allowed_final_response": False,
        "user_confirmed_interruption": True,
    }
    assert state_path.read_bytes() == original_state


def test_report_generates_product_artifacts_and_candidate_lineage(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)

    summary = _generate(reporter, tmp_path, monkeypatch)
    report = tmp_path.joinpath("autofl_final_report.md").read_text(encoding="utf-8")
    saved_summary = json.loads(tmp_path.joinpath("autofl_report_summary.json").read_text(encoding="utf-8"))

    assert summary["schema_version"] == "nvflare.autofl.report.v1"
    assert summary["baseline"]["score"] == 0.5
    assert summary["best"]["name"] == "inherited_tuning"
    assert summary["best"]["score"] == 0.65
    assert summary["best_lineage"] == {
        "candidates": ["baseline", "algorithm_code", "inherited_tuning"],
        "changed_files": ["client.py", "job.py"],
        "complete": True,
    }
    assert saved_summary["best"]["name"] == "inherited_tuning"
    assert "## Executive Summary" in report
    assert "## Literature Review Outcomes" in report
    assert "## Validation And Comparability Notes" in report
    assert "Product Findings" not in report
    assert str(tmp_path.joinpath("progress.png").resolve()) in report


def test_report_synthesizes_literature_against_checkpoint_incumbent(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)

    summary = _generate(reporter, tmp_path, monkeypatch)

    literature = summary["literature_reviews"]
    assert len(literature) == 1
    assert literature[0]["sources"] == ["Li18 arXiv:1812.06127", "Karimireddy19"]
    assert literature[0]["outcome"] == "helped"
    assert literature[0]["incumbent_score"] == 0.5
    assert literature[0]["best_candidate"] == "inherited_tuning"
    assert literature[0]["delta_from_incumbent"] == pytest.approx(0.15)


def test_report_warns_about_executed_budget_and_test_metric(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)

    summary = _generate(reporter, tmp_path, monkeypatch)
    warnings = "\n".join(summary["warnings"])

    assert "aggregation_epochs" in warnings
    assert "autofl.yaml=20, executed=10" in warnings
    assert "test-like metric" in warnings
    assert summary["best_command_changes"]["aggregation_epochs"] == {"baseline": "1", "best": "2"}


def test_report_supports_minimization_and_agent_context_without_git(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path, mode="min")
    context = tmp_path / "agent.json"
    context.write_text(json.dumps({"provider": "codex", "notes": "stopped by user"}), encoding="utf-8")

    summary = _generate(
        reporter,
        tmp_path,
        monkeypatch,
        "--metric",
        "loss",
        "--agent-context",
        str(context),
        "--agent-model",
        "gpt-test",
        "--reasoning-effort",
        "high",
    )

    assert not tmp_path.joinpath(".git").exists()
    assert summary["objective"]["mode"] == "min"
    assert summary["best"]["name"] == "weak_prox"
    assert summary["agent_context"] == {
        "model": "gpt-test",
        "notes": "stopped by user",
        "provider": "codex",
        "reasoning_effort": "high",
    }


def test_report_keeps_existing_plot_when_refresh_fails(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    monkeypatch.setattr(reporter, "default_plotter_path", lambda: tmp_path / "missing_plotter.py")

    summary = reporter.generate(reporter.parse_args([str(tmp_path)]))

    assert tmp_path.joinpath("progress.png").read_bytes() == b"\x89PNG\r\n\x1a\nexisting plot"
    assert any("plotter not found" in warning for warning in summary["warnings"])


def test_report_reads_best_candidate_manifest_when_available(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    manifest_path = tmp_path / ".nvflare/autofl/candidates/inherited_tuning/candidate_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "nvflare.autofl.candidate.v1",
                "candidate_id": "inherited_tuning",
                "base_candidate": "algorithm_code",
                "hypothesis": "tune retained algorithm",
                "run_args": ["--lr", "0.01"],
                "changed_files": [],
                "created_files": [],
                "candidate_source_sha256": "c" * 64,
                "fixed_budget_sha256": "d" * 64,
                "patch_sha256": "b" * 64,
                "status": "keep",
            }
        ),
        encoding="utf-8",
    )

    summary = _generate(reporter, tmp_path, monkeypatch)

    assert summary["best_manifest"]["available"] is True
    assert summary["best_manifest"]["candidate_id"] == "inherited_tuning"
    assert summary["best_manifest"]["budget_sha256"] == "d" * 64


def test_report_requires_a_progress_artifact(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    tmp_path.joinpath("progress.png").unlink()
    monkeypatch.setattr(reporter, "default_plotter_path", lambda: tmp_path / "missing_plotter.py")

    with pytest.raises(ValueError, match="progress plot is unavailable"):
        reporter.generate(reporter.parse_args([str(tmp_path)]))


def test_candidate_lineage_marks_cycles_incomplete():
    reporter = _load_reporter()
    fields = {
        "score": 0.5,
        "runtime_seconds": 1.0,
        "changed_files": "none",
        "diff_summary": "candidate",
        "run_command": "python job.py",
        "artifacts": "runs/candidate",
        "failure_reason": "",
        "candidate_manifest": "",
        "patch_sha256": "",
    }
    first = reporter.RunRecord(index=0, status="keep", name="first", base_candidate="second", **fields)
    second = reporter.RunRecord(index=1, status="keep", name="second", base_candidate="first", **fields)

    assert reporter.candidate_lineage(second, [first, second])["complete"] is False


def test_budget_comparison_accepts_numeric_equivalence():
    reporter = _load_reporter()

    assert reporter.values_equal("8.0", 8)
    assert not reporter.values_equal("8.1", 8)
