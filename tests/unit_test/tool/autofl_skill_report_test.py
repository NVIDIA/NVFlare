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
    "metric_name",
    "metric_source",
    "metric_artifact",
    "candidate_kind",
    "algorithm_family",
    "literature_event_id",
]


def _load_script(name, script_path):
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_reporter():
    repo_root = Path(__file__).parents[3]
    script_path = repo_root / "skills" / "nvflare-autofl-report" / "scripts" / "generate_report.py"
    return _load_script("nvflare_autofl_skill_report", script_path)


def _load_autofl_script(name):
    repo_root = Path(__file__).parents[3]
    script_path = repo_root / "skills" / "nvflare-autofl" / "scripts" / f"{name}.py"
    return _load_script(f"nvflare_autofl_{name}", script_path)


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


def _write_rows(tmp_path, rows):
    with tmp_path.joinpath("results.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_campaign(tmp_path, *, active=False, mode="max"):
    rows = [
        _row(
            "baseline",
            "baseline",
            "0.500000",
            run_command=("python job.py --n_clients 8 --num_rounds 10 --aggregation_epochs 1 --name autofl_baseline"),
            metric_name="test_accuracy",
            metric_source="json:metrics_summary.json",
            metric_artifact="runs/baseline/metrics_summary.json",
        ),
        _row(
            "literature",
            "literature_review_1",
            diff_summary="Review FedProx [src: Li18 arXiv:1812.06127] and SCAFFOLD [src: Karimireddy19].",
            run_command="agent literature review",
            literature_event_id="lit-0001",
        ),
        _row(
            "keep" if mode == "min" else "discard",
            "weak_prox",
            "0.490000",
            base_candidate="baseline",
            diff_summary="Small FedProx term did not help.",
            literature_event_id="lit-0001",
        ),
        _row(
            "discard" if mode == "min" else "keep",
            "algorithm_code",
            "0.600000",
            changed_files="client.py,job.py",
            base_candidate="baseline",
            candidate_manifest=".nvflare/autofl/candidates/algorithm_code/candidate_manifest.json",
            patch_sha256="a" * 64,
            metric_name="test_accuracy",
            metric_source="json:cross_val_results.json",
            metric_artifact="runs/algorithm_code/cross_val_results.json",
            candidate_kind="source_edit",
            algorithm_family="drift_correction",
            literature_event_id="lit-0001",
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
            metric_name="test_accuracy",
            metric_source="json:cross_val_results.json",
            metric_artifact="runs/inherited_tuning/cross_val_results.json",
            candidate_kind="argument_only",
            algorithm_family="drift_correction",
            literature_event_id="lit-0001",
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
            literature_event_id="lit-0001",
        ),
    ]
    _write_rows(tmp_path, rows)

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


@pytest.mark.parametrize("score", ["", "0.900000"])
def test_report_refuses_pending_ledger_rows_even_when_interruption_is_confirmed(tmp_path, monkeypatch, score):
    reporter = _load_reporter()
    rows = _write_campaign(tmp_path, active=True)
    rows.append(_row("candidate", "pending_algo", score, base_candidate="baseline"))
    _write_rows(tmp_path, rows)
    monkeypatch.setattr(reporter, "refresh_plot", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="finalize or abandon pending candidates"):
        reporter.generate(reporter.parse_args([str(tmp_path), "--confirm-interrupted"]))

    assert not tmp_path.joinpath("autofl_final_report.md").exists()


def test_report_refuses_pending_campaign_state(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    state_path = tmp_path / ".nvflare/autofl/campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["pending_candidates"] = 1
    state["pending_candidate_manifest"] = ".nvflare/autofl/candidates/pending/candidate_manifest.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(reporter, "refresh_plot", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="campaign state reports pending_candidates=1"):
        reporter.generate(reporter.parse_args([str(tmp_path), "--confirm-interrupted"]))


def test_report_refuses_pending_manifest_named_only_in_campaign_state(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    state_path = tmp_path / ".nvflare/autofl/campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["pending_candidate_manifest"] = ".nvflare/autofl/candidates/pending/candidate_manifest.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(reporter, "refresh_plot", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="campaign state names"):
        reporter.generate(reporter.parse_args([str(tmp_path), "--confirm-interrupted"]))


@pytest.mark.parametrize("status", ["prepared", "ready_for_external_execution"])
def test_report_refuses_pending_candidate_manifests(tmp_path, monkeypatch, status):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    manifest_path = tmp_path / ".nvflare/autofl/candidates/pending/candidate_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"status": status}), encoding="utf-8")
    monkeypatch.setattr(reporter, "refresh_plot", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="candidate manifests remain prepared"):
        reporter.generate(reporter.parse_args([str(tmp_path), "--confirm-interrupted"]))


@pytest.mark.parametrize("content", ["", '{"status":', "[]"])
def test_report_refuses_unreadable_candidate_manifests(tmp_path, monkeypatch, content):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    manifest_path = tmp_path / ".nvflare/autofl/candidates/unreadable/candidate_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(reporter, "refresh_plot", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="candidate manifests could not be read"):
        reporter.generate(reporter.parse_args([str(tmp_path), "--confirm-interrupted"]))


def test_report_checks_ledger_referenced_candidate_manifest(tmp_path, monkeypatch):
    reporter = _load_reporter()
    rows = _write_campaign(tmp_path)
    manifest_path = tmp_path / "external/candidate_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps({"status": "prepared"}), encoding="utf-8")
    rows[2]["candidate_manifest"] = str(manifest_path.relative_to(tmp_path))
    _write_rows(tmp_path, rows)
    monkeypatch.setattr(reporter, "refresh_plot", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="candidate manifests remain prepared"):
        reporter.generate(reporter.parse_args([str(tmp_path)]))


def test_pending_refusal_does_not_refresh_progress_plot(tmp_path, monkeypatch):
    reporter = _load_reporter()
    rows = _write_campaign(tmp_path)
    rows.append(_row("candidate", "pending_algo"))
    _write_rows(tmp_path, rows)
    progress_path = tmp_path / "progress.png"
    original_progress = progress_path.read_bytes()

    def mutate_progress(*args, **kwargs):
        progress_path.write_bytes(b"mutated")

    monkeypatch.setattr(reporter, "refresh_plot", mutate_progress)

    with pytest.raises(ValueError, match="finalize or abandon pending candidates"):
        reporter.generate(reporter.parse_args([str(tmp_path)]))

    assert progress_path.read_bytes() == original_progress


def test_pending_candidate_returns_cli_exit_2(tmp_path, monkeypatch, capsys):
    reporter = _load_reporter()
    rows = _write_campaign(tmp_path)
    rows.append(_row("candidate", "pending_algo"))
    _write_rows(tmp_path, rows)
    monkeypatch.setattr(reporter, "refresh_plot", lambda *args, **kwargs: None)

    assert reporter.main([str(tmp_path)]) == 2
    assert "finalize or abandon pending candidates" in capsys.readouterr().err


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
    assert summary["artifacts"]["progress_plot_available"] is True
    assert "## Executive Summary" in report
    assert "## Literature Review Outcomes" in report
    assert "## Validation And Comparability Notes" in report
    assert "Product Findings" not in report
    assert str(tmp_path.joinpath("progress.png").resolve()) in report
    assert "Metric extraction source: `json:cross_val_results.json`" in report
    assert "Metric artifact: `runs/inherited_tuning/cross_val_results.json`" in report


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


def test_report_uses_explicit_literature_links_across_later_events(tmp_path, monkeypatch):
    reporter = _load_reporter()
    rows = [
        _row("baseline", "baseline", "0.500000"),
        _row("literature", "literature_review_1", literature_event_id="lit-0001"),
        _row("literature", "literature_review_2", literature_event_id="lit-0002"),
        _row(
            "keep",
            "late_candidate_for_first_review",
            "0.700000",
            candidate_kind="source_edit",
            literature_event_id="lit-0001",
        ),
        _row(
            "discard",
            "candidate_for_second_review",
            "0.600000",
            candidate_kind="source_edit",
            literature_event_id="lit-0002",
        ),
        _row("discard", "unlinked_candidate", "0.650000", candidate_kind="source_edit"),
    ]
    _write_rows(tmp_path, rows)
    tmp_path.joinpath("autofl.yaml").write_text("objective:\n  metric: accuracy\n", encoding="utf-8")
    state_path = tmp_path / ".nvflare/autofl/campaign_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"final_response_allowed": True}), encoding="utf-8")
    tmp_path.joinpath("progress.png").write_bytes(b"\x89PNG\r\n\x1a\nplot")

    summary = _generate(reporter, tmp_path, monkeypatch)

    first, second = summary["literature_reviews"]
    assert first["literature_event_id"] == "lit-0001"
    assert first["candidate_attempts"] == ["late_candidate_for_first_review"]
    assert second["literature_event_id"] == "lit-0002"
    assert second["candidate_attempts"] == ["candidate_for_second_review"]
    assert all("unlinked_candidate" not in item["candidate_attempts"] for item in summary["literature_reviews"])


def test_candidate_name_containing_literature_is_not_a_checkpoint(tmp_path, monkeypatch):
    reporter = _load_reporter()
    rows = [
        _row("baseline", "baseline", "0.500000"),
        _row("literature", "review_fedprox", literature_event_id="lit-0001"),
        _row(
            "keep",
            "literature_review_based_init",
            "0.600000",
            candidate_kind="source_edit",
            literature_event_id="lit-0001",
        ),
        _row(
            "discard",
            "fedprox_followup",
            "0.550000",
            candidate_kind="source_edit",
            literature_event_id="lit-0001",
        ),
    ]
    _write_rows(tmp_path, rows)
    tmp_path.joinpath("autofl.yaml").write_text("objective:\n  metric: accuracy\n", encoding="utf-8")
    state_path = tmp_path / ".nvflare/autofl/campaign_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"final_response_allowed": True}), encoding="utf-8")
    tmp_path.joinpath("progress.png").write_bytes(b"\x89PNG\r\n\x1a\nplot")

    summary = _generate(reporter, tmp_path, monkeypatch)

    assert len(summary["literature_reviews"]) == 1
    assert summary["literature_reviews"][0]["event"] == "review_fedprox"
    assert summary["literature_reviews"][0]["incumbent_score"] == 0.5
    assert summary["literature_reviews"][0]["candidate_attempts"] == [
        "literature_review_based_init",
        "fedprox_followup",
    ]


def test_literature_evidence_preserves_crash_and_blank_discard_status(tmp_path, monkeypatch):
    reporter = _load_reporter()
    rows = _write_campaign(tmp_path)
    rows[-1]["score"] = "0.990000"
    rows.append(
        _row(
            "discard",
            "blank_discard",
            "",
            diff_summary="No metric artifact was produced.",
            literature_event_id="lit-0001",
        )
    )
    _write_rows(tmp_path, rows)

    summary = _generate(reporter, tmp_path, monkeypatch)
    report = tmp_path.joinpath("autofl_final_report.md").read_text(encoding="utf-8")

    assert summary["best"]["name"] == "inherited_tuning"
    assert summary["best_observed"]["name"] == "inherited_tuning"
    assert all(item["name"] != "unstable_branch" for item in summary["milestones"])
    assert "unstable_branch=crash" in report
    assert "blank_discard=n/a" in report


def test_discard_only_campaign_has_observed_but_no_retained_best(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    _write_rows(tmp_path, [_row("discard", "unretained_gain", "0.700000")])

    summary = _generate(reporter, tmp_path, monkeypatch)
    report = tmp_path.joinpath("autofl_final_report.md").read_text(encoding="utf-8")

    assert summary["best"] is None
    assert summary["best_observed"]["name"] == "unretained_gain"
    assert "No scored result was retained" in report
    assert "Best retained candidate" not in report


def test_baseline_named_discard_cannot_become_baseline_or_retained_best(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    _write_rows(
        tmp_path,
        [
            _row("discard", "baseline_plus_prox", "0.900000"),
            _row("baseline", "control", "0.500000"),
            _row("keep", "retained_candidate", "0.700000", base_candidate="control"),
        ],
    )

    summary = _generate(reporter, tmp_path, monkeypatch)
    report = tmp_path.joinpath("autofl_final_report.md").read_text(encoding="utf-8")

    assert summary["baseline"]["name"] == "control"
    assert summary["best"]["name"] == "retained_candidate"
    assert summary["best_observed"]["name"] == "baseline_plus_prox"
    assert any("baseline_plus_prox was not retained" in warning for warning in summary["warnings"])
    assert "Best retained candidate: `retained_candidate`" in report


def test_zero_runtime_is_rendered_as_not_available(tmp_path, monkeypatch):
    reporter = _load_reporter()
    rows = _write_campaign(tmp_path)
    for row in rows:
        row["runtime_seconds"] = ""
    _write_rows(tmp_path, rows)

    summary = _generate(reporter, tmp_path, monkeypatch)
    report = tmp_path.joinpath("autofl_final_report.md").read_text(encoding="utf-8")

    assert summary["runtime_seconds"] == 0
    assert f"across `{len(rows)}` ledger rows in `n/a`." in report
    assert "- Total recorded runtime: `n/a`" in report


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
    assert summary["artifacts"]["progress_plot_available"] is True


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
                "candidate_kind": "argument_only",
                "algorithm_family": "drift_correction",
                "literature_event_id": "lit-0001",
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
    assert summary["best_manifest"]["candidate_kind"] == "argument_only"
    assert summary["best_manifest"]["literature_event_id"] == "lit-0001"


@pytest.mark.parametrize("invalid_content", [None, b"not a png"])
def test_report_degrades_when_progress_artifact_is_unavailable(tmp_path, monkeypatch, invalid_content):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    progress_path = tmp_path.joinpath("progress.png")
    if invalid_content is None:
        progress_path.unlink()
    else:
        progress_path.write_bytes(invalid_content)
    monkeypatch.setattr(reporter, "default_plotter_path", lambda: tmp_path / "missing_plotter.py")

    summary = reporter.generate(reporter.parse_args([str(tmp_path)]))
    report = tmp_path.joinpath("autofl_final_report.md").read_text(encoding="utf-8")

    assert summary["artifacts"]["progress_plot_available"] is False
    assert "Progress plot unavailable" in report
    assert "![Auto-FL progress]" not in report
    assert tmp_path.joinpath("autofl_report_summary.json").is_file()
    if invalid_content is not None:
        assert progress_path.read_bytes() == invalid_content


def test_report_normalizes_malformed_optional_contract_sections(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    tmp_path.joinpath("autofl.yaml").write_text(
        "objective: []\nenvironment: sim\nbudget: null\n",
        encoding="utf-8",
    )

    summary = _generate(reporter, tmp_path, monkeypatch)
    warnings = "\n".join(summary["warnings"])

    assert summary["objective"]["optimization_metric"] == "score"
    assert summary["objective"]["metric_source"] == "NVFlare metric artifacts"
    assert summary["environment"] == "not declared"
    assert summary["declared_fixed_budget"] == {}
    assert "section 'objective' is list" in warnings
    assert "section 'environment' is str" in warnings
    assert "section 'budget' is null" in warnings


def test_report_separates_metric_measurement_and_contract_sources(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    tmp_path.joinpath("autofl.yaml").write_text(
        "objective:\n"
        "  requested_metric: accuracy\n"
        "  optimization_metric: test_accuracy\n"
        "  metric_contract_source: arg:key_metric\n",
        encoding="utf-8",
    )

    summary = _generate(reporter, tmp_path, monkeypatch)

    assert summary["objective"]["metric_source"] == "NVFlare metric artifacts"
    assert summary["objective"]["metric_contract_source"] == "arg:key_metric"


def test_crash_only_attempts_do_not_trigger_test_metric_selection_warning(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    rows = [
        _row("baseline", "baseline", "0.500000", metric_name="test_accuracy"),
        _row("crash", "candidate_1", failure_reason="exit_code=1"),
        _row("crash", "candidate_2", failure_reason="exit_code=1"),
    ]
    _write_rows(tmp_path, rows)

    summary = _generate(reporter, tmp_path, monkeypatch)

    assert not any("selected against a test-like metric" in warning for warning in summary["warnings"])


def test_relative_plotter_path_resolves_from_campaign_directory(tmp_path, monkeypatch):
    reporter = _load_reporter()
    _write_campaign(tmp_path)
    plotter_path = tmp_path / "tools/plot_progress.py"
    plotter_path.parent.mkdir()
    plotter_path.write_text("# test plotter\n", encoding="utf-8")
    captured = {}

    def _capture_plotter(*args):
        captured["path"] = args[-1]

    monkeypatch.setattr(reporter, "refresh_plot", _capture_plotter)

    reporter.generate(reporter.parse_args([str(tmp_path), "--plotter", "tools/plot_progress.py"]))

    assert captured["path"] == plotter_path.resolve()


def test_refresh_plot_reloads_plotter_without_leaking_module(tmp_path):
    reporter = _load_reporter()
    results_path = tmp_path / "results.tsv"
    results_path.write_text("status\tname\n", encoding="utf-8")
    output_path = tmp_path / "progress.png"
    module_name = "nvflare_autofl_report_plotter"

    for marker in ("first", "second"):
        plotter_path = tmp_path / f"plot_{marker}.py"
        plotter_path.write_text(
            "from pathlib import Path\n"
            "def load_results(path):\n"
            "    return []\n"
            "def plot_progress(records, output, mode, metric):\n"
            f"    Path(output).write_text('{marker}', encoding='utf-8')\n",
            encoding="utf-8",
        )

        assert reporter.refresh_plot(results_path, output_path, "max", "accuracy", plotter_path) is None
        assert output_path.read_text(encoding="utf-8") == marker
        assert module_name not in sys.modules


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


def test_report_attempt_and_baseline_rules_match_campaign_guard():
    reporter = _load_reporter()
    guard = _load_autofl_script("campaign_guard")
    cases = [
        {"status": "baseline", "name": "control", "run_command": "python job.py"},
        {"status": "keep", "name": "baseline", "run_command": "python job.py"},
        {"status": "discard", "name": "baseline_seed_1", "run_command": "python job.py"},
        {"status": "discard", "name": "control", "run_command": "python job.py --name baseline"},
        {"status": "keep", "name": "candidate_1", "run_command": "python job.py"},
    ]

    assert reporter.ATTEMPT_STATUSES == guard.ATTEMPT_STATUSES
    for index, case in enumerate(cases):
        record = reporter.RunRecord(
            index=index,
            score=0.5,
            runtime_seconds=1.0,
            changed_files="",
            diff_summary="",
            artifacts="",
            failure_reason="",
            candidate_manifest="",
            base_candidate="",
            patch_sha256="",
            **case,
        )
        assert reporter.is_baseline(record) == guard.is_baseline(case)


@pytest.mark.parametrize("seconds", [0.0, 45.0, 3599.0, 3600.0])
def test_report_runtime_format_matches_progress_plotter(seconds):
    reporter = _load_reporter()
    plotter = _load_autofl_script("plot_progress")

    assert reporter.format_runtime(seconds) == plotter.format_runtime(seconds)
