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
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_plotter():
    repo_root = Path(__file__).parents[3]
    script_path = repo_root / "skills" / "nvflare-autofl" / "scripts" / "plot_progress.py"
    spec = importlib.util.spec_from_file_location("nvflare_autofl_skill_plot_progress", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(plotter, index, score, status="discard", name=None, runtime=300.0, kind="", family="", lit_id=""):
    return plotter.ProgressRecord(
        index=index,
        status=status,
        name=name or f"candidate_{index}",
        score=score,
        runtime_seconds=runtime,
        description=f"candidate {index}",
        kind=kind,
        family=family,
        literature_event_id=lit_id,
    )


def test_robust_y_limits_focus_on_improvement_region():
    plotter = _load_plotter()
    scores = [0.50, 0.55, 0.687] + [0.70 + index * 0.002 for index in range(20)]

    lower, upper = plotter.default_y_limits(scores, baseline=0.687)

    assert lower > 0.60
    assert lower < 0.687
    assert upper > max(scores)


def test_milestone_selection_tracks_the_running_maximum():
    plotter = _load_plotter()
    scores = [0.5, 0.6, 0.55, 0.7, 0.69, 0.8]
    records = [
        _record(plotter, index, score, status="baseline" if index == 0 else "keep")
        for index, score in enumerate(scores)
    ]

    milestones = plotter.select_observed_milestones(records, max_labels=3)

    assert len(milestones) <= 3
    assert milestones[-1][1].score == 0.8


def test_load_results_uses_productized_ledger_fields(tmp_path):
    plotter = _load_plotter()
    ledger = tmp_path / "results.tsv"
    with ledger.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["status", "name", "score", "runtime_seconds", "diff_summary"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "status": "keep",
                "name": "fedavgm",
                "score": "0.75",
                "runtime_seconds": "123.5",
                "diff_summary": "server momentum",
            }
        )

    records = plotter.load_results(ledger)

    # Legacy ledger has no candidate_kind/changed_files columns, so kind falls back to "argument_only".
    assert records == [plotter.ProgressRecord(0, "keep", "fedavgm", 0.75, 123.5, "server momentum", "argument_only")]
    assert plotter.normalize_records(records) == records


def test_load_results_populates_candidate_kind_family_and_literature_link(tmp_path):
    plotter = _load_plotter()
    ledger = tmp_path / "results.tsv"
    fieldnames = [
        "status",
        "name",
        "score",
        "runtime_seconds",
        "diff_summary",
        "changed_files",
        "candidate_kind",
        "algorithm_family",
        "literature_event_id",
    ]
    with ledger.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow({"status": "baseline", "name": "baseline", "score": "0.687", "runtime_seconds": "100"})
        writer.writerow(
            {
                "status": "keep",
                "name": "fedprox_mu",
                "score": "0.75",
                "runtime_seconds": "120",
                "changed_files": "client.py",
                "candidate_kind": "source_edit",
                "algorithm_family": "fedprox",
                "literature_event_id": "lit-0001",
            }
        )
        writer.writerow(
            {
                "status": "discard",
                "name": "higher_lr",
                "score": "0.70",
                "runtime_seconds": "110",
                "candidate_kind": "argument_only",
            }
        )

    records = plotter.load_results(ledger)

    assert [record.kind for record in records] == ["", "source_edit", "argument_only"]
    assert [record.family for record in records] == ["", "fedprox", ""]
    assert [record.literature_event_id for record in records] == ["", "lit-0001", ""]


def test_normalize_records_reads_run_record_attributes_with_changed_files_fallback():
    plotter = _load_plotter()
    runs = [
        SimpleNamespace(
            status="keep",
            name="scaffold_variant",
            score=0.76,
            runtime_seconds=100.0,
            diff_summary="control variates",
            changed_files="client.py",
            candidate_kind="source_edit",
            algorithm_family="scaffold",
            literature_event_id="lit-0002",
        ),
        # Legacy run records without the new columns fall back to changed_files.
        SimpleNamespace(
            status="discard",
            name="legacy_edit",
            score=0.70,
            runtime_seconds=90.0,
            diff_summary="edited trainer",
            changed_files="trainer.py",
        ),
        SimpleNamespace(
            status="discard",
            name="legacy_args",
            score=0.69,
            runtime_seconds=80.0,
            diff_summary="tuned lr",
            changed_files="none",
        ),
    ]

    records = plotter.normalize_records(runs)

    assert [record.kind for record in records] == ["source_edit", "source_edit", "argument_only"]
    assert records[0].family == "scaffold"
    assert records[0].literature_event_id == "lit-0002"
    assert [record.family for record in records[1:]] == ["", ""]
    assert [record.literature_event_id for record in records[1:]] == ["", ""]


def test_zero_score_label_uses_the_actual_score():
    plotter = _load_plotter()
    record = _record(plotter, 1, 0.0)

    offset, _, vertical_alignment = plotter.label_placement(0, record, (0.0, 2.0), (-1.0, 0.1))

    assert offset[1] < 0
    assert vertical_alignment == "top"


def test_rich_progress_plot_renders_png(tmp_path):
    pytest.importorskip("matplotlib")
    plotter = _load_plotter()
    records = [_record(plotter, 0, 0.687, status="baseline", name="baseline")]
    for index in range(1, 21):
        status = "keep" if index in {3, 8, 15} else "discard"
        records.append(_record(plotter, index, 0.69 + index * 0.002, status=status))
    records.insert(10, _record(plotter, 10, None, status="literature", name="literature_review_1", runtime=45.0))
    output = tmp_path / "progress.png"

    baseline, best = plotter.plot_progress(records, output, metric_label="test_accuracy")

    assert baseline == pytest.approx(0.687)
    assert best == pytest.approx(0.73)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 20_000


def test_progress_plot_distinguishes_candidate_kinds(tmp_path):
    pytest.importorskip("matplotlib")
    plotter = _load_plotter()
    records = [
        _record(plotter, 0, 0.687, status="baseline", name="baseline"),
        _record(plotter, 1, None, status="literature", name="lit_review", runtime=45.0, lit_id="lit-0001"),
        _record(plotter, 2, 0.70, status="discard", kind="argument_only"),
        _record(plotter, 3, 0.71, status="discard", kind="source_edit"),
        _record(plotter, 4, 0.72, status="candidate", kind="argument_only"),
        _record(plotter, 5, 0.75, status="keep", kind="source_edit", family="fedprox", lit_id="lit-0001"),
    ]
    output = tmp_path / "progress.png"

    baseline, best = plotter.plot_progress(records, output, metric_label="test_accuracy")

    assert baseline == pytest.approx(0.687)
    assert best == pytest.approx(0.75)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 20_000


def test_plot_cli_rejects_minimization_mode(tmp_path, capsys):
    plotter = _load_plotter()
    ledger = tmp_path / "results.tsv"
    ledger.write_text("status\tname\tscore\nbaseline\tbaseline\t0.5\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        plotter.main([str(ledger), "--output", str(tmp_path / "progress.png"), "--mode", "min"])

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "minimization is not supported" in stderr
    assert "neg_val_loss" in stderr
