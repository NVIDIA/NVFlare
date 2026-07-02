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

import pytest


def _load_plotter():
    repo_root = Path(__file__).parents[3]
    script_path = repo_root / "skills" / "nvflare-autofl" / "scripts" / "plot_progress.py"
    spec = importlib.util.spec_from_file_location("nvflare_autofl_skill_plot_progress", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(plotter, index, score, status="discard", name=None, runtime=300.0):
    return plotter.ProgressRecord(
        index=index,
        status=status,
        name=name or f"candidate_{index}",
        score=score,
        runtime_seconds=runtime,
        description=f"candidate {index}",
    )


def test_robust_y_limits_focus_on_improvement_region_for_maximize():
    plotter = _load_plotter()
    scores = [0.50, 0.55, 0.687] + [0.70 + index * 0.002 for index in range(20)]

    lower, upper = plotter.default_y_limits(scores, baseline=0.687, mode="max")

    assert lower > 0.60
    assert lower < 0.687
    assert upper > max(scores)


def test_robust_y_limits_focus_on_improvement_region_for_minimize():
    plotter = _load_plotter()
    scores = [2.0, 1.8, 0.90] + [0.80 - index * 0.01 for index in range(20)]

    lower, upper = plotter.default_y_limits(scores, baseline=0.90, mode="min")

    assert lower < min(scores)
    assert upper > 0.90
    assert upper < 1.5


@pytest.mark.parametrize(
    "mode,scores,expected_best",
    [
        ("max", [0.5, 0.6, 0.55, 0.7, 0.69, 0.8], 0.8),
        ("min", [0.8, 0.7, 0.75, 0.6, 0.62, 0.5], 0.5),
    ],
)
def test_milestone_selection_supports_both_objective_directions(mode, scores, expected_best):
    plotter = _load_plotter()
    records = [
        _record(plotter, index, score, status="baseline" if index == 0 else "keep")
        for index, score in enumerate(scores)
    ]

    milestones = plotter.select_observed_milestones(records, mode=mode, max_labels=3)

    assert len(milestones) <= 3
    assert milestones[-1][1].score == expected_best


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

    assert records == [plotter.ProgressRecord(0, "keep", "fedavgm", 0.75, 123.5, "server momentum")]
    assert plotter.normalize_records(records) == records


def test_rich_progress_plot_renders_png(tmp_path):
    pytest.importorskip("matplotlib")
    plotter = _load_plotter()
    records = [_record(plotter, 0, 0.687, status="baseline", name="baseline")]
    for index in range(1, 21):
        status = "keep" if index in {3, 8, 15} else "discard"
        records.append(_record(plotter, index, 0.69 + index * 0.002, status=status))
    records.insert(10, _record(plotter, 10, None, status="literature", name="literature_review_1", runtime=45.0))
    output = tmp_path / "progress.png"

    baseline, best = plotter.plot_progress(records, output, mode="max", metric_label="test_accuracy")

    assert baseline == pytest.approx(0.687)
    assert best == pytest.approx(0.73)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 20_000
