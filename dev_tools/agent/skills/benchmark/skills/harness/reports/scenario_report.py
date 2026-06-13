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

"""Scenario-level benchmark report rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..common import write_json


def markdown_cell(value: Any) -> str:
    text = "NA" if value is None or value == "" else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def run_identity_lines(runs: Any) -> list[str]:
    if not isinstance(runs, list) or not runs:
        return []
    lines = [
        "",
        "## Run Identity",
        "",
        "| Run ID | Label | Agent | Model | Model source | Mode |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        label = run.get("label") or run.get("mode") or run.get("run_id")
        lines.append(
            f"| {markdown_cell(run.get('run_id'))} | {markdown_cell(label)} | {markdown_cell(run.get('agent'))} | "
            f"{markdown_cell(run.get('agent_model'))} | {markdown_cell(run.get('model_source'))} | "
            f"{markdown_cell(run.get('mode'))} |"
        )
    return lines


def write_scenario_report(result_root: Path, summary: Mapping[str, Any]) -> None:
    reports_dir = result_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "scenario_report.json", dict(summary))
    lines = [
        f"# Scenario Report: {summary.get('scenario_name')}",
        "",
        f"Result root: `{result_root}`",
        f"Status: `{summary.get('status')}`",
        f"Agent invocation: `{summary.get('agent_invocation') or 'live'}`",
        f"Runs: {summary.get('completed_run_count')}/{summary.get('expanded_case_count')} completed",
    ]
    replay = summary.get("replay") if isinstance(summary.get("replay"), Mapping) else {}
    if replay.get("replayed"):
        lines.extend(
            [
                "",
                "Replay: `true`",
                f"Replayed at: `{replay.get('replayed_at')}`",
                "This report was regenerated from captured artifacts; no agent or Docker run was executed.",
            ]
        )
    lines.extend(run_identity_lines(summary.get("runs")))
    lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "| Label | Runs | Quality pass | Median agent seconds | Median tokens |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    aggregates = (summary.get("aggregate_results") or {}).get("by_label") or {}
    for label, data in sorted(aggregates.items()):
        elapsed = data.get("agent_elapsed_seconds", {}).get("median")
        tokens = data.get("token_count", {}).get("median")
        lines.append(
            f"| {markdown_cell(label)} | {markdown_cell(data.get('run_count'))} | "
            f"{markdown_cell(data.get('quality_pass_count'))} | {markdown_cell(elapsed)} | {markdown_cell(tokens)} |"
        )
    winner = (summary.get("aggregate_results") or {}).get("winner")
    lines.extend(["", "## Winner Policy", "", f"`{summary.get('winner_policy')}`"])
    if winner:
        lines.append(f"\nSelected winner: `{winner.get('label')}`.")
    else:
        lines.append("\nNo winner selected because no compared label passed the quality gate with timing data.")
    (reports_dir / "scenario_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
