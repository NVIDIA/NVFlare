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

"""Run-id based reports for scenarios that are not a single two-mode pair."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping

from ..common import load_json, write_json, write_text_atomic
from ..redaction import redact_text
from .scenario_report import markdown_cell


def _runs(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    runs = summary.get("runs")
    return [run for run in runs if isinstance(run, Mapping)] if isinstance(runs, list) else []


def _failed_behavior(run: Mapping[str, Any]) -> list[tuple[str, str, Mapping[str, Any]]]:
    failed = []
    for category in ("mandatory_behavior", "prohibited_behavior"):
        behavior_map = run.get(category)
        if not isinstance(behavior_map, Mapping):
            continue
        for behavior_id, evidence in behavior_map.items():
            if isinstance(evidence, Mapping) and evidence.get("status") in {"fail", "missing"}:
                failed.append((category, str(behavior_id), evidence))
    return failed


def metrics_payload(summary: Mapping[str, Any], title: str) -> dict[str, Any]:
    rows = []
    for run in _runs(summary):
        rows.append(
            {
                key: run.get(key)
                for key in (
                    "run_id",
                    "agent",
                    "agent_model",
                    "workflow",
                    "job_name",
                    "mode",
                    "status",
                    "quality_gate_passed",
                    "quality_gate_failures",
                    "agent_elapsed_seconds",
                    "token_count",
                    "command_count",
                    "eval_contract_status",
                    "expected_skill_status",
                    "required_behavior_status",
                    "failure_root_cause",
                )
            }
        )
    return {
        "schema_version": "1",
        "title": title,
        "scenario_name": summary.get("scenario_name"),
        "status": summary.get("status"),
        "run_count": len(rows),
        "runs": rows,
    }


def markdown_report(summary: Mapping[str, Any], title: str) -> str:
    lines = [
        f"# {markdown_cell(title)}",
        "",
        f"Scenario: `{markdown_cell(summary.get('scenario_name'))}`",
        f"Status: `{markdown_cell(summary.get('status'))}`",
        "",
        "| Run ID | Agent | Model | Job | Mode | Status | Quality gate | Seconds | Tokens | Root cause |",
        "|---|---|---|---|---|---|---|---:|---:|---|",
    ]
    details = []
    for run in _runs(summary):
        failures = run.get("quality_gate_failures") or []
        lines.append(
            f"| {markdown_cell(run.get('run_id'))} | {markdown_cell(run.get('agent'))} | "
            f"{markdown_cell(run.get('agent_model'))} | {markdown_cell(run.get('job_name'))} | "
            f"{markdown_cell(run.get('mode'))} | {markdown_cell(run.get('status'))} | "
            f"{markdown_cell('pass' if run.get('quality_gate_passed') else ', '.join(map(str, failures)))} | "
            f"{markdown_cell(run.get('agent_elapsed_seconds'))} | {markdown_cell(run.get('token_count'))} | "
            f"{markdown_cell(run.get('failure_root_cause') or 'NA')} |"
        )
        failed = _failed_behavior(run)
        if failed:
            details.extend(["", f"## {markdown_cell(run.get('run_id'))} failed behavior", ""])
            for category, behavior_id, evidence in failed:
                details.append(
                    f"- {markdown_cell(category)} `{markdown_cell(behavior_id)}`: "
                    f"{markdown_cell(evidence.get('status'))} — {markdown_cell(evidence.get('evidence'))}"
                )
    return "\n".join([*lines, *details]) + "\n"


def html_report(summary: Mapping[str, Any], title: str) -> str:
    def escaped(value: Any) -> str:
        return html.escape(redact_text("NA" if value is None or value == "" else value))

    rows = []
    for run in _runs(summary):
        gate = "pass" if run.get("quality_gate_passed") else ", ".join(map(str, run.get("quality_gate_failures") or []))
        cells = [
            run.get("run_id"),
            run.get("agent"),
            run.get("agent_model"),
            run.get("job_name"),
            run.get("mode"),
            run.get("status"),
            gate,
            run.get("agent_elapsed_seconds"),
            run.get("token_count"),
            run.get("failure_root_cause") or "NA",
        ]
        rows.append("<tr>" + "".join(f"<td>{escaped(cell)}</td>" for cell in cells) + "</tr>")
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>"
        + escaped(title)
        + "</title></head><body><h1>"
        + escaped(title)
        + "</h1><table><thead><tr>"
        + "".join(
            f"<th>{heading}</th>"
            for heading in (
                "Run ID",
                "Agent",
                "Model",
                "Job",
                "Mode",
                "Status",
                "Quality gate",
                "Seconds",
                "Tokens",
                "Root cause",
            )
        )
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>\n"
    )


def write_reports(root: Path, title: str) -> None:
    summary = load_json(root / "scenario_summary.json", {}) or {}
    if not isinstance(summary, dict):
        raise ValueError("scenario_summary.json must contain an object")
    payload = metrics_payload(summary, title)
    write_json(root / "metrics_report.json", payload)
    write_text_atomic(root / "metrics_report.md", markdown_report(summary, title))
    write_text_atomic(root / "metrics_report.html", html_report(summary, title))
    write_text_atomic(root / "benchmark_insights.md", markdown_report(summary, "Benchmark Insights and RCA"))
