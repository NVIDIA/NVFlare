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

"""Generate record-driven benchmark metrics reports."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

from ..common import flatten_numbers, load_json, write_json
from ..modes import BENCHMARK_RUNS
from .benchmark_insights import (
    collect_benchmark_runs,
    embedded_bar_chart,
    final_record_path,
    human_readable_status,
    markdown_cell,
    metric_name_for_runs,
    mode_dir_for_benchmark,
    outcome_metrics_table,
    run_analysis,
    status_summary,
)


def collect_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for spec in BENCHMARK_RUNS:
        mode_dir = mode_dir_for_benchmark(root, spec.mode)
        summary = load_json(mode_dir / "run_summary.json", {}) if mode_dir.exists() else {}
        record = load_json(final_record_path(root, spec.mode), {}) if mode_dir.exists() else {}
        activity = load_json(mode_dir / "agent_activity.json", {}) if mode_dir.exists() else {}
        usage = load_json(mode_dir / "agent_usage.json", {}) if mode_dir.exists() else {}
        runtime_image = load_json(mode_dir / "runtime_image.json", {}) if mode_dir.exists() else {}
        if not isinstance(summary, dict):
            summary = {}
        if not isinstance(record, dict):
            record = {}
        if not isinstance(activity, dict):
            activity = {}
        if not isinstance(usage, dict):
            usage = {}
        if not isinstance(runtime_image, dict):
            runtime_image = {}
        runs.append(
            {
                "mode": spec.mode,
                "label": spec.label,
                "available": mode_dir.exists(),
                "skills_enabled": spec.skills_enabled,
                "summary": summary,
                "record": record,
                "activity": activity,
                "usage": usage,
                "runtime_image": runtime_image,
                "metrics": flatten_numbers(
                    {"summary": summary, "record": record, "activity": activity, "usage": usage}
                ),
            }
        )
    return runs


def runs_by_mode_for_insights(root: Path, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    collected = collect_benchmark_runs(root)
    for row in rows:
        run = collected.get(row["mode"])
        if isinstance(run, dict):
            run["label"] = row["label"]
    return collected


def numeric_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2:
        return {}
    without, with_skills = rows
    result: dict[str, Any] = {}
    for key in ("elapsed_seconds", "token_count"):
        left = without["summary"].get(key)
        right = with_skills["summary"].get(key)
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            result[f"{key}_with_skills_minus_without_skills"] = right - left
    return result


def report_summary(
    root: Path,
    title: str,
    rows: list[dict[str, Any]] | None = None,
    insight_runs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = collect_runs(root) if rows is None else rows
    insight_runs = runs_by_mode_for_insights(root, rows) if insight_runs is None else insight_runs
    metric_name = metric_name_for_runs(insight_runs)
    return {
        "title": title,
        "result_root": str(root),
        "status": status_summary(insight_runs),
        "metric_name": metric_name,
        "runs": rows,
        "comparison": numeric_comparison(rows),
    }


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_report(summary: dict[str, Any], insight_runs: dict[str, dict[str, Any]] | None = None) -> str:
    root = Path(summary["result_root"])
    insight_runs = runs_by_mode_for_insights(root, summary["runs"]) if insight_runs is None else insight_runs
    lines = [
        f"# {summary['title']}",
        "",
        f"Result root: `{summary['result_root']}`",
        "",
        f"Status: {summary['status']}",
        "",
        "## Runs",
        "",
        "| Run | Status | Skills | Elapsed seconds | Tokens | Commands | Root cause |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary["runs"]:
        run = insight_runs[row["mode"]]
        run_summary = row["summary"]
        activity = row["activity"]
        root_cause = "NA" if human_readable_status(run) == "passed" else run_analysis(run)
        lines.append(
            f"| {markdown_cell(row['label'])} | {markdown_cell(human_readable_status(run))} | "
            f"{fmt(row['skills_enabled'])} | {fmt(run_summary.get('elapsed_seconds'))} | "
            f"{fmt(run_summary.get('token_count'))} | {fmt(activity.get('command_count'))} | "
            f"{markdown_cell(root_cause)} |"
        )
    lines.extend(["", "## Metrics", "", embedded_bar_chart(insight_runs), "", outcome_metrics_table(insight_runs)])
    comparison = summary.get("comparison") or {}
    if comparison:
        lines.extend(["", "## Comparison", "", "| Metric | Delta |", "|---|---:|"])
        for key, value in sorted(comparison.items()):
            lines.append(f"| {markdown_cell(key)} | {fmt(value)} |")
    return "\n".join(lines) + "\n"


def html_report(summary: dict[str, Any], insight_runs: dict[str, dict[str, Any]] | None = None) -> str:
    root = Path(summary["result_root"])
    insight_runs = runs_by_mode_for_insights(root, summary["runs"]) if insight_runs is None else insight_runs
    rows = []
    for row in summary["runs"]:
        run = insight_runs[row["mode"]]
        run_summary = row["summary"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{html.escape(human_readable_status(run))}</td>"
            f"<td>{html.escape(fmt(row['skills_enabled']))}</td>"
            f"<td>{html.escape(fmt(run_summary.get('elapsed_seconds')))}</td>"
            f"<td>{html.escape(fmt(run_summary.get('token_count')))}</td>"
            "</tr>"
        )
    chart = embedded_bar_chart(insight_runs)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(summary['title'])}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
    th {{ background: #f5f7fa; }}
    .bar-row {{ display: grid; grid-template-columns: 180px 1fr 80px; gap: 12px; align-items: center; margin: 8px 0; }}
    .bar-track {{ background: #e4e7eb; height: 14px; position: relative; }}
    .bar-fill {{ display: block; background: #2f80ed; height: 14px; }}
  </style>
</head>
<body>
  <h1>{html.escape(summary['title'])}</h1>
  <p>Result root: <code>{html.escape(summary['result_root'])}</code></p>
  <p>Status: {html.escape(summary['status'])}</p>
  <table>
    <thead><tr><th>Run</th><th>Status</th><th>Skills</th><th>Elapsed seconds</th><th>Tokens</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {chart}
</body>
</html>
"""


def write_reports(root: Path, title: str) -> dict[str, Any]:
    rows = collect_runs(root)
    insight_runs = runs_by_mode_for_insights(root, rows)
    summary = report_summary(root, title, rows, insight_runs)
    write_json(root / "metrics_report.json", summary)
    (root / "metrics_report.md").write_text(markdown_report(summary, insight_runs), encoding="utf-8")
    (root / "metrics_report.html").write_text(html_report(summary, insight_runs), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--title", default="Agent Skills Benchmark Metrics")
    parser.add_argument(
        "--plots", action="store_true", help="accepted for compatibility; HTML report is always written"
    )
    args = parser.parse_args()
    write_reports(args.root, args.title)
    print(args.root / "metrics_report.html")


if __name__ == "__main__":
    main()
