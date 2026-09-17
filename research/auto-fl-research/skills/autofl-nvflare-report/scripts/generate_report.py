#!/usr/bin/env python3
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

"""Generate a stopped-campaign Auto-FL NVFlare markdown report."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

SOURCE_RE = re.compile(r"\[src:\s*([^\]]+)\]")


@dataclass
class ResultRow:
    index: int
    commit: str
    score: float | None
    runtime_seconds: float
    budget: str
    status: str
    target: str
    description: str
    artifacts: str


def parse_score(value: str) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    return score


def parse_runtime(value: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(seconds) or math.isinf(seconds):
        return 0.0
    return max(0.0, seconds)


def format_runtime(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def pct(value: float) -> str:
    return f"{value:.1f}%"


def load_results(path: Path) -> list[ResultRow]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [
            ResultRow(
                index=i,
                commit=row.get("commit", ""),
                score=parse_score(row.get("score", "")),
                runtime_seconds=parse_runtime(row.get("runtime_seconds", "")),
                budget=row.get("budget", ""),
                status=(row.get("status", "") or "").strip().lower(),
                target=row.get("target", ""),
                description=row.get("description", ""),
                artifacts=row.get("artifacts", ""),
            )
            for i, row in enumerate(reader)
        ]


def git_value(args: list[str], fallback: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip() or fallback
    except Exception:
        return fallback


def default_output_path() -> Path:
    branch = git_value(["branch", "--show-current"], "autofl-campaign")
    safe_branch = re.sub(r"[^A-Za-z0-9._-]+", "-", branch).strip("-") or "autofl-campaign"
    return Path("reports") / f"{safe_branch}-autoresearch-report.md"


def find_baseline(rows: list[ResultRow]) -> ResultRow:
    scored = [row for row in rows if row.score is not None]
    if not scored:
        raise ValueError("No scored rows found in results.tsv")
    for row in scored:
        if "baseline" in row.description.lower():
            return row
    return scored[0]


def running_best_milestones(rows: list[ResultRow], max_rows: int) -> list[tuple[float, ResultRow]]:
    best = -math.inf
    milestones: list[tuple[float, ResultRow]] = []
    for row in rows:
        if row.score is None:
            continue
        if row.score > best:
            delta = row.score - best if best > -math.inf else 0.0
            milestones.append((delta, row))
            best = row.score
    if len(milestones) <= max_rows:
        return milestones
    first = milestones[0]
    last = milestones[-1]
    middle = sorted(milestones[1:-1], key=lambda item: item[0], reverse=True)
    selected = [first, *middle[: max_rows - 2], last]
    return sorted(selected, key=lambda item: item[1].index)


def source_refs(description: str) -> list[str]:
    return [match.strip() for match in SOURCE_RE.findall(description or "")]


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |" for row in rows)
    return "\n".join(out)


def truncate(text: str, limit: int = 96) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def read_agent_context(literal: str | None, file_path: Path | None) -> str:
    parts = []
    if literal:
        parts.append(literal.strip())
    if file_path:
        try:
            text = file_path.read_text(encoding="utf-8").strip()
        except OSError as e:
            text = f"Unable to read {file_path}: {e}"
        if text:
            parts.append(text)
    return "\n\n".join(part for part in parts if part)


def context_summary(text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return "not provided"
    return truncate(first_line, 120)


def markdown_path(path: Path, output_path: Path) -> str:
    if path.is_absolute():
        return path.as_posix()
    output_parent = output_path.parent if output_path.parent != Path("") else Path(".")
    return Path(os.path.relpath(path, output_parent)).as_posix()


def classify_contract(row: ResultRow) -> str:
    budget = row.budget.lower()
    if "--aggregator scaffold" in budget:
        return "Explicit SCAFFOLD metadata mode"
    return "Strict DIFF contract"


def mechanism_hint(row: ResultRow) -> str:
    text = f"{row.description} {row.budget}".lower()
    hints = []
    if "fedavgm" in text or "server_momentum" in text:
        hints.append("server momentum")
    if "server_lr" in text or "lr_s" in text:
        hints.append("server diff amplification")
    if "fedprox" in text or "prox" in text:
        hints.append("FedProx/client drift regularization")
    if "scaffold" in text:
        hints.append("SCAFFOLD control variates")
    if "warmup" in text or "warmup_epochs" in text or " w=" in text:
        hints.append("client LR warmup retuning")
    if "clip" in text:
        hints.append("gradient clipping")
    if "label_smoothing" in text or " ls " in text:
        hints.append("label smoothing")
    if "weight_decay" in text or " wd" in text:
        hints.append("weight decay")
    return ", ".join(dict.fromkeys(hints)) or "configuration change"


def generate_report(
    results_path: Path,
    output_path: Path,
    plot_path: Path | None,
    max_milestones: int,
    agent_cost: str,
    agent_settings: str,
) -> None:
    rows = load_results(results_path)
    if not rows:
        raise ValueError(f"No rows found in {results_path}")
    scored = [row for row in rows if row.score is not None]
    baseline = find_baseline(rows)
    best = max(scored, key=lambda row: row.score if row.score is not None else -math.inf)
    baseline_score = baseline.score or 0.0
    best_score = best.score or 0.0
    absolute_lift = best_score - baseline_score
    relative_lift = absolute_lift / baseline_score if baseline_score else 0.0
    runtime_rows = [row for row in rows if row.runtime_seconds > 0]
    total_runtime = sum(row.runtime_seconds for row in rows)
    average_runtime = total_runtime / len(runtime_rows) if runtime_rows else 0.0
    status_counts = Counter(row.status or "unknown" for row in rows)
    crash_rows = [row for row in rows if row.status == "crash"]
    candidate_rows = [row for row in rows if row.status == "candidate"]
    kept_rows = [row for row in rows if row.status == "keep"]
    milestones = running_best_milestones(rows, max_milestones)

    source_groups: dict[str, list[ResultRow]] = defaultdict(list)
    for row in rows:
        for ref in source_refs(row.description):
            source_groups[ref].append(row)

    branch = git_value(["branch", "--show-current"], "unknown")
    head = git_value(["rev-parse", "--short", "HEAD"], "unknown")

    lines = [
        "# Auto-FL NVFlare Autoresearch Campaign Report",
        "",
        "## Executive Summary",
        "",
        f"- **Branch:** `{branch}` at `{head}`",
        f"- **Rows analyzed:** {len(rows)} total, {len(scored)} scored",
        f"- **Best score:** {best_score:.6f} at experiment `#{best.index}`",
        f"- **Baseline:** {baseline_score:.6f} at experiment `#{baseline.index}`",
        f"- **Lift:** {absolute_lift:+.6f} absolute, {pct(relative_lift * 100)} relative",
        f"- **Runtime cost:** {format_runtime(total_runtime)} aggregate; {format_runtime(average_runtime)} average over {len(runtime_rows)} timed candidates",
        f"- **Agent model/effort:** {context_summary(agent_settings)}",
        f"- **Agent/tooling cost:** {context_summary(agent_cost)}",
        f"- **Best status:** `{best.status or 'unknown'}`; treat as needing reproduction unless independently repeated or marked `keep`.",
        "",
    ]
    if plot_path:
        plot_markdown_path = markdown_path(plot_path, output_path)
        lines.extend([f"- **Progress plot:** `{plot_path}`", ""])
        lines.extend(
            [
                "## Progress Plot",
                "",
                f"![Auto-FL progress]({plot_markdown_path})",
                "",
            ]
        )

    lines.extend(
        [
            "## Best Candidate",
            "",
            table(
                ["field", "value"],
                [
                    ["experiment", f"#{best.index}"],
                    ["score", f"{best_score:.6f}"],
                    ["delta vs baseline", f"{absolute_lift:+.6f}"],
                    ["relative lift", pct(relative_lift * 100)],
                    ["status", best.status or "unknown"],
                    ["commit", best.commit],
                    ["runtime", format_runtime(best.runtime_seconds)],
                    ["target", best.target],
                    ["description", best.description],
                    ["artifact", best.artifacts],
                    ["contract mode", classify_contract(best)],
                ],
            ),
            "",
            "### Exact Budget / Args",
            "",
            "```text",
            best.budget,
            "```",
            "",
        ]
    )

    milestone_rows = []
    for delta, row in milestones:
        score_text = f"{row.score:.6f}" if row.score is not None else ""
        milestone_rows.append(
            [
                f"#{row.index}",
                score_text,
                f"{delta:+.6f}" if delta else "baseline",
                truncate(row.description, 72),
                mechanism_hint(row),
                "; ".join(source_refs(row.description)) or "",
            ]
        )
    lines.extend(
        [
            "## Improvement Path",
            "",
            "Major running-best milestones, selected by first/last and largest score jumps:",
            "",
            table(["experiment", "score", "jump", "description", "likely mechanism", "source refs"], milestone_rows),
            "",
        ]
    )

    lines.extend(
        [
            "## Runtime and Reliability",
            "",
            table(
                ["metric", "value"],
                [
                    ["total aggregate runtime", format_runtime(total_runtime)],
                    ["average runtime per timed candidate", format_runtime(average_runtime)],
                    ["timed candidates", str(len(runtime_rows))],
                    ["candidate rows", str(len(candidate_rows))],
                    ["kept rows", str(len(kept_rows))],
                    ["crash rows", str(len(crash_rows))],
                ],
            ),
            "",
            "The runtime total is aggregate candidate runtime from `runtime_seconds`, not wall-clock elapsed campaign time.",
            "",
        ]
    )

    lines.extend(["## Agent / Tooling Context", "", "### Model / Effort Settings", ""])
    if agent_settings:
        lines.extend(
            [
                "Agent model and effort context provided for this report:",
                "",
                "```text",
                agent_settings,
                "```",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No agent model or effort settings were provided. If the runtime exposes interactive `/model` and `/effort` commands, run them manually and paste the output into the reporting prompt.",
                "",
            ]
        )

    lines.extend(["### Agent / Tooling Cost", ""])
    if agent_cost:
        lines.extend(
            [
                "Agent/tooling cost context provided for this report:",
                "",
                "```text",
                agent_cost,
                "```",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No agent/tooling cost telemetry was provided. The report still includes experiment runtime cost from `results.tsv`.",
                "",
            ]
        )

    if crash_rows:
        crash_summary = Counter(truncate(row.description, 64) for row in crash_rows).most_common(8)
        lines.extend(
            [
                "### Crash / Failure Notes",
                "",
                table(["count", "description"], [[str(count), desc] for desc, count in crash_summary]),
                "",
            ]
        )

    lines.extend(["## Literature-Derived Ideas", ""])
    if source_groups:
        literature_rows = []
        for ref, ref_rows in sorted(source_groups.items()):
            best_ref = max(ref_rows, key=lambda row: row.score if row.score is not None else -math.inf)
            literature_rows.append(
                [
                    ref,
                    str(len(ref_rows)),
                    f"{best_ref.score:.6f}" if best_ref.score is not None else "",
                    truncate(best_ref.description, 72),
                    mechanism_hint(best_ref),
                    "helped" if (best_ref.score or 0.0) > baseline_score else "not confirmed",
                ]
            )
        lines.extend(
            [
                table(
                    ["source ref", "rows", "best score", "best description", "mapped mechanism", "outcome"],
                    literature_rows,
                ),
                "",
                "Source refs are extracted from `[src: ...]` markers in `results.tsv` descriptions. Check `templates/mutation_report.md` or the campaign notes for full citations and URLs.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No `[src: ...]` markers were found in `results.tsv`. Treat this as a gap: paper-derived methods should include compact source refs in the ledger and fuller citations in the campaign notes.",
                "",
            ]
        )

    below_baseline = [row for row in scored if (row.score or 0.0) < baseline_score]
    low_rows = sorted(below_baseline, key=lambda row: row.score or 0.0)[:8]
    lines.extend(["## Null, Worse, or Unstable Ideas", ""])
    if low_rows:
        lines.extend(
            [
                table(
                    ["experiment", "score", "description", "mechanism"],
                    [
                        [
                            f"#{row.index}",
                            f"{row.score:.6f}" if row.score is not None else "",
                            truncate(row.description, 72),
                            mechanism_hint(row),
                        ]
                        for row in low_rows
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(["No scored rows fell below the baseline.", ""])

    lines.extend(
        [
            "## Recommendation",
            "",
            "1. Reproduce the best candidate with multiple seeds or repeated runs before promotion.",
            "2. Promote only changes that preserve the declared contract mode, or keep explicit protocol modes such as SCAFFOLD labeled separately.",
            "3. Use the milestone table to focus follow-up sweeps on mechanisms that created durable running-best jumps.",
            "4. Retire ideas that repeatedly crash, underperform the baseline, or add complexity without a repeatable score lift.",
            "",
            "## Technical Appendix",
            "",
            "### Status Counts",
            "",
            table(["status", "rows"], [[status, str(count)] for status, count in sorted(status_counts.items())]),
            "",
            "### Top Scored Rows",
            "",
            table(
                ["rank", "experiment", "score", "runtime", "status", "description"],
                [
                    [
                        str(rank),
                        f"#{row.index}",
                        f"{row.score:.6f}" if row.score is not None else "",
                        format_runtime(row.runtime_seconds),
                        row.status,
                        truncate(row.description, 88),
                    ]
                    for rank, row in enumerate(
                        sorted(scored, key=lambda row: row.score or -math.inf, reverse=True)[:10],
                        start=1,
                    )
                ],
            ),
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.tsv", help="Path to results.tsv")
    parser.add_argument("--output", default=None, help="Output markdown path")
    parser.add_argument("--plot", default=None, help="Progress plot path to reference")
    parser.add_argument("--max-milestones", type=int, default=10)
    parser.add_argument("--agent-cost", default=None, help="Literal agent/tooling cost summary text")
    parser.add_argument(
        "--agent-cost-file", default=None, help="Path to a text file containing agent/tooling cost output"
    )
    parser.add_argument("--agent-settings", default=None, help="Literal agent model/effort settings text")
    parser.add_argument(
        "--agent-settings-file",
        default=None,
        help="Path to a text file containing agent model/effort settings output",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    output_path = Path(args.output) if args.output else default_output_path()
    plot_path = Path(args.plot) if args.plot else None
    agent_cost_file = Path(args.agent_cost_file) if args.agent_cost_file else None
    agent_settings_file = Path(args.agent_settings_file) if args.agent_settings_file else None
    agent_cost = read_agent_context(args.agent_cost, agent_cost_file)
    agent_settings = read_agent_context(args.agent_settings, agent_settings_file)
    generate_report(results_path, output_path, plot_path, args.max_milestones, agent_cost, agent_settings)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
