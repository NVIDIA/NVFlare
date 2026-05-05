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

"""Generate an autoresearch-style progress plot from Auto-FL results.tsv.

This adapts the plotting idea from karpathy/autoresearch's analysis notebook,
but uses this repo's ledger schema and higher-is-better `score` metric.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


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
        return ""
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def load_results(path: Path) -> list[ResultRow]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for index, row in enumerate(reader):
            rows.append(
                ResultRow(
                    index=index,
                    commit=row.get("commit", ""),
                    score=parse_score(row.get("score", "")),
                    runtime_seconds=parse_runtime(row.get("runtime_seconds", "")),
                    budget=row.get("budget", ""),
                    status=(row.get("status", "") or "").strip().upper(),
                    target=row.get("target", ""),
                    description=row.get("description", ""),
                    artifacts=row.get("artifacts", ""),
                )
            )
    return rows


def cumulative_max(values: list[float]) -> list[float]:
    best = -math.inf
    out = []
    for value in values:
        best = max(best, value)
        out.append(best)
    return out


def truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def compact_label(text: str, limit: int) -> str:
    text = " ".join(str(text).strip().split())
    replacements = {
        "server_momentum": "smom",
        "server_lr": "slr",
        "clip_grad_norm": "clip",
        "eta_min_factor": "eta_min",
        "label_smoothing": "ls",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    for marker in [" (", ";"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    return truncate(text, limit)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = min(max(fraction, 0.0), 1.0) * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]

    lower = ordered[lower_index]
    upper = ordered[upper_index]
    weight = position - lower_index
    return lower + (upper - lower) * weight


def select_observed_milestones(valid: list[ResultRow], max_labels: int) -> list[tuple[float, ResultRow]]:
    if max_labels <= 0:
        return []

    best = -math.inf
    milestones = []
    final_running_best = None
    for row in valid:
        if row.score is None:
            continue
        if row.score <= best:
            continue
        delta = 0.0 if best == -math.inf else row.score - best
        final_running_best = (delta, row)
        if best != -math.inf:
            milestones.append((delta, row))
        best = row.score

    if final_running_best is None:
        return []
    if not milestones:
        return [final_running_best]

    final_index = final_running_best[1].index
    non_final = [item for item in milestones if item[1].index != final_index]
    selected = sorted(non_final, key=lambda item: item[0], reverse=True)[: max_labels - 1]
    selected_indices = {row.index for _, row in selected}
    selected_indices.add(final_index)
    return [(delta, row) for delta, row in milestones if row.index in selected_indices]


def select_literature_labels(literature_rows: list[ResultRow], max_labels: int) -> list[ResultRow]:
    if max_labels <= 0 or not literature_rows:
        return []
    if len(literature_rows) <= max_labels:
        return literature_rows

    latest = literature_rows[-1]
    longest = sorted(literature_rows, key=lambda row: row.runtime_seconds, reverse=True)
    selected = {latest.index}
    for row in longest:
        if len(selected) >= max_labels:
            break
        selected.add(row.index)
    return [row for row in literature_rows if row.index in selected]


def label_placement(
    label_number: int,
    row: ResultRow,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> tuple[tuple[int, int], str, str]:
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-9)
    x_fraction = (row.index - x_min) / x_span
    y_fraction = ((row.score or y_min) - y_min) / y_span

    near_right = x_fraction > 0.72
    near_top = y_fraction > 0.78
    x_offset = -10 if near_right else 10
    y_base = -12 if near_top else 12
    y_step = (label_number % 3) * 8
    y_offset = y_base - y_step if near_top else y_base + y_step
    return (
        (x_offset, y_offset),
        ("right" if near_right else "left"),
        ("top" if near_top else "bottom"),
    )


def default_y_limits(
    scores: list[float], baseline: float, y_min: float | None, full_y_range: bool
) -> tuple[float, float]:
    score_min = min(scores)
    score_max = max(scores)

    if y_min is not None:
        lower = y_min
        span = max(score_max - lower, 0.01)
        return lower, score_max + max(0.01, span * 0.18)

    if full_y_range:
        span = max(score_max - score_min, 0.01)
        return score_min - max(0.01, span * 0.08), score_max + max(0.01, span * 0.16)

    useful_min = min(baseline, percentile(scores, 0.20))
    useful_span = max(score_max - useful_min, 0.01)
    lower = useful_min - max(0.01, useful_span * 0.20)
    upper = score_max + max(0.015, useful_span * 0.35)

    if score_min >= lower:
        full_span = max(score_max - score_min, 0.01)
        lower = score_min - max(0.01, full_span * 0.08)
    return lower, upper


def plot_progress(
    rows: list[ResultRow],
    output: Path,
    max_labels: int,
    max_literature_labels: int,
    y_min: float | None,
    full_y_range: bool,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to generate progress.png. "
            "Install it in the analysis environment and rerun this script."
        ) from exc

    valid = [row for row in rows if row.score is not None and row.status != "CRASH"]
    literature_rows = [row for row in rows if row.status == "LITERATURE"]
    if not valid:
        raise ValueError("No non-crash rows with numeric scores found.")

    baseline = valid[0].score
    best_score = max(row.score for row in valid if row.score is not None)
    best_row = max(valid, key=lambda row: row.score if row.score is not None else -math.inf)

    fig, ax = plt.subplots(figsize=(16, 8))

    styles = {
        "DISCARD": {
            "color": "#cccccc",
            "size": 18,
            "alpha": 0.55,
            "label": "Discarded",
        },
        "CANDIDATE": {
            "color": "#3498db",
            "size": 32,
            "alpha": 0.75,
            "label": "Candidate",
        },
        "KEEP": {"color": "#2ecc71", "size": 72, "alpha": 0.95, "label": "Kept"},
        "CRASH": {"color": "#e74c3c", "size": 34, "alpha": 0.75, "label": "Crash"},
    }

    for status in ["DISCARD", "CANDIDATE", "KEEP"]:
        group = [row for row in valid if row.status == status]
        if not group:
            continue
        style = styles[status]
        ax.scatter(
            [row.index for row in group],
            [row.score for row in group],
            c=style["color"],
            s=style["size"],
            alpha=style["alpha"],
            zorder=4 if status == "KEEP" else 2,
            label=style["label"],
            edgecolors="black" if status == "KEEP" else "none",
            linewidths=0.5 if status == "KEEP" else 0,
        )

    other = [row for row in valid if row.status not in {"DISCARD", "CANDIDATE", "KEEP"}]
    if other:
        ax.scatter(
            [row.index for row in other],
            [row.score for row in other],
            c="#9b59b6",
            s=28,
            alpha=0.65,
            zorder=2,
            label="Other",
        )

    observed_scores = [row.score for row in valid if row.score is not None]
    ax.step(
        [row.index for row in valid],
        cumulative_max(observed_scores),
        where="post",
        color="#27ae60",
        linewidth=2.2,
        alpha=0.75,
        zorder=3,
        label="Running best observed",
    )

    n_total = len(rows)
    n_valid = len(valid)
    n_keep = sum(row.status == "KEEP" for row in rows)
    n_candidate = sum(row.status == "CANDIDATE" for row in rows)
    n_discard = sum(row.status == "DISCARD" for row in rows)
    n_crash = sum(row.status == "CRASH" for row in rows)
    n_literature = len(literature_rows)
    runtime_rows = [row for row in rows if row.runtime_seconds > 0 and row.status != "LITERATURE"]
    literature_runtime = sum(row.runtime_seconds for row in literature_rows)
    total_runtime = sum(row.runtime_seconds for row in runtime_rows)
    average_runtime = total_runtime / len(runtime_rows) if runtime_rows else 0.0
    runtime_label = format_runtime(total_runtime)
    average_runtime_label = format_runtime(average_runtime)
    literature_runtime_label = format_runtime(literature_runtime)
    runtime_title = ""
    if runtime_label and n_literature:
        runtime_title = f", {runtime_label} candidate"
        if average_runtime_label:
            runtime_title += f", {average_runtime_label} avg/candidate"
    elif runtime_label:
        runtime_title = f", {runtime_label} total"
        if average_runtime_label:
            runtime_title += f", {average_runtime_label} avg/candidate"
    if n_literature:
        runtime_title += f", {n_literature} lit"
        if literature_runtime_label:
            runtime_title += f" ({literature_runtime_label})"

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Cross-site score (higher is better)", fontsize=12)
    ax.set_title(
        f"Auto-FL Progress: {n_total} rows, {n_valid} scored, "
        f"{n_keep} kept, {n_candidate} candidate, {n_discard} discarded, "
        f"{n_crash} crash{runtime_title}",
        fontsize=14,
    )
    ax.axhline(
        baseline,
        color="#7f8c8d",
        linewidth=1.2,
        alpha=0.5,
        linestyle="--",
        label="Baseline",
    )
    ax.grid(True, alpha=0.2)

    scores = [row.score for row in valid if row.score is not None]
    y_limits = default_y_limits(scores, baseline, y_min, full_y_range)
    ax.set_ylim(*y_limits)
    ax.set_xlim(-0.5, max(n_total - 0.5, max(row.index for row in valid) + 0.5))

    if literature_rows:
        event_color = "#8e44ad"
        y_span = max(y_limits[1] - y_limits[0], 1e-9)
        event_y = y_limits[1] - y_span * 0.035
        for row in literature_rows:
            ax.axvline(
                row.index,
                color=event_color,
                linestyle=":",
                linewidth=1.1,
                alpha=0.45,
                zorder=1,
            )
        ax.scatter(
            [row.index for row in literature_rows],
            [event_y for _ in literature_rows],
            marker="v",
            c=event_color,
            s=50,
            alpha=0.85,
            zorder=5,
            label="Literature review",
            edgecolors="white",
            linewidths=0.4,
        )
        event_x_limits = ax.get_xlim()
        event_x_span = max(event_x_limits[1] - event_x_limits[0], 1.0)
        for row in select_literature_labels(literature_rows, max_literature_labels):
            runtime = format_runtime(row.runtime_seconds)
            runtime_suffix = f" {runtime}" if runtime else ""
            near_right = (row.index - event_x_limits[0]) / event_x_span > 0.88
            annotation = ax.annotate(
                f"lit #{row.index}{runtime_suffix}: {compact_label(row.description, 30)}",
                (row.index, event_y),
                textcoords="offset points",
                xytext=((-4, -4) if near_right else (4, -4)),
                fontsize=7.0,
                color=event_color,
                alpha=0.9,
                rotation=90,
                ha=("right" if near_right else "left"),
                va="top",
                annotation_clip=True,
            )
            annotation.set_clip_on(True)

    milestone_rows = select_observed_milestones(valid, max_labels)
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    for i, (delta, row) in enumerate(milestone_rows):
        offset, horizontal_align, vertical_align = label_placement(i, row, x_limits, y_limits)
        annotation = ax.annotate(
            f"#{row.index} {row.score:.4f}: {compact_label(row.description, 28)}",
            (row.index, row.score),
            textcoords="offset points",
            xytext=offset,
            fontsize=8.0,
            color="#1a7a3a",
            alpha=0.9,
            rotation=0,
            ha=horizontal_align,
            va=vertical_align,
            annotation_clip=True,
            arrowprops={
                "arrowstyle": "-",
                "color": "#1a7a3a",
                "alpha": 0.35,
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 4,
            },
        )
        annotation.set_clip_on(True)

    summary_lines = [
        f"Baseline: {baseline:.6f}",
        f"Best: {best_score:.6f}",
        f"Delta: {best_score - baseline:+.6f}",
    ]
    if runtime_label:
        summary_lines.append(f"Candidate runtime: {runtime_label}" if n_literature else f"Runtime: {runtime_label}")
    if average_runtime_label:
        summary_lines.append(f"Avg/candidate: {average_runtime_label}")
    if n_literature:
        literature_summary = f"Lit reviews: {n_literature}"
        if literature_runtime_label:
            literature_summary += f" ({literature_runtime_label})"
        summary_lines.append(literature_summary)
    summary_lines.append(f"Best run: #{best_row.index} {truncate(best_row.description, 36)}")
    summary = "\n".join(summary_lines)
    ax.text(
        0.015,
        0.985,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "alpha": 0.9,
        },
    )
    ax.legend(loc="best", fontsize=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close(fig)
    return (
        baseline,
        best_score,
        best_row,
        total_runtime,
        average_runtime,
        len(runtime_rows),
        n_literature,
        literature_runtime,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="results.tsv", help="Path to results TSV ledger.")
    parser.add_argument("--output", default="progress.png", help="Output PNG path.")
    parser.add_argument(
        "--max-labels",
        type=int,
        default=6,
        help="Maximum running-best jumps to annotate; always includes the final running-best experiment.",
    )
    parser.add_argument(
        "--max-literature-labels",
        type=int,
        default=4,
        help="Maximum literature-review event markers to annotate; all events still draw vertical markers.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional lower y-axis bound. Defaults to dynamic score_min - margin.",
    )
    parser.add_argument(
        "--full-y-range",
        action="store_true",
        help="Use the full score range instead of the robust default that focuses on the running-best region.",
    )
    args = parser.parse_args()

    rows = load_results(Path(args.path))
    (
        baseline,
        best_score,
        best_row,
        total_runtime,
        average_runtime,
        runtime_candidate_count,
        literature_count,
        literature_runtime,
    ) = plot_progress(
        rows,
        Path(args.output),
        args.max_labels,
        args.max_literature_labels,
        args.y_min,
        args.full_y_range,
    )
    print(f"Saved {args.output}")
    print(f"baseline={baseline:.6f}")
    print(f"best={best_score:.6f}")
    print(f"delta={best_score - baseline:+.6f}")
    if total_runtime > 0:
        print(f"runtime_seconds={total_runtime:.0f}")
        print(f"runtime={format_runtime(total_runtime)}")
        print(f"avg_runtime_seconds={average_runtime:.0f}")
        print(f"avg_runtime={format_runtime(average_runtime)}")
        print(f"avg_runtime_candidates={runtime_candidate_count}")
    if literature_count:
        print(f"literature_reviews={literature_count}")
        print(f"literature_runtime_seconds={literature_runtime:.0f}")
        if literature_runtime > 0:
            print(f"literature_runtime={format_runtime(literature_runtime)}")
    print(f"best_experiment=#{best_row.index} {best_row.description}")


if __name__ == "__main__":
    sys.exit(main())
