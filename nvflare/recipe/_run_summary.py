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

"""Presentation of existing result artifacts; never determines job success."""

import json
from itertools import islice
from pathlib import Path

from nvflare.fuel.utils.log_utils import _read_log_tail, format_metric_table, wrap_log_message


def summary_header(outcome, elapsed, *, context=""):
    """Use the same heading and elapsed-time alignment for every run outcome."""
    heading = "\n" + " RUN SUMMARY ".center(72, "=")
    return heading + (f"\n\n{context}" if context else "") + f"\n\n  {outcome}".ljust(65) + f"{elapsed:.1f}s"


def run_context(job_name, env):
    """Describe known Recipe context without guessing production participation."""
    name = type(env).__name__
    label = {"SimEnv": "Simulation", "PocEnv": "POC", "ProdEnv": "Production"}.get(name, name)
    count = getattr(env, "num_clients", None)
    # A supplied POC project overrides constructor client counts.
    if type(count) is int and count > 0 and not getattr(env, "project_conf_path", None):
        label += f" · {count} client{'s' if count != 1 else ''}"
    title = "NVIDIA FLARE" + (f" · {_text(job_name)}" if job_name else "")
    return wrap_log_message(f"  {title}\n  {label}")


def _read_json(path):
    # Reporting must not load an arbitrarily large application artifact.
    with path.open("rb") as stream:
        data = stream.read(1048577)
    if len(data) > 1048576:
        return None
    return json.loads(data)


def _text(value):
    return json.dumps(str(value)[:160], ensure_ascii=True)[1:-1]


def result_summary(result):
    """Return a bounded report for standard simulator/downloaded result layouts.

    Missing, custom, oversized, or malformed artifacts are left for inspection in
    the result directory. No model files are deserialized.
    """
    root = Path(result)
    lines = []
    artifacts = []
    # Simulator result root, downloaded admin transfer root, or server run root.
    candidates = [root / "workspace", root]
    candidates.extend(islice(root.glob("server/*"), 20))
    for run_dir in candidates:
        metrics = run_dir / "metrics"
        summary_path = metrics / "metrics_summary.json"
        if summary_path.is_file():
            try:
                round_path = metrics / "round_metrics.jsonl"
                with round_path.open("rb") as stream:
                    data, truncated = _read_log_tail(stream, 1048576, whole_lines=True)
                records = data.splitlines()[-11:]
                truncated = truncated or len(records) > 10
                records = records[-10:]
                rows = []
                for raw in records:
                    try:
                        record = json.loads(raw)
                        values = {m["name"]: m["value"] for m in record["aggregated_metrics"]}
                        round_index = record["round"]
                        label = str(round_index + 1) if type(round_index) is int else "?"
                        rows.append((label, values))
                    except (ValueError, TypeError, KeyError):
                        continue
                if rows:
                    heading = "  Training · aggregated client metrics"
                    lines.extend(["", heading + (" (last 10 rounds)" if truncated else ""), ""])
                    lines.append(format_metric_table(rows, label="Round"))
            except (OSError, ValueError, TypeError, KeyError):
                lines.append("Training details: see the saved metrics artifacts.")
            artifacts.append(f"  Metrics   {summary_path.parent.relative_to(root)}/")
        evaluation_path = run_dir / "cross_site_val" / "cross_val_results.json"
        if evaluation_path.is_file():
            try:
                evaluations = _read_json(evaluation_path)
                if isinstance(evaluations, dict) and evaluations:
                    metric_names = list(
                        dict.fromkeys(
                            key
                            for models in evaluations.values()
                            if isinstance(models, dict)
                            for values in models.values()
                            if isinstance(values, dict)
                            for key in values
                        )
                    )
                    for metric in metric_names[:2]:
                        rows = [
                            (
                                site,
                                {
                                    model: values[metric]
                                    for model, values in models.items()
                                    if isinstance(values, dict) and metric in values
                                },
                            )
                            for site, models in islice(evaluations.items(), 6)
                            if isinstance(models, dict)
                        ]
                        lines.extend(["", f"  Model evaluation · {_text(metric)}", ""])
                        lines.append(format_metric_table(rows, label="Client"))
                    if len(evaluations) > 6 or len(metric_names) > 2:
                        lines.append("  Additional evaluation results are available in the saved artifact.")
            except (OSError, ValueError, TypeError):
                pass
            artifacts.append(f"  Evaluation {evaluation_path.relative_to(root)}")
        app_dir = run_dir / "app_server"
        if app_dir.is_dir():
            if any(p.is_file() and p.suffix in (".pt", ".pth", ".npy", ".npz") for p in app_dir.iterdir()):
                artifacts.insert(0, f"  Models    {app_dir.relative_to(root)}/")
        if summary_path.is_file() or evaluation_path.is_file():
            break
    log_paths = list(islice(root.glob("*/log.txt"), 6))
    if not log_paths:
        log_paths = list(islice(root.glob("workspace/log*.txt"), 6))
    if log_paths:
        artifacts.append("  Logs      " + " · ".join(str(p.relative_to(root)) for p in sorted(log_paths)))
    lines.extend(["", *artifacts])
    return wrap_log_message("\n".join(lines))
