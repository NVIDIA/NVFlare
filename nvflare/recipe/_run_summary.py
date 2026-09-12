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
import textwrap
from itertools import islice
from pathlib import Path

from nvflare.fuel.utils.log_utils import format_metric_summary


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
    # Simulator result root, downloaded admin transfer root, or server run root.
    candidates = [root / "workspace", root]
    candidates.extend(islice(root.glob("server/*"), 20))
    for run_dir in candidates:
        metrics = run_dir / "metrics"
        summary_path = metrics / "metrics_summary.json"
        if summary_path.is_file():
            try:
                summary = _read_json(summary_path)
                if isinstance(summary, dict):
                    lines.append(f"NVIDIA FLARE | {_text(summary.get('job_name', root.name))}")
                round_path = metrics / "round_metrics.jsonl"
                with round_path.open("rb") as stream:
                    stream.seek(0, 2)
                    start = max(0, stream.tell() - 1048576)
                    stream.seek(start)
                    if start:
                        stream.readline(1048576)  # Ignore a partial first record.
                    records = stream.read(1048576).splitlines()[-10:]
                lines.extend(
                    ["", "Training (last recorded rounds)", "  Round  Reporting clients  Aggregated client metrics"]
                )
                for raw in records:
                    try:
                        record = json.loads(raw)
                        metrics_dict = {m["name"]: m["value"] for m in record["aggregated_metrics"]}
                        round_index = record["round"]
                        label = str(round_index + 1) if type(round_index) is int else "?"
                        count = len({s["name"] for s in record["sites"]})
                        lines.append(f"  {label:<7}{count:<19}{format_metric_summary(metrics_dict)}")
                    except (ValueError, TypeError, KeyError):
                        continue
                lines.append("  Values as reported; metric names and units are application-defined.")
            except (OSError, ValueError, TypeError, KeyError):
                lines.append("Training details: see the saved metrics artifacts.")
            lines.append(f"Metrics   {summary_path.relative_to(root)}")
        evaluation_path = run_dir / "cross_site_val" / "cross_val_results.json"
        if evaluation_path.is_file():
            try:
                evaluations = _read_json(evaluation_path)
                if isinstance(evaluations, dict) and evaluations:
                    lines.extend(["", "Model evaluation (saved results)"])
                    for site, models in islice(evaluations.items(), 6):
                        if not isinstance(models, dict):
                            continue
                        for model, values in islice(models.items(), 6):
                            lines.append(f"  {_text(site)} | {_text(model)}")
                            lines.append(f"    {format_metric_summary(values)}")
            except (OSError, ValueError, TypeError):
                pass
            lines.append(f"Evaluation   {evaluation_path.relative_to(root)}")
        app_dir = run_dir / "app_server"
        if app_dir.is_dir():
            # List existing common model artifacts; don't guess a "final" model.
            for path in islice((p for p in app_dir.iterdir() if p.suffix in (".pt", ".pth", ".npy", ".npz")), 6):
                if path.is_file():
                    lines.append(f"Model     {path.relative_to(root)}")
        if summary_path.is_file() or evaluation_path.is_file():
            break
    log_paths = list(islice(root.glob("*/log.txt"), 6))
    if not log_paths:
        log_paths = list(islice(root.glob("workspace/log*.txt"), 6))
    if log_paths:
        lines.append("Logs      " + " | ".join(str(p.relative_to(root)) for p in log_paths))
    return "\n".join(
        textwrap.fill(line, width=80, subsequent_indent="    ", replace_whitespace=False) for line in lines
    )
