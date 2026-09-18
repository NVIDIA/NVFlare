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
import time
from itertools import islice
from pathlib import Path

from nvflare.fuel.utils.log_utils import console_text, format_metric_table, read_log_tail, wrap_log_message


def _print_output(message, *, flush=True):
    """Print optional presentation without disrupting execution on limited streams."""
    try:
        print(console_text(message), flush=flush)
    except (OSError, ValueError):
        # Closed streams, broken pipes, or encoding failures cannot change a job's outcome.
        pass


def run_recipe_job(job, env):
    """Start a deployment and attach the reporting context owned by Recipe."""
    from nvflare.recipe.run import Run

    _print_output(f"\nNVIDIA FLARE · {job.name}", flush=True)
    started_at = time.monotonic()
    job_id = env.deploy(job)
    run = Run(env, job_id)
    run._started_at = started_at
    run._summary_context = run_context(job.name, env)
    return run


def summary_header(outcome, elapsed, *, context=""):
    """Use the same heading and elapsed-time alignment for every run outcome."""
    heading = "\n" + " RUN SUMMARY ".center(72, "=")
    return heading + (f"\n\n{context}" if context else "") + "\n\n" + f"  {outcome}".ljust(64) + f"{elapsed:.1f}s"


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


def _model_artifacts(run_dir, root):
    """Return standard model directories without deserializing their contents."""
    extensions = (".pt", ".pth", ".npy", ".npz", ".h5", ".keras", ".joblib", ".pkl")

    def is_model_file(path):
        name = path.name.lower()
        return path.is_file() and (name.endswith(extensions) or ("model" in name and name.endswith((".json", ".ubj"))))

    locations = []
    for model_dir in (run_dir / "app_server", run_dir / "models"):
        if model_dir.is_dir() and any(is_model_file(path) for path in model_dir.iterdir()):
            locations.append(f"{model_dir.relative_to(root)}/")
    return locations


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
    model_locations = []
    statistics_locations = []
    for run_dir in candidates:
        metrics = run_dir / "metrics"
        summary_path = metrics / "metrics_summary.json"
        if summary_path.is_file():
            try:
                round_path = metrics / "round_metrics.jsonl"
                with round_path.open("rb") as stream:
                    data, truncated = read_log_tail(stream, 1048576, whole_lines=True)
                records = data.splitlines()[-11:]
                truncated = truncated or len(records) > 10
                records = records[-10:]
                aggregate_rows = []
                site_rows = []
                site_title = "Training"
                for raw in records:
                    try:
                        record = json.loads(raw)
                        values = {m["name"]: m["value"] for m in record["aggregated_metrics"]}
                        round_index = record["round"]
                        label = str(round_index + 1) if type(round_index) is int else "?"
                        if values:
                            aggregate_rows.append((label, values))
                            continue
                        site_title = _text(record.get("progress_title", "Training"))
                        sites = record.get("sites", [])
                        if not isinstance(sites, list):
                            continue
                        for site in sites:
                            if not isinstance(site, dict):
                                continue
                            metrics = site.get("progress_metrics") or site.get("metrics")
                            if not isinstance(metrics, list):
                                continue
                            site_values = {
                                metric["name"]: metric["value"]
                                for metric in metrics
                                if isinstance(metric, dict) and "name" in metric and "value" in metric
                            }
                            if site_values:
                                site_rows.append((_text(site.get("name", "unknown")), site_values))
                    except (ValueError, TypeError, KeyError):
                        continue
                if aggregate_rows:
                    heading = "  Training · aggregated client metrics"
                    lines.extend(["", heading + (" (last 10 rounds)" if truncated else ""), ""])
                    lines.append(format_metric_table(aggregate_rows, label="Round"))
                if site_rows:
                    heading = f"  {site_title} · client metrics"
                    lines.extend(["", heading + (" (last 10 rounds)" if truncated else ""), ""])
                    lines.append(format_metric_table(site_rows, label="Client"))
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
        for location in _model_artifacts(run_dir, root):
            if location not in model_locations:
                model_locations.append(location)
        statistics_dir = run_dir / "statistics"
        if statistics_dir.is_dir() and any(path.is_file() for path in statistics_dir.iterdir()):
            location = f"{statistics_dir.relative_to(root)}/"
            if location not in statistics_locations:
                statistics_locations.append(location)
        if summary_path.is_file() or evaluation_path.is_file():
            break
    if model_locations:
        artifacts.insert(0, "  Models    " + " · ".join(model_locations))
    if statistics_locations:
        artifacts.append("  Statistics " + " · ".join(statistics_locations))
    log_paths = list(islice(root.glob("*/log.txt"), 6))
    if not log_paths:
        log_paths = list(islice(root.glob("workspace/log*.txt"), 6))
    if log_paths:
        artifacts.append("  Logs      " + " · ".join(str(p.relative_to(root)) for p in sorted(log_paths)))
    lines.extend(["", *artifacts])
    return wrap_log_message("\n".join(lines))
