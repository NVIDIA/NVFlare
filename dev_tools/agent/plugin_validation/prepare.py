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

"""Prepare synthetic plugin workloads without touching signed skill datasets."""

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def hashes(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def campaign(root, state):
    root.mkdir(parents=True)
    (root / "job.py").write_text('"""Synthetic report evidence only; not an executed training benchmark."""\n')
    fields = [
        "status",
        "name",
        "score",
        "runtime_seconds",
        "changed_files",
        "diff_summary",
        "run_command",
        "artifacts",
        "failure_reason",
        "candidate_manifest",
        "base_candidate",
        "patch_sha256",
        "metric_name",
        "metric_source",
        "metric_artifact",
        "candidate_kind",
        "algorithm_family",
        "literature_event_id",
    ]
    rows = []
    for status, name, score in [("baseline", "baseline", 0.5), ("keep", "candidate-1", 0.7)]:
        row = dict.fromkeys(fields, "")
        row.update(
            status=status,
            name=name,
            score=score,
            runtime_seconds=1,
            changed_files="none",
            diff_summary="Synthetic fixture record",
            run_command="python job.py",
            artifacts=f"runs/{name}",
            base_candidate="baseline" if status == "keep" else "",
            metric_name="validation_accuracy",
            metric_source="synthetic fixture",
            metric_artifact=f"runs/{name}/metrics.json",
            candidate_kind="argument_only",
        )
        rows.append(row)
        write_json(root / "runs" / name / "metrics.json", {"validation_accuracy": score, "synthetic": True})
    if state == "pending":
        row = dict.fromkeys(fields, "")
        row.update(status="candidate", name="unfinished", diff_summary="Synthetic pending candidate")
        rows.append(row)
    with (root / "results.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (root / "autofl.yaml").write_text(
        "schema_version: nvflare.autofl.config.v1\nobjective:\n  requested_metric: validation_accuracy\n"
        "  optimization_metric: validation_accuracy\n  metric_source: synthetic fixture\n  mode: max\n"
        "environment:\n  requested: sim\nbudget:\n  fixed_training_budget:\n    num_clients: 2\n    num_rounds: 1\n"
    )
    write_json(
        root / ".nvflare/autofl/campaign_state.json",
        {
            "reason": "running" if state == "active" else "manual_interrupt",
            "final_response_allowed": state != "active",
            "mode": "max",
            "candidate_cap": None,
            "candidate_cap_source": "uncapped",
            "synthetic": True,
        },
    )
    # The persisted lock is included in the immutable fixture snapshot.
    (root / ".nvflare/autofl/campaign.lock").touch()


def tabular_data(root):
    sites = {}
    for site, offset in [("site-1", 0), ("site-2", 240)]:
        path = root / "data" / site / "data.csv"
        path.parent.mkdir(parents=True)
        rows = [{"x": offset + i, "y": 2 * (offset + i) + 1} for i in range(240)]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["x", "y"])
            writer.writeheader()
            writer.writerows(rows)
        sites[site] = rows
    (root / "README.md").write_text(
        "# Synthetic two-site statistics fixture\n\nEach site owns its data.csv under data/. "
        "Features x and y are numeric; compute count, sum, mean, stddev and var for dataset data. "
        "Keep the recipe's default privacy filters and site boundaries. No external data is needed.\n"
    )
    oracle = {}
    for site, rows in {**sites, "Global": sum(sites.values(), [])}.items():
        oracle[site] = {}
        for feature in ("x", "y"):
            values = [row[feature] for row in rows]
            oracle[site][feature] = {
                "count": len(values),
                "sum": sum(values),
                "mean": statistics.mean(values),
                "stddev": statistics.stdev(values),
                "sample_variance": statistics.variance(values),
            }
    return oracle


def prepare(output, plugin):
    if output.is_relative_to(plugin) or plugin.is_relative_to(output):
        raise ValueError("Output and plugin directories must not overlap")
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    manifest = json.loads((plugin / "plugin.json").read_text())
    if manifest.get("name") != "nvidia-nvflare":
        raise ValueError("Expected the extracted nvidia-nvflare plugin")
    if any(p.is_symlink() for p in plugin.rglob("*")):
        raise ValueError("Use an extracted package containing real files")
    output.mkdir(parents=True)
    work = output / "workspaces"
    work.mkdir()
    cases = []

    def add(case_id, prompt, skills, *, negative_for=None):
        path = work / case_id
        path.mkdir(exist_ok=True)
        cases.append(
            {
                "id": case_id,
                "workspace": f"workspaces/{case_id}",
                "prompt": prompt,
                "expected_skills": skills,
                "negative_for": negative_for or [],
                "status": "not_run",
            }
        )
        return path

    for framework in ("pytorch", "lightning", "huggingface"):
        path = add(
            f"convert-{framework}",
            "Convert train.py to an NVIDIA FLARE job for two sites, one federated round, CPU only. "
            "Preserve the synthetic data and model semantics. Validate in the simulator and export the job. "
            "Use installed dependencies; do not download models/data or submit remote jobs. Report actual artifacts.",
            [f"nvflare-convert-{framework}", "nvflare-shared"],
        )
        shutil.copy2(FIXTURES / framework / "train.py", path / "train.py")
    path = add(
        "statistics",
        "Compute the statistics declared in README.md for ./data using NVIDIA FLARE.",
        ["nvflare-fed-stats"],
    )
    write_json(output / "oracles/statistics.json", tabular_data(path))
    for state in ("stopped", "active", "pending"):
        case_id = f"report-{state}"
        campaign(work / case_id, state)
        if state == "active":
            prompt = (
                "Read the supplied synthetic active campaign snapshot and summarize status; do not stop or finalize it."
            )
            expected, negative = ["nvflare-autofl"], ["nvflare-autofl-report"]
        else:
            prompt = (
                "The supplied synthetic campaign is stopped. Generate its final report if the evidence is complete."
            )
            expected, negative = ["nvflare-autofl-report"], []
        add(case_id, prompt, expected, negative_for=negative)
    path = add(
        "diagnose",
        "Explain the failure in logs/client.log using bounded evidence and suggest a correction. "
        "Do not run a job or fabricate missing inputs.",
        ["nvflare-diagnose-job"],
    )
    (path / "logs").mkdir()
    (path / "logs/client.log").write_text(
        "SYNTHETIC TEST LOG\nERROR Executor: task=train round=0\n"
        'Traceback (most recent call last):\n  File "client.py", line 12, in load_data\n'
        '    open("data/site-1/train.csv")\n'
        "FileNotFoundError: data/site-1/train.csv\nERROR task return_code=EXECUTION_EXCEPTION\n"
    )
    add(
        "orient",
        "Explain which NVIDIA FLARE workflows fit conversion, diagnosis, statistics and optimization. "
        "Ask for concrete project details before choosing a conversion framework.",
        ["nvflare-orient"],
    )
    add(
        "nonflare-negative",
        "Explain how to profile an ordinary non-federated Lightning training loop. "
        "No federated learning or NVIDIA FLARE conversion is requested.",
        [],
        negative_for=["*"],
    )
    path = add(
        "statistics-to-pytorch",
        "For the synthetic sites in ./data, compute the declared statistics, "
        "then convert train.py into a two-site, one-round CPU FLARE job using each site's own x/y rows. "
        "Validate statistics and training execution and export the job; keep site data separate.",
        ["nvflare-fed-stats", "nvflare-convert-pytorch", "nvflare-shared"],
    )
    tabular_data(path)
    shutil.copy2(FIXTURES / "pytorch" / "train.py", path / "train.py")
    write_json(
        output / "cases.json",
        {
            "format": "nvflare.plugin.workflow-preparation.v1",
            "formal_evaluator_dataset": False,
            "plugin": str(plugin),
            "plugin_version": manifest["version"],
            "cases": cases,
            "pending_scenarios": ["bounded live Auto-FL optimization (requires a validated converted job)"],
            "polarity_note": "Active status is negative for the report component, but positive for the full plugin.",
        },
    )
    write_json(output / "source-hashes.json", hashes(work))
    write_json(output / "plugin-hashes.json", hashes(plugin))
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = prepare(args.output.resolve(), args.plugin.resolve())
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Fixture preparation failed: {exc}\n")
    print(f"Prepared {count} scenarios; workflow execution and official evaluation have not run.")


if __name__ == "__main__":
    main()
