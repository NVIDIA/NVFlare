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

"""Summarize matched CIFAR-10 runs with 95% confidence intervals."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

METRICS = ("global_accuracy", "worst_client_accuracy")


def _load_rows(path: str, protocol_version: str | None = None) -> list[dict]:
    rows = []
    with open(path) as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if protocol_version is not None and row.get("protocol_version") != protocol_version:
                continue
            for key in ("method", "alpha", "participation_rate", "seed", *METRICS):
                if key not in row:
                    raise ValueError(f"line {line_number} is missing required field {key!r}")
            rows.append(row)
    if not rows:
        suffix = "" if protocol_version is None else f" for protocol {protocol_version!r}"
        raise ValueError(f"results file contains no runs{suffix}")
    return rows


def _ci(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("confidence interval input must contain finite values")
    mean = float(values.mean())
    if values.size == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "ci95_low": None,
            "ci95_high": None,
            "half_width": None,
        }
    std = float(values.std(ddof=1))
    critical = float(t.ppf(0.975, df=values.size - 1))
    half_width = critical * std / math.sqrt(values.size)
    return {
        "n": int(values.size),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "half_width": half_width,
    }


def summarize(rows: list[dict], reference_method: str = "adaptive") -> dict:
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        unique_key = (
            str(row["method"]),
            float(row["alpha"]),
            float(row["participation_rate"]),
            int(row["seed"]),
        )
        if unique_key in seen:
            raise ValueError(f"duplicate completed run for condition {unique_key}")
        seen.add(unique_key)
        group_key = (float(row["alpha"]), float(row["participation_rate"]), str(row["method"]))
        grouped[group_key].append(row)

    summaries = []
    for (alpha, participation_rate, method), group in sorted(grouped.items()):
        summaries.append(
            {
                "alpha": alpha,
                "participation_rate": participation_rate,
                "method": method,
                "seeds": sorted(int(row["seed"]) for row in group),
                "metrics": {metric: _ci([float(row[metric]) for row in group]) for metric in METRICS},
            }
        )

    paired = []
    conditions = sorted({(float(row["alpha"]), float(row["participation_rate"])) for row in rows})
    for alpha, participation_rate in conditions:
        condition_rows = [
            row
            for row in rows
            if float(row["alpha"]) == alpha and float(row["participation_rate"]) == participation_rate
        ]
        reference = {int(row["seed"]): row for row in condition_rows if str(row["method"]) == reference_method}
        methods = sorted({str(row["method"]) for row in condition_rows if str(row["method"]) != reference_method})
        for method in methods:
            baseline = {int(row["seed"]): row for row in condition_rows if str(row["method"]) == method}
            common_seeds = sorted(set(reference) & set(baseline))
            if not common_seeds:
                continue
            paired.append(
                {
                    "alpha": alpha,
                    "participation_rate": participation_rate,
                    "reference_method": reference_method,
                    "baseline_method": method,
                    "seeds": common_seeds,
                    "delta_reference_minus_baseline": {
                        metric: _ci(
                            [
                                float(reference[seed][metric]) - float(baseline[seed][metric])
                                for seed in common_seeds
                            ]
                        )
                        for metric in METRICS
                    },
                }
            )

    return {"summaries": summaries, "paired_comparisons": paired}


def main(args):
    rows = _load_rows(args.input, protocol_version=args.protocol_version)
    result = summarize(rows, reference_method=args.reference_method)
    if args.protocol_version is not None:
        result["protocol_version"] = args.protocol_version
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL file with one completed run per line")
    parser.add_argument("--output", required=True, help="Destination JSON summary")
    parser.add_argument("--reference_method", default="adaptive")
    parser.add_argument("--protocol_version", default=None)
    main(parser.parse_args())
