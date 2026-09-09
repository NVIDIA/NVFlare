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

"""Summarize matched CIFAR-10 runs with provenance and 95% confidence intervals."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

from protocol import canonical_config_hash

METRICS = ("global_accuracy", "worst_client_accuracy")
ADAPTIVE_TELEMETRY_FIELDS = (
    "adaptive_aggregation_rounds",
    "adaptive_active_rounds",
    "adaptive_activation_rate",
    "adaptive_mean_active_blend_factor",
    "adaptive_max_observed_blend_factor",
    "adaptive_cohort_change_count",
)


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


def _condition_key(row: dict) -> tuple[str, float, float, int]:
    return (
        str(row["method"]),
        float(row["alpha"]),
        float(row["participation_rate"]),
        int(row["seed"]),
    )


def _validate_hash_payload(row: dict, config_key: str, hash_key: str) -> None:
    config = row.get(config_key)
    digest = row.get(hash_key)
    if not isinstance(config, dict) or not isinstance(digest, str) or not digest:
        raise ValueError(f"result row {_condition_key(row)} is missing {config_key}/{hash_key} provenance")
    actual = canonical_config_hash(config)
    if actual != digest:
        raise ValueError(f"result row {_condition_key(row)} has an invalid {hash_key}")


def _validate_nested_provenance(row: dict) -> None:
    method = str(row["method"])
    condition = row.get("condition_config")
    experiment = row.get("experiment_config")
    if not isinstance(condition, dict):
        raise ValueError(f"result row {_condition_key(row)} is missing condition_config provenance")
    expected_condition = {
        "alpha": float(row["alpha"]),
        "participation_rate": float(row["participation_rate"]),
        "seed": int(row["seed"]),
    }
    if condition != expected_condition:
        raise ValueError(f"result row {_condition_key(row)} has inconsistent condition_config provenance")
    method_config = row["method_config"]
    if method_config.get("method") != method:
        raise ValueError(f"result row {_condition_key(row)} has inconsistent method_config provenance")
    if not isinstance(experiment, dict):
        raise ValueError(f"result row {_condition_key(row)} is missing experiment_config provenance")
    expected_experiment = {
        "protocol_version": row.get("protocol_version"),
        "common": row["common_config"],
        "method": method_config,
        "condition": condition,
    }
    if experiment != expected_experiment:
        raise ValueError(f"result row {_condition_key(row)} has inconsistent experiment_config provenance")


def _validate_adaptive_telemetry(row: dict) -> None:
    telemetry = row.get("adaptive_telemetry")
    if not isinstance(telemetry, dict):
        raise ValueError(f"adaptive result row {_condition_key(row)} is missing activation telemetry")
    missing = [key for key in ADAPTIVE_TELEMETRY_FIELDS if key not in telemetry]
    if missing:
        raise ValueError(f"adaptive result row {_condition_key(row)} is missing telemetry fields: {missing}")

    rounds = int(telemetry["adaptive_aggregation_rounds"])
    active_rounds = int(telemetry["adaptive_active_rounds"])
    activation_rate = float(telemetry["adaptive_activation_rate"])
    mean_active_blend = float(telemetry["adaptive_mean_active_blend_factor"])
    max_blend = float(telemetry["adaptive_max_observed_blend_factor"])
    cohort_changes = int(telemetry["adaptive_cohort_change_count"])
    values = (activation_rate, mean_active_blend, max_blend)
    if rounds <= 0 or active_rounds < 0 or active_rounds > rounds or cohort_changes < 0:
        raise ValueError(f"adaptive result row {_condition_key(row)} has invalid activation counts")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"adaptive result row {_condition_key(row)} has non-finite activation telemetry")
    if not 0.0 <= activation_rate <= 1.0 or not 0.0 <= mean_active_blend <= 1.0 or not 0.0 <= max_blend <= 1.0:
        raise ValueError(f"adaptive result row {_condition_key(row)} has out-of-range activation telemetry")
    expected_rate = active_rounds / rounds
    if not math.isclose(activation_rate, expected_rate, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"adaptive result row {_condition_key(row)} has inconsistent activation rate")
    if active_rounds == 0 and (mean_active_blend != 0.0 or max_blend != 0.0):
        raise ValueError(f"adaptive result row {_condition_key(row)} reports blend without active rounds")


def validate_config_provenance(
    rows: list[dict],
    expected_common_hash: str | None = None,
    expected_method_hashes: dict[str, str] | None = None,
) -> None:
    """Reject rows produced by incompatible or internally inconsistent configurations."""

    common_hashes = set()
    per_method_hashes = defaultdict(set)
    expected_method_hashes = expected_method_hashes or {}
    for row in rows:
        _validate_hash_payload(row, "common_config", "common_config_hash")
        _validate_hash_payload(row, "method_config", "method_config_hash")
        _validate_hash_payload(row, "experiment_config", "experiment_config_hash")
        _validate_nested_provenance(row)
        common_hash = row["common_config_hash"]
        method = str(row["method"])
        method_hash = row["method_config_hash"]
        common_hashes.add(common_hash)
        per_method_hashes[method].add(method_hash)
        if expected_common_hash is not None and common_hash != expected_common_hash:
            raise ValueError(f"result row {_condition_key(row)} does not match the requested common configuration")
        expected_method_hash = expected_method_hashes.get(method)
        if expected_method_hash is not None and method_hash != expected_method_hash:
            raise ValueError(f"result row {_condition_key(row)} does not match the requested {method} configuration")
        if method == "adaptive":
            _validate_adaptive_telemetry(row)

    if len(common_hashes) != 1:
        raise ValueError("results mix multiple common experiment configurations")
    mixed_methods = {method: hashes for method, hashes in per_method_hashes.items() if len(hashes) != 1}
    if mixed_methods:
        raise ValueError(f"results mix multiple method configurations: {sorted(mixed_methods)}")


def validate_complete_matrix(
    rows: list[dict],
    methods: list[str],
    alphas: list[float],
    participation_rates: list[float],
    seeds: list[int],
    expected_common_hash: str | None = None,
    expected_method_hashes: dict[str, str] | None = None,
) -> None:
    """Require every requested method/condition/seed exactly once with one configuration."""

    validate_config_provenance(rows, expected_common_hash, expected_method_hashes)
    actual_keys = [_condition_key(row) for row in rows]
    if len(set(actual_keys)) != len(actual_keys):
        raise ValueError("results contain duplicate method/alpha/participation/seed rows")
    actual = set(actual_keys)

    expected = set()
    for method in methods:
        for alpha in alphas:
            for participation in participation_rates:
                if method == "fedopt" and participation < 1.0:
                    continue
                for seed in seeds:
                    expected.add((method, float(alpha), float(participation), int(seed)))

    missing = sorted(expected - actual)
    if missing:
        raise ValueError(
            f"incomplete CIFAR-10 evidence matrix: missing {len(missing)} rows; first entries: {missing[:8]}"
        )


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
        unique_key = _condition_key(row)
        if unique_key in seen:
            raise ValueError(f"duplicate completed run for condition {unique_key}")
        seen.add(unique_key)
        group_key = (float(row["alpha"]), float(row["participation_rate"]), str(row["method"]))
        grouped[group_key].append(row)

    summaries = []
    for (alpha, participation_rate, method), group in sorted(grouped.items()):
        item = {
            "alpha": alpha,
            "participation_rate": participation_rate,
            "method": method,
            "seeds": sorted(int(row["seed"]) for row in group),
            "metrics": {metric: _ci([float(row[metric]) for row in group]) for metric in METRICS},
        }
        if method == "adaptive" and all(isinstance(row.get("adaptive_telemetry"), dict) for row in group):
            item["adaptive_telemetry"] = {
                "activation_rate": _ci(
                    [float(row["adaptive_telemetry"]["adaptive_activation_rate"]) for row in group]
                ),
                "active_rounds": _ci(
                    [float(row["adaptive_telemetry"]["adaptive_active_rounds"]) for row in group]
                ),
                "mean_active_blend_factor": _ci(
                    [float(row["adaptive_telemetry"]["adaptive_mean_active_blend_factor"]) for row in group]
                ),
                "max_observed_blend_factor": _ci(
                    [float(row["adaptive_telemetry"]["adaptive_max_observed_blend_factor"]) for row in group]
                ),
                "cohort_change_count": _ci(
                    [float(row["adaptive_telemetry"]["adaptive_cohort_change_count"]) for row in group]
                ),
            }
        summaries.append(item)

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


def _format_ci(stats: dict) -> str:
    mean = 100.0 * float(stats["mean"])
    if stats["half_width"] is None:
        return f"{mean:.2f}% (n=1; CI unavailable)"
    half_width = 100.0 * float(stats["half_width"])
    return f"{mean:.2f}% ± {half_width:.2f} pp"


def _format_delta(stats: dict) -> str:
    mean = 100.0 * float(stats["mean"])
    if stats["half_width"] is None:
        return f"{mean:+.2f} pp (n=1; CI unavailable)"
    half_width = 100.0 * float(stats["half_width"])
    return f"{mean:+.2f} ± {half_width:.2f} pp"


def _format_activation(item: dict) -> str:
    telemetry = item.get("adaptive_telemetry")
    if not telemetry:
        return "—"
    rate = telemetry["activation_rate"]
    mean = 100.0 * float(rate["mean"])
    if rate["half_width"] is None:
        return f"{mean:.1f}%"
    half_width = 100.0 * float(rate["half_width"])
    return f"{mean:.1f}% ± {half_width:.1f} pp"


def render_markdown(summary: dict) -> str:
    """Render full and partial participation together as main results."""

    lines = [
        "# CIFAR-10 Main Results",
        "",
        "Values are means across matched seeds with two-sided 95% Student-t confidence intervals.",
        "Global and worst-client accuracy use the common post-training evaluator.",
        "Adaptive activation rate is the fraction of valid aggregation rounds with a non-zero blend; a low rate makes conservative fallback behavior explicit.",
        "",
    ]
    groups = defaultdict(list)
    for item in summary["summaries"]:
        groups[(float(item["alpha"]), float(item["participation_rate"]))].append(item)

    for (alpha, participation_rate), items in sorted(groups.items()):
        lines.extend(
            [
                f"## Dirichlet alpha={alpha:g}, participation={participation_rate:.0%}",
                "",
                "| Method | Seeds | Global accuracy | Worst-client accuracy | Adaptive activation |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in sorted(items, key=lambda value: value["method"]):
            lines.append(
                "| {method} | {n} | {global_ci} | {worst_ci} | {activation} |".format(
                    method=item["method"],
                    n=len(item["seeds"]),
                    global_ci=_format_ci(item["metrics"]["global_accuracy"]),
                    worst_ci=_format_ci(item["metrics"]["worst_client_accuracy"]),
                    activation=_format_activation(item),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Paired adaptive-minus-baseline deltas",
            "",
            "Positive values favor adaptive aggregation. Neutral and negative deltas are retained.",
            "",
            "| Alpha | Participation | Baseline | Seeds | Global delta | Worst-client delta |",
            "| ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for item in summary["paired_comparisons"]:
        lines.append(
            "| {alpha:g} | {participation:.0%} | {baseline} | {n} | {global_delta} | {worst_delta} |".format(
                alpha=item["alpha"],
                participation=item["participation_rate"],
                baseline=item["baseline_method"],
                n=len(item["seeds"]),
                global_delta=_format_delta(item["delta_reference_minus_baseline"]["global_accuracy"]),
                worst_delta=_format_delta(item["delta_reference_minus_baseline"]["worst_client_accuracy"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _method_hashes(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--method_config_hash entries must use METHOD=HASH")
        method, digest = value.split("=", 1)
        if not method or not digest:
            raise ValueError("--method_config_hash entries must use non-empty METHOD=HASH")
        result[method] = digest
    return result


def main(args):
    rows = _load_rows(args.input, protocol_version=args.protocol_version)
    expected_method_hashes = _method_hashes(args.method_config_hash)
    if args.require_complete:
        validate_complete_matrix(
            rows,
            args.methods,
            args.alphas,
            args.participation_rates,
            args.seeds,
            expected_common_hash=args.common_config_hash,
            expected_method_hashes=expected_method_hashes,
        )
    elif args.common_config_hash or expected_method_hashes:
        validate_config_provenance(rows, args.common_config_hash, expected_method_hashes)
    result = summarize(rows, reference_method=args.reference_method)
    if args.protocol_version is not None:
        result["protocol_version"] = args.protocol_version
    if args.common_config_hash is not None:
        result["common_config_hash"] = args.common_config_hash
    if expected_method_hashes:
        result["method_config_hashes"] = expected_method_hashes
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.markdown_output:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(result))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL file with one completed run per line")
    parser.add_argument("--output", required=True, help="Destination JSON summary")
    parser.add_argument("--markdown_output", default=None, help="Optional reviewer-facing Markdown main-results table")
    parser.add_argument("--reference_method", default="adaptive")
    parser.add_argument("--protocol_version", default=None)
    parser.add_argument("--common_config_hash", default=None)
    parser.add_argument("--method_config_hash", action="append", default=[])
    parser.add_argument("--require_complete", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["fedavg", "fedprox", "scaffold", "fedce", "adaptive"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.5])
    parser.add_argument("--participation_rates", nargs="+", type=float, default=[1.0, 0.75])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 31, 43, 57])
    main(parser.parse_args())
