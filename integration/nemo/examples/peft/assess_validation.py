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
"""Apply the fixed smoke, continuity, and learning acceptance gates."""

from __future__ import annotations

import argparse
import glob
import json
import math

RELOAD_LOSS_TOLERANCE = 5e-4


def _load(path):
    with open(path) as f:
        return json.load(f)


def _client_manifests(root):
    return [_load(path) for path in sorted(glob.glob(f"{root}/**/round_manifest.json", recursive=True))]


def verify_reload_reproducibility(first: dict, second: dict) -> dict:
    first_validation = first["validation"]
    second_validation = second["validation"]
    exact_metrics = ("response_token_count", "accuracy", "macro_f1", "confusion", "prediction_counts")
    mismatches = [name for name in exact_metrics if first_validation[name] != second_validation[name]]
    if mismatches:
        raise ValueError(f"Native adapter reload changed evaluation metrics: {mismatches}")
    first_loss = float(first_validation["response_token_loss"])
    second_loss = float(second_validation["response_token_loss"])
    loss_delta = abs(first_loss - second_loss)
    if not math.isfinite(first_loss) or not math.isfinite(second_loss) or loss_delta > RELOAD_LOSS_TOLERANCE:
        raise ValueError(
            f"Native adapter reload response-token loss delta {loss_delta} exceeds {RELOAD_LOSS_TOLERANCE}"
        )
    return {
        "exact_metrics": list(exact_metrics),
        "response_token_loss_delta": loss_delta,
        "response_token_loss_tolerance": RELOAD_LOSS_TOLERANCE,
    }


def _last_training_metrics(manifest: dict) -> dict:
    record = manifest.get("automodel_report", {}).get("last_training_record", {})
    return record.get("metrics", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_summary", required=True)
    parser.add_argument("--smoke_client_root", required=True)
    parser.add_argument("--continuity_report", required=True)
    parser.add_argument("--seed_summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    failures = []
    smoke = _client_manifests(args.smoke_client_root)
    if len(smoke) != 1:
        failures.append(f"expected one smoke task, found {len(smoke)}")
    for manifest in smoke:
        if manifest["actual_optimizer_steps"] != 2:
            failures.append(f"smoke actual_optimizer_steps={manifest['actual_optimizer_steps']}, expected 2")
        if not math.isfinite(manifest["update_norm"]) or manifest["update_norm"] <= 0:
            failures.append(f"smoke update norm is not finite and positive: {manifest['update_norm']}")
        trainable = manifest.get("automodel_report", {}).get("trainable_tensor_count", 0)
        if trainable <= 0:
            failures.append("smoke did not report trainable LoRA tensors")
        training_metrics = _last_training_metrics(manifest)
        for name in ("loss", "grad_norm"):
            value = training_metrics.get(name)
            if value is None or not math.isfinite(float(value)):
                failures.append(f"smoke {name} is missing or non-finite: {value}")

    continuity = _load(args.continuity_report)
    if not continuity.get("all_checks_passed") or len(continuity.get("rounds", [])) != 3:
        failures.append("three-round continuity verification did not pass")
    elif sum(len(round_report.get("clients", [])) for round_report in continuity["rounds"]) != 9:
        failures.append("continuity report does not contain all nine client tasks")

    base = _load(args.base_summary)
    base_val_loss = base["validation"]["response_token_loss"]
    base_test_f1 = base["test"]["macro_f1"]
    final_summaries = [_load(path) for path in args.seed_summary]
    if len(final_summaries) != 2:
        failures.append(f"expected two learning seeds, found {len(final_summaries)}")
    for path, summary in zip(args.seed_summary, final_summaries):
        if summary["validation"]["response_token_loss"] >= base_val_loss:
            failures.append(f"{path} did not lower validation response-token loss")
    mean_final_test_f1 = sum(summary["test"]["macro_f1"] for summary in final_summaries) / len(final_summaries)
    if mean_final_test_f1 < base_test_f1:
        failures.append(f"mean final test Macro-F1 {mean_final_test_f1} is below base Macro-F1 {base_test_f1}")

    report = {
        "schema_version": 1,
        "passed": not failures,
        "failures": failures,
        "base_validation_response_token_loss": base_val_loss,
        "base_test_macro_f1": base_test_f1,
        "final_validation_response_token_losses": [
            summary["validation"]["response_token_loss"] for summary in final_summaries
        ],
        "final_test_macro_f1": [summary["test"]["macro_f1"] for summary in final_summaries],
        "mean_final_test_macro_f1": mean_final_test_f1,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
