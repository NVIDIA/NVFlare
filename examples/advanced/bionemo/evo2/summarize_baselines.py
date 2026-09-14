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
"""Validate and summarize the three-site Evo2 local-only baseline campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

EXPECTED_SITE_IDS = (1, 2, 3)
EXPECTED_ROWS = 3000
ROW_IDENTITY = "zero_based_jsonl_line_index"
BIONEMO_ACCURACY_REFERENCE = 0.966
FORMAT_VERSION = 1

_CHECKPOINT_ROLES = ("initialization", "primary_fl_final", "secondary_selected_fl")
_METRICS = ("accuracy", "macro_f1")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SIGNATURE_SPLIT_FIELDS = frozenset(("split_role", "test_file_sha256"))
_CURRENT_SIGNATURE_FIELDS = frozenset(
    (
        "backend",
        "base_checkpoint_sha256",
        "classifier_file_sha256",
        "dataset_manifest_sha256",
        "global_batch_size",
        "lora_alpha",
        "lora_dim",
        "lora_dropout",
        "lora_target_modules",
        "micro_batch_size",
        "peft_mode",
        "row_identity",
        "seed",
        "seq_length",
        "split_role",
        "test_file_sha256",
    )
)


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", required=True)
    parser.add_argument(
        "--site-validation",
        action="append",
        required=True,
        metavar="SITE_ID=PATH",
        help="Validation evaluation for one local site; provide exactly sites 1, 2, and 3",
    )
    parser.add_argument(
        "--site-test",
        action="append",
        required=True,
        metavar="SITE_ID=PATH",
        help="Official-test evaluation for one local site; provide exactly sites 1, 2, and 3",
    )
    parser.add_argument("--primary-fl-validation", required=True)
    parser.add_argument("--primary-fl-test", required=True)
    parser.add_argument("--secondary-selected-fl-validation", required=True)
    parser.add_argument("--secondary-selected-fl-test", required=True)
    parser.add_argument("--initialization-validation", required=True)
    parser.add_argument("--initialization-test", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _load_json(path: str | os.PathLike[str], label: str) -> dict:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{label} JSON not found: {resolved_path}")
    try:
        with resolved_path.open(encoding="utf-8") as file:
            value = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {resolved_path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {resolved_path}")
    return value


def _require_sha256(value, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters.")
    return value


def _load_campaign_manifest(path: str | os.PathLike[str]) -> dict:
    manifest = _load_json(path, "Campaign manifest")
    expected_keys = {"format_version", "checkpoint_sha256", "protocol", "resources"}
    if set(manifest) != expected_keys:
        raise ValueError(
            f"Campaign manifest fields must be exactly {sorted(expected_keys)}, received {sorted(manifest)}."
        )
    if type(manifest["format_version"]) is not int or manifest["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported campaign manifest format_version {manifest['format_version']!r}; expected {FORMAT_VERSION}."
        )
    for name in ("protocol", "resources"):
        if not isinstance(manifest[name], dict):
            raise ValueError(f"Campaign manifest {name} must be a JSON object.")

    checkpoint_sha256 = manifest["checkpoint_sha256"]
    if not isinstance(checkpoint_sha256, dict):
        raise ValueError("Campaign manifest checkpoint_sha256 must be a JSON object.")
    expected_checkpoint_keys = {*_CHECKPOINT_ROLES, "local_sites"}
    if set(checkpoint_sha256) != expected_checkpoint_keys:
        raise ValueError(
            "Campaign manifest checkpoint_sha256 fields must be exactly "
            f"{sorted(expected_checkpoint_keys)}, received {sorted(checkpoint_sha256)}."
        )
    for role in _CHECKPOINT_ROLES:
        _require_sha256(checkpoint_sha256[role], f"Campaign manifest checkpoint_sha256.{role}")

    local_sites = checkpoint_sha256["local_sites"]
    if not isinstance(local_sites, dict) or set(local_sites) != {str(site_id) for site_id in EXPECTED_SITE_IDS}:
        raise ValueError("Campaign manifest checkpoint_sha256.local_sites must contain exactly site IDs 1, 2, and 3.")
    for site_id in EXPECTED_SITE_IDS:
        _require_sha256(
            local_sites[str(site_id)],
            f"Campaign manifest checkpoint_sha256.local_sites.{site_id}",
        )
    return manifest


def _parse_site_paths(values: Sequence[str], option_name: str) -> dict[int, Path]:
    if len(values) != len(EXPECTED_SITE_IDS):
        raise ValueError(f"{option_name} must be provided exactly three times for site IDs 1, 2, and 3.")
    paths = {}
    for value in values:
        site_value, separator, path_value = value.partition("=")
        if not separator or not site_value.isdecimal() or not path_value:
            raise ValueError(f"{option_name} values must use SITE_ID=PATH with site IDs 1, 2, and 3: {value!r}.")
        site_id = int(site_value)
        if site_id not in EXPECTED_SITE_IDS:
            raise ValueError(f"{option_name} contains unsupported site ID {site_id}; expected 1, 2, or 3.")
        if site_id in paths:
            raise ValueError(f"{option_name} contains duplicate site ID {site_id}.")
        paths[site_id] = Path(path_value).expanduser().resolve()
    if set(paths) != set(EXPECTED_SITE_IDS):
        raise ValueError(f"{option_name} must contain exactly site IDs 1, 2, and 3.")
    if len(set(paths.values())) != len(paths):
        raise ValueError(f"{option_name} must reference a distinct JSON file for every site.")
    return dict(sorted(paths.items()))


def _require_probability(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{label} must be finite and between zero and one.")
    return normalized


def _validate_evaluation(report: dict, *, label: str, split_role: str, checkpoint_sha256: str) -> dict:
    observed_checkpoint_sha256 = _require_sha256(report.get("checkpoint_sha256"), f"{label} checkpoint_sha256")
    if observed_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError(
            f"{label} checkpoint SHA-256 does not match the campaign manifest: "
            f"expected {checkpoint_sha256}, observed {observed_checkpoint_sha256}."
        )
    if type(report.get("num_examples")) is not int or report["num_examples"] != EXPECTED_ROWS:
        raise ValueError(f"{label} must report num_examples={EXPECTED_ROWS}.")

    coverage = report.get("evaluation_coverage")
    if not isinstance(coverage, dict):
        raise ValueError(f"{label} is missing evaluation_coverage.")
    expected_coverage = {
        "identity": ROW_IDENTITY,
        "expected_rows": EXPECTED_ROWS,
        "observed_rows": EXPECTED_ROWS,
        "unique_rows": EXPECTED_ROWS,
    }
    coverage_mismatches = {
        name: {"expected": expected, "observed": coverage.get(name)}
        for name, expected in expected_coverage.items()
        if coverage.get(name) != expected
    }
    if coverage_mismatches:
        raise ValueError(f"{label} does not contain complete {EXPECTED_ROWS}-row coverage: {coverage_mismatches}.")
    _require_sha256(coverage.get("sampler_order_sha256"), f"{label} evaluation_coverage.sampler_order_sha256")

    signature = report.get("evaluation_signature")
    if not isinstance(signature, dict):
        raise ValueError(f"{label} is missing evaluation_signature.")
    missing_signature_fields = sorted(_CURRENT_SIGNATURE_FIELDS - set(signature))
    if missing_signature_fields:
        raise ValueError(f"{label} does not use the current evaluation_signature: missing {missing_signature_fields}.")
    if signature.get("backend") != "bionemo" or signature.get("peft_mode") != "lora":
        raise ValueError(f"{label} must be a BioNeMo LoRA evaluation.")
    if signature.get("split_role") != split_role:
        raise ValueError(
            f"{label} evaluation_signature.split_role must be {split_role!r}, "
            f"observed {signature.get('split_role')!r}."
        )
    if signature.get("row_identity") != ROW_IDENTITY:
        raise ValueError(f"{label} evaluation_signature.row_identity must be {ROW_IDENTITY!r}.")
    for field in (
        "base_checkpoint_sha256",
        "classifier_file_sha256",
        "dataset_manifest_sha256",
        "test_file_sha256",
    ):
        _require_sha256(signature.get(field), f"{label} evaluation_signature.{field}")

    dataset_manifest = report.get("dataset_manifest")
    if not isinstance(dataset_manifest, dict):
        raise ValueError(f"{label} must be bound to a prepared dataset manifest.")
    if type(dataset_manifest.get("format_version")) is not int or dataset_manifest["format_version"] != 2:
        raise ValueError(f"{label} dataset manifest must use format_version 2.")
    if dataset_manifest.get("split_role") != split_role:
        raise ValueError(
            f"{label} dataset_manifest.split_role must be {split_role!r}, "
            f"observed {dataset_manifest.get('split_role')!r}."
        )
    if dataset_manifest.get("sha256") != signature["dataset_manifest_sha256"]:
        raise ValueError(f"{label} dataset manifest SHA-256 does not match its evaluation_signature.")
    file_identity = dataset_manifest.get("file_identity")
    if not isinstance(file_identity, dict) or set(file_identity) != {"sha256", "bytes", "rows"}:
        raise ValueError(f"{label} dataset manifest split file identity is malformed.")
    if file_identity.get("sha256") != signature["test_file_sha256"]:
        raise ValueError(f"{label} split file SHA-256 does not match its evaluation_signature.")
    if type(file_identity.get("bytes")) is not int or file_identity["bytes"] <= 0:
        raise ValueError(f"{label} dataset manifest split identity must report a positive byte count.")
    if type(file_identity.get("rows")) is not int or file_identity["rows"] != EXPECTED_ROWS:
        raise ValueError(f"{label} dataset manifest split identity must contain {EXPECTED_ROWS} rows.")

    return {
        "accuracy": _require_probability(report.get("accuracy"), f"{label} accuracy"),
        "checkpoint_sha256": observed_checkpoint_sha256,
        "macro_f1": _require_probability(report.get("macro_f1"), f"{label} macro_f1"),
        "signature": signature,
    }


def _signature_without_split(signature: Mapping) -> dict:
    return {name: value for name, value in signature.items() if name not in _SIGNATURE_SPLIT_FIELDS}


def _validate_matching_signatures(evaluations: Mapping[str, Mapping[str, dict]]) -> dict:
    validation_signatures = {name: value["validation"]["signature"] for name, value in evaluations.items()}
    test_signatures = {name: value["test"]["signature"] for name, value in evaluations.items()}
    reference_name = sorted(evaluations)[0]
    validation_reference = validation_signatures[reference_name]
    test_reference = test_signatures[reference_name]
    if any(signature != validation_reference for signature in validation_signatures.values()):
        raise ValueError("Validation evaluation_signature values do not match across all models.")
    if any(signature != test_reference for signature in test_signatures.values()):
        raise ValueError("Official-test evaluation_signature values do not match across all models.")
    if _signature_without_split(validation_reference) != _signature_without_split(test_reference):
        raise ValueError("Validation and official-test evaluation settings do not match apart from split identity.")
    return {"validation": validation_reference, "test": test_reference}


def _public_metrics(evaluation: dict) -> dict:
    return {name: evaluation[name] for name in ("checkpoint_sha256", *_METRICS)}


def _population_summary(values: Sequence[float]) -> dict:
    mean = math.fsum(values) / len(values)
    population_variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return {
        "max": max(values),
        "mean": mean,
        "min": min(values),
        "population_std": math.sqrt(population_variance),
    }


def _metric_delta(left: Mapping[str, float], right: Mapping[str, float]) -> dict:
    return {metric: left[metric] - right[metric] for metric in _METRICS}


def _fl_deltas(
    fl_evaluations: Mapping[str, dict], local_sites: Mapping[int, Mapping[str, dict]], best_site: int
) -> dict:
    result = {}
    for split_role in ("validation", "test"):
        fl_metrics = fl_evaluations[split_role]
        local_mean = {
            metric: math.fsum(site[split_role][metric] for site in local_sites.values()) / len(local_sites)
            for metric in _METRICS
        }
        result[split_role] = {
            "minus_each_site": {
                str(site_id): _metric_delta(fl_metrics, site[split_role])
                for site_id, site in sorted(local_sites.items())
            },
            "minus_equal_site_mean": _metric_delta(fl_metrics, local_mean),
            "minus_validation_selected_best": _metric_delta(fl_metrics, local_sites[best_site][split_role]),
        }
    return result


def _accuracy_gap(accuracy: float) -> float:
    return accuracy - BIONEMO_ACCURACY_REFERENCE


def summarize(args: argparse.Namespace) -> dict:
    campaign_manifest = _load_campaign_manifest(args.campaign_manifest)
    site_validation_paths = _parse_site_paths(args.site_validation, "--site-validation")
    site_test_paths = _parse_site_paths(args.site_test, "--site-test")
    checkpoint_sha256 = campaign_manifest["checkpoint_sha256"]

    reference_paths = {
        "initialization": {
            "validation": Path(args.initialization_validation).expanduser().resolve(),
            "test": Path(args.initialization_test).expanduser().resolve(),
        },
        "primary_fl_final": {
            "validation": Path(args.primary_fl_validation).expanduser().resolve(),
            "test": Path(args.primary_fl_test).expanduser().resolve(),
        },
        "secondary_selected_fl": {
            "validation": Path(args.secondary_selected_fl_validation).expanduser().resolve(),
            "test": Path(args.secondary_selected_fl_test).expanduser().resolve(),
        },
    }
    all_paths = [*site_validation_paths.values(), *site_test_paths.values()]
    all_paths.extend(path for split_paths in reference_paths.values() for path in split_paths.values())
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("Every validation and official-test input must reference a distinct JSON file.")

    evaluations = {}
    for role, split_paths in reference_paths.items():
        evaluations[role] = {}
        for split_role, path in split_paths.items():
            evaluations[role][split_role] = _validate_evaluation(
                _load_json(path, f"{role} {split_role} evaluation"),
                label=f"{role} {split_role} evaluation",
                split_role=split_role,
                checkpoint_sha256=checkpoint_sha256[role],
            )

    local_sites = {}
    for site_id in EXPECTED_SITE_IDS:
        local_sites[site_id] = {}
        for split_role, path in (
            ("validation", site_validation_paths[site_id]),
            ("test", site_test_paths[site_id]),
        ):
            label = f"site {site_id} {split_role} evaluation"
            local_sites[site_id][split_role] = _validate_evaluation(
                _load_json(path, label),
                label=label,
                split_role=split_role,
                checkpoint_sha256=checkpoint_sha256["local_sites"][str(site_id)],
            )
        evaluations[f"site_{site_id}"] = local_sites[site_id]

    evaluation_signatures = _validate_matching_signatures(evaluations)
    best_site = min(
        EXPECTED_SITE_IDS,
        key=lambda site_id: (
            -local_sites[site_id]["validation"]["macro_f1"],
            -local_sites[site_id]["validation"]["accuracy"],
            site_id,
        ),
    )

    equal_site_summary = {}
    for split_role in ("validation", "test"):
        equal_site_summary[split_role] = {
            metric: _population_summary([local_sites[site_id][split_role][metric] for site_id in EXPECTED_SITE_IDS])
            for metric in _METRICS
        }

    public_local_sites = {
        str(site_id): {
            "site_id": site_id,
            "test": _public_metrics(local_sites[site_id]["test"]),
            "validation": _public_metrics(local_sites[site_id]["validation"]),
        }
        for site_id in EXPECTED_SITE_IDS
    }
    test_mean_accuracy = equal_site_summary["test"]["accuracy"]["mean"]
    result = {
        "bionemo_accuracy": {
            "gap_definition": "observed test accuracy minus BioNeMo reference accuracy",
            "observed_minus_reference": {
                "initialization": _accuracy_gap(evaluations["initialization"]["test"]["accuracy"]),
                "local_equal_site_mean": _accuracy_gap(test_mean_accuracy),
                "local_sites": {
                    str(site_id): _accuracy_gap(local_sites[site_id]["test"]["accuracy"])
                    for site_id in EXPECTED_SITE_IDS
                },
                "local_validation_selected_best": _accuracy_gap(local_sites[best_site]["test"]["accuracy"]),
                "primary_fl_final": _accuracy_gap(evaluations["primary_fl_final"]["test"]["accuracy"]),
                "secondary_selected_fl": _accuracy_gap(evaluations["secondary_selected_fl"]["test"]["accuracy"]),
            },
            "reference_accuracy": BIONEMO_ACCURACY_REFERENCE,
        },
        "checkpoint_sha256": checkpoint_sha256,
        "deltas": {
            "primary_fl_final": _fl_deltas(evaluations["primary_fl_final"], local_sites, best_site),
            "secondary_selected_fl": _fl_deltas(evaluations["secondary_selected_fl"], local_sites, best_site),
        },
        "evaluation_signatures": evaluation_signatures,
        "format_version": FORMAT_VERSION,
        "initialization": {
            split_role: _public_metrics(evaluations["initialization"][split_role])
            for split_role in ("validation", "test")
        },
        "local": {
            "best_site": {
                "selection_rule": "highest validation macro_f1, then accuracy, then lowest site ID",
                "site_id": best_site,
            },
            "equal_site_summary": equal_site_summary,
            "sites": public_local_sites,
        },
        "primary_fl_final": {
            split_role: _public_metrics(evaluations["primary_fl_final"][split_role])
            for split_role in ("validation", "test")
        },
        "protocol": campaign_manifest["protocol"],
        "resources": campaign_manifest["resources"],
        "secondary_selected_fl": {
            split_role: _public_metrics(evaluations["secondary_selected_fl"][split_role])
            for split_role in ("validation", "test")
        },
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, output_path)
    return result


def main(argv: list[str] | None = None) -> None:
    args = define_parser().parse_args(argv)
    result = summarize(args)
    print(
        f"best_local_site={result['local']['best_site']['site_id']}, "
        f"primary_fl_accuracy={result['primary_fl_final']['test']['accuracy']:.6f}, "
        f"local_mean_accuracy={result['local']['equal_site_summary']['test']['accuracy']['mean']:.6f}, "
        f"report={Path(args.output).expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
