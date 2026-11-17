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

"""Prepared-data configuration, manifests, safe result summaries, and source fingerprints."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Any

from fedready.data.qc import VISUAL_QC_BUNDLE_SCHEMA, VISUAL_QC_TRANSFORMS, _safe_visual_qc_transform_candidates
from fedready.utils.logging import payload_digest

try:  # Pillow is optional outside image-processing workflows.
    from PIL import ImageStat
except ImportError:  # pragma: no cover - minimal environments
    ImageStat = None  # type: ignore[assignment]


EXTRACTION_RESULT_SCHEMA = "fedready.local_extraction_result.v1"
TRAINING_TRANSFORM_SCHEMA = "fedready.training_transform_policy.v1"


def _extractor_logic_fingerprint() -> str:
    """Fingerprint extraction sources so cache invalidation cannot rely on a manual bump."""

    package_root = Path(__file__).resolve().parents[1]
    sources = (
        Path(__file__).resolve(),
        package_root / "data" / "extractor.py",
        package_root / "data" / "materializer.py",
        package_root / "data" / "visual_qc.py",
        package_root / "data" / "splits.py",
        package_root / "data" / "parser.py",
        package_root / "data" / "contracts" / "base.py",
        package_root / "data" / "contracts" / "segmentation.py",
        package_root / "data" / "contracts" / "classification.py",
        package_root / "agents" / "local_adapter.py",
        package_root / "data" / "qc.py",
    )
    digest = sha256()
    for source in sources:
        digest.update(source.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return f"source-sha256:{digest.hexdigest()}"


EXTRACTOR_LOGIC_VERSION = _extractor_logic_fingerprint()


@dataclass(frozen=True)
class ExtractionConfig:
    """Controls local extraction output and scan bounds."""

    output_root: str = "data/dataset_fl"
    output_name: str | None = None
    max_samples: int | None = None
    overwrite: bool = False
    validation_fraction: float = 0.1
    split_seed: int = 20260603

    def __post_init__(self) -> None:
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be at least 1 when provided")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")


def _extraction_result(
    manifest: dict[str, Any],
    *,
    label_rule_applied: dict[str, Any],
    warnings: list[str],
    extraction_details: dict[str, Any],
) -> dict[str, Any]:
    """Serialize verifier-owned facts; never copy the private manifest wholesale."""
    result = {
        key: manifest[key]
        for key in (
            "client_id",
            "policy_digest",
            "source_label_type",
            "record_type",
            "local_adapter",
            "conversion_options",
            "counts",
            "failed_pairs",
            "split_plan",
            "screening",
            "verification",
        )
    }
    result.update(
        {
            "schema_version": EXTRACTION_RESULT_SCHEMA,
            "data": _extraction_status(
                extracted_count=manifest["counts"]["total"],
                screening=manifest["screening"],
                verification=manifest["verification"],
            ),
            "label_orientation": {
                "selected_transform": manifest["label_orientation"]["selected_transform"],
                "strategy": manifest["label_orientation"]["strategy"],
                "safe_to_share": True,
            },
            "extraction": {
                "output_root": manifest["output"]["root"],
                "client_folder": manifest["client_id"],
                "output_name": manifest["output"]["run_label"],
                "local_output_path_redacted": True,
                "reused_existing_output": False,
                "visual_qc_available": bool(manifest["visual_qc"].get("available")),
                "target_size": manifest["target_size"],
                "stored_resolution": manifest["stored_resolution"],
                "training_transform_policy": manifest["training_transform_policy"],
                "sample_manifest": manifest["sample_manifest"],
                "sample_manifest_format": manifest["sample_manifest_format"],
                "classification_storage": manifest.get("classification_storage"),
                "image_rule_applied": {
                    "channels": "RGB",
                    "resize": "deferred_to_training_time",
                    "extraction_resize_applied": False,
                },
                "label_rule_applied": label_rule_applied,
                **extraction_details,
            },
            "privacy": {
                "safe_to_share": True,
                "redacted": [
                    "source_local_paths",
                    "source_filenames",
                    "extracted_local_paths",
                    "extracted_filenames",
                    "sample_ids",
                    "raw_images",
                    "raw_masks",
                    "raw_labels",
                    "raw_annotations",
                    "visual_qc_artifact_paths",
                ],
            },
            "visual_qc_artifacts": _visual_qc_safe_summary(manifest["visual_qc"]),
            "warnings": warnings,
        }
    )
    return result


def _failed_extraction_result(
    *,
    client_id: str,
    policy: dict[str, Any],
    source_label_type: str,
    adapter_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Common failed-result envelope; callers supply screening and error details."""
    return {
        "schema_version": EXTRACTION_RESULT_SCHEMA,
        "client_id": client_id,
        "data": "extraction verification failed",
        "policy_digest": policy.get("strategy_digest") or payload_digest(policy),
        "source_label_type": source_label_type,
        "verification": {
            "schema_version": "fedready.local_extraction_verification.v1",
            "valid_case_count": 0,
            "expected_extracted_count": 0,
            "extracted_count": 0,
            "failed_pair_count": 0,
            "matches_execution_target": False,
            "matches_all_valid_cases": False,
            "passed": False,
        },
        "local_adapter": adapter_summary,
        "privacy": {
            "safe_to_share": True,
            "redacted": ["source_local_paths", "source_filenames", "raw_images", "raw_masks", "raw_annotations"],
        },
    }


def _extraction_verification(
    *,
    full_valid_case_count: int,
    expected_extracted_count: int,
    extracted_count: int,
    failed_pair_count: int,
    max_samples: int | None,
    target_terms: set[str],
) -> dict[str, Any]:
    sample_limited = max_samples is not None and full_valid_case_count > max_samples
    passed = extracted_count == expected_extracted_count and failed_pair_count == 0
    return {
        "schema_version": "fedready.local_extraction_verification.v1",
        "target_concepts": sorted(target_terms),
        "scope": "bounded_sample" if sample_limited else "full_valid_cases",
        "sample_limited": sample_limited,
        "max_samples": max_samples,
        "valid_case_count": full_valid_case_count,
        "expected_extracted_count": expected_extracted_count,
        "extracted_count": extracted_count,
        "failed_pair_count": failed_pair_count,
        "matches_execution_target": passed,
        "matches_all_valid_cases": extracted_count == full_valid_case_count and failed_pair_count == 0,
        "passed": passed,
    }


def _extraction_screening(
    *,
    record_count: int,
    valid_case_count: int,
    execution_pair_count: int,
    target_terms: set[str],
) -> dict[str, Any]:
    status = "usable"
    reason_code = None
    reason = "Local label validation found usable image-label pairs for the extraction policy."
    if record_count == 0:
        status = "screened_out"
        reason_code = "NO_USABLE_LABEL_RECORDS"
        reason = "No local label records matched the round-2 extraction policy during local validation."
    elif valid_case_count == 0:
        status = "screened_out"
        reason_code = "NO_VALID_CASES_AFTER_SPLIT_RULE"
        reason = "Local image-label pairs existed, but none remained after local split-rule validation."

    return {
        "schema_version": "fedready.local_label_screening.v1",
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
        "target_concepts": sorted(target_terms),
        "record_count": record_count,
        "valid_case_count": valid_case_count,
        "execution_pair_count": execution_pair_count,
        "safe_to_share": True,
    }


def _extraction_status(
    *,
    extracted_count: int,
    screening: dict[str, Any],
    verification: dict[str, Any],
) -> str:
    if screening.get("status") == "screened_out":
        return "screened out"
    if verification.get("passed") and extracted_count > 0:
        return "extracted"
    if verification.get("passed") and extracted_count == 0:
        return "no valid cases"
    return "extraction verification failed"


def _local_adapter_safe_summary(
    local_adapter: dict[str, Any] | None,
    *,
    adapter_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    status = local_adapter.get("status") if isinstance(local_adapter, dict) else None
    runtime = (
        local_adapter.get("runtime_validation")
        if isinstance(local_adapter, dict) and isinstance(local_adapter.get("runtime_validation"), dict)
        else {}
    )
    visual_qc_owner = str(runtime.get("visual_qc_owner") or "")
    return {
        "schema_version": "fedready.local_adapter_summary.v1",
        "status": status or ("implemented" if adapter_manifest is not None else "not_used"),
        "adapter_kind": local_adapter.get("adapter_kind") if isinstance(local_adapter, dict) else None,
        "source_label_type": local_adapter.get("source_label_type") if isinstance(local_adapter, dict) else None,
        "manifest_available": adapter_manifest is not None,
        "record_count": len(adapter_manifest.get("records", [])) if isinstance(adapter_manifest, dict) else 0,
        "preflight_visual_review_passed": (
            runtime.get("status") == "passed" and visual_qc_owner == "adapter_preflight_local_guardrail"
        ),
        "visual_qc_owner": visual_qc_owner or None,
        "local_paths_redacted": True,
        "script_path_redacted": bool(isinstance(local_adapter, dict) and local_adapter.get("script_path")),
        "safe_to_share": True,
    }


def _extraction_policy_digest(policy: dict[str, Any]) -> str:
    """Digest only fields that affect local PNG extraction."""

    image_rule = policy.get("image_rule", {})
    label_rule = policy.get("label_rule", {})
    split_rule = policy.get("split_rule", {})
    digest_body = {
        "schema_version": "fedready.extraction_policy_digest.v1",
        "client_id": policy.get("client_id"),
        "task": policy.get("task"),
        "site_label_mapping": _site_mapping(policy),
        "label_rule": label_rule if isinstance(label_rule, dict) else {},
        "split_rule": split_rule if isinstance(split_rule, dict) else {},
        "image_extraction_rule": {
            "channels": (image_rule.get("channels") if isinstance(image_rule, dict) else None)
            or "convert_to_rgb_if_needed",
            "stored_resolution": "source_image_resolution",
            "resize": "deferred_to_training_time",
        },
    }
    return payload_digest(digest_body)


def _new_intensity_accumulator() -> dict[str, Any]:
    return {
        "image_count": 0,
        "pixel_count": 0,
        "channel_sum": [0.0, 0.0, 0.0],
        "channel_sum_sq": [0.0, 0.0, 0.0],
    }


def _update_intensity_accumulator(accumulator: dict[str, Any], image: Any) -> None:
    stat = ImageStat.Stat(image.convert("RGB"))
    count = int(stat.count[0]) if stat.count else 0
    if count <= 0:
        return
    accumulator["image_count"] += 1
    accumulator["pixel_count"] += count
    for index in range(3):
        accumulator["channel_sum"][index] += float(stat.sum[index])
        accumulator["channel_sum_sq"][index] += float(stat.sum2[index])


def _finalize_intensity_accumulator(accumulator: dict[str, Any]) -> dict[str, Any]:
    pixel_count = int(accumulator.get("pixel_count", 0))
    if pixel_count <= 0:
        return {
            "schema_version": "fedready.local_image_intensity_stats.v1",
            "available": False,
            "reason": "no successfully extracted images",
            "safe_to_share": False,
        }

    sums = accumulator["channel_sum"]
    sums_sq = accumulator["channel_sum_sq"]
    mean_0_255 = [sums[index] / pixel_count for index in range(3)]
    std_0_255 = [sqrt(max((sums_sq[index] / pixel_count) - (mean_0_255[index] ** 2), 0.0)) for index in range(3)]
    return {
        "schema_version": "fedready.local_image_intensity_stats.v1",
        "available": True,
        "image_count": int(accumulator.get("image_count", 0)),
        "pixel_count": pixel_count,
        "channels": ["R", "G", "B"],
        "channel_mean_0_255": [round(value, 4) for value in mean_0_255],
        "channel_std_0_255": [round(value, 4) for value in std_0_255],
        "channel_mean_0_1": [round(value / 255.0, 6) for value in mean_0_255],
        "channel_std_0_1": [round(value / 255.0, 6) for value in std_0_255],
        "safe_to_share": False,
    }


def _write_training_transform_policy(
    *,
    output_dir: Path,
    client_id: str,
    policy: dict[str, Any],
    target_size: tuple[int, int],
    image_intensity: dict[str, Any],
    counts: dict[str, Any],
    label_rule_applied: dict[str, Any] | None = None,
    storage_sections: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_rule = policy.get("image_rule", {})
    split_rule = policy.get("split_rule", {})
    image_rule = image_rule if isinstance(image_rule, dict) else {}
    split_rule = split_rule if isinstance(split_rule, dict) else {}
    label_rule_applied = label_rule_applied if isinstance(label_rule_applied, dict) else {}
    storage_sections = storage_sections if isinstance(storage_sections, dict) else {}
    training_transform = {
        "image": {
            "input_format": "png",
            "input_mode": "RGB",
            "input_resolution": "source_image_resolution",
            "channels": "RGB",
            "dtype": image_rule.get("dtype", "float32"),
            "resize": {
                "application_time": "training_time",
                "method": "resize_with_aspect_preserving_padding",
                "target_size": list(target_size),
                "image_interpolation": "bilinear",
                "padding_value": [0, 0, 0],
                "can_update_without_reextracting_png": True,
            },
            "intensity": {
                "application_time": "training_time",
                "requested": image_rule.get("intensity", "scale_to_0_1"),
                "scale": "divide_uint8_by_255",
                "site_statistics": image_intensity,
                "can_update_without_reextracting_png": True,
            },
        },
        "label": _label_training_transform(
            label_rule_applied=label_rule_applied,
            storage_sections=storage_sections,
            target_size=target_size,
        ),
        "split_rule": split_rule,
    }
    payload = {
        "schema_version": TRAINING_TRANSFORM_SCHEMA,
        "client_id": client_id,
        "policy_digest": policy.get("strategy_digest") or payload_digest(policy),
        "extraction_policy_digest": _extraction_policy_digest(policy),
        "counts": counts,
        "storage_contract": _training_storage_contract(storage_sections),
        "training_transform": training_transform,
        "privacy": {
            "local_file": True,
            "safe_to_share": False,
            "contains_local_site_intensity_statistics": bool(image_intensity.get("available")),
        },
    }
    (output_dir / "training_transforms.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def _label_training_transform(
    *,
    label_rule_applied: dict[str, Any],
    storage_sections: dict[str, Any],
    target_size: tuple[int, int],
) -> dict[str, Any]:
    if storage_sections.get("classification_storage") is not None:
        return {
            "target_kind": "classification",
            "canonical_labels": label_rule_applied.get("canonical_labels", {}),
            "dtype": "int64",
            "resize": None,
        }
    object_detection_storage = storage_sections.get("object_detection_storage")
    if object_detection_storage is not None:
        return {
            "target_kind": "object_detection",
            "bbox_format": label_rule_applied.get("bbox_format")
            or (object_detection_storage.get("box_format") if isinstance(object_detection_storage, dict) else None),
            "canonical_labels": label_rule_applied.get("canonical_labels", {}),
            "dtype": {"boxes": "float32", "labels": "int64"},
            "resize": {
                "application_time": "training_time",
                "method": "training_code_must_update_geometry_consistently",
                "target_size": list(target_size),
                "can_update_without_reextracting_png": True,
            },
        }
    return {
        "target_kind": "segmentation",
        "input_format": "png",
        "input_mode": "L",
        "input_resolution": "matched_to_source_image_resolution",
        "canonical_labels": label_rule_applied.get("canonical_labels", {}),
        "ignore_label": label_rule_applied.get("ignore_label", 255),
        "mask_dtype": label_rule_applied.get("mask_dtype", "uint8"),
        "resize": {
            "application_time": "training_time",
            "method": "resize_with_aspect_preserving_padding",
            "target_size": list(target_size),
            "mask_interpolation": "nearest",
            "padding_value": 0,
            "can_update_without_reextracting_png": True,
        },
    }


def _training_storage_contract(storage_sections: dict[str, Any]) -> dict[str, Any]:
    labels: dict[str, Any]
    if storage_sections.get("classification_storage") is not None:
        labels = {"kind": "classification", "storage": storage_sections.get("classification_storage")}
    elif storage_sections.get("object_detection_storage") is not None:
        labels = {"kind": "object_detection", "storage": storage_sections.get("object_detection_storage")}
    else:
        labels = {"kind": "segmentation", "storage": storage_sections.get("mask_storage")}
    return {
        "images": "RGB PNG, source image resolution",
        "labels": labels,
        "heavy_image_updates_deferred_to_training": True,
    }


def _extraction_output_dir(
    *,
    client_id: str,
    output_root: str,
    project_root: str | Path | None,
) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        base = Path(project_root).resolve() if project_root is not None else Path.cwd()
        root = base / root
    return root / client_id


def _visual_qc_safe_summary(visual_qc: dict[str, Any]) -> dict[str, Any]:
    artifacts = visual_qc.get("artifacts")
    sample_count = sum(1 for artifact in artifacts if isinstance(artifact, dict)) if isinstance(artifacts, list) else 0
    requested_sample_count = visual_qc.get("requested_sample_count")
    if (
        not isinstance(requested_sample_count, int)
        or isinstance(requested_sample_count, bool)
        or requested_sample_count < 0
    ):
        requested_sample_count = None
    reviewer = visual_qc.get("reviewer")
    if reviewer not in {"client_agent_visual_review", "not_applicable_classification"}:
        reviewer = "client_agent_visual_review"
    return {
        "schema_version": VISUAL_QC_BUNDLE_SCHEMA,
        "available": visual_qc.get("available") is True and sample_count > 0,
        "sample_count": sample_count,
        "requested_sample_count": requested_sample_count,
        "review_required": bool(visual_qc.get("review_required")),
        "visual_qc_owner": str(visual_qc.get("visual_qc_owner") or "client_local_visual_qc"),
        "reviewer": reviewer,
        "transform_candidates": _safe_visual_qc_transform_candidates(
            visual_qc.get("transform_candidates"),
            default=VISUAL_QC_TRANSFORMS,
        ),
        "local_artifact_paths_redacted": True,
    }


def _generated_data_contract(policy: dict[str, Any]) -> dict[str, Any] | None:
    value = policy.get("generated_data_contract")
    return value if isinstance(value, dict) else None


def _site_mapping(policy: dict[str, Any]) -> dict[str, Any]:
    site_mapping = policy.get("site_label_mapping")
    if isinstance(site_mapping, dict):
        return site_mapping
    if "source_label_type" in policy:
        return policy
    raise ValueError("extraction policy missing site_label_mapping")


def _safe_slug(text: str) -> str:
    slug = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.lower())).strip("_")
    return slug or "extracted"


def _dedupe(values: list[str] | tuple[str, ...]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
