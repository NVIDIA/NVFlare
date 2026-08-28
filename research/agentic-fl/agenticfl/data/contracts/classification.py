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

"""Canonical image-level classification prepared-data contract."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

from agenticfl.data.contracts.base import CLASSIFICATION, STANDARD_SPLITS, DataContract, datalist_split_key
from agenticfl.data.parser import _infer_split
from agenticfl.utils.logging import payload_digest
from PIL import Image

SAMPLE_MANIFEST = "samples_classification.json"
SAMPLE_MANIFEST_FORMAT = "monai_decathlon_datalist_json"
SPLIT_ROOTS = ("images",)
RECORD_EXAMPLE = {
    "image_path": "absolute_or_manifest_relative_image_path",
    "label_source_path": "absolute_or_manifest_relative_existing_metadata_file_or_class_folder",
    "label": "non-negative integer class index, e.g. 0 or 1",
    "split": "optional train|validation|test",
}

CONTRACT = DataContract(
    name="canonical_image_classification",
    record_type=CLASSIFICATION,
    sample_manifest=SAMPLE_MANIFEST,
    sample_manifest_format=SAMPLE_MANIFEST_FORMAT,
    record_example=RECORD_EXAMPLE,
    materialized_outputs=("images/<split>/*.png", SAMPLE_MANIFEST),
    description=(
        "Image-level classification tasks. The adapter provides real image paths, existing local evidence "
        "for each class label, and scalar non-negative integer labels. Opaque numeric diagnosis codes require "
        "client-local codebook, legend, schema, or class-name evidence before they can be mapped to task labels."
    ),
    adapter_record_required_fields=("image_path", "label_source_path", "label"),
    adapter_record_optional_fields=("split",),
    manifest_validation={
        "label_value": "label must be a non-negative integer after any task-required harmonization",
        "provenance": "label_source_path must point to a pre-existing client-local metadata file, class folder, legend, README, or codebook that proves the class label",
        "semantic_evidence": "opaque numeric diagnosis codes are not sufficient without an explicit local definition",
    },
    visual_qc={
        "required": False,
        "type": "not_applicable",
        "reason": "image-level scalar labels do not have an image/label overlay to inspect",
    },
)


def split_roots() -> tuple[str, ...]:
    return SPLIT_ROOTS


def manifest_pairs(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    manifest_path = Path(str(manifest.get("_manifest_path", "")))
    base_dir = manifest_path.parent if str(manifest_path) else Path.cwd()
    warnings: list[str] = []
    pairs: list[dict[str, Any]] = []
    records = manifest.get("records", [])
    if not isinstance(records, list):
        return [], ["Local adapter manifest records field was not a list."]
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            warnings.append("Local adapter manifest contained a non-object record.")
            continue
        image_path = _adapter_record_path(record.get("image_path"), base_dir)
        label_source_path = _adapter_record_path(record.get("label_source_path"), base_dir)
        label = label_value(record.get("label"))
        if image_path is None or label_source_path is None or label is None:
            warnings.append(
                "Local adapter classification record was missing image_path, label_source_path, or integer label."
            )
            continue
        if not image_path.exists() or not label_source_path.exists():
            warnings.append("Local adapter classification record referenced an unreadable local image or label source.")
            continue
        if label_source_path == image_path:
            warnings.append("Local adapter classification label source matched the image.")
            continue
        split_value = record.get("split")
        split = str(split_value) if isinstance(split_value, str) else _infer_split(image_path.as_posix())
        if split not in STANDARD_SPLITS:
            split = "unknown"
        pairs.append(
            {
                "image_path": image_path,
                "label_source_path": label_source_path,
                "label": label,
                "split": split,
                "adapter_record_index": index,
                "stable_key": payload_digest(
                    {
                        "image_path": image_path.as_posix(),
                        "label_source_path": label_source_path.as_posix(),
                        "label": label,
                        "index": index,
                    }
                ),
            }
        )
    return pairs, warnings


def materialize_pair(
    *,
    pair: dict[str, Any],
    output_dir: Path,
    source_label_type: str,
    update_intensity: Callable[[dict[str, Any], Any], None],
    intensity_accumulator: dict[str, Any],
) -> dict[str, Any]:
    image_path = Path(pair["image_path"])
    label_source_path = Path(pair["label_source_path"])
    label = int(pair["label"])
    split = pair["split"]
    sample_id = payload_digest(
        {
            "image": image_path.as_posix(),
            "label_source": label_source_path.as_posix(),
            "label": label,
            "adapter_record_index": pair.get("adapter_record_index"),
        }
    )[:16]

    with Image.open(image_path) as source_image:
        rgb = source_image.convert("RGB")
        rgb.load()
        source_size = rgb.size

    image_rel = Path("images") / split / f"{sample_id}.png"
    (output_dir / image_rel).parent.mkdir(parents=True, exist_ok=True)
    rgb.save(output_dir / image_rel)
    update_intensity(intensity_accumulator, rgb)

    return {
        "sample_id": sample_id,
        "split": split,
        "image": image_rel.as_posix(),
        "label": label,
        "source_digest": payload_digest({"adapter_record_index": pair.get("adapter_record_index")}),
        "source_label_type": source_label_type,
        "local_adapter_applied": True,
        "transform": {
            "source_size": list(source_size),
            "stored_size": list(source_size),
            "stored_resolution": "source_image_resolution",
            "extraction_resize_applied": False,
            "training_resize": "deferred_to_training_transforms_json",
        },
    }


def write_sample_manifest(path: Path, *, rows: list[dict[str, Any]], policy: Mapping[str, Any]) -> None:
    grouped = {"training": [], "validation": [], "test": []}
    for row in rows:
        grouped[datalist_split_key(row.get("split"))].append(
            {
                "image": str(row.get("image")),
                "label": int(row.get("label", 0)),
            }
        )
    payload: dict[str, Any] = {
        "training": grouped["training"],
        "validation": grouped["validation"],
        "test": grouped["test"],
        "labels": label_mapping(policy, rows),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def orientation_rule(manifest: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": "agenticfl.local_label_orientation_rule.v1",
        "strategy": "not_applicable_classification",
        "selected_transform": "as_is",
        "reason": "Image-level classification labels do not need orientation adjustment.",
        "safe_to_share": True,
    }


def preview(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = _first_row(rows)
    if row is None:
        return {"available": False, "reason": "no extracted samples", "local_output_path_redacted": True}
    image_path = output_dir / str(row.get("image", ""))
    if not image_path.exists():
        return {"available": False, "reason": "sample image missing", "local_output_path_redacted": True}
    preview_image = output_dir / "sample_image.png"
    with Image.open(image_path) as image:
        image.convert("RGB").save(preview_image)
    return {
        "available": True,
        "image": "sample_image.png",
        "label_kind": "classification",
        "label": row.get("label"),
        "source_sample_split": row.get("split"),
        "local_output_path_redacted": True,
    }


def visual_qc_bundle(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    sample_count: int,
    schema_version: str,
    **_: Any,
) -> dict[str, Any]:
    return visual_qc_not_applicable(schema_version=schema_version, sample_count=sample_count)


def storage_sections(policy: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mask_storage": None,
        "classification_storage": storage_summary(policy, rows),
        "object_detection_storage": None,
    }


def label_rule_applied(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "canonical_labels": (
            policy.get("label_rule", {}).get("canonical_labels", {})
            if isinstance(policy.get("label_rule"), Mapping)
            else {}
        ),
        "label_kind": "classification",
        "mask_dtype": None,
        "bbox_format": None,
        "resize": None,
    }


def label_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        return int(value.strip())
    return None


def label_mapping(policy: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, str]:
    label_rule = policy.get("label_rule", {})
    canonical = label_rule.get("canonical_labels") if isinstance(label_rule, Mapping) else None
    if isinstance(canonical, Mapping):
        mapping: dict[str, str] = {}
        for name, value in canonical.items():
            if isinstance(value, int):
                mapping[str(value)] = str(name)
            elif isinstance(value, str) and re.fullmatch(r"\d+", value):
                mapping[value] = str(name)
        if mapping:
            return dict(sorted(mapping.items(), key=lambda item: int(item[0])))
    labels = sorted({int(row["label"]) for row in rows if isinstance(row.get("label"), int)})
    return {str(label): f"class_{label}" for label in labels}


def storage_summary(policy: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(int(row["label"]) for row in rows if isinstance(row.get("label"), int))
    return {
        "format": SAMPLE_MANIFEST_FORMAT,
        "sample_manifest": SAMPLE_MANIFEST,
        "record_fields": ["image", "label"],
        "labels": label_mapping(policy, rows),
        "class_counts": {str(label): count for label, count in sorted(counts.items())},
        "safe_to_share": True,
    }


def visual_qc_not_applicable(*, schema_version: str, sample_count: int) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "available": False,
        "sample_count": 0,
        "requested_sample_count": sample_count,
        "reason": "image/mask overlay QC is not applicable to image-level classification labels",
        "review_required": False,
        "reviewer": "not_applicable_classification",
        "transform_candidates": [],
        "local_output_path_redacted": True,
    }


def _adapter_record_path(value: Any, base_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _first_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda item: (
            STANDARD_SPLITS.index(item.get("split", "train")) if item.get("split") in STANDARD_SPLITS else 99,
            str(item.get("sample_id", "")),
        ),
    )[0]
