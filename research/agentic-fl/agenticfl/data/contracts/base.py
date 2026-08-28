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

"""Shared prepared-data contract helpers.

The core harness knows only a few generic task-family contracts. Agents own the
client-local code that maps real datasets into these contracts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping

STANDARD_SPLITS = ("train", "validation", "test")
SEGMENTATION = "segmentation"
CLASSIFICATION = "classification"
OBJECT_DETECTION = "object_detection"
GENERATED_DATA_CONTRACT_SCHEMA_VERSION = "agenticfl.generated_data_contract.v1"


@dataclass(frozen=True)
class DataContract:
    """Description of one canonical prepared-data contract."""

    name: str
    record_type: str
    sample_manifest: str
    sample_manifest_format: str
    record_example: Mapping[str, Any]
    materialized_outputs: tuple[str, ...]
    description: str
    adapter_record_required_fields: tuple[str, ...] = ()
    adapter_record_optional_fields: tuple[str, ...] = ()
    manifest_validation: Mapping[str, Any] = field(default_factory=dict)
    visual_qc: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "record_type": self.record_type,
            "sample_manifest": self.sample_manifest,
            "sample_manifest_format": self.sample_manifest_format,
            "record_example": dict(self.record_example),
            "materialized_outputs": list(self.materialized_outputs),
            "description": self.description,
            "adapter_record_required_fields": list(self.adapter_record_required_fields),
            "adapter_record_optional_fields": list(self.adapter_record_optional_fields),
            "manifest_validation": dict(self.manifest_validation),
            "visual_qc": dict(self.visual_qc),
        }


def normalize_record_type(value: Any, *, generated_aliases: Mapping[str, str] | None = None) -> str:
    record_type = _record_type_text(value)
    if record_type in {SEGMENTATION, CLASSIFICATION}:
        return record_type
    if generated_aliases:
        canonical = generated_aliases.get(record_type)
        if canonical:
            canonical_text = _record_type_text(canonical)
            return (
                normalize_record_type(canonical_text)
                if canonical_text in {SEGMENTATION, CLASSIFICATION}
                else canonical_text
            )
    return "unknown"


def infer_record_type(record: Mapping[str, Any]) -> str:
    """Infer built-in adapter record type from required fields."""

    if "mask_path" in record:
        return SEGMENTATION
    if "label" in record:
        return CLASSIFICATION
    return "unknown"


def manifest_record_type(
    manifest: Mapping[str, Any],
    *,
    generated_contract: Mapping[str, Any] | None = None,
) -> str:
    generated_type = generated_contract_record_type(generated_contract)
    aliases = generated_record_type_aliases(generated_contract, canonical=generated_type)
    declared = normalize_record_type(manifest.get("record_type"), generated_aliases=aliases)
    if declared != "unknown":
        return declared
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        return "unknown"
    inferred = {infer_record_type(record) for record in records if isinstance(record, Mapping)}
    inferred.discard("unknown")
    return inferred.pop() if len(inferred) == 1 else "unknown"


def generated_contract_record_type(contract: Mapping[str, Any] | None) -> str:
    """Return the contract-owned generated record family, if one was declared."""

    if not isinstance(contract, Mapping):
        return "unknown"
    for key in ("runtime_record_type", "runtime_executor", "record_family", "record_type"):
        direct = normalize_record_type(contract.get(key))
        if direct != "unknown":
            return direct
        text = _record_type_text(contract.get(key))
        if text:
            return text
    return "unknown"


def generated_record_type_aliases(
    contract: Mapping[str, Any] | None,
    *,
    canonical: str | None = None,
) -> dict[str, str]:
    if not isinstance(contract, Mapping):
        return {}
    resolved = canonical or generated_contract_record_type(contract)
    if resolved == "unknown":
        return {}
    aliases: dict[str, str] = {}
    raw_aliases = contract.get("record_type_aliases")
    if isinstance(raw_aliases, Mapping):
        for alias, value in raw_aliases.items():
            aliases[_record_type_text(alias)] = _record_type_text(value)
    elif isinstance(raw_aliases, list):
        for alias in raw_aliases:
            aliases[_record_type_text(alias)] = resolved
    declared = _record_type_text(contract.get("record_type"))
    if declared:
        aliases[declared] = resolved
    return {key: value for key, value in aliases.items() if key and value}


def generated_contract_field_names(contract: Mapping[str, Any] | None) -> dict[str, set[str]]:
    fields = contract.get("record_fields") if isinstance(contract, Mapping) else None
    required = _field_name_set(fields.get("required") if isinstance(fields, Mapping) else None)
    optional = _field_name_set(fields.get("optional") if isinstance(fields, Mapping) else None)
    if isinstance(fields, Mapping):
        declared = _field_name_set(fields.get("field_requirements"))
        optional.update(declared - required)
    if isinstance(contract, Mapping):
        required.update(_field_name_set(contract.get("adapter_record_required_fields")))
        optional.update(_field_name_set(contract.get("adapter_record_optional_fields")))
        example = contract.get("record_example")
        if isinstance(example, Mapping):
            optional.update(_record_type_text(key) for key in example)
        sample_fields = contract.get("sample_fields")
        if isinstance(sample_fields, list):
            optional.update(_record_type_text(item) for item in sample_fields if _record_type_text(item))
    return {"required": required, "optional": optional, "all": required | optional}


def generated_contract_materialized_field_names(
    contract: Mapping[str, Any] | None,
) -> dict[str, set[str]]:
    if not isinstance(contract, Mapping) or "materialized_record_fields" not in contract:
        return generated_contract_field_names(contract)
    fields = contract.get("materialized_record_fields")
    if not isinstance(fields, Mapping):
        return {"required": set(), "optional": set(), "all": set()}
    required = _field_name_set(fields.get("required"))
    optional = _field_name_set(fields.get("optional"))
    declared = _field_name_set(fields.get("field_requirements"))
    optional.update(declared - required)
    return {"required": required, "optional": optional, "all": required | optional}


def generated_contract_box_field(contract: Mapping[str, Any] | None) -> str:
    names = generated_contract_field_names(contract)["all"]
    for candidate in ("boxes_xyxy", "boxes", "bboxes", "bounding_boxes", "bbox", "box"):
        if candidate in names:
            return candidate
    return ""


def generated_contract_label_field(contract: Mapping[str, Any] | None) -> str:
    names = generated_contract_field_names(contract)["all"]
    for candidate in (
        "labels",
        "box_labels",
        "label_ids",
        "class_ids",
        "category_ids",
        "classes",
        "class_labels",
        "category",
        "class",
    ):
        if candidate in names:
            return candidate
    return ""


def generated_contract_label_ids(contract: Mapping[str, Any] | None) -> set[int]:
    if not isinstance(contract, Mapping):
        return set()
    ids: set[int] = set()
    target_space = contract.get("target_space")
    categories = target_space.get("categories") if isinstance(target_space, Mapping) else None
    if isinstance(categories, list):
        for item in categories:
            if isinstance(item, Mapping):
                _collect_generated_label_id(ids, item.get("id"))
    for key in ("class_map", "labels", "label_space", "classes"):
        value = contract.get(key)
        if isinstance(value, Mapping):
            for raw_key, raw_value in value.items():
                _collect_generated_label_id(ids, raw_key)
                _collect_generated_label_id(ids, raw_value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    _collect_generated_label_id(ids, item.get("id"))
                else:
                    _collect_generated_label_id(ids, item)
    return ids


def _collect_generated_label_id(ids: set[int], value: Any) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        ids.add(int(value))
    elif isinstance(value, str) and value.strip().isdigit():
        ids.add(int(value.strip()))


def generated_contract_bbox_format(contract: Mapping[str, Any] | None) -> str:
    if not isinstance(contract, Mapping):
        return ""
    for key in ("bbox_format", "box_format"):
        value = contract.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    fields = contract.get("record_fields")
    record_shape = contract.get("record_shape") if isinstance(contract, Mapping) else None
    descriptions = [
        _generated_contract_field_description(fields, generated_contract_box_field(contract)),
        _generated_contract_field_description(fields, "bbox_format"),
        _generated_contract_field_description(fields, "box_format"),
        (
            str(record_shape.get(generated_contract_box_field(contract)) or "")
            if isinstance(record_shape, Mapping)
            else ""
        ),
        str(record_shape.get("bbox_format") or "") if isinstance(record_shape, Mapping) else "",
    ]
    if "xyxy" in " ".join(descriptions).lower():
        return "xyxy_absolute_pixels"
    return ""


def generated_contract_visual_qc_required(
    contract: Mapping[str, Any] | None,
) -> bool:
    """Return whether any supported generated-contract QC declaration requires review."""

    if not isinstance(contract, Mapping):
        return False
    qc_requirements = contract.get("qc_requirements")
    candidates = [contract.get("visual_qc"), qc_requirements]
    if isinstance(qc_requirements, Mapping):
        candidates.append(qc_requirements.get("visual_qc"))
    return any(isinstance(candidate, Mapping) and candidate.get("required") is True for candidate in candidates)


def generated_data_contract_validation_errors(contract: Any) -> list[str]:
    """Validate only the executable envelope of an agent-owned data contract."""

    if not isinstance(contract, Mapping):
        return ["generated_data_contract must be a JSON object"]
    errors: list[str] = []
    if contract.get("schema_version") != GENERATED_DATA_CONTRACT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {GENERATED_DATA_CONTRACT_SCHEMA_VERSION}")
    if generated_contract_record_type(contract) == "unknown":
        errors.append("record_type is required")

    sample_manifest = str(contract.get("sample_manifest") or contract.get("samples_file") or "").strip()
    if not sample_manifest:
        errors.append("sample_manifest is required")
    elif not _is_safe_generated_contract_relative_path(sample_manifest):
        errors.append("sample_manifest must be a relative path contained in the client output folder")
    if not str(contract.get("sample_manifest_format") or "").strip():
        errors.append("sample_manifest_format is required")

    required_fields = generated_contract_field_names(contract)["required"]
    if not required_fields:
        errors.append("record_fields.required must declare the adapter record shape")
    elif not ({"image", "image_path"} & required_fields):
        errors.append("required record fields must include image or image_path")

    if "materialized_record_fields" in contract:
        materialized_required = generated_contract_materialized_field_names(contract)["required"]
        if not materialized_required:
            errors.append("materialized_record_fields.required must declare the prepared sample shape")
        elif not ({"image", "image_path"} & materialized_required):
            errors.append("materialized required record fields must include image or image_path")
    return errors


def _is_safe_generated_contract_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value.strip()) and not path.is_absolute() and ".." not in path.parts


def datalist_split_key(split: Any) -> str:
    value = str(split or "train")
    return {"train": "training", "validation": "validation", "test": "test"}.get(value, "training")


def task_family_from_policy(policy: Mapping[str, Any], source_label_type: str = "") -> str:
    generated_type = generated_contract_record_type(policy.get("generated_data_contract"))
    if generated_type != "unknown":
        return generated_type
    site_mapping = policy.get("site_label_mapping")
    site_mapping = site_mapping if isinstance(site_mapping, Mapping) else policy
    candidates = [
        source_label_type,
        str(policy.get("task") or ""),
        str(policy.get("task_description") or ""),
        str(site_mapping.get("canonical_task") or ""),
    ]
    text = " ".join(candidates).lower()
    tokens = set(_tokens(text))
    detection_terms = {"detection", "detector", "detect", "bbox", "box", "boxes", "bounding", "rcnn", "r-cnn", "yolo"}
    if tokens & detection_terms or "bounding box" in text or "r-cnn" in text:
        return "unknown"
    if tokens & {"classification", "classify", "diagnosis", "diagnostic", "disease", "label"}:
        return CLASSIFICATION
    if "class" in source_label_type.lower() or "diagnos" in source_label_type.lower():
        return CLASSIFICATION
    if tokens & {"segmentation", "segment", "mask", "contour", "boundary"}:
        return SEGMENTATION
    return "unknown"


def available_contract_summaries() -> dict[str, dict[str, Any]]:
    from agenticfl.data.contracts import classification, segmentation

    return {
        SEGMENTATION: segmentation.CONTRACT.summary(),
        CLASSIFICATION: classification.CONTRACT.summary(),
    }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.replace("_", " ").replace("-", " ").lower())


def _record_type_text(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value or "").strip().lower().replace("-", "_"))


def _field_name_set(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        name = _field_definition_name(value)
        if name:
            return {name}
        return {_record_type_text(key) for key in value.keys() if _record_type_text(key)}
    if isinstance(value, list):
        names: set[str] = set()
        for item in value:
            if isinstance(item, Mapping):
                names.update(_field_name_set(item))
            else:
                name = _record_type_text(item)
                if name:
                    names.add(name)
        return names
    return set()


def _field_definition_name(value: Mapping[str, Any]) -> str:
    for key in ("name", "field", "field_name"):
        name = _record_type_text(value.get(key))
        if name:
            return name
    return ""


def _generated_contract_field_description(fields: Any, field_name: str) -> str:
    if not field_name or not isinstance(fields, Mapping):
        return ""
    parts: list[str] = []
    for section_name in ("required", "optional", "field_requirements", "field_definitions"):
        section = fields.get(section_name)
        if isinstance(section, Mapping):
            value = section.get(field_name)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, Mapping):
                parts.extend(str(value.get(key) or "") for key in ("type", "description", "format"))
        elif isinstance(section, list):
            for item in section:
                if isinstance(item, Mapping) and _field_definition_name(item) == field_name:
                    parts.extend(str(item.get(key) or "") for key in ("type", "description", "format"))
    return " ".join(part for part in parts if part)


def _explicit_record_type_alias(value: Any) -> str:
    text = _record_type_text(value)
    if text.startswith(f"{OBJECT_DETECTION}_"):
        return OBJECT_DETECTION
    return "unknown"
