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

"""Training-phase contract helpers for canonical prepared-data contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agenticfl.data.contracts import classification, segmentation
from agenticfl.data.contracts.base import (
    CLASSIFICATION,
    SEGMENTATION,
    generated_contract_materialized_field_names,
    generated_contract_record_type,
    generated_contract_visual_qc_required,
    task_family_from_policy,
)
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class TrainingContract:
    """Task-family training contract consumed by the FL training harness."""

    contract_name: str
    record_type: str
    samples_file: str
    sample_manifest_format: str
    sample_fields: tuple[str, ...]
    record_shape: Mapping[str, Any]
    qc_contract: Mapping[str, Any]
    metric_contract: Mapping[str, Any]
    mock_preflight: Mapping[str, Any]
    runtime_checks: tuple[str, ...]

    def as_data_contract(self) -> dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "record_type": self.record_type,
            "samples_file": self.samples_file,
            "sample_manifest_format": self.sample_manifest_format,
            "sample_fields": list(self.sample_fields),
            "record_shape": dict(self.record_shape),
            "qc_contract": dict(self.qc_contract),
            "mock_preflight": _jsonable(self.mock_preflight),
        }


SEGMENTATION_TRAINING = TrainingContract(
    contract_name=segmentation.CONTRACT.name,
    record_type=SEGMENTATION,
    samples_file=segmentation.SAMPLE_MANIFEST,
    sample_manifest_format=segmentation.SAMPLE_MANIFEST_FORMAT,
    sample_fields=("image", "mask", "split"),
    record_shape={
        "image": "relative RGB PNG path under the site folder",
        "mask": "relative single-channel binary PNG path under the site folder",
        "split": "train, validation, or test",
    },
    qc_contract={
        "visual_qc_required": True,
        "visual_qc_type": "local_image_mask_alignment",
        "reason": "spatial labels require local image/label alignment QC before training",
    },
    metric_contract={
        "primary_metric": "validation_mean_dice",
        "safe_metrics": [
            "train_loss",
            "validation_mean_dice",
            "num_train_samples",
            "num_validation_samples",
            "num_steps",
        ],
        "local_metric_artifact": "agenticfl_training_metrics.jsonl",
        "local_metric_artifact_format": "jsonl",
    },
    mock_preflight={
        "image": {"format": "png", "channels": 3, "value": "random_uint8_noise"},
        "target": {
            "field": "mask",
            "format": "png",
            "channels": 1,
            "value": "random_binary_label",
            "stored_values": [0, 255],
        },
        "splits": {"train": 2, "validation": 1},
    },
    runtime_checks=(
        "manifest.json",
        segmentation.SAMPLE_MANIFEST,
        "verification.passed",
        "visual_qc.passed",
        "train_split",
    ),
)

CLASSIFICATION_TRAINING = TrainingContract(
    contract_name=classification.CONTRACT.name,
    record_type=CLASSIFICATION,
    samples_file=classification.SAMPLE_MANIFEST,
    sample_manifest_format=classification.SAMPLE_MANIFEST_FORMAT,
    sample_fields=("image", "label"),
    record_shape={
        "image": "relative RGB PNG path under the site folder",
        "label": "non-negative integer class label",
        "split_layout": "MONAI Decathlon datalist keys: training, validation, test",
    },
    qc_contract={
        "visual_qc_required": False,
        "visual_qc_type": "not_applicable",
        "reason": "scalar image-level labels do not use mask-overlay QC",
    },
    metric_contract={
        "primary_metric": "validation_accuracy",
        "safe_metrics": [
            "train_loss",
            "validation_accuracy",
            "validation_balanced_accuracy",
            "validation_auc",
            "num_train_samples",
            "num_validation_samples",
            "num_steps",
        ],
        "local_metric_artifact": "agenticfl_training_metrics.jsonl",
        "local_metric_artifact_format": "jsonl",
    },
    mock_preflight={
        "image": {"format": "png", "channels": 3, "value": "random_uint8_noise"},
        "target": {"field": "label", "value": "random_integer_class_label", "stored_values": [0, 1]},
        "splits": {"train": 4, "validation": 2},
    },
    runtime_checks=(
        "manifest.json",
        classification.SAMPLE_MANIFEST,
        "verification.passed",
        "train_split",
    ),
)


KNOWN_TRAINING_CONTRACTS = (SEGMENTATION_TRAINING, CLASSIFICATION_TRAINING)


def known_sample_manifests() -> tuple[tuple[str, str], ...]:
    return tuple((contract.samples_file, contract.sample_manifest_format) for contract in KNOWN_TRAINING_CONTRACTS)


def sample_manifest_format_for_file(samples_file: str) -> str:
    for known_file, known_format in known_sample_manifests():
        if samples_file == known_file:
            return known_format
    return ""


def contract_from_task(task: str, *, extraction_summary: Mapping[str, Any] | None = None) -> TrainingContract:
    generated = generated_training_contract_from_summary(extraction_summary)
    if generated is not None:
        return generated
    family = task_family_from_policy({"task": task})
    if family == "unknown" and extraction_summary is not None:
        family = family_from_extraction_summary(extraction_summary)
    return contract_from_record_type(family)


def contract_from_record_type(record_type: str) -> TrainingContract:
    normalized = str(record_type or "").strip().lower()
    if normalized == CLASSIFICATION:
        return CLASSIFICATION_TRAINING
    if normalized == SEGMENTATION:
        return SEGMENTATION_TRAINING
    raise ValueError(
        f"unsupported training data contract record_type={record_type!r}; "
        "run contract generation first or provide an approved generated training contract"
    )


def contract_from_plan(training_plan: Mapping[str, Any]) -> TrainingContract:
    data_contract = training_plan.get("data_contract") if isinstance(training_plan, Mapping) else None
    if not isinstance(data_contract, Mapping):
        raise ValueError("training plan missing data_contract")
    generated = contract_from_generated_data_contract(data_contract.get("generated_data_contract"))
    if generated is None and generated_contract_record_type(data_contract) != "unknown":
        generated = contract_from_generated_data_contract(data_contract)
    if generated is not None:
        return generated
    record_type = str(data_contract.get("record_type") or "").strip().lower()
    if record_type:
        return contract_from_record_type(record_type)
    samples_file = str(data_contract.get("samples_file") or "")
    if samples_file == classification.SAMPLE_MANIFEST:
        return CLASSIFICATION_TRAINING
    if samples_file == segmentation.SAMPLE_MANIFEST:
        return SEGMENTATION_TRAINING
    raise ValueError(
        f"unsupported training data contract samples_file={samples_file!r}; "
        "run contract generation first or provide an approved generated training contract"
    )


def generated_training_contract_from_summary(
    extraction_summary: Mapping[str, Any] | None,
) -> TrainingContract | None:
    if not isinstance(extraction_summary, Mapping):
        return None
    contract = extraction_summary.get("generated_data_contract")
    generated = contract_from_generated_data_contract(contract)
    if generated is not None:
        return generated
    strategy = extraction_summary.get("extraction_strategy")
    if isinstance(strategy, Mapping):
        generated = contract_from_generated_data_contract(strategy.get("generated_data_contract"))
        if generated is not None:
            return generated
    return None


def contract_from_generated_data_contract(contract: Any) -> TrainingContract | None:
    if not isinstance(contract, Mapping):
        return None
    record_type = generated_contract_record_type(contract)
    if record_type in {"unknown", SEGMENTATION, CLASSIFICATION}:
        return None
    samples_file = str(contract.get("sample_manifest") or contract.get("samples_file") or "").strip()
    sample_format = str(contract.get("sample_manifest_format") or "").strip()
    if not samples_file or not sample_format:
        raise ValueError("generated training contract missing sample_manifest/sample_manifest_format")
    field_names = generated_contract_materialized_field_names(contract)
    required_fields = field_names["required"]
    if not required_fields:
        raise ValueError("generated training contract missing required sample fields")
    sample_fields = tuple(sorted(required_fields | ({"split"} if "split" in field_names["all"] else set())))
    qc_contract = _generated_qc_contract(contract)
    return TrainingContract(
        contract_name=str(contract.get("name") or f"generated_{record_type}"),
        record_type=record_type,
        samples_file=samples_file,
        sample_manifest_format=sample_format,
        sample_fields=sample_fields,
        record_shape=_generated_record_shape(contract, sample_fields=sample_fields),
        qc_contract=qc_contract,
        metric_contract=_generated_metric_contract(contract),
        mock_preflight=_generated_mock_preflight(contract, sample_fields=sample_fields),
        runtime_checks=tuple(
            item
            for item in (
                "manifest.json",
                samples_file,
                "verification.passed",
                "visual_qc.passed" if qc_contract.get("visual_qc_required") else "visual_qc.not_required_or_passed",
                "train_split",
            )
        ),
    )


def _generated_qc_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    raw = contract.get("qc_requirements") or contract.get("visual_qc")
    qc = raw if isinstance(raw, Mapping) else {}
    nested = qc.get("visual_qc") if isinstance(qc.get("visual_qc"), Mapping) else {}
    required = generated_contract_visual_qc_required(contract)
    qc_type = str(
        nested.get("type")
        or nested.get("artifact_type")
        or qc.get("type")
        or qc.get("visual_qc_type")
        or "contract_declared_visual_qc"
    ).strip()
    return {
        "visual_qc_required": required,
        "visual_qc_type": qc_type if required else "not_applicable",
        "reason": str(qc.get("reason") or "generated contract controls visual QC requirements"),
    }


def _generated_metric_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    metric = _first_mapping(
        contract.get("metric_contract"),
        _nested_mapping(contract, "training_contract", "metric_contract"),
        _nested_mapping(contract, "training_guidance", "metric_contract"),
    )
    primary = str(
        metric.get("primary_metric")
        or _nested_value(contract, "training_guidance", "primary_metric")
        or "validation_metric"
    ).strip()
    safe_metrics = metric.get("safe_metrics")
    if not isinstance(safe_metrics, list) or not all(
        isinstance(value, str) and value.strip() for value in safe_metrics
    ):
        safe_metrics = [
            "train_loss",
            "validation_loss",
            primary,
            "num_train_samples",
            "num_validation_samples",
            "num_steps",
        ]
    elif primary not in safe_metrics:
        safe_metrics = [*safe_metrics, primary]
    return {
        "primary_metric": primary,
        "safe_metrics": list(dict.fromkeys(str(value).strip() for value in safe_metrics)),
        "local_metric_artifact": str(metric.get("local_metric_artifact") or "agenticfl_training_metrics.jsonl"),
        "local_metric_artifact_format": str(metric.get("local_metric_artifact_format") or "jsonl"),
    }


def _generated_record_shape(contract: Mapping[str, Any], *, sample_fields: tuple[str, ...]) -> dict[str, Any]:
    shape = contract.get("record_shape") if isinstance(contract.get("record_shape"), Mapping) else {}
    fields = contract.get("record_fields") if isinstance(contract.get("record_fields"), Mapping) else {}
    field_definitions = fields.get("field_definitions") if isinstance(fields.get("field_definitions"), Mapping) else {}
    result: dict[str, Any] = {}
    for field in sample_fields:
        value = shape.get(field) if isinstance(shape, Mapping) else None
        if value is None and isinstance(field_definitions, Mapping):
            value = field_definitions.get(field)
        result[field] = value if value is not None else "contract-declared sample field"
    if "split" not in result:
        result["split"] = "train, validation, or test"
    return result


def _generated_mock_preflight(contract: Mapping[str, Any], *, sample_fields: tuple[str, ...]) -> dict[str, Any]:
    source = _first_mapping(
        contract.get("mock_preflight"),
        contract.get("mock_training"),
        contract.get("mock_data"),
        contract.get("mock_data_guidance"),
    )
    splits = source.get("splits") if isinstance(source.get("splits"), Mapping) else {"train": 2, "validation": 1}
    template = _first_mapping(
        source.get("record_template"),
        source.get("sample_record"),
        contract.get("record_example"),
    )
    return {
        "image": {"format": "png", "channels": 3, "value": "random_uint8_noise"},
        "record_template": _jsonable(template),
        "sample_fields": list(sample_fields),
        "splits": dict(splits),
    }


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _nested_mapping(container: Mapping[str, Any], *path: str) -> Any:
    value: Any = container
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value if isinstance(value, Mapping) else None


def _nested_value(container: Mapping[str, Any], *path: str) -> Any:
    value: Any = container
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def label_harmonization_for_training(
    *,
    task: str,
    contract: TrainingContract,
    label_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if contract.record_type != CLASSIFICATION or "binary" not in task.lower():
        return {}
    raw_values = label_summary.get("observed_label_values", [])
    return {
        "required": True,
        "scope": "cross_client_federated_training",
        "shared_label_space": {"0": "negative_or_control", "1": "positive_or_target"},
        "raw_observed_label_values": raw_values if isinstance(raw_values, list) else [],
        "adapter_requirement": (
            "Client data harmonization should emit canonical integer labels 0 and 1. If a local source "
            "has multiple task-positive diagnosis values, collapse them to 1 only when local evidence "
            "defines those values as positive for the requested binary task; otherwise mark the site unfeasible."
        ),
        "training_code_requirement": (
            "Use one shared binary output head/loss across all clients. Do not create per-client output "
            "heads or let one client remain multi-class. As a guard for extracted artifacts admitted by "
            "the data phase, the dataset loader may map nonzero integer labels to 1 for this binary task, "
            "but it must reject missing, negative, non-integer, or semantically unknown labels."
        ),
    }


def harmonized_binary_class_counts(class_counts: Mapping[str, Any]) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for key, value in class_counts.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        label = _coerce_label_value(key)
        if isinstance(label, int):
            counts["0" if label == 0 else "1"] += count
    return {key: value for key, value in counts.items() if value}


def _coerce_label_value(value: Any) -> int | str:
    text = str(value)
    return int(text) if text.isdigit() else text


def family_from_extraction_summary(extraction_summary: Mapping[str, Any]) -> str:
    formats: set[str] = set()
    results = extraction_summary.get("extraction_results")
    if isinstance(results, Mapping):
        for result in results.values():
            if not isinstance(result, Mapping):
                continue
            observed = result_sample_manifest_format(result)
            if observed:
                formats.add(observed)
    if formats == {classification.SAMPLE_MANIFEST_FORMAT}:
        return CLASSIFICATION
    if formats == {segmentation.SAMPLE_MANIFEST_FORMAT}:
        return SEGMENTATION
    generated = generated_training_contract_from_summary(extraction_summary)
    if generated is not None:
        return generated.record_type
    return "unknown"


def result_sample_manifest_format(result: Mapping[str, Any]) -> str:
    direct = result.get("sample_manifest_format")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    extraction = result.get("extraction")
    if isinstance(extraction, Mapping):
        value = extraction.get("sample_manifest_format")
        if isinstance(value, str) and value.strip():
            return value.strip()
        storage = extraction.get("classification_storage")
        if isinstance(storage, Mapping):
            value = storage.get("format")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def read_split_counts(path: Path, *, sample_manifest_format: str = "") -> tuple[dict[str, int], int]:
    if not path.exists():
        return {}, 0
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and any(key in payload for key in ("training", "validation", "test")):
            split_keys = {"train": "training", "validation": "validation", "test": "test"}
            counts = {
                split: len(payload.get(key) or []) if isinstance(payload.get(key), list) else 0
                for split, key in split_keys.items()
            }
            return counts, sum(counts.values())
    split_counts: dict[str, int] = {}
    sample_count = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"sample manifest row must contain an object: {path}")
        split = str(record.get("split") or "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
        sample_count += 1
    return split_counts, sample_count


def write_mock_preflight_dataset(
    *,
    dataset_root: Path,
    contract: TrainingContract,
    client_id: str,
    target_size: tuple[int, int],
    label_values: list[int] | None = None,
) -> None:
    site_dir = dataset_root / client_id
    width, height = int(target_size[0]), int(target_size[1])
    splits = contract.mock_preflight.get("splits") if isinstance(contract.mock_preflight, Mapping) else None
    if not isinstance(splits, Mapping):
        splits = {"train": 2, "validation": 1}
    records: list[dict[str, Any]] = []
    label_values = sorted({int(value) for value in (label_values or [])}) or [0, 1]
    sample_index = 0
    for split in ("train", "validation"):
        count = int(splits.get(split, 0) or 0)
        for index in range(count):
            stem = f"{split}_{index:03d}"
            image_rel = f"images/{split}/{stem}.png"
            image_path = site_dir / image_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            _write_noise_image(image_path, width=width, height=height, seed=sample_index + 1)
            record_template = contract.mock_preflight.get("record_template")
            if isinstance(record_template, Mapping):
                records.append(
                    _mock_record_from_template(
                        record_template,
                        site_dir=site_dir,
                        image_rel=image_rel,
                        split=split,
                        stem=stem,
                    )
                )
                sample_index += 1
                continue
            target = contract.mock_preflight.get("target")
            target_field = target.get("field") if isinstance(target, Mapping) else "mask"
            if target_field == "label":
                records.append(
                    {"image": image_rel, "label": label_values[sample_index % len(label_values)], "split": split}
                )
            elif target_field == "mask":
                mask_rel = f"masks/{split}/{stem}.png"
                mask_path = site_dir / mask_rel
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                _write_binary_mask(mask_path, width=width, height=height)
                records.append({"image": image_rel, "mask": mask_rel, "split": split})
            else:
                records.append({"image": image_rel, "split": split})
            sample_index += 1
    site_dir.mkdir(parents=True, exist_ok=True)
    target = contract.mock_preflight.get("target")
    target_field = target.get("field") if isinstance(target, Mapping) else "mask"
    if target_field == "label":
        classification.write_sample_manifest(site_dir / contract.samples_file, rows=records, policy={})
    else:
        (site_dir / contract.samples_file).write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )


def _mock_record_from_template(
    template: Mapping[str, Any],
    *,
    site_dir: Path,
    image_rel: str,
    split: str,
    stem: str,
) -> dict[str, Any]:
    record = _jsonable(template)
    if "image_path" in record and "image" not in record:
        record["image_path"] = image_rel
    else:
        record["image"] = image_rel
    record["split"] = split
    for field, value in list(record.items()):
        if field in {"image", "image_path", "split"}:
            continue
        if isinstance(value, str) and (
            field.endswith("_path") or field in {"label_source", "label_source_path", "mask", "mask_path"}
        ):
            suffix = Path(value).suffix or ".json"
            rel = f"mock_annotations/{split}/{stem}_{field}{suffix}"
            target = site_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}\n", encoding="utf-8")
            record[field] = rel
    return record


def mock_visual_qc_payload(
    *, contract: TrainingContract, sample_count: int, client_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if contract.qc_contract.get("visual_qc_required"):
        return (
            {"available": True, "sample_count": sample_count},
            {
                "schema_version": "agenticfl.extraction_visual_qc_decision.v1",
                "client_id": client_id,
                "status": "passed",
                "passed": True,
                "reviewed": True,
                "selected_transform": "as_is",
            },
        )
    return (
        {
            "schema_version": "agenticfl.local_visual_qc_bundle.v1",
            "available": False,
            "sample_count": 0,
            "requested_sample_count": sample_count,
            "reason": "visual QC is not required by this training contract",
            "review_required": False,
            "transform_candidates": [],
            "local_output_path_redacted": True,
        },
        {
            "schema_version": "agenticfl.extraction_visual_qc_decision.v1",
            "client_id": client_id,
            "status": "not_performed",
            "passed": None,
            "reviewed": False,
            "reason": "not required by training contract",
        },
    )


def mock_label_values(contract: TrainingContract, label_values: list[int] | None = None) -> dict[str, int]:
    target = contract.mock_preflight.get("target")
    field = target.get("field") if isinstance(target, Mapping) else "mask"
    if field == "label":
        values = sorted({int(value) for value in (label_values or [])}) or [0, 1]
        return {f"class_{value}": value for value in values}
    if field == "mask":
        return {"background": 0, "target": 1}
    return {}


def _write_noise_image(path: Path, *, width: int, height: int, seed: int) -> None:
    image = Image.new("RGB", (width, height))
    pixels = [
        (
            (x * 11 + y * 7 + seed * 13) % 256,
            (x * 5 + y * 17 + seed * 19) % 256,
            (x * 23 + y * 3 + seed * 29) % 256,
        )
        for y in range(height)
        for x in range(width)
    ]
    image.putdata(pixels)
    image.save(path)


def _write_binary_mask(path: Path, *, width: int, height: int) -> None:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    margin = max(2, min(width, height) // 4)
    draw.ellipse((margin, margin, width - margin - 1, height - margin - 1), fill=255)
    mask.save(path)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))
