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

"""Prepare digest-bound task references from existing prepared-data records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

TASK_EXAMPLE_SCHEMA = "agenticfl.task_example_manifest.v3"
TASK_KEYS = ("cup_seg", "disc_seg", "disc_detec", "glaucoma_cls")

_TASK_DEFINITIONS: dict[str, dict[str, str]] = {
    "cup_seg": {
        "task": "optic cup segmentation",
        "record_type": "segmentation",
        "description": "Final expected cup segmentation form: RGB fundus image plus binary cup mask.",
    },
    "disc_seg": {
        "task": "optic disc segmentation",
        "record_type": "segmentation",
        "description": "Final expected disc segmentation form: RGB fundus image plus binary disc mask.",
    },
    "disc_detec": {
        "task": "optic disc detection",
        "record_type": "generated_contract",
        "description": (
            "Generated-contract example: optic-disc detection starts from this real image and annotation pair; "
            "the fixed harness does not provide a built-in detection renderer."
        ),
    },
    "glaucoma_cls": {
        "task": "binary glaucoma classification",
        "record_type": "classification",
        "description": "Final expected classification form: RGB retinal fundus image plus canonical integer label metadata.",
    },
}


def prepare_reference_bundle(source_config: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Select one real prepared record per task and write canonical references."""

    config_path = Path(source_config).expanduser().resolve()
    config = _read_json_object(config_path)
    configured_examples = config.get("examples")
    if not isinstance(configured_examples, list):
        raise ValueError("reference source config examples must be a list")

    sources: dict[str, Mapping[str, Any]] = {}
    for item in configured_examples:
        if not isinstance(item, Mapping):
            raise ValueError("each reference source must be an object")
        task_key = item.get("task_key")
        if not isinstance(task_key, str) or task_key not in TASK_KEYS:
            raise ValueError(f"reference source task_key must be one of {TASK_KEYS}")
        if task_key in sources:
            raise ValueError(f"duplicate reference source for {task_key}")
        sources[task_key] = item
    missing = [task_key for task_key in TASK_KEYS if task_key not in sources]
    if missing:
        raise ValueError(f"reference source config is missing tasks: {', '.join(missing)}")

    root = Path(output_dir).expanduser().resolve()
    examples = [
        _prepare_example(
            task_key=task_key,
            source=sources[task_key],
            config_dir=config_path.parent,
            root=root,
        )
        for task_key in TASK_KEYS
    ]
    manifest = {
        "schema_version": TASK_EXAMPLE_SCHEMA,
        "description": (
            "Canonical final expected-form examples selected from existing prepared retinal records for bounded "
            "raw-input and contract-owned visual review."
        ),
        "examples": examples,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def _prepare_example(*, task_key: str, source: Mapping[str, Any], config_dir: Path, root: Path) -> dict[str, Any]:
    manifest_value = source.get("sample_manifest")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ValueError(f"{task_key} requires a non-empty sample_manifest")
    source_manifest = _resolve_path(manifest_value, base=config_dir)
    records = _load_records(source_manifest, requested_split=source.get("split"))
    record, source_record_index = _select_record(records, source=source, task_key=task_key)

    destination_dir = root / task_key
    destination_dir.mkdir(parents=True, exist_ok=True)
    image_source = _record_path(record, fields=("image", "image_path"), manifest_dir=source_manifest.parent)
    image_destination = destination_dir / "image.png"
    image_size = _write_rgb_png(image_source, image_destination)

    definition = _TASK_DEFINITIONS[task_key]
    if task_key in {"cup_seg", "disc_seg"}:
        mask_source = _record_path(record, fields=("mask", "mask_path"), manifest_dir=source_manifest.parent)
        mask_destination = destination_dir / "mask.png"
        _write_mask_png(mask_source, mask_destination, expected_size=image_size)
        final_expected_form: dict[str, Any] = {
            "image_path": f"{task_key}/image.png",
            "mask_format": "binary_png",
            "mask_path": f"{task_key}/mask.png",
        }
    elif task_key == "disc_detec":
        annotation_destination = destination_dir / "anno.jsonl"
        annotation = _detection_annotation(record, source=source, image_size=image_size)
        _write_jsonl(annotation_destination, annotation)
        final_expected_form = {
            "annotation_format": "jsonl_object_detection_records",
            "annotation_path": "disc_detec/anno.jsonl",
            "bbox_format": annotation["bbox_format"],
            "box_field": "boxes",
            "image_path": "disc_detec/image.png",
            "label_field": "labels",
            "record_type_hint": "object_detection_bbox",
            "renderer_policy": "generated_contract_or_agent_owned",
        }
    else:
        label = record.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label not in {0, 1}:
            raise ValueError("glaucoma_cls selected record must have canonical integer label 0 or 1")
        label_destination = destination_dir / "label.json"
        _write_json(
            label_destination,
            {
                "image_path": "image.png",
                "label": label,
                "label_meaning": ("glaucoma_positive" if label == 1 else "glaucoma_negative"),
                "label_space": {"0": "negative/control", "1": "positive/target"},
                "record_type": "classification",
            },
        )
        final_expected_form = {
            "image_path": "glaucoma_cls/image.png",
            "label_field": "label",
            "label_format": "json_classification_label",
            "label_path": "glaucoma_cls/label.json",
            "label_space": {"0": "negative/control", "1": "positive/target"},
        }

    asset_sha256 = {
        field: f"sha256:{_sha256(root / value)}"
        for field, value in final_expected_form.items()
        if field.endswith("_path") and isinstance(value, str)
    }
    source_summary: dict[str, Any] = {
        "kind": "prepared_client_record",
        "sample_manifest_sha256": f"sha256:{_sha256(source_manifest)}",
        "record_index": source_record_index,
    }
    if isinstance(record.get("split"), str):
        source_summary["split"] = record["split"]
    if isinstance(record.get("sample_id"), str):
        source_summary["sample_id"] = record["sample_id"]
    return {
        "task": definition["task"],
        "task_key": task_key,
        "record_type": definition["record_type"],
        "description": definition["description"],
        "source": source_summary,
        "final_expected_form": final_expected_form,
        "asset_sha256": asset_sha256,
    }


def _load_records(path: Path, *, requested_split: Any) -> list[tuple[int, Mapping[str, Any]]]:
    if not path.is_file():
        raise ValueError(f"sample manifest is not a file: {path}")
    if path.suffix.casefold() == ".jsonl":
        records: list[tuple[int, Mapping[str, Any]]] = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"sample manifest row {index} must be an object")
            records.append((index, value))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
            values = payload["records"]
        elif isinstance(payload, Mapping):
            split_key = _json_split_key(payload, requested_split)
            values = payload.get(split_key) if split_key is not None else None
            if not isinstance(values, list):
                raise ValueError(f"could not find a record list in {path}")
        else:
            raise ValueError(f"sample manifest must contain a list or object: {path}")
        records = []
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise ValueError(f"sample manifest record {index} must be an object")
            records.append((index, value))

    if isinstance(requested_split, str) and requested_split.strip():
        requested = _normalise_split(requested_split)
        split_records = [
            item for item in records if _normalise_split(str(item[1].get("split") or requested)) == requested
        ]
        if split_records:
            records = split_records
    if not records:
        raise ValueError(f"sample manifest has no selectable records: {path}")
    return records


def _json_split_key(payload: Mapping[str, Any], requested_split: Any) -> str | None:
    candidates: list[str] = []
    if isinstance(requested_split, str) and requested_split.strip():
        candidates.extend((requested_split, _normalise_split(requested_split)))
    candidates.extend(("training", "train", "validation", "test"))
    for candidate in candidates:
        if isinstance(payload.get(candidate), list):
            return candidate
    return None


def _normalise_split(value: str) -> str:
    normalised = value.strip().casefold()
    return "train" if normalised == "training" else normalised


def _select_record(
    records: list[tuple[int, Mapping[str, Any]]],
    *,
    source: Mapping[str, Any],
    task_key: str,
) -> tuple[Mapping[str, Any], int]:
    sample_id = source.get("sample_id")
    if sample_id is not None:
        records = [item for item in records if item[1].get("sample_id") == sample_id]
    label = source.get("label")
    if label is not None:
        records = [item for item in records if item[1].get("label") == label]
    if not records:
        raise ValueError(f"{task_key} source selectors matched no records")
    record_index = source.get("record_index", 0)
    if isinstance(record_index, bool) or not isinstance(record_index, int) or record_index < 0:
        raise ValueError(f"{task_key} record_index must be a non-negative integer")
    try:
        return records[record_index][1], records[record_index][0]
    except IndexError as exc:
        raise ValueError(f"{task_key} record_index {record_index} is out of range") from exc


def _record_path(record: Mapping[str, Any], *, fields: tuple[str, ...], manifest_dir: Path) -> Path:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            path = _resolve_path(value, base=manifest_dir)
            if not path.is_file():
                raise ValueError(f"selected record {field} is not a file: {path}")
            return path
    raise ValueError(f"selected record requires one of these path fields: {', '.join(fields)}")


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _write_rgb_png(source: Path, destination: Path) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        size = rgb.size
        rgb.save(destination, format="PNG")
    return size


def _write_mask_png(source: Path, destination: Path, *, expected_size: tuple[int, int]) -> None:
    with Image.open(source) as image:
        mask = image.convert("L")
        if mask.size != expected_size:
            raise ValueError(f"reference mask size {mask.size} does not match image size {expected_size}")
        mask.save(destination, format="PNG")


def _detection_annotation(
    record: Mapping[str, Any], *, source: Mapping[str, Any], image_size: tuple[int, int]
) -> dict[str, Any]:
    boxes = record.get("boxes")
    labels = record.get("labels")
    if not isinstance(boxes, list) or not boxes or not all(isinstance(box, list) and len(box) == 4 for box in boxes):
        raise ValueError("disc_detec selected record requires non-empty four-value boxes")
    if not isinstance(labels, list) or len(labels) != len(boxes):
        raise ValueError("disc_detec selected record requires one label per box")
    bbox_format = record.get("bbox_format") or source.get("bbox_format")
    if bbox_format != "xyxy_abs":
        raise ValueError("disc_detec reference requires explicitly declared bbox_format=xyxy_abs")
    return {
        "bbox_format": bbox_format,
        "boxes": boxes,
        "image_path": "image.png",
        "image_size": {"height": image_size[1], "width": image_size[0]},
        "labels": labels,
        "record_type": "object_detection",
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"reference source config is not a file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("reference source config must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-config",
        required=True,
        help="JSON config selecting real prepared records.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive canonical references.",
    )
    args = parser.parse_args(argv)
    manifest = prepare_reference_bundle(args.source_config, args.output_dir)
    print(f"Prepared {len(manifest['examples'])} data-derived references in {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
