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

"""Client-local AgenticFL training-site preparation launcher.

This script is packaged into each NVFlare client app. It validates the local
prepared dataset folder at client runtime, then delegates to the generated
training entry script with the original training arguments. The server exporter
can therefore avoid reading client-local manifest.json or sample manifests while
still producing a runnable SimEnv job.
"""

from __future__ import annotations

import argparse
import json
import re
import runpy
import sys
from pathlib import Path
from typing import Any

from agenticfl.data.parser import validate_client_id
from agenticfl.data.qc import visual_qc_decision_passed

REPORT_SCHEMA = "agenticfl.client_training_runtime_prep.v1"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--agenticfl-entry-script", required=True)
    parser.add_argument("--agenticfl-expected-train", type=int, default=None)
    parser.add_argument("--agenticfl-expected-validation", type=int, default=None)
    parser.add_argument("--agenticfl-expected-test", type=int, default=None)
    parser.add_argument("--agenticfl-prep-report", default="agenticfl_client_training_prep.json")
    parser.add_argument("--agenticfl-package-dir", required=True)
    parser.add_argument("--agenticfl-record-type", required=True)
    parser.add_argument("--agenticfl-samples-file", required=True)
    parser.add_argument("--agenticfl-sample-manifest-format", default="")
    parser.add_argument("--agenticfl-required-fields", default="")
    parser.add_argument("--agenticfl-visual-qc-required", default="true")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--client-id", required=True)
    args, remaining = parser.parse_known_args(argv)

    custom_root = Path(__file__).resolve().parent
    entry_script = Path(args.agenticfl_entry_script)
    if not entry_script.is_absolute():
        entry_script = custom_root / entry_script
    entry_script = entry_script.resolve()
    if not entry_script.exists():
        raise FileNotFoundError(f"generated training entry script is missing: {entry_script}")
    package_root = next(
        (
            candidate
            for candidate in (entry_script.parent, *entry_script.parents)
            if candidate.name == args.agenticfl_package_dir
        ),
        None,
    )
    if package_root is None:
        raise RuntimeError("generated training entry script is outside the packaged training module")

    dataset_root = _resolve_dataset_root(args.dataset_root)
    report = _validate_prepared_dataset(
        dataset_root=dataset_root,
        client_id=args.client_id,
        expected_train=args.agenticfl_expected_train,
        expected_validation=args.agenticfl_expected_validation,
        expected_test=args.agenticfl_expected_test,
        record_type=args.agenticfl_record_type,
        samples_file=args.agenticfl_samples_file,
        sample_manifest_format=args.agenticfl_sample_manifest_format,
        required_fields=_parse_required_fields(args.agenticfl_required_fields),
        visual_qc_required=_as_bool(args.agenticfl_visual_qc_required),
    )
    _write_report(Path(args.agenticfl_prep_report), report)

    sys.argv = [
        str(entry_script),
        "--dataset-root",
        str(dataset_root),
        "--client-id",
        args.client_id,
        *remaining,
    ]
    runpy.run_path(str(entry_script), run_name="__main__")


def _resolve_dataset_root(value: str) -> Path:
    root = Path(value).expanduser()
    if root.is_absolute():
        return root.resolve()
    candidates = [Path.cwd(), *Path.cwd().parents]
    for base in candidates:
        candidate = (base / root).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / root).resolve()


def _validate_prepared_dataset(
    *,
    dataset_root: Path,
    client_id: str,
    expected_train: int | None,
    expected_validation: int | None,
    expected_test: int | None,
    record_type: str,
    samples_file: str,
    sample_manifest_format: str = "",
    required_fields: list[str] | None = None,
    visual_qc_required: bool = True,
) -> dict[str, Any]:
    client_id = validate_client_id(client_id)
    site_dir = dataset_root / client_id
    manifest_path = site_dir / "manifest.json"
    if not str(record_type or "").strip():
        raise RuntimeError("client training runtime missing data contract record_type")
    if not str(samples_file or "").strip():
        raise RuntimeError("client training runtime missing data contract samples_file")
    if not site_dir.exists():
        raise FileNotFoundError(f"client prepared dataset folder is missing: {site_dir}")
    record_family = _record_family(record_type)
    samples_path = _resolve_site_relative_existing_file(site_dir, samples_file, field="samples_file")
    manifest = _read_json_object(manifest_path)
    _validate_manifest_contract(
        manifest,
        expected_record_type=record_family,
        expected_samples_file=samples_file,
        expected_sample_manifest_format=sample_manifest_format,
    )
    split_counts = _read_split_counts(
        samples_path,
        site_dir=site_dir,
        record_type=record_family,
        sample_manifest_format=sample_manifest_format,
        required_fields=required_fields or [],
    )
    verification = manifest.get("verification") if isinstance(manifest.get("verification"), dict) else {}
    visual_qc = manifest.get("visual_qc_decision")
    if not isinstance(visual_qc, dict):
        legacy_visual_qc = manifest.get("visual_qc")
        visual_qc = legacy_visual_qc if isinstance(legacy_visual_qc, dict) and "passed" in legacy_visual_qc else {}
    if verification.get("passed") is not True:
        raise RuntimeError("client prepared dataset verification did not pass")
    if visual_qc_required and not visual_qc_decision_passed(
        visual_qc,
        label_orientation=(
            manifest.get("label_orientation") if isinstance(manifest.get("label_orientation"), dict) else None
        ),
    ):
        raise RuntimeError("client prepared dataset visual_qc_decision.passed is not a strict task-aligned pass")
    if split_counts.get("train", 0) <= 0:
        raise RuntimeError("client prepared dataset has no train samples")
    _assert_expected_count("train", split_counts.get("train", 0), expected_train)
    _assert_expected_count("validation", split_counts.get("validation", 0), expected_validation)
    _assert_expected_count("test", split_counts.get("test", 0), expected_test)
    return {
        "schema_version": REPORT_SCHEMA,
        "client_id": client_id,
        "status": "ready",
        "dataset_root_resolved_client_local": True,
        "manifest_available": True,
        "samples_file": samples_file,
        "sample_manifest_format": sample_manifest_format,
        "required_fields": required_fields or [],
        "record_type": record_family,
        "samples_file_available": True,
        "split_counts": split_counts,
        "verification_passed": True,
        "visual_qc_required": visual_qc_required,
        "visual_qc_passed": visual_qc.get("passed") if visual_qc_required else None,
        "privacy": {
            "safe_to_share": False,
            "reason": "client-local runtime preparation report; do not return through FLARE",
        },
    }


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "required"}


def _parse_required_fields(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [item.strip() for item in text.split(",") if item.strip()]
    if not isinstance(payload, list):
        return []
    result: list[str] = []
    for item in payload:
        if isinstance(item, str) and item.strip() and item.strip() not in result:
            result.append(item.strip())
    return result


def _record_family(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if normalized in {"segmentation", "image_mask", "image_mask_pairs", "image_mask_records"}:
        return "segmentation"
    if normalized in {"classification", "image_classification", "image_level_classification"}:
        return "classification"
    return normalized


def _validate_manifest_contract(
    manifest: dict[str, Any],
    *,
    expected_record_type: str,
    expected_samples_file: str,
    expected_sample_manifest_format: str,
) -> None:
    declared_record_type = manifest.get("record_type")
    if not isinstance(declared_record_type, str) or not declared_record_type.strip():
        raise RuntimeError("client prepared dataset manifest missing record_type")
    if _record_family(declared_record_type) != expected_record_type:
        raise RuntimeError(
            "client prepared dataset manifest record_type mismatch: "
            f"expected {expected_record_type}, observed {declared_record_type}"
        )
    declared_samples_file = manifest.get("sample_manifest") or manifest.get("samples_file")
    if not isinstance(declared_samples_file, str) or not declared_samples_file.strip():
        raise RuntimeError("client prepared dataset manifest missing sample_manifest")
    if Path(declared_samples_file).as_posix() != Path(expected_samples_file).as_posix():
        raise RuntimeError(
            "client prepared dataset manifest sample_manifest mismatch: "
            f"expected {expected_samples_file}, observed {declared_samples_file}"
        )
    declared_format = manifest.get("sample_manifest_format")
    if not isinstance(declared_format, str) or not declared_format.strip():
        raise RuntimeError("client prepared dataset manifest missing sample_manifest_format")
    if expected_sample_manifest_format and declared_format != expected_sample_manifest_format:
        raise RuntimeError(
            "client prepared dataset manifest sample_manifest_format mismatch: "
            f"expected {expected_sample_manifest_format}, observed {declared_format}"
        )


def _resolve_site_relative_existing_file(site_dir: Path, rel_path: str, *, field: str) -> Path:
    path = Path(str(rel_path or ""))
    if not str(rel_path or "").strip():
        raise FileNotFoundError(f"client prepared dataset {field} is missing")
    if path.is_absolute():
        raise RuntimeError(f"client prepared dataset {field} must be site-folder-relative")
    root = site_dir.resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"client prepared dataset {field} escapes the site folder")
    if not target.is_file():
        raise FileNotFoundError(f"client prepared dataset {field} file is missing")
    return target


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required client-local file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"required client-local JSON file is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"required client-local JSON file must contain an object: {path}")
    return payload


def _read_split_counts(
    path: Path,
    *,
    site_dir: Path,
    record_type: str,
    sample_manifest_format: str = "",
    required_fields: list[str] | None = None,
) -> dict[str, int]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required client-local samples file is missing: {path}") from exc
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and any(key in payload for key in ("training", "validation", "test")):
            split_keys = {"train": "training", "validation": "validation", "test": "test"}
            counts: dict[str, int] = {}
            for split, key in split_keys.items():
                rows = payload.get(key) or []
                if not isinstance(rows, list):
                    raise ValueError(f"client-local sample manifest {key} must be a list: {path}")
                for index, record in enumerate(rows, start=1):
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"client-local sample manifest {key} row {index} must contain an object: {path}"
                        )
                    _validate_sample_record(
                        record,
                        site_dir=site_dir,
                        record_type=record_type,
                        split=split,
                        row_index=index,
                        required_fields=required_fields or [],
                    )
                counts[split] = len(rows)
            return counts
    counts: dict[str, int] = {}
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"client-local sample manifest contains invalid JSONL: {path}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"client-local sample manifest row must contain an object: {path}")
        split = str(record.get("split") or "")
        _validate_sample_record(
            record,
            site_dir=site_dir,
            record_type=record_type,
            split=split,
            row_index=index,
            required_fields=required_fields or [],
        )
        counts[split] = counts.get(split, 0) + 1
    return counts


def _validate_sample_record(
    record: dict[str, Any],
    *,
    site_dir: Path,
    record_type: str,
    split: str,
    row_index: int,
    required_fields: list[str] | None = None,
) -> None:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"client-local sample manifest row {row_index} has invalid split")
    if record_type == "segmentation":
        _resolve_site_relative_existing_file(site_dir, _required_string(record, "image", row_index), field="image")
        _resolve_site_relative_existing_file(site_dir, _required_string(record, "mask", row_index), field="mask")
        return
    if record_type == "classification":
        _resolve_site_relative_existing_file(site_dir, _required_string(record, "image", row_index), field="image")
        label = record.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise ValueError(f"client-local sample manifest row {row_index} missing non-negative integer label")
        return
    _validate_generic_sample_record(
        record,
        site_dir=site_dir,
        required_fields=required_fields or [],
        row_index=row_index,
    )


def _validate_generic_sample_record(
    record: dict[str, Any],
    *,
    site_dir: Path,
    required_fields: list[str],
    row_index: int,
) -> None:
    missing_fields = [
        field
        for field in required_fields
        if field != "split"
        and (
            field not in record
            or record.get(field) is None
            or record.get(field) == ""
            or (isinstance(record.get(field), list) and not record.get(field))
        )
    ]
    if missing_fields:
        raise ValueError(
            f"client-local sample manifest row {row_index} missing required fields: " + ", ".join(missing_fields)
        )
    image_value = record.get("image") or record.get("image_path")
    if not isinstance(image_value, str) or not image_value.strip():
        raise ValueError(f"client-local sample manifest row {row_index} missing image/image_path")
    _resolve_site_relative_existing_file(site_dir, image_value, field="image")
    for field, value in record.items():
        if field in {"image", "image_path", "split"}:
            continue
        if isinstance(value, str) and _is_path_field(field):
            _resolve_site_relative_existing_file(site_dir, value, field=field)


def _is_path_field(field: str) -> bool:
    return field in {"mask", "mask_path", "label_source", "label_source_path"} or field.endswith("_path")


def _required_string(record: dict[str, Any], field: str, row_index: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"client-local sample manifest row {row_index} missing {field}")
    return value


def _assert_expected_count(split: str, observed: int, expected: int | None) -> None:
    if expected is not None and expected >= 0 and observed != expected:
        raise RuntimeError(f"client prepared dataset {split} count mismatch: expected {expected}, observed {observed}")


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
