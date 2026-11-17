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

"""Execution and verification of agent-generated client-local data materializers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from fedready.data.contracts import (
    STANDARD_SPLITS,
    generated_contract_materialized_field_names,
    generated_contract_record_type,
    generated_contract_visual_qc_required,
    generated_data_contract_validation_errors,
)
from fedready.data.extraction_artifacts import (
    EXTRACTOR_LOGIC_VERSION,
    ExtractionConfig,
    _dedupe,
    _extraction_policy_digest,
    _extraction_result,
    _extraction_screening,
    _extraction_verification,
    _failed_extraction_result,
    _finalize_intensity_accumulator,
    _generated_data_contract,
    _local_adapter_safe_summary,
    _new_intensity_accumulator,
    _update_intensity_accumulator,
    _write_training_transform_policy,
)
from fedready.data.qc import VISUAL_QC_BUNDLE_SCHEMA, VISUAL_QC_TRANSFORMS, _safe_visual_qc_transform_candidates
from fedready.data.visual_qc import _resolve_output_dir_relative_path, _validate_visual_qc_artifact_references
from fedready.utils.logging import payload_digest

try:  # Pillow is optional outside image-processing workflows.
    from PIL import Image
except ImportError:  # pragma: no cover - minimal environments
    Image = None  # type: ignore[assignment]


def _extract_generated_contract_dataset(
    *,
    client_id: str,
    policy: dict[str, Any],
    cfg: ExtractionConfig,
    output_dir: Path,
    output_name: str,
    target_size: tuple[int, int],
    target_terms: set[str],
    source_label_type: str,
    conversion_options: list[str],
    local_adapter: dict[str, Any] | None,
    adapter_manifest: dict[str, Any],
    adapter_warnings: list[str],
    generated_contract: dict[str, Any] | None,
    generated_materializer: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(generated_contract, dict):
        return _generated_materializer_failure_result(
            client_id=client_id,
            policy=policy,
            source_label_type=source_label_type,
            local_adapter=local_adapter,
            adapter_manifest=adapter_manifest,
            reason="Generated materializer was provided without a generated data contract.",
            warnings=adapter_warnings,
        )
    contract_errors = generated_data_contract_validation_errors(generated_contract)
    if contract_errors:
        return _generated_materializer_failure_result(
            client_id=client_id,
            policy=policy,
            source_label_type=source_label_type,
            local_adapter=local_adapter,
            adapter_manifest=adapter_manifest,
            reason="Generated data contract failed validation: " + "; ".join(contract_errors),
            warnings=adapter_warnings,
        )
    if output_dir.exists() and not cfg.overwrite:
        raise FileExistsError(f"extraction output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    materializer_result = _run_generated_data_materializer(
        output_dir=output_dir,
        adapter_manifest=adapter_manifest,
        policy=policy,
        generated_contract=generated_contract,
        generated_materializer=generated_materializer,
    )
    if materializer_result.get("status") != "passed":
        reason_code = _generated_materializer_result_issue_code(materializer_result)
        return _generated_materializer_failure_result(
            client_id=client_id,
            policy=policy,
            source_label_type=source_label_type,
            local_adapter=local_adapter,
            adapter_manifest=adapter_manifest,
            reason_code=reason_code,
            reason=_generated_materializer_public_reason(reason_code),
            warnings=adapter_warnings,
        )

    report = materializer_result.get("report") if isinstance(materializer_result.get("report"), dict) else {}
    sample_manifest_name = str(generated_contract.get("sample_manifest") or "").strip()
    sample_manifest_format = str(generated_contract.get("sample_manifest_format") or "").strip()
    if not sample_manifest_name or not sample_manifest_format:
        return _generated_materializer_failure_result(
            client_id=client_id,
            policy=policy,
            source_label_type=source_label_type,
            local_adapter=local_adapter,
            adapter_manifest=adapter_manifest,
            reason="Generated data contract did not declare sample_manifest and sample_manifest_format",
            warnings=adapter_warnings,
        )
    try:
        _require_matching_generated_manifest_declaration(
            report,
            source="materializer report",
            expected_sample_manifest=sample_manifest_name,
            expected_sample_manifest_format=sample_manifest_format,
        )
        _require_matching_generated_manifest_declaration(
            generated_materializer,
            source="materializer spec",
            expected_sample_manifest=sample_manifest_name,
            expected_sample_manifest_format=sample_manifest_format,
        )
        sample_manifest_path = _resolve_output_dir_relative_path(
            output_dir,
            sample_manifest_name,
            field="sample_manifest",
            require_existing=True,
            artifact_kind="generated sample manifest",
        )
        sample_rows = _read_generated_sample_rows(sample_manifest_path, sample_manifest_format=sample_manifest_format)
        _validate_generated_sample_rows(
            rows=sample_rows,
            output_dir=output_dir,
            generated_contract=generated_contract,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _generated_materializer_failure_result(
            client_id=client_id,
            policy=policy,
            source_label_type=source_label_type,
            local_adapter=local_adapter,
            adapter_manifest=adapter_manifest,
            reason=f"Generated materializer wrote an invalid sample manifest: {exc}",
            warnings=adapter_warnings,
        )

    counts: Counter[str] = Counter()
    for row in sample_rows:
        split = str(row.get("split") or "train")
        if split not in STANDARD_SPLITS:
            split = "train"
        counts[split] += 1
    adapter_record_count = len(adapter_manifest.get("records", [])) if isinstance(adapter_manifest, dict) else 0
    extracted_count = sum(counts.values())
    failed_pair_count = max(adapter_record_count - extracted_count, 0)
    screening = _extraction_screening(
        record_count=adapter_record_count,
        valid_case_count=extracted_count,
        execution_pair_count=extracted_count,
        target_terms=target_terms,
    )
    verification = _extraction_verification(
        full_valid_case_count=adapter_record_count,
        expected_extracted_count=adapter_record_count,
        extracted_count=extracted_count,
        failed_pair_count=failed_pair_count,
        max_samples=cfg.max_samples,
        target_terms=target_terms,
    )
    try:
        visual_qc = _generated_visual_qc_bundle(
            report=report,
            generated_contract=generated_contract,
            output_dir=output_dir,
        )
    except ValueError as exc:
        return _generated_materializer_failure_result(
            client_id=client_id,
            policy=policy,
            source_label_type=source_label_type,
            local_adapter=local_adapter,
            adapter_manifest=adapter_manifest,
            reason=f"Generated materializer wrote invalid visual QC artifacts: {exc}",
            warnings=adapter_warnings,
        )
    if generated_contract_visual_qc_required(generated_contract) and not visual_qc.get("available"):
        verification = {**verification, "passed": False, "generated_contract_visual_qc_missing": True}
    preview = _generated_preview(output_dir=output_dir, rows=sample_rows, report=report)
    storage_sections = _generated_storage_sections(
        generated_contract=generated_contract, rows=sample_rows, report=report
    )
    intensity_stats = _generated_intensity_stats(output_dir=output_dir, rows=sample_rows)
    _write_training_transform_policy(
        output_dir=output_dir,
        client_id=client_id,
        policy=policy,
        target_size=target_size,
        image_intensity=intensity_stats,
        counts={"total": extracted_count, "by_split": dict(sorted(counts.items()))},
        label_rule_applied=_generated_label_rule_applied(generated_contract),
        storage_sections=storage_sections,
    )
    split_plan = {
        "schema_version": "fedready.local_split_plan.v1",
        "source": "server_generated_contract_materializer",
        "preserve_existing_splits": True,
        "sample_limited": cfg.max_samples is not None and adapter_record_count > (cfg.max_samples or 0),
    }
    label_orientation = {
        "schema_version": "fedready.local_label_orientation_rule.v1",
        "strategy": "generated_contract_materializer",
        "selected_transform": "as_is",
        "reason": "Generated spatial materializers write canonical as-is output geometry.",
        "safe_to_share": True,
    }
    manifest = {
        "schema_version": "fedready.local_extracted_manifest.v1",
        "extractor_logic_version": EXTRACTOR_LOGIC_VERSION,
        "client_id": client_id,
        "policy_digest": policy.get("strategy_digest") or payload_digest(policy),
        "extraction_policy_digest": _extraction_policy_digest(policy),
        "target_size": list(target_size),
        "stored_resolution": "source_image_resolution",
        "training_transform_policy": "training_transforms.json",
        "source_label_type": source_label_type,
        "record_type": generated_contract_record_type(generated_contract),
        "local_adapter": _local_adapter_safe_summary(local_adapter, adapter_manifest=adapter_manifest),
        "conversion_options": conversion_options,
        "image_storage": {
            "format": "png",
            "mode": "RGB",
            "resolution": "source_image_resolution",
            "extraction_resize_applied": False,
            "training_dtype": policy.get("image_rule", {}).get("dtype", "float32"),
            "training_intensity": policy.get("image_rule", {}).get("intensity", "scale_to_0_1"),
            "training_transform_deferred": True,
        },
        "mask_storage": storage_sections.get("mask_storage"),
        "classification_storage": storage_sections.get("classification_storage"),
        "object_detection_storage": storage_sections.get("object_detection_storage"),
        "counts": {"total": extracted_count, "by_split": dict(sorted(counts.items()))},
        "failed_pairs": {"materializer_drop_count": failed_pair_count} if failed_pair_count else {},
        "split_plan": split_plan,
        "label_orientation": label_orientation,
        "screening": screening,
        "verification": verification,
        "preview": preview,
        "visual_qc": visual_qc,
        "sample_manifest": sample_manifest_name,
        "sample_manifest_format": sample_manifest_format,
        "generated_data_contract": _safe_generated_contract_summary(generated_contract),
        "generated_data_materializer": _safe_generated_materializer_summary(generated_materializer),
        "output": {
            "layout": "project_client_folder",
            "root": cfg.output_root,
            "client_folder": client_id,
            "run_label": output_name,
            "local_output_path_redacted": True,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return _extraction_result(
        manifest,
        label_rule_applied=_generated_label_rule_applied(generated_contract),
        warnings=_dedupe(
            [
                *_stable_warning_codes(adapter_warnings, fallback_code="ADAPTER_WARNING"),
                *_stable_warning_codes(
                    materializer_result.get("warnings", []), fallback_code="GENERATED_MATERIALIZER_WARNING"
                ),
            ]
        ),
        extraction_details={
            "preview_available": preview.get("available") is True,
            "object_detection_storage": storage_sections.get("object_detection_storage"),
        },
    )


def _run_generated_data_materializer(
    *,
    output_dir: Path,
    adapter_manifest: dict[str, Any],
    policy: dict[str, Any],
    generated_contract: dict[str, Any],
    generated_materializer: dict[str, Any],
) -> dict[str, Any]:
    try:
        workspace = output_dir / "_generated_materializer_runtime"
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        source_files = generated_materializer.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            raise ValueError("generated materializer did not include source_files")
        source_file_count = generated_materializer.get("source_file_count")
        if (
            isinstance(source_file_count, bool)
            or not isinstance(source_file_count, int)
            or source_file_count != len(source_files)
        ):
            raise ValueError("generated materializer source_file_count mismatch")
        entry_script = generated_materializer.get("entry_script")
        if not isinstance(entry_script, str) or not entry_script.strip():
            raise ValueError("generated materializer missing entry_script")

        validated_sources: list[tuple[Path, str]] = []
        source_manifest: list[dict[str, str]] = []
        resolved_targets: set[Path] = set()
        for entry in source_files:
            if not isinstance(entry, dict):
                raise ValueError("generated materializer source file entry was not an object")
            rel_path = entry.get("path")
            content = entry.get("content")
            expected_sha = entry.get("sha256")
            if not isinstance(rel_path, str) or not rel_path.strip() or not isinstance(content, str):
                raise ValueError("generated materializer source file missing path or content")
            if not isinstance(expected_sha, str) or not expected_sha:
                raise ValueError(f"generated materializer source checksum missing for {rel_path}")
            if payload_digest(content) != expected_sha:
                raise ValueError(f"generated materializer source checksum mismatch for {rel_path}")
            target = (workspace / rel_path).resolve()
            if target != workspace and workspace not in target.parents:
                raise ValueError(f"generated materializer source file escapes workspace: {rel_path}")
            if target in resolved_targets:
                raise ValueError(f"generated materializer source file target is duplicated: {rel_path}")
            resolved_targets.add(target)
            validated_sources.append((target, content))
            source_manifest.append({"path": rel_path, "sha256": expected_sha})

        expected_source_digest = generated_materializer.get("source_digest")
        if not isinstance(expected_source_digest, str) or not expected_source_digest:
            raise ValueError("generated materializer source_digest missing")
        actual_source_digest = payload_digest(
            {
                "entry_script": entry_script,
                "source_files": source_manifest,
            }
        )
        if actual_source_digest != expected_source_digest:
            raise ValueError("generated materializer source_digest mismatch")

        for target, content in validated_sources:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        entry_path = (workspace / entry_script).resolve()
        if entry_path != workspace and workspace not in entry_path.parents:
            raise ValueError("generated materializer entry_script escapes workspace")
        if not entry_path.is_file():
            raise ValueError("generated materializer entry_script was not written")
        adapter_manifest_path = workspace / "adapter_manifest.json"
        data_contract_path = workspace / "generated_data_contract.json"
        policy_path = workspace / "policy.json"
        report_path = workspace / "materializer_report.json"
        adapter_manifest_path.write_text(json.dumps(adapter_manifest, indent=2, sort_keys=True), encoding="utf-8")
        data_contract_path.write_text(json.dumps(generated_contract, indent=2, sort_keys=True), encoding="utf-8")
        policy_path.write_text(
            json.dumps(_redact_policy_for_materializer(policy), indent=2, sort_keys=True), encoding="utf-8"
        )
        command = [
            sys.executable,
            str(entry_path),
            "--adapter-manifest",
            str(adapter_manifest_path),
            "--output-dir",
            str(output_dir),
            "--data-contract",
            str(data_contract_path),
            "--policy",
            str(policy_path),
            "--report-path",
            str(report_path),
        ]
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=_generated_materializer_environment(workspace),
            capture_output=True,
            text=True,
            timeout=_generated_materializer_timeout_seconds(),
            check=False,
        )
        diagnostic = _short_materializer_diagnostic(completed.stdout, completed.stderr)
        if completed.returncode != 0:
            return {
                "status": "failed",
                "issue_code": "GENERATED_MATERIALIZER_PROCESS_FAILED",
                "local_diagnostic": diagnostic or "generated materializer exited with a nonzero status",
                "return_code": completed.returncode,
                "warnings": [],
            }
        if not report_path.is_file():
            return {
                "status": "failed",
                "issue_code": "GENERATED_MATERIALIZER_REPORT_MISSING",
                "local_diagnostic": "generated materializer did not write report_path",
                "return_code": completed.returncode,
                "warnings": [],
            }
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("generated materializer report was not a JSON object")
        if report.get("schema_version") != "fedready.generated_materializer_report.v1":
            raise ValueError("generated materializer report schema_version mismatch")
        if report.get("status") not in {None, "passed", "completed"}:
            raise ValueError("generated materializer report did not indicate success")
        return {
            "status": "passed",
            "local_diagnostic": diagnostic,
            "return_code": completed.returncode,
            "report": report,
            "warnings": [str(item) for item in report.get("warnings", []) if isinstance(item, str)],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "issue_code": "GENERATED_MATERIALIZER_TIMEOUT",
            "local_diagnostic": str(exc),
            "warnings": [],
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "issue_code": "GENERATED_MATERIALIZER_RUNTIME_ERROR",
            "local_diagnostic": str(exc),
            "warnings": [],
        }


def _require_matching_generated_manifest_declaration(
    container: dict[str, Any],
    *,
    source: str,
    expected_sample_manifest: str,
    expected_sample_manifest_format: str,
) -> None:
    sample_manifest = container.get("sample_manifest") if isinstance(container, dict) else None
    if isinstance(sample_manifest, str) and sample_manifest.strip():
        if _relative_path_text(sample_manifest) != _relative_path_text(expected_sample_manifest):
            raise ValueError(f"{source} sample_manifest does not match the approved generated contract")
    sample_manifest_format = container.get("sample_manifest_format") if isinstance(container, dict) else None
    if isinstance(sample_manifest_format, str) and sample_manifest_format.strip():
        if sample_manifest_format.strip() != expected_sample_manifest_format:
            raise ValueError(f"{source} sample_manifest_format does not match the approved generated contract")


def _relative_path_text(value: str) -> str:
    return Path(value).as_posix()


def _stable_warning_codes(raw_warnings: Any, *, fallback_code: str) -> list[str]:
    if not isinstance(raw_warnings, list):
        return []
    codes: list[str] = []
    redacted_count = 0
    for warning in raw_warnings[:20]:
        candidate: Any = None
        if isinstance(warning, dict):
            candidate = warning.get("code") or warning.get("reason_code")
        elif isinstance(warning, str):
            candidate = warning.strip()
        if isinstance(candidate, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{2,80}", candidate.strip()):
            codes.append(candidate.strip())
        elif warning:
            redacted_count += 1
    if redacted_count:
        codes.append(fallback_code)
    return _dedupe(codes)


def _generated_materializer_timeout_seconds() -> int:
    raw = os.environ.get("FEDREADY_GENERATED_MATERIALIZER_TIMEOUT_SECONDS", "3600")
    try:
        value = int(raw)
    except ValueError:
        value = 3600
    return max(60, value)


def _generated_materializer_environment(workspace: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "HOME": str(workspace),
        "TMPDIR": str(workspace / "tmp"),
    }
    (workspace / "tmp").mkdir(parents=True, exist_ok=True)
    return env


def _read_generated_sample_rows(path: Path, *, sample_manifest_format: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"sample manifest does not exist: {path.name}")
    if sample_manifest_format.endswith("json") and not sample_manifest_format.startswith("jsonl"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = []
            for key in ("training", "validation", "test", "records", "samples"):
                value = payload.get(key)
                if isinstance(value, list):
                    for row in value:
                        if isinstance(row, dict):
                            split = {"training": "train"}.get(key, key)
                            rows.append({**row, "split": row.get("split") or split})
            return rows
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        raise ValueError("JSON sample manifest was not an object or list")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("JSONL sample manifest contained a non-object row")
        rows.append(row)
    return rows


def _validate_generated_sample_rows(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    generated_contract: dict[str, Any],
) -> None:
    if not rows:
        raise ValueError("sample manifest contained no records")
    field_names = generated_contract_materialized_field_names(generated_contract)
    required_fields = field_names["required"]
    all_fields = field_names["all"]
    if not required_fields:
        raise ValueError("generated contract must declare required sample fields")
    path_fields = _generated_sample_path_fields(all_fields)
    for index, row in enumerate(rows, start=1):
        missing_fields = [
            field
            for field in sorted(required_fields)
            if field not in row
            or row.get(field) is None
            or row.get(field) == ""
            or (isinstance(row.get(field), list) and not row.get(field))
        ]
        if missing_fields:
            raise ValueError(f"row {index} missing required fields: {', '.join(missing_fields)}")
        split = str(row.get("split") or "")
        if split not in STANDARD_SPLITS:
            raise ValueError(f"row {index} has invalid split")
        image_value = row.get("image") or row.get("image_path")
        if not isinstance(image_value, str) or not image_value.strip():
            raise ValueError(f"row {index} missing image path")
        _resolve_output_dir_relative_path(
            output_dir,
            image_value,
            field="image",
            require_existing=True,
            artifact_kind="generated sample manifest row",
        )
        for field in sorted(path_fields - {"image", "image_path"}):
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                _resolve_output_dir_relative_path(
                    output_dir,
                    value,
                    field=field,
                    require_existing=True,
                    artifact_kind="generated sample manifest row",
                )


def _generated_sample_path_fields(fields: set[str]) -> set[str]:
    return {
        field
        for field in fields
        if field in {"image", "image_path", "mask", "mask_path", "label_source", "label_source_path"}
        or field.endswith("_path")
    }


def _generated_visual_qc_bundle(
    *,
    report: dict[str, Any],
    generated_contract: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    visual_qc = report.get("visual_qc") if isinstance(report.get("visual_qc"), dict) else {}
    if visual_qc.get("available") is True and isinstance(visual_qc.get("artifacts"), list):
        artifacts = [artifact for item in visual_qc["artifacts"] if (artifact := _generated_visual_qc_artifact(item))]
        transform_candidates = _safe_visual_qc_transform_candidates(
            visual_qc.get("transform_candidates"),
            default=("as_is",),
        )
        bundle = {
            "schema_version": VISUAL_QC_BUNDLE_SCHEMA,
            "available": bool(artifacts),
            "sample_count": len(artifacts),
            "requested_sample_count": len(artifacts),
            "review_required": generated_contract_visual_qc_required(generated_contract),
            "visual_qc_owner": "generated_materializer",
            "reviewer": "client_agent_visual_review",
            "transform_candidates": transform_candidates,
            "purpose": "local generated-contract visual verification",
            "artifacts": artifacts,
            "local_output_path_redacted": True,
        }
        _validate_visual_qc_artifact_references(bundle, output_dir=output_dir, require_existing=True)
        return bundle
    return {
        "schema_version": VISUAL_QC_BUNDLE_SCHEMA,
        "available": False,
        "reason": "generated materializer did not provide local visual QC artifacts",
        "sample_count": 0,
        "artifacts": [],
        "review_required": generated_contract_visual_qc_required(generated_contract),
        "visual_qc_owner": "generated_materializer",
        "transform_candidates": ["as_is"],
        "local_output_path_redacted": True,
    }


def _generated_visual_qc_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    artifact: dict[str, Any] = {}
    sample_index = value.get("sample_index")
    if isinstance(sample_index, int) and not isinstance(sample_index, bool) and sample_index >= 0:
        artifact["sample_index"] = sample_index
    split = value.get("split")
    if split in STANDARD_SPLITS:
        artifact["split"] = split
    for key in ("image", "label", "overlay", "candidate_sheet"):
        path = value.get(key)
        if isinstance(path, str) and path.strip():
            artifact[key] = path.strip()
    candidate_overlays = value.get("candidate_overlays")
    if isinstance(candidate_overlays, dict):
        safe_overlays = {
            transform: path.strip()
            for transform, path in candidate_overlays.items()
            if transform in VISUAL_QC_TRANSFORMS and isinstance(path, str) and path.strip()
        }
        if safe_overlays:
            artifact["candidate_overlays"] = safe_overlays
    if not any(key in artifact for key in ("image", "label", "overlay", "candidate_sheet", "candidate_overlays")):
        return {}
    artifact["local_artifact_path_redacted"] = True
    return artifact


def _generated_preview(*, output_dir: Path, rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    preview = report.get("preview") if isinstance(report.get("preview"), dict) else None
    if isinstance(preview, dict) and "available" in preview:
        return preview
    if not rows:
        return {"available": False, "reason": "no extracted samples", "local_output_path_redacted": True}
    image_value = rows[0].get("image") or rows[0].get("image_path")
    if not isinstance(image_value, str):
        return {"available": False, "reason": "sample image missing", "local_output_path_redacted": True}
    source = output_dir / image_value
    if not source.exists():
        return {"available": False, "reason": "sample image missing", "local_output_path_redacted": True}
    preview_image = output_dir / "sample_image.png"
    with Image.open(source) as image:
        image.convert("RGB").save(preview_image)
    return {
        "available": True,
        "image": "sample_image.png",
        "label_kind": "generated_contract",
        "source_sample_split": rows[0].get("split"),
        "local_output_path_redacted": True,
    }


def _generated_storage_sections(
    *, generated_contract: dict[str, Any], rows: list[dict[str, Any]], report: dict[str, Any]
) -> dict[str, Any]:
    del generated_contract, rows, report
    return {"mask_storage": None, "classification_storage": None, "object_detection_storage": None}


def _generated_label_rule_applied(generated_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_kind": generated_contract_record_type(generated_contract),
        "canonical_labels": generated_contract.get("class_map")
        or generated_contract.get("target_space")
        or generated_contract.get("label_space"),
        "resize": None,
    }


def _generated_intensity_stats(*, output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    accumulator = _new_intensity_accumulator()
    for row in rows[: min(len(rows), 64)]:
        image_value = row.get("image") or row.get("image_path")
        if not isinstance(image_value, str):
            continue
        image_path = output_dir / image_value
        if not image_path.exists():
            continue
        try:
            with Image.open(image_path) as image:
                _update_intensity_accumulator(accumulator, image.convert("RGB"))
        except OSError:
            continue
    return _finalize_intensity_accumulator(accumulator)


def _safe_generated_contract_summary(generated_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": generated_contract.get("schema_version"),
        "name": generated_contract.get("name") or generated_contract.get("contract_id"),
        "record_type": generated_contract.get("record_type"),
        "sample_manifest": generated_contract.get("sample_manifest"),
        "sample_manifest_format": generated_contract.get("sample_manifest_format"),
        "safe_to_share": True,
    }


def _safe_generated_materializer_summary(generated_materializer: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": generated_materializer.get("schema_version"),
        "status": generated_materializer.get("status"),
        "record_type": generated_materializer.get("record_type"),
        "source_digest": generated_materializer.get("source_digest"),
        "source_file_count": generated_materializer.get("source_file_count"),
        "safe_to_share": True,
    }


def _redact_policy_for_materializer(policy: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(policy)
    redacted.pop("generated_data_materializer", None)
    return redacted


def _short_materializer_diagnostic(stdout: str | None, stderr: str | None) -> str:
    text = "\n".join(part for part in (stdout or "", stderr or "") if part).strip()
    if len(text) > 1600:
        return text[:1600] + "..."
    return text


def _generated_materializer_result_issue_code(result: dict[str, Any]) -> str:
    code = result.get("issue_code")
    if isinstance(code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{2,80}", code):
        return code
    return "GENERATED_MATERIALIZER_FAILED"


def _generated_materializer_public_reason(reason_code: str) -> str:
    reasons = {
        "GENERATED_MATERIALIZER_PROCESS_FAILED": "Generated data materializer exited unsuccessfully inside the client boundary.",
        "GENERATED_MATERIALIZER_REPORT_MISSING": "Generated data materializer did not produce the required harness report.",
        "GENERATED_MATERIALIZER_TIMEOUT": "Generated data materializer exceeded the client-local execution timeout.",
        "GENERATED_MATERIALIZER_RUNTIME_ERROR": "Generated data materializer failed client-local harness validation.",
        "GENERATED_MATERIALIZER_FAILED": "Generated data materializer failed inside the client boundary.",
    }
    return reasons.get(reason_code, reasons["GENERATED_MATERIALIZER_FAILED"])


def _generated_materializer_failure_result(
    *,
    client_id: str,
    policy: dict[str, Any],
    source_label_type: str,
    local_adapter: dict[str, Any] | None,
    adapter_manifest: dict[str, Any] | None,
    reason: str,
    warnings: list[str],
    reason_code: str = "SERVER_GENERATED_MATERIALIZER_FAILED",
) -> dict[str, Any]:
    return {
        **_failed_extraction_result(
            client_id=client_id,
            policy=policy,
            source_label_type=source_label_type,
            adapter_summary=_local_adapter_safe_summary(local_adapter, adapter_manifest=adapter_manifest),
        ),
        "screening": {
            "schema_version": "fedready.local_label_screening.v1",
            "status": "failed",
            "reason_code": reason_code,
            "reason": reason,
            "safe_to_share": True,
        },
        "warnings": _dedupe([reason_code, *_stable_warning_codes(warnings, fallback_code="ADAPTER_WARNING")]),
    }


def _generated_data_materializer(policy: dict[str, Any]) -> dict[str, Any] | None:
    value = policy.get("generated_data_materializer")
    return value if isinstance(value, dict) and value.get("status") == "implemented" else None


def _generated_materializer_required_result(
    *,
    client_id: str,
    policy: dict[str, Any],
    source_label_type: str,
    local_adapter: dict[str, Any] | None,
    adapter_manifest: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    generated_contract = _generated_data_contract(policy)
    generated_type = generated_contract_record_type(generated_contract)
    reason = (
        "This task uses a server-generated data contract. The shared extractor does not materialize "
        "generated task outputs itself; a server-authored generated-contract materializer must be "
        "produced and shipped with the extraction policy before client-local extraction can continue."
    )
    return {
        **_failed_extraction_result(
            client_id=client_id,
            policy=policy,
            source_label_type=source_label_type,
            adapter_summary=_local_adapter_safe_summary(local_adapter, adapter_manifest=adapter_manifest),
        ),
        "screening": {
            "schema_version": "fedready.local_label_screening.v1",
            "status": "capability_gap",
            "reason_code": "SERVER_GENERATED_MATERIALIZER_REQUIRED",
            "reason": reason,
            "generated_record_type": generated_type,
            "safe_to_share": True,
        },
        "agent_action_required": {
            "required": True,
            "kind": "server_generated_contract_materializer",
            "implemented_by_baseline": False,
            "safe_to_share": True,
        },
        "warnings": _dedupe([reason, *warnings]),
    }
