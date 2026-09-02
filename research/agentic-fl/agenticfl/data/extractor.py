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

"""Policy-driven local dataset extraction for AgenticFL client agents.

This module is intended to run on a client after the server has sent a
site-specific extraction policy through FLARE. It resolves only the local
client data path, writes extracted PNG images and canonical masks into the
project-level ``data/dataset_fl/<client_id>/`` folder, records deferred
training-time transform parameters locally, and returns aggregate metadata that
is safe to share.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Any, Callable

from agenticfl.agents.local_adapter import (
    UNLIMITED_LOCAL_ADAPTER_RECORDS,
    ClientDataProvenanceSnapshot,
    execute_local_adapter,
)
from agenticfl.data.contracts import (
    RUNTIME_CONTRACTS,
    STANDARD_SPLITS,
    generated_contract_materialized_field_names,
    generated_contract_record_type,
    generated_contract_visual_qc_required,
    generated_data_contract_validation_errors,
    infer_record_type,
    manifest_record_type,
    runtime_contract_for_record_type,
    task_family_from_policy,
)
from agenticfl.data.parser import _load_json, _tokens, list_client_ids, validate_client_id
from agenticfl.data.qc import VISUAL_QC_TRANSFORMS
from agenticfl.utils.logging import canonical_json, payload_digest

try:  # Pillow is optional in package metadata, but available in vlm_env.
    from PIL import Image, ImageDraw, ImageStat
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageStat = None  # type: ignore[assignment]


EXTRACTION_RESULT_SCHEMA = "agenticfl.local_extraction_result.v1"
SITE_EXTRACTION_POLICY_SCHEMA = "agenticfl.site_extraction_policy.v1"
TRAINING_TRANSFORM_SCHEMA = "agenticfl.training_transform_policy.v1"
VISUAL_QC_BUNDLE_SCHEMA = "agenticfl.local_visual_qc_bundle.v1"
VISUAL_QC_CONTEXT_SCHEMA = "agenticfl.extraction_visual_qc_context.v1"
LOCAL_ADAPTER_SPEC_SCHEMA = "agenticfl.local_adapter_spec.v1"
LOCAL_ADAPTER_MANIFEST_SCHEMA = "agenticfl.local_adapter_manifest.v1"


def _extractor_logic_fingerprint() -> str:
    """Fingerprint extraction sources so cache invalidation cannot rely on a manual bump."""

    package_root = Path(__file__).resolve().parents[1]
    sources = (
        Path(__file__).resolve(),
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
COMMON_SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
VISUAL_QC_PANEL_MAX_DIMENSION = int(os.environ.get("AGENTICFL_VISUAL_QC_PANEL_MAX_DIMENSION", "512"))
VISUAL_QC_INLINE_MAX_BYTES = 175_000
_visual_qc_max_bytes_env = os.environ.get("AGENTICFL_VISUAL_QC_MAX_BYTES")
VISUAL_QC_MAX_BYTES = int(_visual_qc_max_bytes_env) if _visual_qc_max_bytes_env else None
VISUAL_QC_MIN_LONG_SIDE = 192
VISUAL_QC_PALETTE_COLORS = 128


LOCAL_ADAPTER_SOURCE_ROLE_TOKENS = {
    "image",
    "images",
    "img",
    "input",
    "inputs",
    "original",
    "photo",
    "raw",
    "scan",
    "source",
    "volume",
}


@dataclass(frozen=True)
class ExtractionConfig:
    """Controls local extraction output and scan bounds."""

    output_root: str = "data/dataset_fl"
    output_name: str | None = None
    max_samples: int | None = None
    max_scan_files: int = 200_000
    overwrite: bool = False
    validation_fraction: float = 0.1
    split_seed: int = 20260603
    mask_threshold: int = 0

    def __post_init__(self) -> None:
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be at least 1 when provided")
        if self.max_scan_files < 1:
            raise ValueError("max_scan_files must be at least 1")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not 0 <= self.mask_threshold <= 255:
            raise ValueError("mask_threshold must be between 0 and 255")


def load_site_extraction_policy(strategy_path: str | Path, client_id: str) -> dict[str, Any]:
    """Extract one client's site policy from a full server state or strategy JSON."""

    raw = _load_json(Path(strategy_path))
    strategy = raw.get("extraction_strategy") if isinstance(raw.get("extraction_strategy"), dict) else raw
    if not isinstance(strategy, dict):
        raise ValueError("extraction strategy must be a JSON object")

    per_site = strategy.get("per_site_label_mapping")
    if not isinstance(per_site, dict):
        if strategy.get("site_label_mapping"):
            return strategy
        raise ValueError("extraction strategy missing per_site_label_mapping")

    site_mapping = per_site.get(client_id)
    if not isinstance(site_mapping, dict):
        return {
            "schema_version": SITE_EXTRACTION_POLICY_SCHEMA,
            "client_id": client_id,
            "applicable": False,
            "reason": "client_id is not present in per_site_label_mapping",
            "strategy_digest": strategy.get("strategy_digest"),
        }

    return {
        "schema_version": SITE_EXTRACTION_POLICY_SCHEMA,
        "client_id": client_id,
        "applicable": True,
        "strategy_digest": strategy.get("strategy_digest") or payload_digest(strategy),
        "image_rule": strategy.get("image_rule", {}),
        "label_rule": strategy.get("label_rule", {}),
        "split_rule": strategy.get("split_rule", {}),
        "site_label_mapping": site_mapping,
    }


def extract_site_dataset(
    site_meta_path: str | Path,
    client_id: str,
    *,
    policy: dict[str, Any],
    project_root: str | Path | None = None,
    config: ExtractionConfig | None = None,
    local_adapter: dict[str, Any] | None = None,
    local_adapter_provenance_snapshot: ClientDataProvenanceSnapshot | None = None,
) -> dict[str, Any]:
    """Resolve one client's data path from site metadata and extract locally."""

    meta_path = Path(site_meta_path).resolve()
    root = Path(project_root).resolve() if project_root is not None else meta_path.parent.parent.resolve()
    data_path = _resolve_site_data_path(meta_path, client_id, project_root=root)
    result = extract_dataset(
        data_path,
        client_id=client_id,
        policy=policy,
        config=config,
        project_root=root,
        local_adapter=local_adapter,
        local_adapter_provenance_snapshot=local_adapter_provenance_snapshot,
    )
    result["site_meta"] = {"client_id_resolved": client_id, "data_path_redacted": True}
    return result


def extract_dataset(
    data_path: str | Path,
    *,
    client_id: str,
    policy: dict[str, Any],
    config: ExtractionConfig | None = None,
    project_root: str | Path | None = None,
    local_adapter: dict[str, Any] | None = None,
    local_adapter_provenance_snapshot: ClientDataProvenanceSnapshot | None = None,
) -> dict[str, Any]:
    """Apply a client-local adapter manifest and write extracted artifacts."""

    if Image is None:
        raise RuntimeError("Pillow is required for image extraction")

    cfg = config or ExtractionConfig()
    if policy.get("applicable") is False:
        return _not_applicable_result(client_id, policy, "extraction policy is not applicable to this client")

    site_mapping = _site_mapping(policy)
    generated_contract = _generated_data_contract(policy)
    source_label_type = str(site_mapping.get("source_label_type", "unknown"))
    policy_record_type = task_family_from_policy(policy, source_label_type)
    target_size = _target_size(policy)
    target_terms = _target_terms(policy)
    conversion_options = [str(item) for item in site_mapping.get("conversion_options", []) if isinstance(item, str)]
    root = Path(data_path)
    output_name = cfg.output_name or _safe_slug(str(site_mapping.get("canonical_task") or "extracted"))
    output_dir = _extraction_output_dir(
        client_id=client_id,
        output_root=cfg.output_root,
        project_root=project_root,
    )
    deferred_adapter_staging_dir: Path | None = None
    deferred_adapter_output_prepared = False
    if isinstance(local_adapter, dict) and _local_adapter_full_run_deferred(local_adapter):
        if output_dir.exists():
            if not cfg.overwrite:
                raise FileExistsError(f"extraction output already exists: {output_dir}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        deferred_result = _run_deferred_local_adapter_full(
            local_adapter=local_adapter,
            data_path=root,
            output_dir=output_dir,
            max_records=cfg.max_samples,
            provenance_snapshot=local_adapter_provenance_snapshot,
            data_contract=generated_contract,
        )
        deferred_adapter_staging_dir = deferred_result.get("staging_dir")
        if deferred_result.get("status") != "passed":
            diagnostic = str(deferred_result.get("diagnostic") or "deferred full local adapter run failed")
            _cleanup_deferred_adapter_staging(deferred_adapter_staging_dir)
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
            return _adapter_manifest_required_result(
                client_id=client_id,
                policy=policy,
                source_label_type=source_label_type,
                local_adapter=local_adapter,
                warnings=[diagnostic],
                client_local_diagnostic=diagnostic,
            )
        local_adapter = _local_adapter_with_deferred_full_result(
            local_adapter=local_adapter,
            manifest_path=deferred_result["manifest_path"],
            record_count=int(deferred_result["record_count"]),
        )
        deferred_adapter_output_prepared = True

    adapter_manifest, adapter_warnings = _load_local_adapter_manifest(local_adapter, client_id=client_id)
    adapter_status = local_adapter.get("status") if isinstance(local_adapter, dict) else None
    if isinstance(local_adapter, dict):
        if adapter_status == "unfeasible":
            return _adapter_unfeasible_result(
                client_id=client_id,
                policy=policy,
                source_label_type=source_label_type,
                local_adapter=local_adapter,
            )
        if adapter_status != "implemented" or adapter_manifest is None:
            return _adapter_manifest_required_result(
                client_id=client_id,
                policy=policy,
                source_label_type=source_label_type,
                local_adapter=local_adapter,
                warnings=adapter_warnings,
            )
    elif local_adapter is None:
        return _adapter_manifest_required_result(
            client_id=client_id,
            policy=policy,
            source_label_type=source_label_type,
            local_adapter=None,
            warnings=["Extraction requires a validated client-local adapter manifest."],
        )

    adapter_record_type = (
        _adapter_manifest_record_type(adapter_manifest, generated_contract=generated_contract)
        if adapter_manifest is not None
        else "unknown"
    )
    record_type = adapter_record_type if adapter_record_type != "unknown" else policy_record_type
    contract_runtime = runtime_contract_for_record_type(record_type)
    if contract_runtime is None:
        generated_materializer = _generated_data_materializer(policy)
        if generated_materializer is not None:
            generated_result = _extract_generated_contract_dataset(
                client_id=client_id,
                policy=policy,
                cfg=cfg,
                output_dir=output_dir,
                output_name=output_name,
                target_size=target_size,
                target_terms=target_terms,
                source_label_type=source_label_type,
                conversion_options=conversion_options,
                local_adapter=local_adapter,
                adapter_manifest=adapter_manifest,
                adapter_warnings=adapter_warnings,
                generated_contract=generated_contract,
                generated_materializer=generated_materializer,
            )
            _cleanup_deferred_adapter_staging(deferred_adapter_staging_dir)
            return generated_result
        _cleanup_deferred_adapter_staging(deferred_adapter_staging_dir)
        return _generated_materializer_required_result(
            client_id=client_id,
            policy=policy,
            source_label_type=source_label_type,
            local_adapter=local_adapter,
            adapter_manifest=adapter_manifest,
            warnings=adapter_warnings,
        )
    contract = contract_runtime.CONTRACT

    if not cfg.overwrite and local_adapter is None:
        existing = _reuse_existing_extraction(
            output_dir=output_dir,
            client_id=client_id,
            policy=policy,
            cfg=cfg,
            output_name=output_name,
            target_size=target_size,
            source_label_type=source_label_type,
            conversion_options=conversion_options,
        )
        if existing is not None:
            return existing

    if output_dir.exists() and not deferred_adapter_output_prepared:
        if not cfg.overwrite:
            raise FileExistsError(f"extraction output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    _ensure_split_dirs(output_dir, root_names=contract_runtime.split_roots())

    scan_warnings: list[str] = []
    adapter_pairs, label_warnings = contract_runtime.manifest_pairs(adapter_manifest)
    candidate_pairs = adapter_pairs
    pair_warnings: list[str] = []
    valid_pairs, split_plan = _organize_adapter_pairs_for_fl_splits(adapter_pairs, policy, cfg)
    primary_record_count = len(adapter_pairs)
    selected_label_record_count = len(adapter_pairs)
    full_valid_case_count = len(valid_pairs)
    label_orientation = contract_runtime.orientation_rule(adapter_manifest)
    if cfg.max_samples is not None:
        pairs, split_plan = _bounded_sample_pairs_for_default_fl_splits(
            valid_pairs,
            policy,
            cfg,
            stable_key=lambda pair: str(pair.get("stable_key", pair.get("adapter_record_index", ""))),
            source="bounded_client_local_adapter_manifest",
        )
    else:
        pairs = valid_pairs
    screening = _extraction_screening(
        primary_record_count=primary_record_count,
        selected_label_record_count=selected_label_record_count,
        paired_case_count=len(candidate_pairs),
        valid_case_count=full_valid_case_count,
        execution_pair_count=len(pairs),
        target_terms=target_terms,
    )

    verification = _extraction_verification(
        full_valid_case_count=full_valid_case_count,
        expected_extracted_count=len(pairs),
        extracted_count=0,
        failed_pair_count=0,
        max_samples=cfg.max_samples,
        target_terms=target_terms,
    )
    if screening.get("status") == "screened_out" and not pairs:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        visual_qc = _contract_visual_qc_bundle(
            contract_runtime=contract_runtime,
            output_dir=output_dir,
            rows=[],
            sample_count=_visual_qc_sample_count(policy),
        )
        return {
            "schema_version": EXTRACTION_RESULT_SCHEMA,
            "client_id": client_id,
            "data": "screened out",
            "screening": screening,
            "policy_digest": policy.get("strategy_digest") or payload_digest(policy),
            "source_label_type": source_label_type,
            "local_adapter": (
                _local_adapter_safe_summary(local_adapter, adapter_manifest=adapter_manifest)
                if local_adapter is not None or adapter_manifest is not None
                else None
            ),
            "conversion_options": conversion_options,
            "counts": {"total": 0, "by_split": {}},
            "failed_pairs": {},
            "split_plan": split_plan,
            "label_orientation": {
                "selected_transform": label_orientation["selected_transform"],
                "strategy": label_orientation["strategy"],
                "safe_to_share": True,
            },
            "verification": verification,
            "extraction": {
                "output_root": cfg.output_root,
                "client_folder": client_id,
                "output_name": output_name,
                "local_output_path_redacted": True,
                "materialized_output": False,
                "reused_existing_output": False,
                "preview_available": False,
                "visual_qc_available": False,
                "target_size": list(target_size),
                "stored_resolution": "source_image_resolution",
                "training_transform_policy": None,
                "image_rule_applied": {},
                "label_rule_applied": {},
            },
            "privacy": {
                "safe_to_share": True,
                "redacted": [
                    "source_local_paths",
                    "source_filenames",
                    "sample_ids",
                    "raw_images",
                    "raw_masks",
                    "raw_annotations",
                ],
            },
            "visual_qc_artifacts": _visual_qc_safe_summary(visual_qc),
            "warnings": _dedupe(scan_warnings + label_warnings + pair_warnings + adapter_warnings),
        }

    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []
    intensity_accumulator = _new_intensity_accumulator()
    sample_manifest_name = contract.sample_manifest
    sample_manifest_format = contract.sample_manifest_format
    sample_manifest_path = output_dir / sample_manifest_name
    for pair in pairs:
        try:
            row = contract_runtime.materialize_pair(
                pair=pair,
                output_dir=output_dir,
                source_label_type=source_label_type,
                intensity_accumulator=intensity_accumulator,
                update_intensity=_update_intensity_accumulator,
            )
        except Exception:  # noqa: BLE001 - local data formats vary; count and continue.
            failures[pair["split"]] += 1
            continue
        counts[row["split"]] += 1
        sample_rows.append(row)
    contract_runtime.write_sample_manifest(
        sample_manifest_path,
        rows=sample_rows,
        policy=policy,
    )

    extracted_count = sum(counts.values())
    failed_pair_count = sum(failures.values())
    preview = contract_runtime.preview(output_dir, sample_rows)
    visual_qc = _contract_visual_qc_bundle(
        contract_runtime=contract_runtime,
        output_dir=output_dir,
        rows=sample_rows,
        sample_count=_visual_qc_sample_count(policy),
    )
    storage_sections = contract_runtime.storage_sections(policy, sample_rows)
    intensity_stats = _finalize_intensity_accumulator(intensity_accumulator)
    training_transform_policy = _write_training_transform_policy(
        output_dir=output_dir,
        client_id=client_id,
        policy=policy,
        target_size=target_size,
        image_intensity=intensity_stats,
        counts={"total": extracted_count, "by_split": dict(sorted(counts.items()))},
        label_rule_applied=contract_runtime.label_rule_applied(policy),
        storage_sections=storage_sections,
    )
    verification = _extraction_verification(
        full_valid_case_count=full_valid_case_count,
        expected_extracted_count=len(pairs),
        extracted_count=extracted_count,
        failed_pair_count=failed_pair_count,
        max_samples=cfg.max_samples,
        target_terms=target_terms,
    )

    manifest = {
        "schema_version": "agenticfl.local_extracted_manifest.v1",
        "extractor_logic_version": EXTRACTOR_LOGIC_VERSION,
        "client_id": client_id,
        "policy_digest": policy.get("strategy_digest") or payload_digest(policy),
        "extraction_policy_digest": _extraction_policy_digest(policy),
        "target_size": list(target_size),
        "stored_resolution": "source_image_resolution",
        "training_transform_policy": "training_transforms.json",
        "source_label_type": source_label_type,
        "record_type": contract.record_type,
        "local_adapter": (
            _local_adapter_safe_summary(local_adapter, adapter_manifest=adapter_manifest)
            if local_adapter is not None or adapter_manifest is not None
            else None
        ),
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
        "failed_pairs": dict(sorted(failures.items())),
        "split_plan": split_plan,
        "label_orientation": label_orientation,
        "screening": screening,
        "verification": verification,
        "preview": preview,
        "visual_qc": visual_qc,
        "sample_manifest": sample_manifest_name,
        "sample_manifest_format": sample_manifest_format,
        "output": {
            "layout": "project_client_folder",
            "root": cfg.output_root,
            "client_folder": client_id,
            "run_label": output_name,
            "local_output_path_redacted": True,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _cleanup_deferred_adapter_staging(deferred_adapter_staging_dir)

    return {
        "schema_version": EXTRACTION_RESULT_SCHEMA,
        "client_id": client_id,
        "data": _extraction_status(
            extracted_count=extracted_count,
            screening=screening,
            verification=verification,
        ),
        "screening": screening,
        "policy_digest": manifest["policy_digest"],
        "source_label_type": source_label_type,
        "record_type": contract.record_type,
        "local_adapter": (
            _local_adapter_safe_summary(local_adapter, adapter_manifest=adapter_manifest)
            if local_adapter is not None or adapter_manifest is not None
            else None
        ),
        "conversion_options": conversion_options,
        "counts": manifest["counts"],
        "failed_pairs": manifest["failed_pairs"],
        "split_plan": split_plan,
        "label_orientation": {
            "selected_transform": label_orientation["selected_transform"],
            "strategy": label_orientation["strategy"],
            "safe_to_share": True,
        },
        "verification": verification,
        "extraction": {
            "output_root": cfg.output_root,
            "client_folder": client_id,
            "output_name": output_name,
            "local_output_path_redacted": True,
            "reused_existing_output": False,
            "preview_available": preview["available"],
            "preview_label_value_scale": preview.get("label_value_scale"),
            "visual_qc_available": bool(visual_qc.get("available")),
            "target_size": list(target_size),
            "stored_resolution": "source_image_resolution",
            "training_transform_policy": "training_transforms.json",
            "sample_manifest": sample_manifest_name,
            "sample_manifest_format": sample_manifest_format,
            "classification_storage": storage_sections.get("classification_storage"),
            "image_rule_applied": {
                "channels": "RGB",
                "resize": "deferred_to_training_time",
                "extraction_resize_applied": False,
            },
            "label_rule_applied": contract_runtime.label_rule_applied(policy),
        },
        "privacy": {
            "safe_to_share": True,
            "redacted": [
                "source_local_paths",
                "source_filenames",
                "extracted_local_paths",
                "extracted_filenames",
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
        "visual_qc_artifacts": _visual_qc_safe_summary(visual_qc),
        "warnings": _dedupe(scan_warnings + label_warnings + pair_warnings + adapter_warnings),
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
        "schema_version": "agenticfl.local_extraction_verification.v1",
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
    primary_record_count: int,
    selected_label_record_count: int,
    paired_case_count: int,
    valid_case_count: int,
    execution_pair_count: int,
    target_terms: set[str],
) -> dict[str, Any]:
    status = "usable"
    reason_code = None
    reason = "Local label validation found usable image-label pairs for the extraction policy."
    if selected_label_record_count == 0:
        status = "screened_out"
        reason_code = "NO_USABLE_LABEL_RECORDS"
        reason = "No local label records matched the round-2 extraction policy during local validation."
    elif paired_case_count == 0:
        status = "screened_out"
        reason_code = "NO_IMAGE_LABEL_PAIRS"
        reason = "Local label records existed, but generic local pairing found no usable image-label pairs."
    elif valid_case_count == 0:
        status = "screened_out"
        reason_code = "NO_VALID_CASES_AFTER_SPLIT_RULE"
        reason = "Local image-label pairs existed, but none remained after local split-rule validation."

    return {
        "schema_version": "agenticfl.local_label_screening.v1",
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
        "target_concepts": sorted(target_terms),
        "primary_record_count": primary_record_count,
        "selected_label_record_count": selected_label_record_count,
        "paired_case_count": paired_case_count,
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


def _local_adapter_full_run_deferred(local_adapter: dict[str, Any]) -> bool:
    runtime = local_adapter.get("runtime_validation") if isinstance(local_adapter, dict) else None
    return isinstance(runtime, dict) and runtime.get("full_dataset_deferred_to_extractor") is True


def _deferred_local_adapter_min_records(local_adapter: dict[str, Any]) -> int:
    runtime = (
        local_adapter.get("runtime_validation") if isinstance(local_adapter.get("runtime_validation"), dict) else {}
    )
    value = runtime.get("min_records")
    if isinstance(value, int) and value > 0:
        return value
    declared_count = local_adapter.get("record_count")
    if isinstance(declared_count, int) and declared_count > 0:
        return 1
    return 1


def _deferred_local_adapter_timeout_seconds() -> int:
    raw = os.environ.get("AGENTICFL_LOCAL_ADAPTER_RUNTIME_TIMEOUT_SECONDS", "3600")
    try:
        value = int(raw)
    except ValueError:
        value = 3600
    return max(60, value)


def _run_deferred_local_adapter_full(
    *,
    local_adapter: dict[str, Any],
    data_path: Path,
    output_dir: Path,
    max_records: int | None,
    provenance_snapshot: ClientDataProvenanceSnapshot | None,
    data_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    script_value = local_adapter.get("script_path")
    if not isinstance(script_value, str) or not script_value.strip():
        return {"status": "failed", "diagnostic": "deferred local adapter is missing script_path"}
    script_path = Path(script_value).expanduser().resolve()
    if script_path.name != "adapter.py" or not script_path.is_file():
        return {"status": "failed", "diagnostic": "deferred local adapter script_path is not a readable adapter.py"}

    workspace = script_path.parent
    staging_dir = output_dir / "_adapter_generated_labels"
    manifest_path = output_dir / "_adapter_manifest.json"
    state_path = output_dir / "_adapter_runtime_state.json"
    bounded_max = int(max_records) if max_records is not None else UNLIMITED_LOCAL_ADAPTER_RECORDS
    result = execute_local_adapter(
        workspace=workspace,
        local_data_path=data_path,
        max_records=bounded_max,
        min_records=_deferred_local_adapter_min_records(local_adapter),
        timeout_seconds=_deferred_local_adapter_timeout_seconds(),
        manifest_name=str(manifest_path),
        output_dir_name=str(staging_dir),
        state_file_name=str(state_path),
        stage="full_dataset_data_dir",
        provenance_snapshot=provenance_snapshot,
        data_contract=data_contract,
        allowed_generated_roots=(staging_dir,),
    )
    return {
        "status": result.status,
        "diagnostic": result.diagnostic,
        "record_count": result.record_count,
        "manifest_path": manifest_path,
        "staging_dir": staging_dir,
    }


def _local_adapter_with_deferred_full_result(
    *,
    local_adapter: dict[str, Any],
    manifest_path: Path,
    record_count: int,
) -> dict[str, Any]:
    updated = dict(local_adapter)
    runtime = updated.get("runtime_validation")
    runtime = dict(runtime) if isinstance(runtime, dict) else {}
    stages = [str(value) for value in runtime.get("stages", []) if isinstance(value, str)]
    if "full_dataset_data_dir" not in stages:
        stages.append("full_dataset_data_dir")
    runtime.update(
        {
            "schema_version": "agenticfl.local_adapter_runtime_attestation.v1",
            "status": "passed",
            "stages": stages,
            "full_dataset_deferred_to_extractor": False,
            "full_dataset_materialized_in_data_dir": True,
            "harness_owned": True,
        }
    )
    updated.update(
        {
            "manifest_path": str(manifest_path),
            "record_count": int(record_count),
            "runtime_validation": runtime,
        }
    )
    return updated


def _cleanup_deferred_adapter_staging(staging_dir: Path | None) -> None:
    if staging_dir is None:
        return
    keep_raw = os.environ.get("AGENTICFL_KEEP_DEFERRED_ADAPTER_STAGING", "").strip().lower()
    if keep_raw in {"1", "true", "yes", "on"}:
        return
    staging_dir = Path(staging_dir)
    shutil.rmtree(staging_dir, ignore_errors=True)
    for artifact_name in ("_adapter_manifest.json", "_adapter_runtime_state.json"):
        try:
            (staging_dir.parent / artifact_name).unlink(missing_ok=True)
        except OSError:
            pass


def _load_local_adapter_manifest(
    local_adapter: dict[str, Any] | None,
    *,
    client_id: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(local_adapter, dict):
        return None, []
    if local_adapter.get("status") != "implemented":
        return None, [
            f"Local adapter status was {local_adapter.get('status', 'unknown')}; no adapter manifest consumed."
        ]
    manifest_value = local_adapter.get("manifest_path") or local_adapter.get("adapter_manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        return None, ["Local adapter reported implemented but did not provide an adapter manifest path."]
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.exists():
        return None, ["Local adapter manifest path was reported but the local file was not readable."]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, ["Local adapter manifest was not valid JSON."]
    if not isinstance(manifest, dict):
        return None, ["Local adapter manifest was not a JSON object."]
    if manifest.get("schema_version") != LOCAL_ADAPTER_MANIFEST_SCHEMA:
        return None, ["Local adapter manifest schema_version did not match agenticfl.local_adapter_manifest.v1."]
    manifest_client = manifest.get("client_id")
    if isinstance(manifest_client, str) and manifest_client != client_id:
        return None, ["Local adapter manifest client_id did not match the local client."]
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        return None, ["Local adapter manifest did not contain any records."]
    manifest["_manifest_path"] = str(manifest_path.resolve())
    return manifest, []


def _adapter_manifest_record_type(
    manifest: dict[str, Any],
    *,
    generated_contract: dict[str, Any] | None = None,
) -> str:
    return manifest_record_type(manifest, generated_contract=generated_contract)


def _adapter_record_type(record: dict[str, Any]) -> str:
    return infer_record_type(record)


def _organize_adapter_pairs_for_fl_splits(
    pairs: list[dict[str, Any]],
    policy: dict[str, Any],
    config: ExtractionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignments, plan = _assign_fl_splits(
        item_count=len(pairs),
        current_split=lambda index: str(pairs[index].get("split", "unknown")),
        stable_key=lambda index: str(pairs[index].get("stable_key", index)),
        config=config,
    )
    adjusted = [{**pair, "split": assignments[index]} for index, pair in enumerate(pairs)]
    split_rule = policy.get("split_rule", {})
    if isinstance(split_rule, dict):
        plan["policy_split_rule"] = {
            "preserve_existing_splits": bool(split_rule.get("preserve_existing_splits", True)),
            "if_validation_missing": split_rule.get("if_validation_missing"),
        }
    plan["source"] = "client_local_adapter_manifest"
    return adjusted, plan


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
        "schema_version": "agenticfl.local_adapter_summary.v1",
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


def _adapter_unfeasible_result(
    *,
    client_id: str,
    policy: dict[str, Any],
    source_label_type: str,
    local_adapter: dict[str, Any],
) -> dict[str, Any]:
    reason = local_adapter.get("reason")
    if not isinstance(reason, str) or not reason:
        reason = "Client-local adapter determined that this dataset cannot produce valid task-aligned records for the active contract."
    return {
        "schema_version": EXTRACTION_RESULT_SCHEMA,
        "client_id": client_id,
        "data": "screened out",
        "policy_digest": policy.get("strategy_digest") or payload_digest(policy),
        "source_label_type": source_label_type,
        "screening": {
            "schema_version": "agenticfl.local_label_screening.v1",
            "status": "screened_out",
            "reason_code": "LOCAL_ADAPTER_DECLARED_UNFEASIBLE",
            "reason": reason,
            "safe_to_share": True,
        },
        "verification": {
            "schema_version": "agenticfl.local_extraction_verification.v1",
            "valid_case_count": 0,
            "expected_extracted_count": 0,
            "extracted_count": 0,
            "failed_pair_count": 0,
            "matches_execution_target": True,
            "matches_all_valid_cases": True,
            "passed": True,
        },
        "local_adapter": _local_adapter_safe_summary(local_adapter, adapter_manifest=None),
        "privacy": {
            "safe_to_share": True,
            "redacted": ["source_local_paths", "source_filenames", "raw_images", "raw_masks", "raw_annotations"],
        },
    }


def _adapter_manifest_required_result(
    *,
    client_id: str,
    policy: dict[str, Any],
    source_label_type: str,
    local_adapter: dict[str, Any] | None,
    warnings: list[str],
    client_local_diagnostic: str | None = None,
) -> dict[str, Any]:
    reason = "Live extraction requires an implemented client-local adapter with a valid manifest."
    result = {
        "schema_version": EXTRACTION_RESULT_SCHEMA,
        "client_id": client_id,
        "data": "extraction verification failed",
        "policy_digest": policy.get("strategy_digest") or payload_digest(policy),
        "source_label_type": source_label_type,
        "screening": {
            "schema_version": "agenticfl.local_label_screening.v1",
            "status": "failed",
            "reason_code": "LOCAL_ADAPTER_MANIFEST_REQUIRED",
            "reason": reason,
            "safe_to_share": True,
        },
        "verification": {
            "schema_version": "agenticfl.local_extraction_verification.v1",
            "valid_case_count": 0,
            "expected_extracted_count": 0,
            "extracted_count": 0,
            "failed_pair_count": 0,
            "matches_execution_target": False,
            "matches_all_valid_cases": False,
            "passed": False,
        },
        "local_adapter": _local_adapter_safe_summary(local_adapter, adapter_manifest=None),
        "privacy": {
            "safe_to_share": True,
            "redacted": ["source_local_paths", "source_filenames", "raw_images", "raw_masks", "raw_annotations"],
        },
        "warnings": _dedupe(
            ["LOCAL_ADAPTER_MANIFEST_REQUIRED", *_stable_warning_codes(warnings, fallback_code="ADAPTER_WARNING")]
        ),
    }
    if client_local_diagnostic:
        result["_client_local_diagnostic"] = client_local_diagnostic
    return result


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
        primary_record_count=adapter_record_count,
        selected_label_record_count=adapter_record_count,
        paired_case_count=adapter_record_count,
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
    if _generated_contract_requires_visual_qc(generated_contract) and not visual_qc.get("available"):
        verification = {**verification, "passed": False, "generated_contract_visual_qc_missing": True}
    preview = _generated_preview(output_dir=output_dir, rows=sample_rows, report=report)
    storage_sections = _generated_storage_sections(
        generated_contract=generated_contract, rows=sample_rows, report=report
    )
    intensity_stats = _generated_intensity_stats(output_dir=output_dir, rows=sample_rows)
    training_transform_policy = _write_training_transform_policy(
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
        "schema_version": "agenticfl.local_split_plan.v1",
        "source": "server_generated_contract_materializer",
        "preserve_existing_splits": True,
        "sample_limited": cfg.max_samples is not None and adapter_record_count > (cfg.max_samples or 0),
    }
    label_orientation = {
        "schema_version": "agenticfl.local_label_orientation_rule.v1",
        "strategy": "generated_contract_materializer",
        "selected_transform": "as_is",
        "reason": "Generated spatial materializers write canonical as-is output geometry.",
        "safe_to_share": True,
    }
    manifest = {
        "schema_version": "agenticfl.local_extracted_manifest.v1",
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
    return {
        "schema_version": EXTRACTION_RESULT_SCHEMA,
        "client_id": client_id,
        "data": _extraction_status(
            extracted_count=extracted_count,
            screening=screening,
            verification=verification,
        ),
        "screening": screening,
        "policy_digest": manifest["policy_digest"],
        "source_label_type": source_label_type,
        "record_type": generated_contract_record_type(generated_contract),
        "local_adapter": _local_adapter_safe_summary(local_adapter, adapter_manifest=adapter_manifest),
        "conversion_options": conversion_options,
        "counts": manifest["counts"],
        "failed_pairs": manifest["failed_pairs"],
        "split_plan": split_plan,
        "label_orientation": {
            "selected_transform": "as_is",
            "strategy": "generated_contract_materializer",
            "safe_to_share": True,
        },
        "verification": verification,
        "extraction": {
            "output_root": cfg.output_root,
            "client_folder": client_id,
            "output_name": output_name,
            "local_output_path_redacted": True,
            "reused_existing_output": False,
            "preview_available": preview.get("available") is True,
            "visual_qc_available": bool(visual_qc.get("available")),
            "target_size": list(target_size),
            "stored_resolution": "source_image_resolution",
            "training_transform_policy": "training_transforms.json",
            "sample_manifest": sample_manifest_name,
            "sample_manifest_format": sample_manifest_format,
            "classification_storage": storage_sections.get("classification_storage"),
            "object_detection_storage": storage_sections.get("object_detection_storage"),
            "image_rule_applied": {
                "channels": "RGB",
                "resize": "deferred_to_training_time",
                "extraction_resize_applied": False,
            },
            "label_rule_applied": _generated_label_rule_applied(generated_contract),
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
        "visual_qc_artifacts": _visual_qc_safe_summary(visual_qc),
        "warnings": _dedupe(
            [
                *_stable_warning_codes(adapter_warnings, fallback_code="ADAPTER_WARNING"),
                *_stable_warning_codes(
                    materializer_result.get("warnings", []),
                    fallback_code="GENERATED_MATERIALIZER_WARNING",
                ),
            ]
        ),
    }


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
        if report.get("schema_version") != "agenticfl.generated_materializer_report.v1":
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
    raw = os.environ.get("AGENTICFL_GENERATED_MATERIALIZER_TIMEOUT_SECONDS", "3600")
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
            "review_required": _generated_contract_requires_visual_qc(generated_contract),
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
        "review_required": _generated_contract_requires_visual_qc(generated_contract),
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


def _generated_contract_requires_visual_qc(generated_contract: dict[str, Any]) -> bool:
    return generated_contract_visual_qc_required(generated_contract)


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
        "schema_version": EXTRACTION_RESULT_SCHEMA,
        "client_id": client_id,
        "data": "extraction verification failed",
        "policy_digest": policy.get("strategy_digest") or payload_digest(policy),
        "source_label_type": source_label_type,
        "screening": {
            "schema_version": "agenticfl.local_label_screening.v1",
            "status": "failed",
            "reason_code": reason_code,
            "reason": reason,
            "safe_to_share": True,
        },
        "verification": {
            "schema_version": "agenticfl.local_extraction_verification.v1",
            "valid_case_count": 0,
            "expected_extracted_count": 0,
            "extracted_count": 0,
            "failed_pair_count": 0,
            "matches_execution_target": False,
            "matches_all_valid_cases": False,
            "passed": False,
        },
        "local_adapter": _local_adapter_safe_summary(local_adapter, adapter_manifest=adapter_manifest),
        "privacy": {
            "safe_to_share": True,
            "redacted": ["source_local_paths", "source_filenames", "raw_images", "raw_masks", "raw_annotations"],
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
        "schema_version": EXTRACTION_RESULT_SCHEMA,
        "client_id": client_id,
        "data": "extraction verification failed",
        "policy_digest": policy.get("strategy_digest") or payload_digest(policy),
        "source_label_type": source_label_type,
        "screening": {
            "schema_version": "agenticfl.local_label_screening.v1",
            "status": "capability_gap",
            "reason_code": "SERVER_GENERATED_MATERIALIZER_REQUIRED",
            "reason": reason,
            "generated_record_type": generated_type,
            "safe_to_share": True,
        },
        "verification": {
            "schema_version": "agenticfl.local_extraction_verification.v1",
            "valid_case_count": 0,
            "expected_extracted_count": 0,
            "extracted_count": 0,
            "failed_pair_count": 0,
            "matches_execution_target": False,
            "matches_all_valid_cases": False,
            "passed": False,
        },
        "local_adapter": _local_adapter_safe_summary(local_adapter, adapter_manifest=adapter_manifest),
        "agent_action_required": {
            "required": True,
            "kind": "server_generated_contract_materializer",
            "implemented_by_baseline": False,
            "safe_to_share": True,
        },
        "privacy": {
            "safe_to_share": True,
            "redacted": ["source_local_paths", "source_filenames", "raw_images", "raw_masks", "raw_annotations"],
        },
        "warnings": _dedupe([reason, *warnings]),
    }


def _extraction_policy_digest(policy: dict[str, Any]) -> str:
    """Digest only fields that affect local PNG extraction."""

    image_rule = policy.get("image_rule", {})
    label_rule = policy.get("label_rule", {})
    split_rule = policy.get("split_rule", {})
    digest_body = {
        "schema_version": "agenticfl.extraction_policy_digest.v1",
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
            "schema_version": "agenticfl.local_image_intensity_stats.v1",
            "available": False,
            "reason": "no successfully extracted images",
            "safe_to_share": False,
        }

    sums = accumulator["channel_sum"]
    sums_sq = accumulator["channel_sum_sq"]
    mean_0_255 = [sums[index] / pixel_count for index in range(3)]
    std_0_255 = [sqrt(max((sums_sq[index] / pixel_count) - (mean_0_255[index] ** 2), 0.0)) for index in range(3)]
    return {
        "schema_version": "agenticfl.local_image_intensity_stats.v1",
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


def _read_training_transform_intensity(output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    transform_path = output_dir / str(manifest.get("training_transform_policy", "training_transforms.json"))
    try:
        transform_policy = json.loads(transform_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        transform_policy = {}
    training_transform = transform_policy.get("training_transform")
    if isinstance(training_transform, dict):
        image = training_transform.get("image")
        if isinstance(image, dict):
            intensity = image.get("intensity")
            if isinstance(intensity, dict) and isinstance(intensity.get("site_statistics"), dict):
                return intensity["site_statistics"]
    legacy_stats = manifest.get("image_intensity_statistics")
    return legacy_stats if isinstance(legacy_stats, dict) else {}


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


def _reuse_existing_extraction(
    *,
    output_dir: Path,
    client_id: str,
    policy: dict[str, Any],
    cfg: ExtractionConfig,
    output_name: str,
    target_size: tuple[int, int],
    source_label_type: str,
    conversion_options: list[str],
) -> dict[str, Any] | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not _manifest_matches_policy(
        manifest=manifest,
        client_id=client_id,
        policy=policy,
        target_size=target_size,
        source_label_type=source_label_type,
    ):
        return None

    counts = manifest.get("counts", {})
    total = counts.get("total") if isinstance(counts, dict) else None
    verification = manifest.get("verification", {})
    if not isinstance(total, int) or total <= 0 or not isinstance(verification, dict) or not verification.get("passed"):
        return None
    if verification.get("max_samples") != cfg.max_samples:
        return None

    contract_runtime = _runtime_contract_from_manifest(manifest)
    if contract_runtime is None:
        return None
    sample_manifest = manifest.get("sample_manifest")
    if not isinstance(sample_manifest, str) or not sample_manifest.strip():
        return None
    expected_sample_manifest = contract_runtime.CONTRACT.sample_manifest
    if _relative_path_text(sample_manifest) != _relative_path_text(expected_sample_manifest):
        return None
    try:
        samples_path = _resolve_output_dir_relative_path(
            output_dir,
            expected_sample_manifest,
            field="sample_manifest",
            require_existing=True,
            artifact_kind="prepared sample manifest",
        )
    except ValueError:
        return None
    rows = _read_sample_rows(samples_path)
    if len(rows) != total:
        return None

    rows, split_plan, reorganized = _reorganize_existing_sample_rows(output_dir, rows, cfg, contract_runtime)
    contract_runtime.write_sample_manifest(samples_path, rows=rows, policy=policy)
    manifest["counts"] = {"total": len(rows), "by_split": _row_split_counts(rows)}
    manifest["split_plan"] = split_plan
    manifest["preview"] = contract_runtime.preview(output_dir, rows)
    manifest["visual_qc"] = _contract_visual_qc_bundle(
        contract_runtime=contract_runtime,
        output_dir=output_dir,
        rows=rows,
        sample_count=_visual_qc_sample_count(policy),
    )
    manifest["policy_digest"] = policy.get("strategy_digest") or payload_digest(policy)
    manifest["extraction_policy_digest"] = _extraction_policy_digest(policy)
    manifest["target_size"] = list(target_size)
    manifest["stored_resolution"] = "source_image_resolution"
    manifest["training_transform_policy"] = "training_transforms.json"
    training_transform_policy = _write_training_transform_policy(
        output_dir=output_dir,
        client_id=client_id,
        policy=policy,
        target_size=target_size,
        image_intensity=_read_training_transform_intensity(output_dir, manifest),
        counts=manifest["counts"],
        label_rule_applied=contract_runtime.label_rule_applied(policy),
        storage_sections={
            "mask_storage": manifest.get("mask_storage"),
            "classification_storage": manifest.get("classification_storage"),
            "object_detection_storage": manifest.get("object_detection_storage"),
        },
    )
    manifest.setdefault("output", {})
    if isinstance(manifest["output"], dict):
        manifest["output"]["layout"] = "project_client_folder"
        manifest["output"]["client_folder"] = client_id
        manifest["output"]["run_label"] = output_name
        manifest["output"]["local_output_path_redacted"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    warnings = ["Existing verified extracted output was reused without reprocessing local source data."]
    if reorganized:
        warnings.append("Existing extracted samples were reorganized into train/validation/test folders.")
    return _result_from_manifest(
        manifest=manifest,
        client_id=client_id,
        policy=policy,
        output_name=output_name,
        source_label_type=source_label_type,
        conversion_options=conversion_options,
        reused_existing_output=True,
        warnings=warnings,
    )


def _runtime_contract_from_manifest(manifest: dict[str, Any]):
    sample_manifest = manifest.get("sample_manifest")
    sample_manifest_format = manifest.get("sample_manifest_format")
    for runtime in RUNTIME_CONTRACTS.values():
        contract = runtime.CONTRACT
        if sample_manifest == contract.sample_manifest or sample_manifest_format == contract.sample_manifest_format:
            return runtime
    return None


def _manifest_matches_policy(
    *,
    manifest: dict[str, Any],
    client_id: str,
    policy: dict[str, Any],
    target_size: tuple[int, int],
    source_label_type: str,
) -> bool:
    expected_digest = policy.get("strategy_digest") or payload_digest(policy)
    expected_extraction_digest = _extraction_policy_digest(policy)
    return (
        manifest.get("client_id") == client_id
        and manifest.get("extractor_logic_version") == EXTRACTOR_LOGIC_VERSION
        and manifest.get("extraction_policy_digest", expected_digest) == expected_extraction_digest
        and manifest.get("source_label_type") == source_label_type
    )


def _read_sample_rows(samples_path: Path) -> list[dict[str, Any]]:
    if not samples_path.exists():
        return []
    rows = []
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(row, dict):
            return []
        rows.append(row)
    return rows


def _write_sample_rows(samples_path: Path, rows: list[dict[str, Any]]) -> None:
    samples_path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _reorganize_existing_sample_rows(
    output_dir: Path,
    rows: list[dict[str, Any]],
    cfg: ExtractionConfig,
    contract_runtime: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    assignments, plan = _assign_fl_splits(
        item_count=len(rows),
        current_split=lambda index: str(rows[index].get("split", "unknown")),
        stable_key=lambda index: str(rows[index].get("sample_id") or rows[index].get("image") or index),
        config=cfg,
    )
    root_names = tuple(contract_runtime.split_roots())
    reorganized = False
    updated_rows: list[dict[str, Any]] = []
    _ensure_split_dirs(output_dir, root_names=root_names)
    for index, row in enumerate(rows):
        split = assignments[index]
        updated = {**row, "split": split}
        if "image" in row and "images" in root_names:
            image_rel = _relocate_sample_file(output_dir, row.get("image"), "images", split)
            reorganized = reorganized or image_rel != row.get("image")
            updated["image"] = image_rel
        if "mask" in row and "masks" in root_names:
            mask_rel = _relocate_sample_file(output_dir, row.get("mask"), "masks", split)
            reorganized = reorganized or mask_rel != row.get("mask")
            updated["mask"] = mask_rel
        reorganized = reorganized or row.get("split") != split
        updated_rows.append(updated)
    _remove_nonstandard_split_dirs(output_dir, root_names=root_names)
    return updated_rows, plan, reorganized


def _relocate_sample_file(output_dir: Path, value: Any, root_name: str, split: str) -> str:
    old_rel = Path(str(value)) if isinstance(value, str) and value else Path(root_name) / split / "missing.png"
    new_rel = Path(root_name) / split / old_rel.name
    source = output_dir / old_rel
    target = output_dir / new_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target and source.exists():
        if target.exists():
            source.unlink()
        else:
            shutil.move(str(source), str(target))
    return new_rel.as_posix()


def _remove_nonstandard_split_dirs(output_dir: Path, *, root_names: Sequence[str]) -> None:
    for root_name in root_names:
        root = output_dir / root_name
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name not in STANDARD_SPLITS:
                shutil.rmtree(child)


def _local_adapter_task_tokens(task: str | None) -> set[str]:
    generic = {"image", "imaging", "task", "segmentation", "classification"}
    words = [token for token in _tokens(task or "") if token not in generic]
    tokens = set(words)
    if len(words) >= 2:
        tokens.add("".join(word[0] for word in words if word))
    return tokens


def apply_automatic_orientation_repair(
    *,
    extraction_result: dict[str, Any],
    decision: dict[str, Any],
    project_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a consensus visual-QC transform to built-in segmentation masks.

    Generated spatial contracts retain ownership of their target geometry and
    are intentionally not rewritten by this generic repair path.
    """

    selected_transform = decision.get("selected_transform")
    if (
        selected_transform not in {"hflip", "vflip", "rot180"}
        or decision.get("status") != "failed"
        or decision.get("passed") is not False
        or decision.get("reviewed") is not True
        or decision.get("consensus_reached") is not True
    ):
        return extraction_result, decision
    if extraction_result.get("record_type") != "segmentation":
        return extraction_result, decision
    if Image is None:
        raise RuntimeError("Pillow is required for automatic orientation repair")

    extraction = extraction_result.get("extraction")
    extraction = extraction if isinstance(extraction, dict) else {}
    output_root = str(extraction.get("output_root") or ExtractionConfig.output_root)
    client_id = extraction.get("client_folder") or extraction_result.get("client_id")
    if not isinstance(client_id, str) or not client_id:
        raise ValueError("extraction result is missing client folder for orientation repair")
    output_dir = _extraction_output_dir(
        client_id=client_id,
        output_root=output_root,
        project_root=project_root,
    )
    manifest_path = output_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"extracted manifest is missing for orientation repair: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("extracted manifest is invalid for orientation repair") from exc
    if not isinstance(manifest, dict) or manifest.get("record_type") != "segmentation":
        raise ValueError("automatic orientation repair requires a built-in segmentation manifest")
    samples_file = str(manifest.get("sample_manifest") or "samples.jsonl")
    samples_path = _resolve_output_dir_relative_path(
        output_dir,
        samples_file,
        field="sample_manifest",
        require_existing=True,
        artifact_kind="orientation repair input",
    )

    rows: list[dict[str, Any]] = []
    pending_masks: list[tuple[Path, Path]] = []
    seen_masks: set[Path] = set()
    try:
        for line_number, line in enumerate(samples_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"segmentation sample manifest row {line_number} is invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"segmentation sample manifest row {line_number} must be an object")
            mask_path = _resolve_output_dir_relative_path(
                output_dir,
                row.get("mask"),
                field=f"row_{line_number}.mask",
                require_existing=True,
                artifact_kind="orientation repair input",
            )
            if mask_path in seen_masks:
                raise ValueError("segmentation sample manifest reuses a mask path")
            seen_masks.add(mask_path)
            with Image.open(mask_path) as source_mask:
                repaired_mask = _apply_orientation_transform(source_mask.convert("L"), selected_transform)
                temporary_mask = mask_path.with_name(f".{mask_path.stem}.orientation-repair.tmp{mask_path.suffix}")
                repaired_mask.save(temporary_mask, format="PNG")
            pending_masks.append((temporary_mask, mask_path))
            row["label_orientation"] = selected_transform
            rows.append(row)
    except Exception:
        for temporary_mask, _ in pending_masks:
            temporary_mask.unlink(missing_ok=True)
        raise
    if not rows:
        raise ValueError("segmentation sample manifest has no masks to repair")
    for temporary_mask, mask_path in pending_masks:
        temporary_mask.replace(mask_path)
    repaired_mask_count = len(pending_masks)

    temporary_samples = samples_path.with_name(f".{samples_path.name}.orientation-repair.tmp")
    temporary_samples.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    temporary_samples.replace(samples_path)

    requested_sample_count = 3
    existing_visual_qc = manifest.get("visual_qc")
    if isinstance(existing_visual_qc, dict) and isinstance(existing_visual_qc.get("requested_sample_count"), int):
        requested_sample_count = max(1, int(existing_visual_qc["requested_sample_count"]))
    segmentation_runtime = runtime_contract_for_record_type("segmentation")
    repaired_visual_qc = _contract_visual_qc_bundle(
        contract_runtime=segmentation_runtime,
        output_dir=output_dir,
        rows=rows,
        sample_count=requested_sample_count,
    )
    repaired_preview = segmentation_runtime.preview(output_dir, rows)

    label_orientation = {
        "schema_version": "agenticfl.local_label_orientation_rule.v1",
        "strategy": "client_local_vlm_automatic_repair",
        "selected_transform": "as_is",
        "applied_transform": selected_transform,
        "reason": "Applied the consensus local-VLM transform to every prepared segmentation mask.",
        "safe_to_share": True,
    }
    manifest["label_orientation"] = label_orientation
    manifest["visual_qc"] = repaired_visual_qc
    manifest["preview"] = repaired_preview
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.orientation-repair.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary_manifest.replace(manifest_path)

    repaired_decision = {
        **decision,
        "status": "passed",
        "passed": True,
        "selected_transform": "as_is",
        "pre_repair_selected_transform": selected_transform,
        "selected_transform_counts": {"as_is": int(decision.get("reviewed_sample_count") or 1)},
        "recommendation": "continue_to_training_with_repaired_orientation",
        "orientation_repair": {
            "schema_version": "agenticfl.orientation_repair.v1",
            "applied": True,
            "applied_transform": selected_transform,
            "repaired_mask_count": repaired_mask_count,
            "scope": "all_prepared_segmentation_masks",
        },
    }
    repaired_extraction = {
        **extraction,
        "preview_available": bool(repaired_preview.get("available")),
        "visual_qc_available": bool(repaired_visual_qc.get("available")),
    }
    repaired_result = {
        **extraction_result,
        "label_orientation": label_orientation,
        "visual_qc_artifacts": _visual_qc_safe_summary(repaired_visual_qc),
        "extraction": repaired_extraction,
    }
    return repaired_result, repaired_decision


def _apply_orientation_transform(label: Any, transform: str) -> Any:
    transpose = Image.Transpose if hasattr(Image, "Transpose") else Image
    if transform == "hflip":
        return label.transpose(transpose.FLIP_LEFT_RIGHT)
    if transform == "vflip":
        return label.transpose(transpose.FLIP_TOP_BOTTOM)
    if transform == "rot180":
        return label.transpose(transpose.ROTATE_180)
    raise ValueError(f"unsupported automatic orientation transform: {transform}")


def persist_visual_qc_decision(
    *,
    extraction_result: dict[str, Any],
    decision: dict[str, Any],
    project_root: str | Path | None = None,
) -> Path:
    """Persist the final client-local QC decision beside extraction artifacts."""

    if not isinstance(decision, dict):
        raise TypeError("visual QC decision must be an object")
    status = decision.get("status")
    passed = decision.get("passed")
    if not isinstance(status, str) or not status:
        raise ValueError("visual QC decision is missing status")
    if not isinstance(passed, bool):
        raise ValueError("visual QC decision is missing boolean passed")
    if (status == "passed") != passed:
        raise ValueError("visual QC decision status and passed value are inconsistent")

    extraction = extraction_result.get("extraction")
    extraction = extraction if isinstance(extraction, dict) else {}
    output_root = str(extraction.get("output_root") or ExtractionConfig.output_root)
    client_id = extraction.get("client_folder") or extraction_result.get("client_id")
    if not isinstance(client_id, str) or not client_id:
        raise ValueError("extraction result is missing client folder for visual QC persistence")

    manifest_path = (
        _extraction_output_dir(
            client_id=client_id,
            output_root=output_root,
            project_root=project_root,
        )
        / "manifest.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"extracted manifest is missing for visual QC persistence: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"extracted manifest is invalid for visual QC persistence: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("extracted manifest must contain an object")
    manifest_client_id = manifest.get("client_id")
    if manifest_client_id not in {None, client_id}:
        raise ValueError("visual QC decision client does not match extracted manifest")
    decision_client_id = decision.get("client_id")
    if decision_client_id not in {None, client_id}:
        raise ValueError("visual QC decision client does not match extraction result")

    serialized_decision = json.loads(json.dumps(decision))
    manifest["visual_qc_decision"] = serialized_decision
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest_path


def build_visual_qc_context(
    *,
    extraction_result: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a client-local visual QC context with artifact paths for an agent.

    This object is intended for the local file-backed agent mailbox only. It is
    not safe to return through FLARE because it contains client-local artifact
    paths.
    """

    extraction = extraction_result.get("extraction") if isinstance(extraction_result.get("extraction"), dict) else {}
    output_root = extraction.get("output_root") or ExtractionConfig.output_root
    client_folder = extraction.get("client_folder") or extraction_result.get("client_id")
    if not isinstance(client_folder, str) or not client_folder:
        return _visual_qc_context_unavailable("extraction result missing client folder")

    output_root_path = Path(str(output_root))
    if not output_root_path.is_absolute():
        base = Path(project_root).resolve() if project_root is not None else Path.cwd()
        output_root_path = base / output_root_path
    output_dir = output_root_path / client_folder
    manifest_path = output_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _visual_qc_context_unavailable("local extracted manifest was not readable")

    visual_qc = manifest.get("visual_qc") if isinstance(manifest.get("visual_qc"), dict) else {}
    review_required = visual_qc.get("review_required") is not False
    visual_qc_owner = str(visual_qc.get("visual_qc_owner") or "client_local_visual_qc")
    if not visual_qc.get("available"):
        return _visual_qc_context_unavailable(
            str(visual_qc.get("reason") or "visual QC artifacts unavailable"),
            review_required=review_required,
            visual_qc_owner=visual_qc_owner,
        )

    all_artifacts = []
    for artifact in visual_qc.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        item: dict[str, Any] = {
            "sample_index": artifact.get("sample_index"),
            "split": artifact.get("split"),
        }
        try:
            for key in ("image", "label", "overlay"):
                rel = artifact.get(key)
                if isinstance(rel, str) and rel:
                    item[f"{key}_path"] = str(
                        _resolve_visual_qc_artifact_path(output_dir, rel, field=key, require_existing=True)
                    )
            rel = artifact.get("candidate_sheet")
            if isinstance(rel, str) and rel:
                item["candidate_sheet_path"] = str(
                    _resolve_visual_qc_artifact_path(
                        output_dir,
                        rel,
                        field="candidate_sheet",
                        require_existing=True,
                    )
                )
            candidates = artifact.get("candidate_overlays")
            if isinstance(candidates, dict):
                item["candidate_overlay_paths"] = {
                    str(transform): str(
                        _resolve_visual_qc_artifact_path(
                            output_dir,
                            rel,
                            field="candidate_overlays",
                            require_existing=True,
                        )
                    )
                    for transform, rel in candidates.items()
                    if isinstance(rel, str) and rel
                }
        except ValueError as exc:
            return _visual_qc_context_unavailable(str(exc))
        all_artifacts.append(item)
    artifacts = _select_visual_qc_context_artifacts(all_artifacts, visual_qc=visual_qc)
    return {
        "schema_version": VISUAL_QC_CONTEXT_SCHEMA,
        "available": bool(artifacts),
        "client_id": extraction_result.get("client_id"),
        "sample_count": len(artifacts),
        "available_artifact_count": len(all_artifacts),
        "review_required": review_required,
        "visual_qc_owner": visual_qc_owner,
        "purpose": str(
            visual_qc.get("purpose") or "client-agent visual verification of extracted image/label alignment"
        ),
        "transform_candidates": [
            str(item)
            for item in (
                visual_qc.get("transform_candidates")
                if isinstance(visual_qc.get("transform_candidates"), list)
                else list(VISUAL_QC_TRANSFORMS)
            )
        ],
        "artifacts": artifacts,
        "local_paths_for_agent_only": True,
        "do_not_return_artifact_paths": True,
        "safe_to_share": False,
    }


def _validate_visual_qc_artifact_references(
    visual_qc: dict[str, Any],
    *,
    output_dir: Path,
    require_existing: bool,
) -> None:
    artifacts = visual_qc.get("artifacts")
    if not isinstance(artifacts, list):
        return
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        for key in ("image", "label", "overlay", "candidate_sheet"):
            rel = artifact.get(key)
            if isinstance(rel, str) and rel:
                _resolve_visual_qc_artifact_path(output_dir, rel, field=key, require_existing=require_existing)
        candidates = artifact.get("candidate_overlays")
        if isinstance(candidates, dict):
            for rel in candidates.values():
                if isinstance(rel, str) and rel:
                    _resolve_visual_qc_artifact_path(
                        output_dir,
                        rel,
                        field="candidate_overlays",
                        require_existing=require_existing,
                    )


def _resolve_visual_qc_artifact_path(
    output_dir: Path,
    rel_path: str,
    *,
    field: str,
    require_existing: bool,
) -> Path:
    return _resolve_output_dir_relative_path(
        output_dir,
        rel_path,
        field=field,
        require_existing=require_existing,
        artifact_kind="visual QC artifact",
    )


def _resolve_output_dir_relative_path(
    output_dir: Path,
    rel_path: str,
    *,
    field: str,
    require_existing: bool,
    artifact_kind: str,
) -> Path:
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError(f"{artifact_kind} {field} is missing")
    path = Path(rel_path)
    if path.is_absolute():
        raise ValueError(f"{artifact_kind} {field} must be output-dir-relative")
    root = output_dir.resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"{artifact_kind} {field} escapes client output directory")
    if require_existing and not target.is_file():
        raise ValueError(f"{artifact_kind} {field} is missing")
    return target


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


def _safe_visual_qc_transform_candidates(value: Any, *, default: tuple[str, ...]) -> list[str]:
    if not isinstance(value, list):
        return list(default)
    selected = {item for item in value if isinstance(item, str)}
    candidates = [transform for transform in VISUAL_QC_TRANSFORMS if transform in selected]
    return candidates or list(default)


def _visual_qc_sample_count(policy: dict[str, Any]) -> int:
    visual_qc_rule = policy.get("visual_qc_rule")
    if isinstance(visual_qc_rule, dict):
        value = visual_qc_rule.get("sample_count")
        if isinstance(value, int) and value > 0:
            return value
    return 3


def _select_visual_qc_context_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    visual_qc: dict[str, Any],
) -> list[dict[str, Any]]:
    if not artifacts:
        return []
    limit = _visual_qc_context_artifact_limit(visual_qc)
    if limit <= 0 or len(artifacts) <= limit:
        return artifacts
    if limit == 1:
        return [artifacts[len(artifacts) // 2]]
    last_index = len(artifacts) - 1
    indices = sorted({round(index * last_index / (limit - 1)) for index in range(limit)})
    return [artifacts[index] for index in indices]


def _visual_qc_context_artifact_limit(visual_qc: dict[str, Any]) -> int:
    for key in ("review_sample_count", "requested_sample_count"):
        value = visual_qc.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                continue
            if parsed > 0:
                return parsed
    value = os.environ.get("AGENTICFL_VISUAL_QC_REVIEW_SAMPLE_COUNT", "3")
    try:
        return max(1, int(value))
    except ValueError:
        return 3


def _visual_qc_context_unavailable(
    reason: str,
    *,
    review_required: bool = True,
    visual_qc_owner: str = "client_local_visual_qc",
) -> dict[str, Any]:
    return {
        "schema_version": VISUAL_QC_CONTEXT_SCHEMA,
        "available": False,
        "sample_count": 0,
        "artifacts": [],
        "review_required": review_required,
        "visual_qc_owner": visual_qc_owner,
        "reason": reason,
        "local_paths_for_agent_only": True,
        "do_not_return_artifact_paths": True,
        "safe_to_share": False,
    }


def _result_from_manifest(
    *,
    manifest: dict[str, Any],
    client_id: str,
    policy: dict[str, Any],
    output_name: str,
    source_label_type: str,
    conversion_options: list[str],
    reused_existing_output: bool,
    warnings: list[str],
) -> dict[str, Any]:
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {"total": 0, "by_split": {}}
    extracted_count = counts.get("total", 0) if isinstance(counts.get("total"), int) else 0
    screening = manifest.get("screening")
    if not isinstance(screening, dict):
        screening = {
            "schema_version": "agenticfl.local_label_screening.v1",
            "status": "usable",
            "reason_code": None,
            "reason": "Existing extracted manifest contains usable extracted cases.",
            "safe_to_share": True,
        }
    verification = manifest.get("verification") if isinstance(manifest.get("verification"), dict) else {}
    preview = manifest.get("preview") if isinstance(manifest.get("preview"), dict) else {"available": False}
    visual_qc = manifest.get("visual_qc") if isinstance(manifest.get("visual_qc"), dict) else {"available": False}
    label_orientation = manifest.get("label_orientation") if isinstance(manifest.get("label_orientation"), dict) else {}
    return {
        "schema_version": EXTRACTION_RESULT_SCHEMA,
        "client_id": client_id,
        "data": _extraction_status(
            extracted_count=extracted_count,
            screening=screening,
            verification=verification,
        ),
        "screening": screening,
        "policy_digest": manifest.get("policy_digest") or policy.get("strategy_digest") or payload_digest(policy),
        "source_label_type": source_label_type,
        "conversion_options": conversion_options,
        "local_adapter": manifest.get("local_adapter") if isinstance(manifest.get("local_adapter"), dict) else None,
        "counts": counts,
        "failed_pairs": manifest.get("failed_pairs", {}),
        "split_plan": manifest.get("split_plan", {}),
        "label_orientation": {
            "selected_transform": label_orientation.get("selected_transform", "as_is"),
            "strategy": label_orientation.get("strategy", "not_applied_by_minimal_extractor"),
            "safe_to_share": True,
        },
        "verification": verification,
        "extraction": {
            "output_root": (
                manifest.get("output", {}).get("root")
                if isinstance(manifest.get("output"), dict)
                else ExtractionConfig.output_root
            ),
            "client_folder": client_id,
            "output_name": output_name,
            "local_output_path_redacted": True,
            "reused_existing_output": reused_existing_output,
            "preview_available": bool(preview.get("available")),
            "preview_label_value_scale": preview.get("label_value_scale"),
            "visual_qc_available": bool(visual_qc.get("available")),
            "target_size": manifest.get("target_size"),
            "stored_resolution": manifest.get("stored_resolution", "source_image_resolution"),
            "training_transform_policy": manifest.get("training_transform_policy", "training_transforms.json"),
            "image_rule_applied": {
                "channels": "RGB",
                "resize": "deferred_to_training_time",
                "extraction_resize_applied": False,
            },
            "label_rule_applied": {
                "canonical_labels": policy.get("label_rule", {}).get("canonical_labels", {}),
                "mask_dtype": "uint8",
                "resize": "deferred_to_training_time_nearest_neighbor",
            },
        },
        "privacy": {
            "safe_to_share": True,
            "redacted": [
                "source_local_paths",
                "source_filenames",
                "extracted_local_paths",
                "extracted_filenames",
                "extracted_local_paths",
                "extracted_filenames",
                "sample_ids",
                "raw_images",
                "raw_masks",
                "raw_labels",
                "raw_annotations",
                "preview_filenames",
                "visual_qc_artifact_paths",
            ],
        },
        "visual_qc_artifacts": _visual_qc_safe_summary(visual_qc),
        "warnings": _dedupe(warnings),
    }


def _ensure_split_dirs(output_dir: Path, *, root_names: tuple[str, ...]) -> None:
    for root_name in root_names:
        for split in STANDARD_SPLITS:
            (output_dir / root_name / split).mkdir(parents=True, exist_ok=True)


def _contract_visual_qc_bundle(
    *,
    contract_runtime: Any,
    output_dir: Path,
    rows: list[dict[str, Any]],
    sample_count: int,
) -> dict[str, Any]:
    return contract_runtime.visual_qc_bundle(
        output_dir=output_dir,
        rows=rows,
        sample_count=sample_count,
        schema_version=VISUAL_QC_BUNDLE_SCHEMA,
        max_dimension=VISUAL_QC_PANEL_MAX_DIMENSION,
        max_bytes=VISUAL_QC_MAX_BYTES,
        min_long_side=VISUAL_QC_MIN_LONG_SIDE,
        palette_colors=VISUAL_QC_PALETTE_COLORS,
    )


def _row_split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        split = row.get("split")
        if split in STANDARD_SPLITS:
            counts[str(split)] += 1
    return dict(sorted(counts.items()))


def _no_orientation_adjustment() -> dict[str, Any]:
    return {
        "schema_version": "agenticfl.local_label_orientation_rule.v1",
        "strategy": "not_applied_by_minimal_extractor",
        "selected_transform": "as_is",
        "reason": (
            "The minimal shared extractor does not infer or repair label orientation. "
            "A real client agent must supply any site-local orientation decision in a future adapter path."
        ),
        "safe_to_share": True,
    }


def _bounded_sample_pairs_for_default_fl_splits(
    pairs: list[dict[str, Any]],
    policy: dict[str, Any],
    config: ExtractionConfig,
    *,
    stable_key: Callable[[dict[str, Any]], str],
    source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a bounded subset and apply the default FL train/val/test split.

    Bounded benchmark runs should not inherit source train/test-only splits. They
    compare a fixed-size sample from each site, so the fixed-size subset gets the
    common 80/10/10 train/validation/test split regardless of source folder names.
    """

    max_samples = config.max_samples if config.max_samples is not None else len(pairs)
    selected_indices = _bounded_sample_indices(
        item_count=len(pairs),
        max_samples=max_samples,
        stable_key=lambda index: stable_key(pairs[index]),
        config=config,
    )
    selected = [pairs[index] for index in selected_indices]
    assignments: list[str | None] = [None] * len(selected)
    _assign_common_ratio(
        indices=list(range(len(selected))),
        assignments=assignments,
        stable_key=lambda index: stable_key(selected[index]),
        config=config,
    )
    final_assignments = [split if split in STANDARD_SPLITS else "train" for split in assignments]
    adjusted = [{**pair, "split": final_assignments[index]} for index, pair in enumerate(selected)]
    input_counts = Counter(
        str(pair.get("split", "unknown")) if pair.get("split") in STANDARD_SPLITS else "unknown" for pair in pairs
    )
    output_counts = Counter(final_assignments)
    plan = {
        "schema_version": "agenticfl.extracted_split_plan.v1",
        "strategy": "bounded_sample_common_ratio_split",
        "official_split_detected": False,
        "source": source,
        "default_ratio": COMMON_SPLIT_RATIOS,
        "seed": config.split_seed,
        "bounded_sample_split": True,
        "max_samples": max_samples,
        "full_valid_case_count": len(pairs),
        "selected_case_count": len(selected),
        "created_splits": sorted({split for split in final_assignments}),
        "input_counts": {split: input_counts.get(split, 0) for split in (*STANDARD_SPLITS, "unknown")},
        "output_counts": {split: output_counts.get(split, 0) for split in STANDARD_SPLITS},
        "safe_to_share": True,
    }
    split_rule = policy.get("split_rule", {})
    if isinstance(split_rule, dict):
        plan["policy_split_rule"] = {
            "preserve_existing_splits": bool(split_rule.get("preserve_existing_splits", True)),
            "if_validation_missing": split_rule.get("if_validation_missing"),
            "overridden_for_bounded_sample": True,
        }
    return adjusted, plan


def _bounded_sample_indices(
    *,
    item_count: int,
    max_samples: int,
    stable_key: Callable[[int], str],
    config: ExtractionConfig,
) -> list[int]:
    indices = list(range(item_count))
    if max_samples >= item_count:
        return indices
    selected = sorted(indices, key=lambda index: _split_hash(config.split_seed, f"bounded:{stable_key(index)}"))[
        :max_samples
    ]
    return sorted(selected)


def _assign_fl_splits(
    *,
    item_count: int,
    current_split: Any,
    stable_key: Any,
    config: ExtractionConfig,
) -> tuple[list[str], dict[str, Any]]:
    raw_splits = [current_split(index) for index in range(item_count)]
    input_counts = Counter(split if split in STANDARD_SPLITS else "unknown" for split in raw_splits)
    known_splits = {split for split in STANDARD_SPLITS if input_counts[split] > 0}
    official_split_detected = len(known_splits) >= 2 or bool(known_splits & {"validation", "test"})

    assignments: list[str | None] = [None] * item_count
    strategy = "common_ratio_split"
    created_splits: list[str] = []

    if official_split_detected:
        strategy = "preserve_official_splits"
        unknown_indices: list[int] = []
        for index, split in enumerate(raw_splits):
            if split in STANDARD_SPLITS:
                assignments[index] = split
            else:
                unknown_indices.append(index)
        if unknown_indices:
            strategy = "preserve_official_splits_with_common_ratio_for_unknown"
            created_splits.append("unknown_to_standard")
            _assign_common_ratio(indices=unknown_indices, assignments=assignments, stable_key=stable_key, config=config)
    else:
        _assign_common_ratio(
            indices=list(range(item_count)), assignments=assignments, stable_key=stable_key, config=config
        )
        created_splits.append("all_cases")

    if item_count:
        if (
            "validation" not in {split for split in assignments if split in STANDARD_SPLITS}
            and config.validation_fraction > 0
        ):
            if _move_train_subset(
                assignments=assignments,
                stable_key=stable_key,
                config=config,
                target_split="validation",
                fraction=config.validation_fraction,
            ):
                created_splits.append("validation")
        if "test" not in {split for split in assignments if split in STANDARD_SPLITS}:
            if _move_train_subset(
                assignments=assignments,
                stable_key=stable_key,
                config=config,
                target_split="test",
                fraction=COMMON_SPLIT_RATIOS["test"],
            ):
                created_splits.append("test")

    final_assignments = [split if split in STANDARD_SPLITS else "train" for split in assignments]
    output_counts = Counter(final_assignments)
    plan = {
        "schema_version": "agenticfl.extracted_split_plan.v1",
        "strategy": strategy,
        "official_split_detected": official_split_detected,
        "source": "source_path_tokens" if official_split_detected else "common_ratio_80_10_10",
        "default_ratio": COMMON_SPLIT_RATIOS,
        "seed": config.split_seed,
        "created_splits": sorted(set(created_splits)),
        "input_counts": {split: input_counts.get(split, 0) for split in (*STANDARD_SPLITS, "unknown")},
        "output_counts": {split: output_counts.get(split, 0) for split in STANDARD_SPLITS},
        "safe_to_share": True,
    }
    return final_assignments, plan


def _assign_common_ratio(
    *,
    indices: list[int],
    assignments: list[str | None],
    stable_key: Any,
    config: ExtractionConfig,
) -> None:
    counts = _common_split_counts(len(indices))
    ordered = sorted(indices, key=lambda index: _split_hash(config.split_seed, stable_key(index)))
    validation_end = counts["validation"]
    test_end = validation_end + counts["test"]
    for offset, index in enumerate(ordered):
        if offset < validation_end:
            assignments[index] = "validation"
        elif offset < test_end:
            assignments[index] = "test"
        else:
            assignments[index] = "train"


def _common_split_counts(total: int) -> dict[str, int]:
    if total <= 0:
        return {"train": 0, "validation": 0, "test": 0}
    if total < 3:
        return {"train": total, "validation": 0, "test": 0}
    validation = max(1, round(total * COMMON_SPLIT_RATIOS["validation"]))
    test = max(1, round(total * COMMON_SPLIT_RATIOS["test"]))
    if validation + test >= total:
        validation = 1
        test = 1 if total > 2 else 0
    return {"train": total - validation - test, "validation": validation, "test": test}


def _move_train_subset(
    *,
    assignments: list[str | None],
    stable_key: Any,
    config: ExtractionConfig,
    target_split: str,
    fraction: float,
) -> bool:
    train_indices = [index for index, split in enumerate(assignments) if split == "train"]
    if len(train_indices) < 2:
        return False
    move_count = max(1, round(len(train_indices) * fraction))
    move_count = min(move_count, len(train_indices) - 1)
    ordered = sorted(
        train_indices, key=lambda index: _split_hash(config.split_seed, f"{target_split}:{stable_key(index)}")
    )
    for index in ordered[:move_count]:
        assignments[index] = target_split
    return move_count > 0


def _split_hash(seed: int, key: str) -> str:
    return sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


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


def _target_size(policy: dict[str, Any]) -> tuple[int, int]:
    image_rule = policy.get("image_rule", {})
    value = image_rule.get("target_size") if isinstance(image_rule, dict) else None
    if isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) and item > 0 for item in value):
        return int(value[0]), int(value[1])
    return 512, 512


def _target_terms(policy: dict[str, Any]) -> set[str]:
    site_mapping = _site_mapping(policy)
    terms = set()
    for value in site_mapping.get("matched_terms", []):
        if isinstance(value, str):
            terms.update(_tokens(value))
    label_rule = policy.get("label_rule", {})
    canonical = label_rule.get("canonical_labels", {}) if isinstance(label_rule, dict) else {}
    if isinstance(canonical, dict):
        for label, value in canonical.items():
            if label != "background" and value == 1:
                terms.update(_tokens(label))
    canonical_task = str(site_mapping.get("canonical_task", ""))
    terms.update(_local_adapter_task_tokens(canonical_task))
    return _expand_target_terms(terms)


def _expand_target_terms(terms: set[str]) -> set[str]:
    expanded: set[str] = set()
    for term in terms:
        words = _tokens(term)
        expanded.update(words)
        if len(words) >= 2:
            expanded.add("".join(word[0] for word in words if word))
    return expanded


def _resolve_site_data_path(
    site_meta_path: str | Path,
    client_id: str,
    *,
    project_root: str | Path | None = None,
) -> Path:
    meta_path = Path(site_meta_path)
    list_client_ids(meta_path)
    client_id = validate_client_id(client_id)
    site_meta = _load_json(meta_path)
    clients = site_meta.get("clients")
    if not isinstance(clients, list):
        raise ValueError("site metadata must contain a clients list")
    selected = next(
        (client for client in clients if isinstance(client, dict) and client.get("client_id") == client_id),
        None,
    )
    if selected is None:
        raise ValueError(f"client_id not found in site metadata: {client_id}")
    data_path_value = selected.get("data_path")
    if not isinstance(data_path_value, str) or not data_path_value:
        raise ValueError(f"client_id has no valid data_path: {client_id}")
    root = Path(project_root) if project_root is not None else meta_path.parent.parent
    data_path = Path(data_path_value)
    return data_path if data_path.is_absolute() else root / data_path


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


def _not_applicable_result(client_id: str, policy: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": EXTRACTION_RESULT_SCHEMA,
        "client_id": client_id,
        "data": "not applicable",
        "policy_digest": policy.get("strategy_digest") or payload_digest(policy),
        "reason": reason,
        "privacy": {
            "safe_to_share": True,
            "redacted": ["source_local_paths", "source_filenames", "raw_images", "raw_masks"],
        },
    }


def _unsupported_label_format_result(client_id: str, policy: dict[str, Any], source_label_type: str) -> dict[str, Any]:
    return {
        "schema_version": EXTRACTION_RESULT_SCHEMA,
        "client_id": client_id,
        "data": "unsupported by minimal extractor",
        "policy_digest": policy.get("strategy_digest") or payload_digest(policy),
        "screening": {
            "schema_version": "agenticfl.local_label_screening.v1",
            "status": "capability_gap",
            "reason_code": "NON_BINARY_LABEL_FORMAT_REQUIRES_AGENT_ADAPTER",
            "reason": (
                "The minimal shared extractor supports native binary segmentation masks only. "
                "No built-in adapter, orientation repair, contour parser, channel parser, or index parser was used."
            ),
            "source_label_type": source_label_type,
            "safe_to_share": True,
        },
        "agent_action_required": {
            "required": True,
            "kind": "client_local_label_adapter",
            "source_label_type": source_label_type,
            "implemented_by_baseline": False,
            "safe_to_share": True,
        },
        "privacy": {
            "safe_to_share": True,
            "redacted": ["source_local_paths", "source_filenames", "raw_images", "raw_masks", "raw_annotations"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a local AgenticFL client dataset.")
    parser.add_argument("data_path", nargs="?", help="Local dataset path to extract.")
    parser.add_argument("--site-meta", help="Resolve data_path from a site-meta.json file.")
    parser.add_argument("--client-id", required=True, help="Client id to extract.")
    parser.add_argument("--project-root", help="Project root for relative site metadata paths.")
    parser.add_argument("--strategy", required=True, help="Server final_state.json or extraction_strategy.json.")
    parser.add_argument("--output-root", default=ExtractionConfig.output_root)
    parser.add_argument("--output-name", help="Optional extracted output folder name.")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-scan-files", type=int, default=ExtractionConfig.max_scan_files)
    parser.add_argument("--validation-fraction", type=float, default=ExtractionConfig.validation_fraction)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing extracted output folder.")
    args = parser.parse_args(argv)

    policy = load_site_extraction_policy(args.strategy, args.client_id)
    config = ExtractionConfig(
        output_root=args.output_root,
        output_name=args.output_name,
        max_samples=args.max_samples,
        max_scan_files=args.max_scan_files,
        validation_fraction=args.validation_fraction,
        overwrite=args.overwrite,
    )
    if args.site_meta:
        result = extract_site_dataset(
            args.site_meta,
            args.client_id,
            policy=policy,
            project_root=args.project_root,
            config=config,
        )
    else:
        if not args.data_path:
            parser.error("data_path is required unless --site-meta is used")
        result = extract_dataset(
            args.data_path,
            client_id=args.client_id,
            policy=policy,
            config=config,
            project_root=args.project_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
