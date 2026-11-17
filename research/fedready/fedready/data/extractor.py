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

"""Policy-driven local dataset extraction for FedReady client agents.

This module is intended to run on a client after the server has sent a
site-specific extraction policy through FLARE. It resolves only the local
client data path, writes extracted PNG images and canonical masks into the
project-level ``data/dataset_fl/<client_id>/`` folder, records deferred
training-time transform parameters locally, and returns aggregate metadata that
is safe to share."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from fedready.agents.local_adapter import (
    UNLIMITED_LOCAL_ADAPTER_RECORDS,
    ClientDataProvenanceSnapshot,
    execute_local_adapter,
)
from fedready.data.contracts import (
    STANDARD_SPLITS,
    manifest_record_type,
    runtime_contract_for_record_type,
    task_family_from_policy,
)
from fedready.data.extraction_artifacts import (
    EXTRACTION_RESULT_SCHEMA,
    EXTRACTOR_LOGIC_VERSION,
    ExtractionConfig,
    _dedupe,
    _extraction_output_dir,
    _extraction_policy_digest,
    _extraction_result,
    _extraction_screening,
    _extraction_verification,
    _failed_extraction_result,
    _finalize_intensity_accumulator,
    _generated_data_contract,
    _local_adapter_safe_summary,
    _new_intensity_accumulator,
    _safe_slug,
    _site_mapping,
    _update_intensity_accumulator,
    _visual_qc_safe_summary,
    _write_training_transform_policy,
)
from fedready.data.materializer import (
    _extract_generated_contract_dataset,
    _generated_data_materializer,
    _generated_materializer_required_result,
    _stable_warning_codes,
)
from fedready.data.parser import _load_json, _tokens, list_client_ids, validate_client_id
from fedready.data.splits import _bounded_sample_pairs_for_default_fl_splits, _organize_adapter_pairs_for_fl_splits
from fedready.data.visual_qc import _contract_visual_qc_bundle, _visual_qc_sample_count
from fedready.utils.logging import payload_digest

try:  # Pillow is optional outside image-processing workflows.
    from PIL import Image
except ImportError:  # pragma: no cover - minimal environments
    Image = None  # type: ignore[assignment]


SITE_EXTRACTION_POLICY_SCHEMA = "fedready.site_extraction_policy.v1"
LOCAL_ADAPTER_MANIFEST_SCHEMA = "fedready.local_adapter_manifest.v1"


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
        manifest_record_type(adapter_manifest, generated_contract=generated_contract)
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

    if output_dir.exists() and not deferred_adapter_output_prepared:
        if not cfg.overwrite:
            raise FileExistsError(f"extraction output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    _ensure_split_dirs(output_dir, root_names=contract_runtime.split_roots())

    adapter_pairs, label_warnings = contract_runtime.manifest_pairs(adapter_manifest)
    valid_pairs, split_plan = _organize_adapter_pairs_for_fl_splits(adapter_pairs, policy, cfg)
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
        record_count=len(adapter_pairs),
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
            "warnings": _dedupe(label_warnings + adapter_warnings),
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
    _write_training_transform_policy(
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
        "schema_version": "fedready.local_extracted_manifest.v1",
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

    return _extraction_result(
        manifest,
        label_rule_applied=contract_runtime.label_rule_applied(policy),
        warnings=_dedupe(label_warnings + adapter_warnings),
        extraction_details={
            "preview_available": preview["available"],
            "preview_label_value_scale": preview.get("label_value_scale"),
        },
    )


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
    raw = os.environ.get("FEDREADY_LOCAL_ADAPTER_RUNTIME_TIMEOUT_SECONDS", "3600")
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
            "schema_version": "fedready.local_adapter_runtime_attestation.v1",
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
    keep_raw = os.environ.get("FEDREADY_KEEP_DEFERRED_ADAPTER_STAGING", "").strip().lower()
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
        return None, ["Local adapter manifest schema_version did not match fedready.local_adapter_manifest.v1."]
    manifest_client = manifest.get("client_id")
    if isinstance(manifest_client, str) and manifest_client != client_id:
        return None, ["Local adapter manifest client_id did not match the local client."]
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        return None, ["Local adapter manifest did not contain any records."]
    manifest["_manifest_path"] = str(manifest_path.resolve())
    return manifest, []


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
            "schema_version": "fedready.local_label_screening.v1",
            "status": "screened_out",
            "reason_code": "LOCAL_ADAPTER_DECLARED_UNFEASIBLE",
            "reason": reason,
            "safe_to_share": True,
        },
        "verification": {
            "schema_version": "fedready.local_extraction_verification.v1",
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
        **_failed_extraction_result(
            client_id=client_id,
            policy=policy,
            source_label_type=source_label_type,
            adapter_summary=_local_adapter_safe_summary(local_adapter, adapter_manifest=None),
        ),
        "screening": {
            "schema_version": "fedready.local_label_screening.v1",
            "status": "failed",
            "reason_code": "LOCAL_ADAPTER_MANIFEST_REQUIRED",
            "reason": reason,
            "safe_to_share": True,
        },
        "warnings": _dedupe(
            ["LOCAL_ADAPTER_MANIFEST_REQUIRED", *_stable_warning_codes(warnings, fallback_code="ADAPTER_WARNING")]
        ),
    }
    if client_local_diagnostic:
        result["_client_local_diagnostic"] = client_local_diagnostic
    return result


def _local_adapter_task_tokens(task: str | None) -> set[str]:
    generic = {"image", "imaging", "task", "segmentation", "classification"}
    words = [token for token in _tokens(task or "") if token not in generic]
    tokens = set(words)
    if len(words) >= 2:
        tokens.add("".join(word[0] for word in words if word))
    return tokens


def _ensure_split_dirs(output_dir: Path, *, root_names: tuple[str, ...]) -> None:
    for root_name in root_names:
        for split in STANDARD_SPLITS:
            (output_dir / root_name / split).mkdir(parents=True, exist_ok=True)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a local FedReady client dataset.")
    parser.add_argument("data_path", nargs="?", help="Local dataset path to extract.")
    parser.add_argument("--site-meta", help="Resolve data_path from a site-meta.json file.")
    parser.add_argument("--client-id", required=True, help="Client id to extract.")
    parser.add_argument("--project-root", help="Project root for relative site metadata paths.")
    parser.add_argument("--strategy", required=True, help="Server final_state.json or extraction_strategy.json.")
    parser.add_argument("--output-root", default=ExtractionConfig.output_root)
    parser.add_argument("--output-name", help="Optional extracted output folder name.")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--validation-fraction", type=float, default=ExtractionConfig.validation_fraction)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing extracted output folder.")
    args = parser.parse_args(argv)

    policy = load_site_extraction_policy(args.strategy, args.client_id)
    config = ExtractionConfig(
        output_root=args.output_root,
        output_name=args.output_name,
        max_samples=args.max_samples,
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


if __name__ == "__main__":
    raise SystemExit(main())
