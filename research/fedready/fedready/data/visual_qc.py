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

"""Client-local visual-review artifacts, persisted decisions, and orientation repair."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fedready.data.contracts import runtime_contract_for_record_type
from fedready.data.extraction_artifacts import ExtractionConfig, _extraction_output_dir, _visual_qc_safe_summary
from fedready.data.qc import VISUAL_QC_BUNDLE_SCHEMA, VISUAL_QC_TRANSFORMS
from fedready.utils.logging import canonical_json

try:  # Pillow is optional outside image-processing workflows.
    from PIL import Image
except ImportError:  # pragma: no cover - minimal environments
    Image = None  # type: ignore[assignment]


VISUAL_QC_CONTEXT_SCHEMA = "fedready.extraction_visual_qc_context.v1"
VISUAL_QC_PANEL_MAX_DIMENSION = int(os.environ.get("FEDREADY_VISUAL_QC_PANEL_MAX_DIMENSION", "512"))
_visual_qc_max_bytes_env = os.environ.get("FEDREADY_VISUAL_QC_MAX_BYTES")
VISUAL_QC_MAX_BYTES = int(_visual_qc_max_bytes_env) if _visual_qc_max_bytes_env else None
VISUAL_QC_MIN_LONG_SIDE = 192
VISUAL_QC_PALETTE_COLORS = 128


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
        "schema_version": "fedready.local_label_orientation_rule.v1",
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
            "schema_version": "fedready.orientation_repair.v1",
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
    value = os.environ.get("FEDREADY_VISUAL_QC_REVIEW_SAMPLE_COUNT", "3")
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
