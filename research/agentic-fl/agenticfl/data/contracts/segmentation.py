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

"""Canonical segmentation prepared-data contract."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from agenticfl.data.contracts.base import SEGMENTATION, STANDARD_SPLITS, DataContract
from agenticfl.data.parser import _infer_split
from agenticfl.data.qc import VISUAL_QC_TRANSFORMS
from agenticfl.utils.logging import canonical_json, payload_digest
from PIL import Image, ImageDraw

SAMPLE_MANIFEST = "samples.jsonl"
SAMPLE_MANIFEST_FORMAT = "jsonl_image_mask_records"
SPLIT_ROOTS = ("images", "masks")
RECORD_EXAMPLE = {
    "image_path": "absolute_or_manifest_relative_image_path",
    "label_source_path": "absolute_or_manifest_relative_existing_label_or_annotation_path",
    "mask_path": "absolute_or_manifest_relative_binary_mask_path_under_adapter_workspace",
    "split": "optional train|validation|test",
}

CONTRACT = DataContract(
    name="canonical_segmentation",
    record_type=SEGMENTATION,
    sample_manifest=SAMPLE_MANIFEST,
    sample_manifest_format=SAMPLE_MANIFEST_FORMAT,
    record_example=RECORD_EXAMPLE,
    materialized_outputs=("images/<split>/*.png", "masks/<split>/*.png", SAMPLE_MANIFEST),
    description=(
        "Spatial image-to-mask tasks. The adapter provides real image paths, existing local label-source "
        "paths, and generated single-channel PNG binary masks under the adapter workspace."
    ),
    adapter_record_required_fields=("image_path", "label_source_path", "mask_path"),
    adapter_record_optional_fields=("split",),
    manifest_validation={
        "mask_storage": "mask_path must point under the adapter workspace, not to the source dataset",
        "mask_encoding": "PNG-readable single-channel binary mask with background encoded as 0",
        "pairing": "image_path and mask_path must resolve to readable images with identical dimensions",
        "provenance": "label_source_path must point to a pre-existing client-local label or annotation source and differ from image_path/mask_path",
    },
    visual_qc={
        "required": True,
        "type": "local_image_mask_overlay",
        "pass_condition": "local VLM consensus confirms the red overlay covers the requested foreground target",
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
        mask_path = _adapter_record_path(record.get("mask_path"), base_dir)
        if image_path is None or label_source_path is None or mask_path is None:
            warnings.append("Local adapter manifest record was missing image_path, label_source_path, or mask_path.")
            continue
        if not image_path.exists() or not label_source_path.exists() or not mask_path.exists():
            warnings.append("Local adapter manifest referenced an unreadable local image, label source, or mask.")
            continue
        if label_source_path in {image_path, mask_path}:
            warnings.append("Local adapter manifest label source matched the image or generated mask.")
            continue
        split_value = record.get("split")
        split = str(split_value) if isinstance(split_value, str) else _infer_split(image_path.as_posix())
        if split not in STANDARD_SPLITS:
            split = _infer_split(mask_path.as_posix())
        if split not in STANDARD_SPLITS:
            split = "unknown"
        pairs.append(
            {
                "image_path": image_path,
                "label_source_path": label_source_path,
                "mask_path": mask_path,
                "split": split,
                "adapter_record_index": index,
                "stable_key": payload_digest(
                    {
                        "image_path": image_path.as_posix(),
                        "label_source_path": label_source_path.as_posix(),
                        "mask_path": mask_path.as_posix(),
                        "index": index,
                    }
                ),
            }
        )
    if pairs:
        has_background, has_foreground = _adapter_manifest_mask_content(pairs)
        if not has_foreground:
            return [], [*warnings, "Local adapter masks contained no nonzero task foreground."]
        if not has_background:
            return [], [*warnings, "Local adapter masks contained no background encoded as 0."]
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
    mask_path = Path(pair["mask_path"])
    split = pair["split"]
    sample_id = payload_digest(
        {
            "image": image_path.as_posix(),
            "mask": mask_path.as_posix(),
            "adapter_record_index": pair.get("adapter_record_index"),
        }
    )[:16]

    with Image.open(image_path) as source_image:
        rgb = source_image.convert("RGB")
        rgb.load()
        source_size = rgb.size
    with Image.open(mask_path) as source_mask:
        mask = source_mask.convert("L")
        if mask.size != source_size:
            raise ValueError("client-local adapter image and mask dimensions differ")
        colors = mask.getcolors(maxcolors=3)
        if colors is None or len(colors) > 2:
            raise ValueError("client-local adapter mask is not binary")
        values = {int(value) for _, value in colors}
        if 0 not in values:
            raise ValueError("client-local adapter mask does not encode background as 0")
        converted_mask = mask.point(lambda value: 1 if value > 0 else 0, mode="L")

    image_rel = Path("images") / split / f"{sample_id}.png"
    mask_rel = Path("masks") / split / f"{sample_id}.png"
    (output_dir / image_rel).parent.mkdir(parents=True, exist_ok=True)
    (output_dir / mask_rel).parent.mkdir(parents=True, exist_ok=True)
    rgb.save(output_dir / image_rel)
    converted_mask.save(output_dir / mask_rel)
    update_intensity(intensity_accumulator, rgb)

    return {
        "sample_id": sample_id,
        "split": split,
        "image": image_rel.as_posix(),
        "mask": mask_rel.as_posix(),
        "source_digest": payload_digest({"adapter_record_index": pair.get("adapter_record_index")}),
        "source_label_type": source_label_type,
        "local_adapter_applied": True,
        "label_orientation": "as_is",
        "transform": {
            "source_size": list(source_size),
            "stored_size": list(source_size),
            "stored_resolution": "source_image_resolution",
            "extraction_resize_applied": False,
            "training_resize": "deferred_to_training_transforms_json",
        },
    }


def write_sample_manifest(path: Path, *, rows: list[dict[str, Any]], policy: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def orientation_rule(manifest: dict[str, Any] | None) -> dict[str, Any]:
    transform = "as_is"
    if isinstance(manifest, dict):
        value = manifest.get("selected_transform") or manifest.get("label_orientation")
        if isinstance(value, str) and value in VISUAL_QC_TRANSFORMS:
            transform = value
    return {
        "schema_version": "agenticfl.local_label_orientation_rule.v1",
        "strategy": "client_local_adapter_manifest",
        "selected_transform": transform,
        "reason": "Label orientation was supplied by a client-local adapter manifest.",
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
    label_path = output_dir / str(row.get("mask", ""))
    if not label_path.exists():
        return {"available": False, "reason": "sample label missing", "local_output_path_redacted": True}
    preview_label = output_dir / "sample_label.png"
    with Image.open(label_path) as label:
        label.convert("L").point(lambda value: 255 if value > 0 else 0, mode="L").save(preview_label)
    return {
        "available": True,
        "image": "sample_image.png",
        "label": "sample_label.png",
        "label_kind": "segmentation_mask",
        "label_value_scale": "0_or_255",
        "source_sample_split": row.get("split"),
        "local_output_path_redacted": True,
    }


def visual_qc_bundle(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    sample_count: int,
    schema_version: str,
    max_dimension: int,
    max_bytes: int | None,
    min_long_side: int,
    palette_colors: int,
) -> dict[str, Any]:
    selected_rows = _select_visual_qc_rows(rows, sample_count)
    if not selected_rows:
        return {
            "schema_version": schema_version,
            "available": False,
            "reason": "no extracted samples",
            "local_output_path_redacted": True,
        }

    qc_dir = output_dir / "visual_qc"
    if qc_dir.exists():
        shutil.rmtree(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, row in enumerate(selected_rows, start=1):
        image_path = output_dir / str(row.get("image", ""))
        label_path = output_dir / str(row.get("mask", ""))
        if not image_path.exists() or not label_path.exists():
            warnings.append(f"sample_{index:02d}: image or label missing")
            continue
        image_rel = Path("visual_qc") / f"sample_{index:02d}_image.png"
        label_rel = Path("visual_qc") / f"sample_{index:02d}_label.png"
        candidate_sheet_rel = Path("visual_qc") / f"sample_{index:02d}_candidate_transforms.png"
        candidate_overlay_rels = {
            transform: Path("visual_qc") / f"sample_{index:02d}_overlay_{transform}.png"
            for transform in VISUAL_QC_TRANSFORMS
        }
        try:
            write_visual_qc_artifacts(
                image_path=image_path,
                label_path=label_path,
                image_out=output_dir / image_rel,
                label_out=output_dir / label_rel,
                candidate_sheet_out=output_dir / candidate_sheet_rel,
                candidate_overlay_outs={
                    transform: output_dir / rel for transform, rel in candidate_overlay_rels.items()
                },
                max_dimension=max_dimension,
                max_bytes=max_bytes,
                min_long_side=min_long_side,
                palette_colors=palette_colors,
            )
        except Exception as exc:  # noqa: BLE001 - QC artifacts should not fail extraction.
            warnings.append(f"sample_{index:02d}: failed to render QC artifact ({type(exc).__name__})")
            continue
        artifacts.append(
            {
                "sample_index": index,
                "split": row.get("split"),
                "image": image_rel.as_posix(),
                "label": label_rel.as_posix(),
                "candidate_sheet": candidate_sheet_rel.as_posix(),
                "candidate_overlays": {transform: rel.as_posix() for transform, rel in candidate_overlay_rels.items()},
                "label_value_scale": "0_or_255",
                "local_artifact_path_redacted": True,
            }
        )

    payload = {
        "schema_version": schema_version,
        "available": bool(artifacts),
        "sample_count": len(artifacts),
        "requested_sample_count": sample_count,
        "artifact_root": "visual_qc",
        "artifacts": artifacts,
        "warnings": warnings,
        "review_required": bool(artifacts),
        "reviewer": "client_agent_visual_review",
        "transform_candidates": list(VISUAL_QC_TRANSFORMS),
        "local_output_path_redacted": True,
    }
    (qc_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def storage_sections(policy: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mask_storage": {
            "format": "png",
            "mode": "L",
            "resolution": "matched_to_source_image_resolution",
            "extraction_resize_applied": False,
            "dtype": policy.get("label_rule", {}).get("mask_dtype", "uint8"),
            "canonical_labels": policy.get("label_rule", {}).get("canonical_labels", {}),
        },
        "classification_storage": None,
        "object_detection_storage": None,
    }


def label_rule_applied(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_labels": policy.get("label_rule", {}).get("canonical_labels", {}),
        "label_kind": "segmentation_mask",
        "mask_dtype": "uint8",
        "bbox_format": None,
        "resize": "deferred_to_training_time_nearest_neighbor",
    }


def storage_summary() -> dict[str, object]:
    return {
        "format": SAMPLE_MANIFEST_FORMAT,
        "sample_manifest": SAMPLE_MANIFEST,
        "record_fields": ["image", "mask", "split"],
        "safe_to_share": True,
    }


def write_visual_qc_artifacts(
    *,
    image_path: Path,
    label_path: Path,
    image_out: Path,
    label_out: Path,
    candidate_sheet_out: Path,
    candidate_overlay_outs: dict[str, Path],
    max_dimension: int = 512,
    max_bytes: int | None = None,
    min_long_side: int = 192,
    palette_colors: int = 128,
) -> None:
    with Image.open(image_path) as image_raw, Image.open(label_path) as label_raw:
        image = image_raw.convert("RGB")
        label = label_raw.convert("L")
        if label.size != image.size:
            label = label.resize(image.size, _resampling("mask"))
        image, label = _resize_qc_pair(image, label, max_dimension=max_dimension)
        label_preview = label.point(lambda value: 255 if value > 0 else 0, mode="L")
        _save_visual_qc_png(
            image, image_out, max_bytes=max_bytes, min_long_side=min_long_side, palette_colors=palette_colors
        )
        _save_visual_qc_png(
            label_preview, label_out, max_bytes=max_bytes, min_long_side=min_long_side, palette_colors=palette_colors
        )
        overlays = {}
        for transform, out_path in candidate_overlay_outs.items():
            overlay = _overlay_label(image, _apply_label_transform(label, transform))
            _save_visual_qc_png(
                overlay, out_path, max_bytes=max_bytes, min_long_side=min_long_side, palette_colors=palette_colors
            )
            overlays[transform] = overlay
        _write_candidate_sheet(
            overlays,
            candidate_sheet_out,
            max_bytes=max_bytes,
            min_long_side=min_long_side,
            palette_colors=palette_colors,
        )


def _adapter_record_path(value: Any, base_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _adapter_manifest_mask_content(pairs: list[dict[str, Any]]) -> tuple[bool, bool]:
    has_background = False
    has_foreground = False
    for pair in pairs:
        try:
            with Image.open(Path(pair["mask_path"])) as mask:
                minimum, maximum = mask.convert("L").getextrema()
        except (OSError, ValueError):
            continue
        has_background = has_background or minimum == 0
        has_foreground = has_foreground or maximum > 0
    return has_background, has_foreground


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


def _select_visual_qc_rows(rows: list[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    if sample_count <= 0 or not rows:
        return []
    ordered = sorted(
        rows,
        key=lambda item: (
            STANDARD_SPLITS.index(item.get("split", "train")) if item.get("split") in STANDARD_SPLITS else 99,
            str(item.get("sample_id", "")),
        ),
    )
    if len(ordered) <= sample_count:
        return ordered
    if sample_count == 1:
        return [ordered[0]]
    last_index = len(ordered) - 1
    indices = sorted({round(index * last_index / (sample_count - 1)) for index in range(sample_count)})
    return [ordered[index] for index in indices]


def _overlay_label(image: Any, label: Any) -> Any:
    alpha = label.point(lambda value: 150 if value > 0 else 0, mode="L")
    red = Image.new("RGBA", image.size, (255, 0, 0, 0))
    red.putalpha(alpha)
    return Image.alpha_composite(image.convert("RGBA"), red).convert("RGB")


def _apply_label_transform(label: Any, transform: str) -> Any:
    if transform == "as_is":
        return label
    transpose = getattr(Image, "Transpose", Image)
    if transform == "hflip":
        return label.transpose(getattr(transpose, "FLIP_LEFT_RIGHT"))
    if transform == "vflip":
        return label.transpose(getattr(transpose, "FLIP_TOP_BOTTOM"))
    if transform == "rot180":
        return label.transpose(getattr(transpose, "ROTATE_180"))
    raise ValueError(f"Unsupported visual QC transform: {transform}")


def _write_candidate_sheet(
    overlays: dict[str, Any],
    out_path: Path,
    *,
    max_bytes: int | None,
    min_long_side: int,
    palette_colors: int,
) -> None:
    if not overlays:
        return
    ordered = [transform for transform in VISUAL_QC_TRANSFORMS if transform in overlays]
    first = overlays[ordered[0]]
    panel_width, panel_height = first.size
    label_height = 24
    sheet = Image.new("RGB", (panel_width * 2, (panel_height + label_height) * 2), color=(0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for index, transform in enumerate(ordered):
        row = index // 2
        col = index % 2
        x = col * panel_width
        y = row * (panel_height + label_height)
        sheet.paste(overlays[transform], (x, y + label_height))
        draw.text((x + 6, y + 4), transform, fill=(255, 255, 255))
    _save_visual_qc_png(
        sheet, out_path, max_bytes=max_bytes, min_long_side=min_long_side, palette_colors=palette_colors
    )


def _save_visual_qc_png(
    image: Any,
    out_path: Path,
    *,
    max_bytes: int | None,
    min_long_side: int,
    palette_colors: int,
) -> None:
    if max_bytes is None or max_bytes <= 0:
        image.save(out_path)
        return
    working = image.convert("RGB") if image.mode in {"RGB", "RGBA"} else image
    while True:
        candidate = _quantize_visual_qc_image(working, palette_colors=palette_colors)
        buffer = io.BytesIO()
        candidate.save(buffer, format="PNG", optimize=True, compress_level=9)
        payload = buffer.getvalue()
        width, height = working.size
        if len(payload) <= max_bytes or max(width, height) <= min_long_side:
            out_path.write_bytes(payload)
            return
        shrink = max(0.75, min(0.95, (max_bytes / float(len(payload))) ** 0.5 * 0.95))
        next_size = (max(1, int(round(width * shrink))), max(1, int(round(height * shrink))))
        if next_size == working.size:
            out_path.write_bytes(payload)
            return
        working = working.resize(next_size, _resampling("image"))


def _quantize_visual_qc_image(image: Any, *, palette_colors: int) -> Any:
    if image.mode != "RGB":
        return image
    quantize = getattr(Image, "Quantize", None)
    method = getattr(quantize, "MEDIANCUT", 0) if quantize is not None else getattr(Image, "MEDIANCUT", 0)
    try:
        return image.quantize(colors=palette_colors, method=method)
    except Exception:  # noqa: BLE001 - compression must not block QC artifact creation.
        return image


def _resize_qc_pair(image: Any, label: Any, *, max_dimension: int) -> tuple[Any, Any]:
    width, height = image.size
    largest = max(width, height)
    if largest <= max_dimension:
        return image, label
    scale = max_dimension / float(largest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(size, _resampling("image")), label.resize(size, _resampling("mask"))


def _resampling(kind: str) -> Any:
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, "NEAREST" if kind == "mask" else "LANCZOS", 0 if kind == "mask" else 1)
