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

"""Read observed metric values from captured runtime artifacts."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping

from .quality_signals import (
    canonical_metric_name,
    is_numeric_metric_value,
    metric_value_entries,
    reported_metric_payload,
)
from .safe_paths import UnsafeArtifactPath, path_beneath

METRIC_NAME_KEYS = frozenset({"name", "metric", "metric_name", "key", "tag"})
METRIC_VALUE_KEYS = frozenset({"value", "score", "scalar", "mean"})
ARTIFACT_SUFFIXES = frozenset({".json", ".jsonl"})
MAX_METRIC_ARTIFACTS = 500
MAX_METRIC_ARTIFACT_BYTES = 512 * 1024
MAX_METRIC_JSONL_LINES = 10000
MAX_METRIC_JSON_NODES = 20000
MAX_METRIC_JSON_DEPTH = 64


def captured_metric_artifact_paths(manifest: Mapping[str, Any], manifest_path: Path) -> list[Path]:
    """Return copied structured artifact paths from a workspace-delta manifest."""

    delta_dir = manifest_path.parent / "workspace_delta"
    paths: list[Path] = []
    seen: set[Path] = set()
    for key in ("runtime_artifacts", "changed_files", "workspace_added_files", "workspace_modified_files"):
        items = manifest.get(key)
        if not isinstance(items, list):
            continue
        category_paths: list[Path] = []
        expected_prefix = "runtime_artifacts" if key == "runtime_artifacts" else "changed_files"
        for item in items[:MAX_METRIC_ARTIFACTS]:
            if not isinstance(item, dict):
                continue
            artifact_text = str(item.get("artifact_path") or "")
            if not artifact_text:
                continue
            try:
                candidate = path_beneath(
                    delta_dir,
                    artifact_text,
                    label=f"workspace delta {key} artifact_path",
                    required_prefix=expected_prefix,
                )
                file_stat = os.lstat(candidate)
            except (OSError, UnsafeArtifactPath):
                continue
            if stat.S_ISREG(file_stat.st_mode) and candidate.suffix.lower() in ARTIFACT_SUFFIXES:
                category_paths.append(candidate)
        for candidate in sorted(category_paths, key=lambda path: path.as_posix()):
            if candidate not in seen:
                paths.append(candidate)
                seen.add(candidate)
    return paths


def load_artifact_payloads(path: Path) -> Iterable[Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return []
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > MAX_METRIC_ARTIFACT_BYTES:
            return []
        chunks = []
        remaining = MAX_METRIC_ARTIFACT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_METRIC_ARTIFACT_BYTES:
            return []
        text = raw.decode("utf-8", errors="replace")
    finally:
        os.close(descriptor)
    if path.suffix.lower() == ".jsonl":
        payloads = []
        for line in text.splitlines()[:MAX_METRIC_JSONL_LINES]:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if payload_within_limits(payload):
                payloads.append(payload)
        return payloads
    try:
        payload = json.loads(text)
    except Exception:
        return []
    return [payload] if payload_within_limits(payload) else []


def payload_within_limits(payload: Any) -> bool:
    pending = [(payload, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > MAX_METRIC_JSON_NODES or depth > MAX_METRIC_JSON_DEPTH:
            return False
        if isinstance(value, Mapping):
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
    return True


def _json_path(parts: tuple[str, ...]) -> str:
    return ".".join(part for part in parts if part)


def _artifact_label(path_parts: tuple[str, ...], source_path: Path) -> str:
    context = _json_path(path_parts)
    if any(token in context.lower() for token in ("site", "client")):
        return f"artifact site validation metric {context or source_path.name}"
    return f"artifact aggregated validation metric {context or source_path.name}"


def metric_entries_from_payload(
    payload: Any,
    expected_metric: str,
    source_path: Path,
    path_parts: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    expected = canonical_metric_name(expected_metric)
    entries: list[dict[str, Any]] = []
    if isinstance(payload, Mapping):
        name_values = [
            value
            for key, value in payload.items()
            if str(key).lower() in METRIC_NAME_KEYS and canonical_metric_name(value) == expected
        ]
        if name_values:
            for key, value in payload.items():
                if str(key).lower() in METRIC_VALUE_KEYS and is_numeric_metric_value(value):
                    entries.append({"label": _artifact_label(path_parts + (str(key),), source_path), "value": value})
        for key, value in payload.items():
            key_name = canonical_metric_name(key)
            next_parts = path_parts + (str(key),)
            if key_name == expected and is_numeric_metric_value(value):
                entries.append({"label": _artifact_label(next_parts, source_path), "value": value})
            elif isinstance(value, str):
                for entry in metric_value_entries(expected, value):
                    labeled = dict(entry)
                    labeled.setdefault("label", _artifact_label(next_parts, source_path))
                    entries.append(labeled)
            else:
                entries.extend(metric_entries_from_payload(value, expected, source_path, next_parts))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            entries.extend(metric_entries_from_payload(item, expected, source_path, path_parts + (f"[{index}]",)))
    elif isinstance(payload, str):
        entries.extend(metric_value_entries(expected, payload))
    return entries


def validation_metric_from_artifact(path: Path, expected_metric: str | None) -> dict[str, Any]:
    metric_name = canonical_metric_name(expected_metric)
    if not metric_name:
        return {}
    entries: list[dict[str, Any]] = []
    for payload in load_artifact_payloads(path):
        entries.extend(metric_entries_from_payload(payload, metric_name, path))
    if not entries:
        return {}
    result = reported_metric_payload(metric_name, entries)
    if not result.get("reported_values"):
        return {}
    result["source"] = "metrics_artifact"
    result["source_path"] = str(path)
    return result


def validation_metric_from_workspace_delta_manifest(
    manifest: Mapping[str, Any], manifest_path: Path, expected_metric: str | None = None
) -> dict[str, Any]:
    for path in captured_metric_artifact_paths(manifest, manifest_path):
        metric = validation_metric_from_artifact(path, expected_metric)
        if metric:
            return metric
    return {}
