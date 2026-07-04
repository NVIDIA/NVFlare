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

"""Deterministic behavior checks over captured benchmark artifacts."""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .device_selection import (
    DEVICE_SELECTION_BEHAVIOR_ID,
    DeviceSelectionResult,
    check_device_selection,
    source_uses_gpu_when_available,
)

NO_HARDCODED_DEVICE_BEHAVIOR_ID = "no-hardcoded-device-rewrite"
MAX_PYTHON_FILES = 100
MAX_PYTHON_FILE_BYTES = 1024 * 1024
MAX_PYTHON_TOTAL_BYTES = 4 * 1024 * 1024
IGNORED_DIRS = {".git", ".hg", ".svn", ".venv", "__pycache__", "env", "node_modules", "outputs", "venv"}


def _path_has_symlink(root: Path, relative: Path) -> bool:
    current = root
    try:
        if stat.S_ISLNK(os.lstat(current).st_mode):
            return True
        for part in relative.parts:
            current = current / part
            if stat.S_ISLNK(os.lstat(current).st_mode):
                return True
    except OSError:
        return True
    return False


def _read_bounded_regular_file(root: Path, relative: Path) -> str | None:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        return None
    if _path_has_symlink(root, relative):
        return None
    path = root.joinpath(relative)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > MAX_PYTHON_FILE_BYTES:
            return None
        chunks = []
        remaining = MAX_PYTHON_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > MAX_PYTHON_FILE_BYTES:
            return None
        return data.decode("utf-8", errors="replace")
    finally:
        os.close(descriptor)


def _source_python_files(root: Path) -> tuple[list[tuple[str, str]], bool]:
    files: list[tuple[str, str]] = []
    total_bytes = 0
    truncated = False
    if not root.is_dir() or root.is_symlink():
        return files, truncated
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = sorted(
            name for name in dirnames if name not in IGNORED_DIRS and not (current / name).is_symlink()
        )
        for filename in sorted(filenames):
            path = current / filename
            if path.suffix.lower() != ".py" or path.is_symlink():
                continue
            if len(files) >= MAX_PYTHON_FILES:
                truncated = True
                return files, truncated
            relative = path.relative_to(root)
            text = _read_bounded_regular_file(root, relative)
            if text is None:
                continue
            encoded_bytes = len(text.encode("utf-8"))
            if total_bytes + encoded_bytes > MAX_PYTHON_TOTAL_BYTES:
                truncated = True
                return files, truncated
            files.append((relative.as_posix(), text))
            total_bytes += encoded_bytes
    return files, truncated


def _safe_artifact_relative_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or pure.parts[0] != "changed_files":
        return None
    return Path(*pure.parts)


def _generated_python_files(
    workspace_delta: dict[str, Any], workspace_delta_manifest_path: Path
) -> tuple[list[tuple[str, str]], bool]:
    files: list[tuple[str, str]] = []
    total_bytes = 0
    truncated = False
    delta_root = workspace_delta_manifest_path.parent / "workspace_delta"
    entries = workspace_delta.get("changed_files")
    if not isinstance(entries, list) or not delta_root.is_dir() or delta_root.is_symlink():
        return files, truncated
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        display_path = entry.get("path")
        if not isinstance(display_path, str) or Path(display_path).suffix.lower() != ".py":
            continue
        relative = _safe_artifact_relative_path(entry.get("artifact_path"))
        if relative is None:
            continue
        if len(files) >= MAX_PYTHON_FILES:
            truncated = True
            break
        text = _read_bounded_regular_file(delta_root, relative)
        if text is None:
            continue
        encoded_bytes = len(text.encode("utf-8"))
        if total_bytes + encoded_bytes > MAX_PYTHON_TOTAL_BYTES:
            truncated = True
            break
        files.append((display_path, text))
        total_bytes += encoded_bytes
    return files, truncated


def _combine_device_results(results: Iterable[tuple[str, DeviceSelectionResult]]) -> DeviceSelectionResult:
    evaluated = list(results)
    failed = [(path, result) for path, result in evaluated if result.status == "fail"]
    passed = [(path, result) for path, result in evaluated if result.status == "pass"]
    if failed:
        path, result = failed[0]
        conflict = " Conflicting preserved-device evidence was also found." if passed else ""
        return DeviceSelectionResult("fail", f"{path}: {result.evidence}{conflict}")
    if passed:
        path, result = passed[0]
        return DeviceSelectionResult("pass", f"{path}: {result.evidence}")
    if evaluated:
        path, result = evaluated[0]
        return DeviceSelectionResult("missing", f"{path}: {result.evidence}")
    return DeviceSelectionResult("missing", "No changed Python artifact was captured for deterministic device review.")


def _behavior_map(record: dict[str, Any], category: str) -> dict[str, Any]:
    value = record.get(category)
    if not isinstance(value, dict):
        value = {}
        record[category] = value
    return value


def apply_deterministic_behavior_checks(
    record: dict[str, Any],
    *,
    input_dir: Path,
    workspace_delta: dict[str, Any],
    workspace_delta_manifest_path: Path,
) -> None:
    """Merge authoritative static checks into an agent-produced run record."""

    source_files, source_truncated = _source_python_files(input_dir)
    applicable_sources = [(path, text) for path, text in source_files if source_uses_gpu_when_available(text)]
    mandatory = _behavior_map(record, "mandatory_behavior")
    prohibited = _behavior_map(record, "prohibited_behavior")
    device_was_declared = DEVICE_SELECTION_BEHAVIOR_ID in mandatory or NO_HARDCODED_DEVICE_BEHAVIOR_ID in prohibited
    if not applicable_sources and not device_was_declared:
        return

    generated_files, generated_truncated = _generated_python_files(workspace_delta, workspace_delta_manifest_path)
    if applicable_sources:
        source_path, source_text = applicable_sources[0]
        result = _combine_device_results(
            (path, check_device_selection(source_text, generated_text)) for path, generated_text in generated_files
        )
    else:
        source_path = None
        result = DeviceSelectionResult(
            "missing",
            "The declared device-selection behavior is not applicable to any bounded input Python file; "
            "the fixture or contract does not match the captured input.",
        )

    mandatory[DEVICE_SELECTION_BEHAVIOR_ID] = {
        "status": result.status,
        "evidence": result.evidence,
        "source": "harness_static_analysis",
    }
    if result.status == "not_applicable":
        prohibited_status = "not_applicable"
        prohibited_evidence = result.evidence
    elif result.status == "fail":
        prohibited_status = "fail"
        prohibited_evidence = result.evidence
    else:
        prohibited_status = "pass"
        prohibited_evidence = "No changed Python artifact hard-codes CPU or GPU in place of the source conditional."
    prohibited[NO_HARDCODED_DEVICE_BEHAVIOR_ID] = {
        "status": prohibited_status,
        "evidence": prohibited_evidence,
        "source": "harness_static_analysis",
    }
    checks = record.get("deterministic_behavior_checks")
    if not isinstance(checks, dict):
        checks = {}
        record["deterministic_behavior_checks"] = checks
    checks[DEVICE_SELECTION_BEHAVIOR_ID] = {
        "checker": "python_ast_device_selection_v1",
        "status": result.status,
        "source_path": source_path,
        "input_python_file_count": len(source_files),
        "generated_python_file_count": len(generated_files),
        "input_scan_truncated": source_truncated,
        "generated_scan_truncated": generated_truncated,
    }
