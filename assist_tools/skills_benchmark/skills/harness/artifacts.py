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

"""Workspace artifact capture and bounded report artifact collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

SOURCE_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
DELTA_SUFFIXES = SOURCE_SUFFIXES | {".log"}
REPORT_SUFFIXES = DELTA_SUFFIXES
SOURCE_NAMES = {"dockerfile", "makefile", "requirements.txt"}
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "data",
    "env",
    "node_modules",
    "outputs",
    "venv",
}
IGNORED_SUFFIXES = {".ckpt", ".npy", ".npz", ".onnx", ".pth", ".pt", ".pkl"}
GENERATED_REPORT_ARTIFACTS = {
    "comprehensive_report.json",
    "comprehensive_report.md",
    "metrics_plots.svg",
    "metrics_plots.png",
    "metrics_report.pdf",
    "benchmark_insights.md",
    # Legacy generated report name; keep excluded for older result roots.
    "direct_report.md",
}
STRUCTURE_FILE_NAMES = {"client.py", "model.py", "job.py", "prepare_data.py", "download_data.py"}

WORKSPACE_MAX_FILE_BYTES = 2 * 1024 * 1024
WORKSPACE_MAX_TOTAL_BYTES = 25 * 1024 * 1024
WORKSPACE_MAX_FILES = 500
WORKSPACE_MAX_FINAL_FILES = 500
REPORT_MAX_FILE_BYTES = 512 * 1024
REPORT_MAX_TOTAL_BYTES = 10 * 1024 * 1024
REPORT_MAX_FILES = 500


def is_source_like(path: Path, *, include_logs: bool = False) -> bool:
    suffixes = DELTA_SUFFIXES if include_logs else SOURCE_SUFFIXES
    lower_name = path.name.lower()
    return (
        path.suffix.lower() in suffixes
        or lower_name in SOURCE_NAMES
        or lower_name.startswith("readme")
        or (lower_name.startswith("requirements") and path.suffix.lower() == ".txt")
    )


def should_skip_rel(rel: Path) -> bool:
    return any(part in IGNORED_DIRS for part in rel.parts) or rel.suffix.lower() in IGNORED_SUFFIXES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files_no_symlink_dirs(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for filename in filenames:
            yield current / filename


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_workspace_baseline(workspace_root: Path, out: Path) -> None:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(iter_files_no_symlink_dirs(workspace_root)):
        if not path.is_file() or path.is_symlink():
            continue
        rel_path = path.relative_to(workspace_root)
        rel = rel_path.as_posix()
        if should_skip_rel(rel_path) or not is_source_like(path):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > WORKSPACE_MAX_FILE_BYTES:
            continue
        files[rel] = {"sha256": sha256(path), "size_bytes": size}

    out.write_text(
        json.dumps(
            {
                "root": str(workspace_root),
                "max_file_bytes": WORKSPACE_MAX_FILE_BYTES,
                "source_file_count": len(files),
                "files": files,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def copy_limited(src: Path, dst: Path, copied_state: dict[str, int], skipped: list[dict[str, Any]]) -> int | None:
    try:
        size = src.stat().st_size
    except OSError as exc:
        skipped.append({"path": str(src), "reason": f"stat_failed:{exc}"})
        return None
    if size > WORKSPACE_MAX_FILE_BYTES:
        skipped.append({"path": str(src), "reason": "file_size_limit", "size_bytes": size})
        return None
    if copied_state["files"] >= WORKSPACE_MAX_FILES:
        skipped.append({"path": str(src), "reason": "file_count_limit", "size_bytes": size})
        return None
    if copied_state["bytes"] + size > WORKSPACE_MAX_TOTAL_BYTES:
        skipped.append({"path": str(src), "reason": "total_size_limit", "size_bytes": size})
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied_state["files"] += 1
    copied_state["bytes"] += size
    return size


def capture_workspace_delta(
    workspace_root: Path,
    baseline_path: Path,
    delta_root: Path,
    manifest_path: Path,
    runtime_artifact_root: Path,
    *,
    delta_scope: str = "workspace",
    include_runtime_artifacts: bool = True,
) -> None:
    baseline = load_json(baseline_path)
    baseline_files = baseline.get("files") if isinstance(baseline.get("files"), dict) else {}
    delta_root.mkdir(parents=True, exist_ok=True)
    changed_root = delta_root / "changed_files"
    runtime_root = delta_root / "runtime_artifacts"
    copied_state = {"files": 0, "bytes": 0}

    changed_files: list[dict[str, Any]] = []
    deleted_files: list[dict[str, Any]] = []
    skipped_files: list[dict[str, Any]] = []
    current_files: dict[str, dict[str, Any]] = {}
    modified_files: list[dict[str, Any]] = []
    added_files: list[dict[str, Any]] = []

    for path in sorted(iter_files_no_symlink_dirs(workspace_root)):
        if not path.is_file() or path.is_symlink():
            continue
        rel_path = path.relative_to(workspace_root)
        rel = rel_path.as_posix()
        if should_skip_rel(rel_path) or not is_source_like(path, include_logs=True):
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            skipped_files.append({"path": rel, "reason": f"stat_failed:{exc}"})
            continue
        if size > WORKSPACE_MAX_FILE_BYTES:
            skipped_files.append({"path": rel, "reason": "file_size_limit", "size_bytes": size})
            continue
        file_hash = sha256(path)
        current_files[rel] = {"sha256": file_hash, "size_bytes": size}
        before = baseline_files.get(rel)
        if before and before.get("sha256") == file_hash:
            continue
        status = "modified" if before else "added"
        copied_size = copy_limited(path, changed_root / rel_path, copied_state, skipped_files)
        if copied_size is not None:
            entry = {
                "path": rel,
                "status": status,
                "size_bytes": copied_size,
                "sha256": file_hash,
                "artifact_path": (changed_root / rel_path).relative_to(delta_root).as_posix(),
            }
            changed_files.append(entry)
            if status == "modified":
                modified_files.append(entry)
            else:
                added_files.append(entry)

    for rel in sorted(set(baseline_files) - set(current_files)):
        deleted_files.append({"path": rel, "status": "deleted"})

    workspace_changes = [*added_files, *modified_files, *deleted_files]

    runtime_artifacts: list[dict[str, Any]] = []
    if include_runtime_artifacts:
        runtime_sources = [
            ("runtime_job_config", runtime_artifact_root / "job_config"),
        ]
        for label, root in runtime_sources:
            if not root.is_dir():
                continue
            for path in sorted(iter_files_no_symlink_dirs(root)):
                if not path.is_file() or path.is_symlink():
                    continue
                rel_path = path.relative_to(root)
                rel = rel_path.as_posix()
                if should_skip_rel(rel_path) or not is_source_like(path, include_logs=True):
                    continue
                dst = runtime_root / label / rel_path
                copied_size = copy_limited(path, dst, copied_state, skipped_files)
                if copied_size is not None:
                    runtime_artifacts.append(
                        {
                            "path": f"{label}/{rel}",
                            "source_path": str(path),
                            "size_bytes": copied_size,
                            "artifact_path": dst.relative_to(delta_root).as_posix(),
                        }
                    )

    final_files: list[dict[str, Any]] = []
    final_structure_files: list[dict[str, Any]] = []
    for rel, meta in sorted(current_files.items()):
        entry = {"path": rel, "size_bytes": meta.get("size_bytes"), "sha256": meta.get("sha256")}
        if len(final_files) < WORKSPACE_MAX_FINAL_FILES:
            final_files.append(entry)
        if Path(rel).name.lower() in STRUCTURE_FILE_NAMES:
            final_structure_files.append(entry)
    manifest = {
        "schema_version": "2",
        "delta_scope": delta_scope,
        "workspace_root": str(workspace_root),
        "delta_dir": str(delta_root),
        "changed_files_dir": str(changed_root),
        "runtime_artifacts_dir": str(runtime_root),
        "limits": {
            "max_file_bytes": WORKSPACE_MAX_FILE_BYTES,
            "max_total_bytes": WORKSPACE_MAX_TOTAL_BYTES,
            "max_files": WORKSPACE_MAX_FILES,
            "max_final_files": WORKSPACE_MAX_FINAL_FILES,
        },
        "initial_source_file_count": len(baseline_files),
        "final_source_file_count": len(current_files),
        "changed_file_count": len(changed_files),
        "deleted_file_count": len(deleted_files),
        "workspace_added_file_count": len(added_files),
        "workspace_modified_file_count": len(modified_files),
        "workspace_deleted_baseline_file_count": len(deleted_files),
        "workspace_change_count": len(workspace_changes),
        "workspace_changes_allowed": delta_scope == "agent_workspace",
        "legacy_compat": {
            "input_mutation_fields_removed": True,
            "source_input_policy_manifest": "input_delta_manifest.json",
        },
        "runtime_artifact_count": len(runtime_artifacts),
        "copied_file_count": copied_state["files"],
        "copied_bytes": copied_state["bytes"],
        "final_file_manifest_count": len(final_files),
        "final_files_truncated": len(current_files) > len(final_files),
        "final_structure_file_count": len(final_structure_files),
        "final_structure_files": final_structure_files,
        "final_files": final_files,
        "changed_files": changed_files,
        "deleted_files": deleted_files,
        "workspace_added_files": added_files,
        "workspace_modified_files": modified_files,
        "workspace_deleted_baseline_files": deleted_files,
        "workspace_changes": workspace_changes,
        "runtime_artifacts": runtime_artifacts,
        "skipped_files": skipped_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def collect_report_artifacts(root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(iter_files_no_symlink_dirs(root)):
        if len(artifacts) >= REPORT_MAX_FILES:
            break
        if not path.is_file() or path.is_symlink():
            continue
        rel_path = path.relative_to(root)
        rel = rel_path.as_posix()
        if rel in GENERATED_REPORT_ARTIFACTS or should_skip_rel(rel_path):
            continue
        suffix = path.suffix.lower()
        if suffix not in REPORT_SUFFIXES:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if total_bytes >= REPORT_MAX_TOTAL_BYTES:
            break
        read_limit = min(size, REPORT_MAX_FILE_BYTES, REPORT_MAX_TOTAL_BYTES - total_bytes)
        try:
            with path.open("rb") as stream:
                raw = stream.read(read_limit)
        except OSError as exc:
            artifacts.append(
                {
                    "relative_path": rel,
                    "size_bytes": size,
                    "captured_bytes": 0,
                    "truncated": False,
                    "line_count": 0,
                    "kind": suffix.lstrip("."),
                    "content": "",
                    "read_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        total_bytes += len(raw)
        text = raw.decode("utf-8", errors="replace")
        truncated = size > read_limit
        entry: dict[str, Any] = {
            "relative_path": rel,
            "size_bytes": size,
            "captured_bytes": len(raw),
            "truncated": truncated,
            "line_count": 0 if text == "" else text.count("\n") + (0 if text.endswith("\n") else 1),
            "kind": suffix.lstrip("."),
            "content": text,
        }

        if not truncated and suffix == ".json":
            try:
                entry["json"] = json.loads(text)
            except json.JSONDecodeError as exc:
                entry["json_parse_error"] = str(exc)
        elif suffix == ".jsonl":
            record_count = 0
            parse_errors = 0
            for line in text.splitlines():
                if not line.strip():
                    continue
                record_count += 1
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
            entry["jsonl_record_count"] = record_count
            entry["jsonl_parse_errors"] = parse_errors

        artifacts.append(entry)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("workspace_root", type=Path)
    baseline.add_argument("out", type=Path)

    delta = subparsers.add_parser("delta")
    delta.add_argument("workspace_root", type=Path)
    delta.add_argument("baseline_path", type=Path)
    delta.add_argument("delta_root", type=Path)
    delta.add_argument("manifest_path", type=Path)
    delta.add_argument("runtime_artifact_root", type=Path)

    args = parser.parse_args()
    if args.command == "baseline":
        write_workspace_baseline(args.workspace_root, args.out)
    elif args.command == "delta":
        capture_workspace_delta(
            args.workspace_root,
            args.baseline_path,
            args.delta_root,
            args.manifest_path,
            args.runtime_artifact_root,
        )


if __name__ == "__main__":
    main()
