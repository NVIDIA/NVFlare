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

"""Deterministic, bounded source-file traversal."""

from __future__ import annotations

import ast
from itertools import islice
from pathlib import Path

from nvflare.tool.agent.inspection.source import analyze_tree
from nvflare.tool.agent.inspection.types import FileFacts, SourceScan

SKIPPED_DIRECTORIES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "venv",
}
SENSITIVE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
SENSITIVE_NAMES = {"id_dsa", "id_ecdsa", "id_ed25519", "id_rsa"}
MAX_WALK_ENTRIES = 50_000


def scan_sources(target: Path, *, redact: bool, max_files: int, max_file_bytes: int) -> SourceScan:
    scan = SourceScan(target=target)
    if target.is_symlink():
        _finding(scan, ".", "SYMLINK_SKIPPED")
        return scan
    if target.is_file():
        scan.files_considered = 1
        scan.files_seen.add(".")
        if _is_sensitive(target):
            _finding(scan, ".", "SENSITIVE_FILE_SKIPPED")
        elif target.suffix == ".py":
            _read_python(target, ".", scan, max_file_bytes, redact)
        return scan

    stack = [target]
    exhausted = False
    while stack and not exhausted:
        directory = stack.pop()
        remaining_entries = MAX_WALK_ENTRIES - scan.entries_visited
        try:
            admitted = list(islice(directory.iterdir(), remaining_entries + 1))
        except OSError:
            scan.complete = False
            _finding(scan, _relative(directory, target), "UNREADABLE_DIRECTORY")
            continue
        if len(admitted) > remaining_entries:
            scan.entries_visited = MAX_WALK_ENTRIES
            scan.complete = False
            _finding(scan, _relative(directory, target), "TRAVERSAL_LIMIT_REACHED")
            break
        children = sorted(admitted, key=lambda item: item.name)
        directories: list[Path] = []
        for child in children:
            if scan.entries_visited >= MAX_WALK_ENTRIES:
                scan.complete = False
                _finding(scan, _relative(directory, target), "TRAVERSAL_LIMIT_REACHED")
                exhausted = True
                break
            scan.entries_visited += 1
            rel = _relative(child, target)
            if child.is_symlink():
                _finding(scan, rel, "SYMLINK_SKIPPED")
                continue
            if child.is_dir():
                if child.name not in SKIPPED_DIRECTORIES and not child.name.startswith("."):
                    directories.append(child)
                else:
                    _finding(scan, rel, "DIRECTORY_SKIPPED")
                continue
            if scan.files_considered >= max_files:
                scan.complete = False
                _finding(scan, rel, "FILE_LIMIT_REACHED")
                exhausted = True
                break
            scan.files_considered += 1
            if not child.is_file():
                _finding(scan, rel, "NON_REGULAR_FILE_SKIPPED")
                continue
            scan.files_seen.add(rel)
            if _is_sensitive(child):
                _finding(scan, rel, "SENSITIVE_FILE_SKIPPED")
            elif child.suffix == ".py":
                _read_python(child, rel, scan, max_file_bytes, redact)
        stack.extend(reversed(directories))
    return scan


def _read_python(path: Path, rel: str, scan: SourceScan, max_file_bytes: int, redact: bool) -> None:
    try:
        with path.open("rb") as stream:
            raw = stream.read(max_file_bytes + 1)
        if len(raw) > max_file_bytes:
            scan.complete = False
            _finding(scan, rel, "FILE_TOO_LARGE")
            return
    except OSError:
        scan.complete = False
        _finding(scan, rel, "UNREADABLE_FILE")
        return
    try:
        source = raw.decode("utf-8-sig")
    except UnicodeError:
        scan.complete = False
        _finding(scan, rel, "NON_UTF8_FILE")
        return
    scan.files_read += 1
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as error:
        scan.complete = False
        line = getattr(error, "lineno", 0) or 0
        _finding(scan, rel, "PYTHON_PARSE_ERROR", line)
        scan.facts[rel] = FileFacts(path=rel, complete=False, is_job_py=path.name == "job.py")
        return
    except RecursionError:
        scan.complete = False
        _finding(scan, rel, "PYTHON_AST_DEPTH_LIMIT")
        scan.facts[rel] = FileFacts(path=rel, complete=False, is_job_py=path.name == "job.py")
        return
    try:
        scan.facts[rel] = analyze_tree(
            tree, rel, is_job_py=path.name == "job.py", findings=scan.findings, redact=redact
        )
    except RecursionError:
        scan.complete = False
        _finding(scan, rel, "PYTHON_AST_DEPTH_LIMIT")
        scan.facts[rel] = FileFacts(path=rel, complete=False, is_job_py=path.name == "job.py")


def _is_sensitive(path: Path) -> bool:
    return path.name.lower() in SENSITIVE_NAMES or path.suffix.lower() in SENSITIVE_SUFFIXES


def _finding(scan: SourceScan, file: str, code: str, line: int = 0) -> None:
    scan.findings.append({"file": file, "line": line, "code": code})


def _relative(path: Path, target: Path) -> str:
    return path.relative_to(target).as_posix() or "."
