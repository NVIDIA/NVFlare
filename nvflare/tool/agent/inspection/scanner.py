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

"""Bounded filesystem scanning for agent inspection."""

import ast
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Optional

from nvflare.tool.agent.inspection.models import MAX_EVIDENCE_PER_BUCKET, InspectionFacts, redact_literal
from nvflare.tool.agent.inspection.python_scanner import _PythonInspector

DEFAULT_MAX_FILES = 250
DEFAULT_MAX_FILE_BYTES = 512 * 1024
# After max_files is reached, the inspector accounts for a bounded number of
# unvisited files/directories so callers can see that classification is
# incomplete without turning the cap into an unbounded full-tree walk.
MAX_FILE_LIMIT_ACCOUNTED_SKIPS = 10000
PYTHON_SUFFIXES = {".py"}
SKIPPED_DIR_NAMES = {
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
SENSITIVE_FILE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
SENSITIVE_FILE_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
_INCOMPLETE_SCAN_SKIP_CODES = {
    "DIRECTORY_NOT_SCANNED_FILE_LIMIT",
    "FILE_LIMIT_REACHED",
    "FILE_TOO_LARGE",
    "NON_UTF8_FILE",
    "UNREADABLE_DIRECTORY",
    "UNREADABLE_FILE",
}

# Framework detection (import roots, symbols, evidence weights, recommended
# skills, and family/promotion rules) lives in nvflare.tool.agent.frameworks.
# This engine stays framework-agnostic; add a framework there, not here.


@dataclass
class _InspectStateBuilder:
    root: Path
    redact: bool
    entries_visited: int = 0
    files_considered: int = 0
    files_scanned: int = 0
    bytes_scanned: int = 0
    files_skipped_count: int = 0
    file_limit_reached: bool = False
    file_limit_accounted_skips: int = 0
    file_limit_skip_accounting_truncated: bool = False
    classification_incomplete: bool = False
    files_skipped: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    framework_evidence: dict[str, list[dict]] = field(default_factory=dict)
    flare_imports: list[dict] = field(default_factory=list)
    flare_calls: set[str] = field(default_factory=set)
    flare_calls_by_file: dict[str, set[str]] = field(default_factory=dict)
    # framework name -> FLARE conversion-integration call names (e.g. Lightning
    # flare.patch). Populated by framework detectors; used by _conversion_state.
    integration_signals: dict[str, set[str]] = field(default_factory=dict)
    integration_signal_files: set[str] = field(default_factory=set)
    file_imports: dict[str, set[str]] = field(default_factory=dict)
    entry_points: list[dict] = field(default_factory=list)
    job_py: Optional[str] = None
    job_py_paths: list[str] = field(default_factory=list)
    sim_env_used: bool = False
    sim_env_files: set[str] = field(default_factory=set)
    export_support: bool = False
    export_support_files: set[str] = field(default_factory=set)
    exported_job_markers: list[str] = field(default_factory=list)
    exported_job_marker_paths: list[Path] = field(default_factory=list)
    distributed_patterns: list[dict] = field(default_factory=list)
    dynamic_patterns: list[dict] = field(default_factory=list)
    absolute_path_findings: list[dict] = field(default_factory=list)
    # file -> list of (start_line, end_line) for every class definition. Used to
    # decide whether base-framework usage lives inside a superset model class
    # body (e.g. torch calls inside a LightningModule) versus standalone.
    class_body_ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def freeze(self) -> InspectionFacts:
        return InspectionFacts(
            **{
                fact_field.name: _freeze_fact_value(getattr(self, fact_field.name))
                for fact_field in fields(InspectionFacts)
            }
        )


def _freeze_fact_value(value):
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, set):
        return frozenset(value)
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_fact_value(item) for key, item in value.items()})
    return value


def scan_path(
    path: Path | str,
    *,
    redact: bool = True,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> InspectionFacts:
    """Collect static facts without importing or executing user code."""
    target = _normalized_inspect_target(path)
    state = _InspectStateBuilder(root=target, redact=redact)

    if not target.exists() and not target.is_symlink():
        raise FileNotFoundError(f"inspect path does not exist: {path}")

    if target.is_symlink():
        _record_symlink_skip(target, state)
    elif target.is_file():
        _inspect_file(target, state, max_file_bytes)
    else:
        _inspect_dir(target, state, max_files=max_files, max_file_bytes=max_file_bytes)
    return state.freeze()


def _inspect_dir(root: Path, state: _InspectStateBuilder, *, max_files: int, max_file_bytes: int) -> None:
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError as e:
            _add_skip(state, _skip_entry(directory, state, "UNREADABLE_DIRECTORY", "could not read directory", e))
            continue

        for index, child in enumerate(children):
            if child.is_symlink():
                _record_symlink_skip(child, state)
                continue
            if child.is_dir():
                if _should_skip_dir(child, root):
                    _add_skip(state, _skip_entry(child, state, "DIRECTORY_SKIPPED", "directory skipped"))
                    continue
                stack.append(child)
                continue
            if state.entries_visited >= max_files:
                _record_unvisited_due_to_file_limit(state, root, stack, children[index:])
                return
            state.entries_visited += 1
            if not child.is_file():
                continue
            _inspect_file(child, state, max_file_bytes)
            if state.entries_visited >= max_files:
                _record_unvisited_due_to_file_limit(state, root, stack, children[index + 1 :])
                return


def _record_unvisited_due_to_file_limit(
    state: _InspectStateBuilder, root: Path, pending_stack: list[Path], remaining_children: list[Path]
) -> None:
    directories = list(pending_stack)
    limit_left_unvisited_entries = bool(directories)
    for child in remaining_children:
        try:
            if child.is_symlink():
                if not _record_symlink_skip_after_file_limit(child, state):
                    break
                continue
            if child.is_file():
                limit_left_unvisited_entries = True
                if not _add_file_limit_skip(state, child):
                    break
                continue
            if not child.is_dir() or _should_skip_dir(child, root):
                continue
        except OSError:
            limit_left_unvisited_entries = True
            if not _add_skip_after_file_limit(
                state, _skip_entry(child, state, "UNREADABLE_FILE", "could not stat file")
            ):
                break
            continue
        directories.append(child)
        limit_left_unvisited_entries = True

    if not limit_left_unvisited_entries:
        return

    state.file_limit_reached = True
    state.classification_incomplete = True

    seen = set()
    for directory in directories:
        key = str(directory)
        if key in seen:
            continue
        seen.add(key)
        if not _add_directory_not_scanned_due_to_file_limit(state, directory):
            break
        _record_unvisited_files_under_file_limit(directory, state, root)
        if state.file_limit_skip_accounting_truncated:
            break


def _record_unvisited_files_under_file_limit(directory: Path, state: _InspectStateBuilder, root: Path) -> None:
    stack = [directory]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda p: p.name)
        except OSError as e:
            if not _add_skip_after_file_limit(
                state, _skip_entry(current, state, "UNREADABLE_DIRECTORY", "could not read directory", e)
            ):
                return
            continue
        for child in children:
            try:
                if child.is_symlink():
                    if not _record_symlink_skip_after_file_limit(child, state):
                        return
                elif child.is_dir():
                    if not _should_skip_dir(child, root):
                        stack.append(child)
                elif child.is_file():
                    if not _add_file_limit_skip(state, child):
                        return
            except OSError:
                if not _add_skip_after_file_limit(
                    state, _skip_entry(child, state, "UNREADABLE_FILE", "could not stat file")
                ):
                    return


def _account_file_limit_skip(state: _InspectStateBuilder) -> bool:
    if state.file_limit_accounted_skips >= MAX_FILE_LIMIT_ACCOUNTED_SKIPS:
        state.file_limit_skip_accounting_truncated = True
        return False
    state.file_limit_accounted_skips += 1
    return True


def _add_file_limit_skip(state: _InspectStateBuilder, path: Path) -> bool:
    if not _account_file_limit_skip(state):
        return False
    state.files_considered += 1
    _add_skip(state, _skip_entry(path, state, "FILE_LIMIT_REACHED", "file scan limit reached"))
    return True


def _add_skip_after_file_limit(state: _InspectStateBuilder, entry: dict) -> bool:
    if not _account_file_limit_skip(state):
        return False
    _add_skip(state, entry)
    return True


def _add_directory_not_scanned_due_to_file_limit(state: _InspectStateBuilder, directory: Path) -> bool:
    return _add_skip_after_file_limit(
        state,
        _skip_entry(
            directory,
            state,
            "DIRECTORY_NOT_SCANNED_FILE_LIMIT",
            "directory not scanned because file scan limit was reached",
        ),
    )


def _inspect_file(path: Path, state: _InspectStateBuilder, max_file_bytes: int) -> None:
    state.files_considered += 1
    rel_path = _display_path(path, state.root, state.redact)
    if _is_sensitive_file(path):
        _add_skip(state, _skip_entry(path, state, "SENSITIVE_FILE_SKIPPED", "sensitive file skipped"))
        return
    if _is_exported_job_marker(path):
        state.exported_job_markers.append(rel_path)
        state.exported_job_marker_paths.append(path)
    if path.suffix not in PYTHON_SUFFIXES:
        return

    try:
        size = path.stat().st_size
    except OSError as e:
        _add_skip(state, _skip_entry(path, state, "UNREADABLE_FILE", "could not stat file", e))
        return
    if size > max_file_bytes:
        _add_skip(state, _skip_entry(path, state, "FILE_TOO_LARGE", "file exceeds static inspection cap"))
        return

    try:
        # utf-8-sig strips a leading BOM (Windows/Notepad-authored sources) that
        # would otherwise reach ast.parse as U+FEFF and raise SyntaxError, losing
        # all framework/entry-point evidence for the file. It decodes plain UTF-8
        # identically, so NON_UTF8_FILE handling is unaffected.
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        _add_skip(state, _skip_entry(path, state, "NON_UTF8_FILE", "file is not UTF-8 text"))
        return
    except OSError as e:
        _add_skip(state, _skip_entry(path, state, "UNREADABLE_FILE", "could not read file", e))
        return

    state.files_scanned += 1
    state.bytes_scanned += size
    if path.name == "job.py":
        state.job_py_paths.append(rel_path)
        # Keep the public summary pointed at the authoritative root candidate
        # when one exists; a later nested fixture must not overwrite it.
        if state.job_py is None or len(Path(rel_path).parts) == 1:
            state.job_py = rel_path

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as e:
        state.findings.append(
            {
                "code": "PYTHON_PARSE_ERROR",
                "severity": "warning",
                "file": rel_path,
                "line": e.lineno,
                "message": "Python file could not be parsed statically.",
            }
        )
        return
    except RecursionError:
        _record_python_ast_depth_limit(state, rel_path)
        return

    # Register every successfully parsed Python file, including leaf modules
    # with no imports, so local imports can resolve to the complete scanned graph.
    state.file_imports.setdefault(rel_path, set())
    visitor = _PythonInspector(path, rel_path, state)
    try:
        visitor.visit(tree)
        visitor.finalize()
        _add_entry_point(path, rel_path, tree, state)
    except RecursionError:
        _record_python_ast_depth_limit(state, rel_path)


def _record_python_ast_depth_limit(state: _InspectStateBuilder, rel_path: str) -> None:
    state.classification_incomplete = True
    state.findings.append(
        {
            "code": "PYTHON_AST_DEPTH_LIMIT",
            "severity": "warning",
            "file": rel_path,
            "line": None,
            "message": "Python file exceeds the safe static-inspection AST depth.",
        }
    )


def _add_entry_point(path: Path, rel_path: str, tree: ast.Module, state: _InspectStateBuilder) -> None:
    functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    main_guard = any(_is_main_guard(node) for node in tree.body if isinstance(node, ast.If))
    likely = path.name in {"client.py", "server.py", "train.py", "trainer.py", "main.py", "job.py"} or main_guard
    if likely or any(name in {"main", "train", "fit", "evaluate"} for name in functions):
        state.entry_points.append(
            {
                "path": rel_path,
                "kind": "python_script",
                "functions": functions[:MAX_EVIDENCE_PER_BUCKET],
                "main_guard": main_guard,
            }
        )


def _is_main_guard(node: ast.If) -> bool:
    left = getattr(node.test, "left", None)
    comparators = getattr(node.test, "comparators", [])
    if not isinstance(left, ast.Name) or left.id != "__name__" or not comparators:
        return False
    value = comparators[0]
    return isinstance(value, ast.Constant) and value.value == "__main__"


def _record_symlink_skip(path: Path, state: _InspectStateBuilder) -> None:
    _add_skip(state, _symlink_skip_entry(path, state))


def _record_symlink_skip_after_file_limit(path: Path, state: _InspectStateBuilder) -> bool:
    return _add_skip_after_file_limit(state, _symlink_skip_entry(path, state))


def _symlink_skip_entry(path: Path, state: _InspectStateBuilder) -> dict:
    try:
        target = os.readlink(path)
    except OSError:
        target = ""
    return {
        "code": "SYMLINK_SKIPPED",
        "path": _display_path(path, state.root, state.redact),
        "target": redact_literal(target, state.redact),
        "message": "symlink was not followed during static inspection",
    }


def _add_skip(state: _InspectStateBuilder, entry: dict) -> None:
    state.files_skipped_count += 1
    if entry.get("code") in _INCOMPLETE_SCAN_SKIP_CODES:
        state.classification_incomplete = True
    if len(state.files_skipped) < MAX_EVIDENCE_PER_BUCKET:
        state.files_skipped.append(entry)


def _skip_entry(path: Path, state: _InspectStateBuilder, code: str, message: str, error: Exception = None) -> dict:
    result = {"code": code, "path": _display_path(path, state.root, state.redact), "message": message}
    if error is not None:
        result["error_type"] = type(error).__name__
    return result


def _should_skip_dir(path: Path, root: Path) -> bool:
    if path == root:
        return False
    return path.name in SKIPPED_DIR_NAMES or path.name.startswith(".")


def _is_sensitive_file(path: Path) -> bool:
    return path.name in SENSITIVE_FILE_NAMES or path.suffix.lower() in SENSITIVE_FILE_SUFFIXES


def _is_exported_job_marker(path: Path) -> bool:
    return path.name in {"meta.json", "config_fed_server.json", "config_fed_client.json"}


def _display_path(path: Path, root: Path, redact: bool) -> str:
    base = root if root.is_dir() and not root.is_symlink() else root.parent
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        if redact and path.is_absolute():
            return f"<REDACTED_PATH>/{path.name}"
        return str(path)


def _inspected_target_path(path: Path) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


def _normalized_inspect_target(path: Path | str) -> Path:
    return Path(_inspected_target_path(Path(path).expanduser()))
