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

"""Static read-only inspection for agent workflows."""

import ast
import copy
import errno
import os
import re
import stat
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nvflare
from nvflare.tool.agent import frameworks
from nvflare.tool.agent.frameworks.base import DetectContext

DEFAULT_MAX_FILES = 250
DEFAULT_MAX_FILE_BYTES = 512 * 1024
MAX_EVIDENCE_PER_BUCKET = 12
# Internal complexity budgets complement the public file/byte limits without
# changing the v1 JSON contract. They bound inputs that contain very broad or
# deeply nested directory trees and syntactically dense Python files.
MAX_DIRECTORIES_VISITED = 1000
MAX_DIRECTORY_ENTRIES = 10000
MAX_DIRECTORY_DEPTH = 32
MAX_AST_NODES = 50000
MAX_IMPORTS_PER_FILE = 1000
MAX_REACHABILITY_EDGES = 10000
MAX_REACHABILITY_DEPTH = 64
# Backstop for evidence collected per framework bucket. Far above the display
# cap so ordinary ranking/detection uses the complete weighted evidence; inputs
# that reach this pathological guard are reported as truncated.
MAX_EVIDENCE_COLLECT = 10000
# Packaging root dirs whose leading segment is not part of the import path
# (PyPA src-layout), so `src/pkg/mod.py` is importable as `pkg.mod`.
_PACKAGE_ROOT_DIR_NAMES = {"src"}

_INCOMPLETE_SKIP_CODES = {
    "AST_NODE_LIMIT_REACHED",
    "DIRECTORY_DEPTH_LIMIT_REACHED",
    "DIRECTORY_ENTRY_LIMIT_REACHED",
    "DIRECTORY_LIMIT_REACHED",
    "DIRECTORY_NOT_SCANNED_FILE_LIMIT",
    "FILE_LIMIT_REACHED",
    "FILE_TOO_LARGE",
    "IMPORT_GRAPH_LIMIT_REACHED",
    "NON_REGULAR_FILE",
    "NON_UTF8_FILE",
    "PYTHON_PARSE_ERROR",
    "UNREADABLE_DIRECTORY",
    "UNREADABLE_FILE",
}

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
SECRET_NAME_PATTERN = re.compile(r"(api[_-]?key|secret|token|password|passwd|credential|access[_-]?key)", re.I)

# Framework detection (import roots, symbols, evidence weights, recommended
# skills, and family/promotion rules) lives in nvflare.tool.agent.frameworks.
# This engine stays framework-agnostic; add a framework there, not here.


@dataclass
class InspectState:
    root: Path
    redact: bool
    entries_visited: int = 0
    directories_visited: int = 0
    directories_discovered: int = 0
    files_considered: int = 0
    files_scanned: int = 0
    bytes_scanned: int = 0
    files_skipped_count: int = 0
    files_skipped: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    framework_evidence: dict[str, list[dict]] = field(default_factory=dict)
    flare_imports: list[dict] = field(default_factory=list)
    flare_calls: set[str] = field(default_factory=set)
    # framework name -> FLARE conversion-integration call names (e.g. Lightning
    # flare.patch). Populated by framework detectors; used by _conversion_state.
    integration_signals: dict[str, set[str]] = field(default_factory=dict)
    file_imports: dict[str, set[str]] = field(default_factory=dict)
    entry_points: list[dict] = field(default_factory=list)
    # Routing considers only the strongest launch candidates. Keeping the
    # priority outside the public entry-point dictionaries preserves the JSON
    # schema while preventing generic helper functions from becoming roots when
    # a real main guard or conventional launch script exists.
    entry_point_priorities: dict[str, int] = field(default_factory=dict)
    job_py: Optional[str] = None
    sim_env_used: bool = False
    export_support: bool = False
    exported_job_markers: list[str] = field(default_factory=list)
    distributed_patterns: list[dict] = field(default_factory=list)
    dynamic_patterns: list[dict] = field(default_factory=list)
    absolute_path_findings: list[dict] = field(default_factory=list)
    # file -> list of (start_line, end_line) for every class definition. Used to
    # decide whether base-framework usage lives inside a superset model class
    # body (e.g. torch calls inside a LightningModule) versus standalone.
    class_body_ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # Cache for _local_files_by_module: built once after the scan populates
    # file_imports, reused by the reachability cache.
    local_files_by_module_cache: Optional[dict[str, set[str]]] = field(default=None, repr=False, compare=False)
    # Union of files reachable from the preferred entry points. Framework
    # selection asks about many evidence items, so compute the import graph once
    # rather than traversing it once per item.
    reachable_files_cache: Optional[set[str]] = field(default=None, repr=False, compare=False)
    import_limit_reported: set[str] = field(default_factory=set, repr=False, compare=False)
    evidence_limit_reported: set[str] = field(default_factory=set, repr=False, compare=False)
    evidence_indices_by_weight: dict[str, dict[int, deque[int]]] = field(
        default_factory=dict, repr=False, compare=False
    )
    reachability_limit_reported: bool = field(default=False, repr=False, compare=False)
    bucket_observed_counts: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    bucket_collected_counts: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    truncated_buckets: set[str] = field(default_factory=set, repr=False, compare=False)
    incomplete_reasons: set[str] = field(default_factory=set, repr=False, compare=False)


def inspect_path(
    path: Path | str,
    *,
    redact: bool = True,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict:
    """Inspect a path without importing or executing user code."""
    target = Path(path).expanduser()
    state = InspectState(root=target, redact=redact)

    if not target.exists() and not target.is_symlink():
        raise FileNotFoundError(f"inspect path does not exist: {path}")

    if target.is_symlink():
        _record_symlink_skip(target, state)
    elif target.is_file():
        _inspect_file(target, state, max_file_bytes)
    else:
        _inspect_dir(target, state, max_files=max_files, max_file_bytes=max_file_bytes)

    ranked_frameworks = _rank_frameworks(state)
    detected_framework = _detect_primary_framework(state, ranked_frameworks)
    ranked_frameworks = _order_frameworks_for_display(ranked_frameworks, detected_framework)
    conversion_state = _conversion_state(state, detected_framework)
    target_type = _target_type(target, state, detected_framework, conversion_state)
    displayed_findings = _findings_for_output(state)

    return {
        "schema_version": "1",
        "nvflare_version": nvflare.__version__,
        "path": _inspected_target_path(target),
        "target_type": target_type,
        "static_only": True,
        "redaction": "on" if redact else "off",
        "completeness": {
            "complete": not state.incomplete_reasons,
            "reasons": sorted(state.incomplete_reasons),
            "truncated_buckets": sorted(state.truncated_buckets),
            "output_truncated_buckets": _output_truncated_buckets(state),
            "bucket_counts": {
                name: {
                    "observed": count,
                    "collected": state.bucket_collected_counts.get(name, 0),
                }
                for name, count in sorted(state.bucket_observed_counts.items())
            },
        },
        "limits": {
            "max_files": max_files,
            "max_file_bytes": max_file_bytes,
            "max_evidence_per_bucket": MAX_EVIDENCE_PER_BUCKET,
            "max_evidence_collected_per_bucket": MAX_EVIDENCE_COLLECT,
            "max_directories": MAX_DIRECTORIES_VISITED,
            "max_directory_entries": MAX_DIRECTORY_ENTRIES,
            "max_directory_depth": MAX_DIRECTORY_DEPTH,
            "max_ast_nodes_per_file": MAX_AST_NODES,
            "max_imports_per_file": MAX_IMPORTS_PER_FILE,
            "max_reachability_edges": MAX_REACHABILITY_EDGES,
            "max_reachability_depth": MAX_REACHABILITY_DEPTH,
        },
        "scan": {
            "entries_visited": state.entries_visited,
            "directories_discovered": state.directories_discovered,
            "directories_visited": state.directories_visited,
            "files_considered": state.files_considered,
            "files_scanned": state.files_scanned,
            "bytes_scanned": state.bytes_scanned,
            "files_skipped_count": state.files_skipped_count,
            "files_skipped_truncated": state.files_skipped_count > len(state.files_skipped),
            "files_skipped": state.files_skipped,
        },
        "frameworks": ranked_frameworks,
        "entry_points": state.entry_points[:MAX_EVIDENCE_PER_BUCKET],
        "flare_integration": {
            "present": bool(state.flare_imports or state.flare_calls),
            "imports": state.flare_imports[:MAX_EVIDENCE_PER_BUCKET],
            "calls": sorted(state.flare_calls)[:MAX_EVIDENCE_PER_BUCKET],
        },
        "conversion_state": conversion_state,
        "job": {
            "job_py": state.job_py,
            "sim_env_used": state.sim_env_used,
            "export_support": state.export_support,
            "exported_job_markers": state.exported_job_markers[:MAX_EVIDENCE_PER_BUCKET],
        },
        "patterns": {
            "distributed": state.distributed_patterns[:MAX_EVIDENCE_PER_BUCKET],
            "dynamic": state.dynamic_patterns[:MAX_EVIDENCE_PER_BUCKET],
            "absolute_data_paths": state.absolute_path_findings[:MAX_EVIDENCE_PER_BUCKET],
        },
        "findings": displayed_findings,
        "skill_selection": _skill_selection(detected_framework, conversion_state, state),
        "recommended_next_commands": _recommended_next_commands(detected_framework, conversion_state, state),
    }


def _inspect_dir(root: Path, state: InspectState, *, max_files: int, max_file_bytes: int) -> None:
    stack = [root]
    state.directories_discovered = 1
    while stack:
        directory = stack.pop()
        if state.directories_visited >= MAX_DIRECTORIES_VISITED:
            _add_skip(
                state,
                _skip_entry(
                    directory,
                    state,
                    "DIRECTORY_LIMIT_REACHED",
                    "directory traversal limit reached",
                ),
            )
            return
        state.directories_visited += 1
        try:
            children, entries_truncated = _bounded_directory_children(directory)
        except OSError as e:
            _add_skip(state, _skip_entry(directory, state, "UNREADABLE_DIRECTORY", "could not read directory", e))
            continue

        if entries_truncated:
            _add_skip(
                state,
                _skip_entry(
                    directory,
                    state,
                    "DIRECTORY_ENTRY_LIMIT_REACHED",
                    "directory entry limit reached; remaining entries were not inspected",
                ),
            )
            # The filesystem does not guarantee directory iteration order. Do
            # not derive routing evidence from an arbitrary bounded prefix.
            continue

        for index, child in enumerate(children):
            if child.is_symlink():
                _record_symlink_skip(child, state)
                continue
            if child.is_dir():
                if _should_skip_dir(child, root):
                    _add_skip(state, _skip_entry(child, state, "DIRECTORY_SKIPPED", "directory skipped"))
                    continue
                if _directory_depth(child, root) > MAX_DIRECTORY_DEPTH:
                    _add_skip(
                        state,
                        _skip_entry(
                            child,
                            state,
                            "DIRECTORY_DEPTH_LIMIT_REACHED",
                            "directory depth limit reached",
                        ),
                    )
                    continue
                if state.directories_discovered >= MAX_DIRECTORIES_VISITED:
                    _add_skip(
                        state,
                        _skip_entry(
                            child,
                            state,
                            "DIRECTORY_LIMIT_REACHED",
                            "directory discovery limit reached",
                        ),
                    )
                    continue
                state.directories_discovered += 1
                stack.append(child)
                continue
            if state.entries_visited >= max_files:
                _add_skip(state, _skip_entry(child, state, "FILE_LIMIT_REACHED", "file scan limit reached"))
                _record_unvisited_directories_due_to_file_limit(state, root, stack, children[index + 1 :])
                return
            state.entries_visited += 1
            if not child.is_file() and child.suffix not in PYTHON_SUFFIXES:
                continue
            _inspect_file(child, state, max_file_bytes)
            if state.entries_visited >= max_files:
                _record_next_file_due_to_file_limit(state, children[index + 1 :])
                _record_unvisited_directories_due_to_file_limit(state, root, stack, children[index + 1 :])
                return


def _bounded_directory_children(directory: Path) -> tuple[list[Path], bool]:
    """Read at most the per-directory entry budget.

    ``sorted(directory.iterdir())`` first materializes every entry, which makes
    the nominal file cap ineffective against a directory containing millions of
    subdirectories. Consume one sentinel beyond the budget, then sort only the
    bounded prefix for stable processing within that prefix.
    """
    children = []
    entries_truncated = False
    for child in directory.iterdir():
        if len(children) >= MAX_DIRECTORY_ENTRIES:
            return [], True
        children.append(child)
    return sorted(children, key=lambda path: path.name), entries_truncated


def _directory_depth(path: Path, root: Path) -> int:
    try:
        return len(path.relative_to(root).parts)
    except ValueError:
        return MAX_DIRECTORY_DEPTH + 1


def _record_unvisited_directories_due_to_file_limit(
    state: InspectState, root: Path, pending_stack: list[Path], remaining_children: list[Path]
) -> None:
    directories = list(pending_stack)
    for child in remaining_children:
        try:
            if child.is_symlink() or not child.is_dir() or _should_skip_dir(child, root):
                continue
        except OSError:
            continue
        directories.append(child)

    seen = set()
    for directory in directories:
        key = str(directory)
        if key in seen:
            continue
        seen.add(key)
        _add_skip(
            state,
            _skip_entry(
                directory,
                state,
                "DIRECTORY_NOT_SCANNED_FILE_LIMIT",
                "directory not scanned because file scan limit was reached",
            ),
        )


def _record_next_file_due_to_file_limit(state: InspectState, remaining_children: list[Path]) -> None:
    for child in remaining_children:
        try:
            if child.is_symlink() or child.is_dir():
                continue
        except OSError:
            continue
        _add_skip(state, _skip_entry(child, state, "FILE_LIMIT_REACHED", "file scan limit reached"))
        return


@dataclass(frozen=True)
class _BoundedTextRead:
    text: Optional[str]
    byte_count: int = 0
    code: Optional[str] = None
    message: Optional[str] = None
    error: Optional[Exception] = None


def _read_bounded_regular_text(path: Path, max_file_bytes: int) -> _BoundedTextRead:
    """Read one regular file through one no-follow, nonblocking descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as e:
        if e.errno in {errno.ELOOP, getattr(errno, "EMLINK", errno.ELOOP)}:
            return _BoundedTextRead(None, code="SYMLINK_SKIPPED", message="symlink was not followed", error=e)
        return _BoundedTextRead(None, code="UNREADABLE_FILE", message="could not open file", error=e)

    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            return _BoundedTextRead(
                None,
                code="NON_REGULAR_FILE",
                message="path is not a regular file",
            )
        if max_file_bytes < 0 or opened.st_size > max_file_bytes:
            return _BoundedTextRead(
                None,
                code="FILE_TOO_LARGE",
                message="file exceeds static inspection cap",
            )

        chunks = []
        byte_count = 0
        read_limit = max_file_bytes + 1
        while byte_count < read_limit:
            chunk = os.read(descriptor, min(64 * 1024, read_limit - byte_count))
            if not chunk:
                break
            chunks.append(chunk)
            byte_count += len(chunk)
        data = b"".join(chunks)
        if len(data) > max_file_bytes:
            return _BoundedTextRead(
                None,
                code="FILE_TOO_LARGE",
                message="file exceeds static inspection cap",
            )
        try:
            return _BoundedTextRead(data.decode("utf-8"), byte_count=len(data))
        except UnicodeDecodeError:
            return _BoundedTextRead(None, code="NON_UTF8_FILE", message="file is not UTF-8 text")
    except OSError as e:
        return _BoundedTextRead(None, code="UNREADABLE_FILE", message="could not read file", error=e)
    finally:
        os.close(descriptor)


def _inspect_file(path: Path, state: InspectState, max_file_bytes: int) -> None:
    state.files_considered += 1
    rel_path = _display_path(path, state.root, state.redact)
    if _is_sensitive_file(path):
        _add_skip(state, _skip_entry(path, state, "SENSITIVE_FILE_SKIPPED", "sensitive file skipped"))
        return
    if _is_exported_job_marker(path):
        _append_bounded_list(state, "exported_job_markers", state.exported_job_markers, rel_path)
    if path.suffix not in PYTHON_SUFFIXES:
        return

    read_result = _read_bounded_regular_text(path, max_file_bytes)
    if read_result.code:
        _add_skip(
            state,
            _skip_entry(path, state, read_result.code, read_result.message or "could not read file", read_result.error),
        )
        return
    text = read_result.text or ""

    state.files_scanned += 1
    state.bytes_scanned += read_result.byte_count
    if path.name == "job.py":
        state.job_py = rel_path

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as e:
        _mark_incomplete(state, "PYTHON_PARSE_ERROR")
        _append_finding(
            state,
            {
                "code": "PYTHON_PARSE_ERROR",
                "severity": "warning",
                "file": rel_path,
                "line": e.lineno,
                "message": "Python file could not be parsed statically.",
            },
        )
        return

    if _ast_node_limit_exceeded(tree):
        _add_skip(
            state,
            _skip_entry(
                path,
                state,
                "AST_NODE_LIMIT_REACHED",
                "Python AST exceeds the static inspection node limit",
            ),
        )
        return

    visitor = _PythonInspector(path, rel_path, state)
    visitor.collect_module_bindings(tree)
    visitor.visit(tree)
    _add_entry_point(path, rel_path, tree, state)


def _ast_node_limit_exceeded(tree: ast.AST) -> bool:
    for index, _node in enumerate(ast.walk(tree), start=1):
        if index > MAX_AST_NODES:
            return True
    return False


class _PythonInspector(ast.NodeVisitor):
    def __init__(self, path: Path, rel_path: str, state: InspectState):
        self.path = path
        self.rel_path = rel_path
        self.state = state
        self._detectors = frameworks.detectors()
        self._detector_states = {detector.name: detector.new_file_state() for detector in self._detectors}
        self._scope_kind = "module"
        self._class_enclosing_states: list[dict] = []
        self._ctx = DetectContext(
            self._emit_framework_evidence,
            self._add_flare_call,
            self._add_integration_signal,
        )

    def collect_module_bindings(self, tree: ast.Module) -> None:
        """Collect module-scope imports before inspecting uses.

        Function bodies may validly reference a global imported later in the
        module because the function is called only after module initialization.
        A single source-order pass misses those uses. The binding pre-pass skips
        nested scopes, then the normal visitor records evidence and handles
        scope-local imports in source order as before.
        """
        _ModuleImportBindingCollector(self).visit(tree)

    def _emit_framework_evidence(self, framework: str, kind: str, value: str, lineno) -> None:
        _append_framework_evidence(self.state, framework, _evidence(self.rel_path, lineno, kind, value))

    def _add_integration_signal(self, framework: str, name: str) -> None:
        _add_bounded_set(
            self.state,
            "integration_signals",
            self.state.integration_signals.setdefault(framework, set()),
            name,
        )

    def _add_flare_call(self, call_name: str) -> None:
        _add_bounded_set(self.state, "flare_calls", self.state.flare_calls, call_name)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record_import(alias.name, node.lineno, classify=not self._is_local_absolute_import(alias.name))
            self._bind_import(alias)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        resolved_module = _resolve_import_from_module(self.rel_path, module, node.level)
        classify = node.level == 0 and not self._is_local_absolute_import(module)
        self._record_import(resolved_module, node.lineno, classify=classify)
        self._record_import_from_modules(module, node.level, node.names)
        self._bind_import_from(module, node.level, node.names)
        for alias in node.names:
            if node.level == 0 and alias.name in {"FedJob", "FLModel", "SimEnv"}:
                _append_bounded_list(
                    self.state,
                    "flare_imports",
                    self.state.flare_imports,
                    _evidence(self.rel_path, node.lineno, "from_import", f"{module}.{alias.name}"),
                )
        self.generic_visit(node)

    def _bind_import(self, alias: ast.alias) -> None:
        if self._is_local_absolute_import(alias.name):
            return
        for detector in self._detectors:
            detector.on_import(alias, self._detector_states[detector.name], self._ctx)

    def _bind_import_from(self, module: str, level: int, aliases: list[ast.alias]) -> None:
        # A relative import is local by definition. Passing its raw spelling
        # (``from .lightning import Trainer`` -> ``lightning``) to a detector
        # would falsely bind a local symbol as the external framework.
        if level or self._is_local_absolute_import(module):
            return
        for detector in self._detectors:
            detector.on_import_from(module, aliases, self._detector_states[detector.name], self._ctx)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        end_lineno = getattr(node, "end_lineno", None) or node.lineno
        _append_class_body_range(self.state, self.rel_path, (node.lineno, end_lineno))
        for base in node.bases:
            base_name = _symbol_name(base)
            if not base_name:
                continue
            for detector in self._detectors:
                detector.on_class_base(base_name, node.lineno, self._detector_states[detector.name], self._ctx)
        self._visit_binding_scope(node, "class")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_binding_scope(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_binding_scope(node, "function")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_binding_scope(node, "function")

    def _lexically_visible_states(self) -> dict:
        if self._scope_kind == "class" and self._class_enclosing_states:
            return self._class_enclosing_states[-1]
        return self._detector_states

    def _visit_binding_scope(self, node: ast.AST, scope_kind: str) -> None:
        enclosing_states = self._lexically_visible_states()
        previous_states = self._detector_states
        previous_kind = self._scope_kind
        self._detector_states = copy.deepcopy(enclosing_states)
        self._scope_kind = scope_kind
        if scope_kind == "class":
            self._class_enclosing_states.append(enclosing_states)
        try:
            self.generic_visit(node)
        finally:
            if scope_kind == "class":
                self._class_enclosing_states.pop()
            self._detector_states = previous_states
            self._scope_kind = previous_kind

    def _is_local_absolute_import(self, module: str) -> bool:
        return _local_absolute_import_root_exists(self.state, self.rel_path, module)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        if call_name:
            self._record_call(call_name, node.lineno)
            for detector in self._detectors:
                detector.on_call(call_name, node.lineno, self._detector_states[detector.name], self._ctx)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._inspect_secret_assignment(node.targets, node.value, getattr(node, "lineno", None))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._inspect_secret_assignment([node.target], node.value, getattr(node, "lineno", None))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self._inspect_string_literal(node.value, getattr(node, "lineno", None))
        self.generic_visit(node)

    def _record_import(self, module: str, lineno: int, *, classify: bool = True) -> None:
        if not module:
            return
        self._add_file_import(module)
        if not classify:
            return
        framework = frameworks.framework_for_import(module)
        if framework:
            _append_framework_evidence(self.state, framework, _evidence(self.rel_path, lineno, "import", module))
        if module == "nvflare" or module.startswith("nvflare."):
            _append_bounded_list(
                self.state,
                "flare_imports",
                self.state.flare_imports,
                _evidence(self.rel_path, lineno, "import", module),
            )
        if module in {"hydra", "omegaconf"} or module.startswith(("hydra.", "omegaconf.")):
            _append_bounded_list(
                self.state,
                "dynamic_patterns",
                self.state.dynamic_patterns,
                _evidence(self.rel_path, lineno, "dynamic_config", module),
            )
        if module == "torch.distributed" or module.startswith("torch.distributed."):
            _append_bounded_list(
                self.state,
                "distributed_patterns",
                self.state.distributed_patterns,
                _evidence(self.rel_path, lineno, "distributed_import", module),
            )
        if module == "accelerate" or module.startswith("accelerate."):
            _append_bounded_list(
                self.state,
                "distributed_patterns",
                self.state.distributed_patterns,
                _evidence(self.rel_path, lineno, "accelerate_import", module),
            )

    def _record_import_from_modules(self, module: str, level: int, aliases: list[ast.alias]) -> None:
        resolved_module = _resolve_import_from_module(self.rel_path, module, level)
        if resolved_module:
            self._add_file_import(resolved_module)
        for alias in aliases:
            if alias.name == "*":
                continue
            self._add_file_import(f"{resolved_module}.{alias.name}" if resolved_module else alias.name)

    def _add_file_import(self, module: str) -> None:
        imports = self.state.file_imports.setdefault(self.rel_path, set())
        if module in imports:
            _observe_bucket(self.state, "file_imports", collected=False)
            return
        total_imports = self.state.bucket_collected_counts.get("file_imports", 0)
        limit_key = "*" if total_imports >= MAX_EVIDENCE_COLLECT else self.rel_path
        if total_imports >= MAX_EVIDENCE_COLLECT or len(imports) >= MAX_IMPORTS_PER_FILE:
            _observe_bucket(self.state, "file_imports", collected=False)
            if limit_key not in self.state.import_limit_reported:
                self.state.import_limit_reported.add(limit_key)
                _mark_bucket_truncated(self.state, "file_imports")
                _add_limit_finding(
                    self.state,
                    "IMPORT_GRAPH_LIMIT_REACHED",
                    None if limit_key == "*" else self.rel_path,
                    (
                        "Import-graph collection reached the global limit."
                        if limit_key == "*"
                        else "Import-graph collection was truncated for this file."
                    ),
                )
            return
        imports.add(module)
        _observe_bucket(self.state, "file_imports", collected=True)
        self.state.local_files_by_module_cache = None
        self.state.reachable_files_cache = None

    def _record_call(self, call_name: str, lineno: int) -> None:
        # Generic FLARE / distributed / dynamic-dispatch signals only. Ranked
        # framework activity (pytorch_call, lightning_trainer) and conversion
        # signals (flare.patch) are recorded by framework detectors via on_call.
        if call_name.startswith("flare.") or call_name.startswith("nvflare."):
            self._add_flare_call(call_name)
        if call_name in {"FedJob", "FLModel", "SimEnv"}:
            self._add_flare_call(call_name)
        if call_name == "SimEnv" or call_name.endswith(".SimEnv"):
            self.state.sim_env_used = True
        if call_name.endswith(".export"):
            self.state.export_support = True
        if call_name in {"importlib.import_module", "__import__", "getattr"}:
            _append_bounded_list(
                self.state,
                "dynamic_patterns",
                self.state.dynamic_patterns,
                _evidence(self.rel_path, lineno, "dynamic_dispatch", call_name),
            )
        if call_name == "torch.compile":
            _append_bounded_list(
                self.state,
                "dynamic_patterns",
                self.state.dynamic_patterns,
                _evidence(self.rel_path, lineno, "torch_compile", call_name),
            )
        if call_name.endswith(("DataParallel", "FSDP", "Accelerator")):
            _append_bounded_list(
                self.state,
                "distributed_patterns",
                self.state.distributed_patterns,
                _evidence(self.rel_path, lineno, "distributed_call", call_name),
            )

    def _inspect_secret_assignment(self, targets: list[ast.AST], value: ast.AST, lineno: Optional[int]) -> None:
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            return
        for target in targets:
            name = _target_name(target)
            if name and SECRET_NAME_PATTERN.search(name):
                _append_finding(
                    self.state,
                    {
                        "code": "SECRET_LITERAL_REDACTED",
                        "severity": "warning",
                        "file": self.rel_path,
                        "line": lineno,
                        "name": name,
                        "value": "<REDACTED>" if self.state.redact else value.value,
                    },
                )

    def _inspect_string_literal(self, value: str, lineno: Optional[int]) -> None:
        if _looks_like_absolute_path(value):
            _append_bounded_list(
                self.state,
                "absolute_path_findings",
                self.state.absolute_path_findings,
                {
                    "code": "ABSOLUTE_DATA_PATH",
                    "severity": "warning",
                    "file": self.rel_path,
                    "line": lineno,
                    "pattern_type": "absolute_path_literal",
                    "value": _redact_literal(value, self.state.redact),
                },
            )


class _ModuleImportBindingCollector(ast.NodeVisitor):
    """Pre-bind module-scope imports without recording duplicate evidence."""

    def __init__(self, inspector: _PythonInspector):
        self._inspector = inspector

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._inspector._bind_import(alias)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._inspector._bind_import_from(node.module or "", node.level, node.names)

    # Imports inside these scopes are not module globals. The main inspection
    # pass still visits them in source order, preserving existing local-binding
    # detection without leaking them into earlier sibling functions.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass


def _rank_frameworks(state: InspectState) -> list[dict]:
    scores = {framework: _evidence_score(evidence) for framework, evidence in state.framework_evidence.items()}
    total_score = sum(scores.values())
    ranked_with_scores = []
    for framework, evidence in state.framework_evidence.items():
        score = scores[framework]
        confidence = 0.0
        if total_score:
            # Any static import evidence should register clearly, but cap below
            # certainty because static evidence alone does not prove active use.
            # Detector-declared weights ensure an active model/class or trainer
            # call is stronger than the same number of incidental imports.
            confidence = min(0.99, 0.45 + (score / total_score) * 0.5)
        ranked_with_scores.append(
            (
                score,
                {
                    "name": framework,
                    "confidence": round(confidence, 2),
                    "evidence": evidence[:MAX_EVIDENCE_PER_BUCKET],
                    "contradicting_evidence": [],
                },
            )
        )
    ranked_with_scores.sort(key=lambda scored: (-scored[0], scored[1]["name"]))
    return [item for _score, item in ranked_with_scores]


def _detect_primary_framework(state: InspectState, ranked: list[dict]) -> Optional[str]:
    if not ranked:
        return None
    primary = _primary_by_confidence_and_entry_context(state, ranked)
    return frameworks.resolve_primary_framework(primary, state.framework_evidence, _FamilyResolver(state))


def _primary_by_confidence_and_entry_context(state: InspectState, ranked: list[dict]) -> str:
    # Frameworks are ranked by (weighted confidence, name), which is blind
    # to reachability: an incidental torch/sklearn utility in an unreachable
    # helper file can outrank the framework the entry point actually uses. When
    # any framework's evidence is tied to the entry context (the inspected file
    # or a file reachable from an entry point), prefer the highest-confidence
    # such framework over a higher-count-but-unreachable one. Same-family
    # (base/superset) promotion is resolved afterward by
    # resolve_primary_framework. Fall back to the raw ranking only when nothing
    # is reachable (e.g. a model-only directory with no entry point).
    #
    # A numerical utility (numpy) is skipped here: an incidental `import numpy`
    # in the entry must not win over a real convertible framework whose code is
    # loaded dynamically or lives outside the entry file. Utilities are still
    # ranked and reported; they just never become the entry-context primary or
    # fallback primary while a non-utility framework exists.
    #
    # Selection order:
    #   1. Highest-confidence non-utility framework with ACTIVE evidence tied to
    #      the entry context (a real model/usage the entry actually reaches).
    #   2. Else highest-confidence non-utility framework with ANY evidence tied
    #      to the entry context (an import-only entry framework such as sklearn
    #      when no active framework is reachable there).
    #   3. Else highest-confidence non-utility framework; a utility wins only
    #      when it is the sole detected framework.
    # Step 1 before step 2 means an actively-used torch model reachable from the
    # entry outranks import-only entry evidence, without demoting a genuinely
    # entry-owned import-only framework when nothing active is reachable.
    #
    # DESIGN DECISION (do not revert to a pure entry-context or pure score rule):
    # this two-step order is the agreed reconciliation of two review positions.
    #   - When the entry reaches an ACTIVE convertible model (e.g. a torch
    #     nn.Module) alongside import-only sklearn, route to the active framework
    #     (recommend a conversion) rather than abstaining on the sklearn import.
    #   - When the torch model is NOT reachable from the entry (a stray helper),
    #     the entry-owned import-only framework still wins (sklearn -> abstain),
    #     preserving the earlier sklearn-entry fix. Both hold simultaneously.
    for item in ranked:
        name = item["name"]
        if name in frameworks.UTILITY_FRAMEWORKS:
            continue
        active = [e for e in state.framework_evidence.get(name, []) if frameworks.is_active_evidence(name, e)]
        if active and _framework_evidence_tied_to_entry_context(state, active):
            return name
    for item in ranked:
        name = item["name"]
        if name in frameworks.UTILITY_FRAMEWORKS:
            continue
        if _framework_evidence_tied_to_entry_context(state, state.framework_evidence.get(name, [])):
            return name
    for item in ranked:
        if item["name"] not in frameworks.UTILITY_FRAMEWORKS:
            return item["name"]
    return ranked[0]["name"]


class _FamilyResolver:
    """Adapter giving family-owning detectors the engine's generic helpers.

    A detector that resolves a base/superset family conflict (for example
    Lightning over PyTorch) reads collected evidence, weighted scores, and
    entry-context/import-graph checks through this adapter, so the engine holds
    no framework-specific promotion logic.
    """

    def __init__(self, state: InspectState):
        self._state = state

    def evidence(self, framework: str) -> list[dict]:
        return self._state.framework_evidence.get(framework, [])

    def active_evidence(self, framework: str) -> list[dict]:
        return [item for item in self.evidence(framework) if frameworks.is_active_evidence(framework, item)]

    def score(self, evidence: list[dict]) -> int:
        return _evidence_score(evidence)

    def tied_to_entry_context(self, evidence: list[dict]) -> bool:
        return _framework_evidence_tied_to_entry_context(self._state, evidence)

    def has_inspected_file_or_entry_point(self) -> bool:
        return self._state.root.is_file() or bool(self._state.entry_points)

    def evidence_outside_files(self, evidence: list[dict], reference_evidence: list[dict]) -> list[dict]:
        reference_files = {item["file"] for item in reference_evidence}
        return [item for item in evidence if item["file"] not in reference_files]

    def evidence_outside_class_bodies(self, evidence: list[dict], class_evidence: list[dict]) -> list[dict]:
        # Exclude items whose (file, line) falls within the body of a class named
        # in ``class_evidence`` (matched by that class's definition line). Lets a
        # family member (Lightning) claim base-framework (torch) calls inside its
        # model class bodies without absorbing torch used in a sibling class or at
        # module level in the same file.
        ranges_by_file: dict[str, list[tuple[int, int]]] = {}
        for item in class_evidence:
            file_path = item["file"]
            def_line = item.get("line")
            for start, end in self._state.class_body_ranges.get(file_path, []):
                if start == def_line:
                    ranges_by_file.setdefault(file_path, []).append((start, end))
        return [
            item for item in evidence if not _line_within_ranges(item.get("line"), ranges_by_file.get(item["file"]))
        ]


def _line_within_ranges(line: Optional[int], ranges: Optional[list[tuple[int, int]]]) -> bool:
    if line is None or not ranges:
        return False
    return any(start <= line <= end for start, end in ranges)


def _framework_evidence_tied_to_entry_context(state: InspectState, evidence: list[dict]) -> bool:
    if _framework_evidence_tied_to_inspected_file_or_entry_point(state, evidence):
        return True
    if state.root.is_file():
        return False
    return any(_entry_point_imports_file(state, item["file"]) for item in evidence)


def _framework_evidence_tied_to_inspected_file_or_entry_point(state: InspectState, evidence: list[dict]) -> bool:
    if state.root.is_file():
        inspected_file = _display_path(state.root, state.root, state.redact)
        return any(item["file"] == inspected_file for item in evidence)
    entry_point_paths = {entry["path"] for entry in _preferred_entry_points(state)}
    return any(item["file"] in entry_point_paths for item in evidence)


def _entry_point_imports_file(state: InspectState, evidence_file: str) -> bool:
    if not _module_names_for_file(evidence_file):
        return False
    return evidence_file in _reachable_files_from_entry_points(state)


def _preferred_entry_points(state: InspectState) -> list[dict]:
    if not state.entry_points:
        return []
    highest_priority = max(state.entry_point_priorities.get(item["path"], 0) for item in state.entry_points)
    return [
        item for item in state.entry_points if state.entry_point_priorities.get(item["path"], 0) == highest_priority
    ]


def _reachable_files_from_entry_points(state: InspectState) -> set[str]:
    if state.reachable_files_cache is not None:
        return state.reachable_files_cache

    # Match on resolved files, not only module names: root and src-layout copies
    # may share a module name, and _local_files_for_import applies the importing
    # file's packaging-root context before returning concrete file paths.
    local_files_by_module = _local_files_by_module(state)
    pending = deque((entry["path"], 0) for entry in _preferred_entry_points(state))
    expanded_files = set()
    reachable_files = set()
    edges_visited = 0
    truncated = False

    while pending and not truncated:
        importing_file, depth = pending.popleft()
        if importing_file in expanded_files:
            continue
        expanded_files.add(importing_file)
        imports = state.file_imports.get(importing_file, set())
        if depth >= MAX_REACHABILITY_DEPTH:
            if any(
                _local_files_for_import(import_name, importing_file, local_files_by_module) for import_name in imports
            ):
                truncated = True
            continue
        for import_name in sorted(imports):
            for imported_file in sorted(_local_files_for_import(import_name, importing_file, local_files_by_module)):
                edges_visited += 1
                if edges_visited > MAX_REACHABILITY_EDGES:
                    truncated = True
                    break
                if imported_file not in reachable_files:
                    reachable_files.add(imported_file)
                    pending.append((imported_file, depth + 1))
            if truncated:
                break

    if truncated and not state.reachability_limit_reported:
        state.reachability_limit_reported = True
        _add_limit_finding(
            state,
            "IMPORT_REACHABILITY_LIMIT_REACHED",
            None,
            "Entry-point import reachability was truncated by a complexity limit.",
        )
    state.reachable_files_cache = reachable_files
    return reachable_files


def _local_files_by_module(state: InspectState) -> dict[str, set[str]]:
    # Register every candidate module name for a file, including the src-layout
    # root-stripped name (mypkg.loop from src/mypkg/loop.py). When a name is
    # claimed by both a root-level file and a src/ copy, the collision is
    # resolved per-import by _prefer_shared_packaging_root using the importing
    # file's own packaging root, so neither a stale src/ copy nor a stale
    # root-level copy can steal the actively-imported module in either direction.
    #
    # Memoized on the state: entry-context reachability calls this once per
    # evidence item, and file_imports is fully populated during the scan before
    # detection runs, so the map is built once instead of O(evidence) times over
    # the now-uncapped evidence lists.
    if state.local_files_by_module_cache is not None:
        return state.local_files_by_module_cache
    files_by_module: dict[str, set[str]] = {}
    for file_path in state.file_imports:
        for module_name in _module_names_for_file(file_path):
            files_by_module.setdefault(module_name, set()).add(file_path)
    state.local_files_by_module_cache = files_by_module
    return files_by_module


def _packaging_root_of(file_path: str) -> str:
    parts = Path(file_path).parts
    if parts and parts[0] in _PACKAGE_ROOT_DIR_NAMES:
        return parts[0]
    return ""


def _prefer_shared_packaging_root(files: set[str], importing_file: str) -> set[str]:
    # When an import resolves to copies in different packaging roots (a root-level
    # file and a src/ copy of the same module path), prefer the copy sharing the
    # importing file's packaging root. Fall back to all matches when none share
    # it (e.g. a root-level entry importing a src-layout package).
    if len(files) <= 1:
        return files
    importing_root = _packaging_root_of(importing_file)
    same_root = {file_path for file_path in files if _packaging_root_of(file_path) == importing_root}
    return same_root or files


def _local_files_for_import(
    import_name: str, importing_file: str, local_files_by_module: dict[str, set[str]]
) -> set[str]:
    files = set()
    exact_candidates = _exact_module_candidates_for_import(import_name, importing_file, local_files_by_module)
    resolved_modules = set()
    for module_name in exact_candidates:
        module_files = local_files_by_module.get(module_name, set())
        if module_files:
            resolved_modules.add(module_name)
            files.update(module_files)
    # Only follow a package's ``__init__.py`` once the full imported module path resolves to a
    # local file. Otherwise an external absolute import (e.g. ``import lightning.pytorch``) whose
    # leading segment happens to match an unrelated local package (a top-level ``lightning/``)
    # would resolve that package's ``__init__.py`` and incorrectly promote it.
    if not files:
        return files
    for module_name in _package_module_prefix_candidates_for_resolved(resolved_modules, exact_candidates):
        files.update(
            file_path
            for file_path in local_files_by_module.get(module_name, set())
            if _is_package_module_file(file_path)
        )
    return _prefer_shared_packaging_root(files, importing_file)


def _exact_module_candidates_for_import(
    import_name: str, importing_file: str, local_files_by_module: dict[str, set[str]]
) -> set[str]:
    candidates = {import_name} if import_name else set()
    context_prefix = _import_context_prefix(importing_file)
    if context_prefix:
        for module_name in list(candidates):
            prefixed = f"{context_prefix}.{module_name}"
            # Single-segment imports always take the importing file's context prefix so a sibling
            # module (``from block import ...`` next to the script) resolves. Dotted imports only
            # take it when the full context-prefixed module resolves to a local file. This keeps
            # nested local dotted imports reachable (``from layers.block import ...`` in
            # ``models/train.py`` resolving ``models/layers/block.py``) without promoting an
            # external absolute import (``import lightning.pytorch``) onto an unrelated local
            # package.
            if _is_single_segment_import(module_name) or prefixed in local_files_by_module:
                candidates.add(prefixed)
    return candidates


def _package_module_prefix_candidates_for_resolved(resolved_modules: set[str], exact_candidates: set[str]) -> set[str]:
    # Package-prefix candidates let us follow the __init__.py of a package whose full path
    # resolves locally (e.g. ``import pkg.sub`` reaching ``pkg/__init__.py``). Derive the prefixes
    # from the exact module candidates that actually resolved to local files, not from the raw
    # import name. A context-resolved import (``from layers.block import ...`` in ``models/train.py``
    # resolving ``models.layers.block``) must follow ``models.layers`` prefixes, not the raw
    # ``layers`` segment that could match an unrelated top-level ``layers/`` package. The exact
    # candidates are excluded so we only follow parent packages, not the fully resolved module file.
    candidates: set[str] = set()
    for module_name in resolved_modules:
        candidates.update(_module_name_prefixes(module_name))
    candidates.difference_update(exact_candidates)
    return candidates


def _is_single_segment_import(import_name: str) -> bool:
    return bool(import_name) and "." not in import_name


def _module_name_prefixes(module_name: str) -> set[str]:
    parts = [part for part in module_name.split(".") if part]
    return {".".join(parts[:index]) for index in range(1, len(parts) + 1)}


def _is_package_module_file(file_path: str) -> bool:
    return Path(file_path).name == "__init__.py"


def _file_module_parts(file_path: str) -> Optional[tuple[str, ...]]:
    if not file_path.endswith(".py"):
        return None
    path = Path(file_path)
    parts = path.parent.parts if path.name == "__init__.py" else path.with_suffix("").parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    return parts


def _module_names_for_file(file_path: str) -> set[str]:
    parts = _file_module_parts(file_path)
    if not parts:
        return set()
    names = {".".join(parts)}
    # src-layout: a file under a packaging root (src/) is imported by its
    # package path without the root, so an entry point's `import mypkg.loop`
    # reaches src/mypkg/loop.py. Offer the root-stripped module name too.
    if len(parts) > 1 and parts[0] in _PACKAGE_ROOT_DIR_NAMES:
        names.add(".".join(parts[1:]))
    return names


def _import_context_prefix(file_path: str) -> str:
    if not file_path.endswith(".py"):
        return ""
    path = Path(file_path)
    parts = path.parent.parts
    if any(part in {"", ".", ".."} for part in parts):
        return ""
    return ".".join(parts)


def _resolve_import_from_module(importing_file: str, module: str, level: int) -> str:
    if level <= 0:
        return module
    # The same keep-formula is correct for both plain modules and __init__.py.
    package_parts = Path(importing_file).parent.parts
    keep = max(0, len(package_parts) - level + 1)
    parts = list(package_parts[:keep])
    if module:
        parts.extend(module.split("."))
    return ".".join(part for part in parts if part)


def _local_absolute_import_root_exists(state: InspectState, importing_file: str, module: str) -> bool:
    """Return whether an absolute import root is shadowed by project content."""

    if not module:
        return False
    import_root = module.split(".", 1)[0]
    project_root = state.root if state.root.is_dir() else state.root.parent
    importing_parent = project_root / Path(importing_file).parent
    bases = {project_root, importing_parent, project_root / "src"}
    for base in bases:
        candidates = (base / f"{import_root}.py", base / import_root)
        for candidate in candidates:
            try:
                candidate_stat = candidate.lstat()
            except OSError:
                continue
            if (
                stat.S_ISREG(candidate_stat.st_mode)
                or stat.S_ISDIR(candidate_stat.st_mode)
                or stat.S_ISLNK(candidate_stat.st_mode)
            ):
                return True
    return False


def _evidence_score(evidence: list[dict]) -> int:
    weights = frameworks.evidence_weights()
    return sum(weights.get(item["kind"], 1) for item in evidence)


def _order_frameworks_for_display(ranked: list[dict], detected_framework: Optional[str]) -> list[dict]:
    # Surface the detected primary framework first so callers reading
    # frameworks[0] always stay aligned with the routing decision, including the
    # family case (a PyTorch base detected while a higher-confidence Lightning
    # member is present). sorted() is stable, so every other framework keeps its
    # confidence-ranked order. When nothing was detected, the order is unchanged.
    if not detected_framework:
        return ranked
    return sorted(ranked, key=lambda item: item["name"] != detected_framework)


def _conversion_state(state: InspectState, detected_framework: Optional[str]) -> str:
    if state.exported_job_markers:
        return "exported_job"
    if state.job_py or state.sim_env_used:
        return "flare_job"
    if _has_conversion_integration(state):
        return "client_api_converted"
    if {"flare.receive", "flare.send"} <= state.flare_calls or "FLModel" in state.flare_calls:
        return "client_api_converted"
    if state.flare_imports or state.flare_calls:
        return "partial_client_api"
    if detected_framework:
        return "not_converted"
    return "unknown"


def _has_conversion_integration(state: InspectState) -> bool:
    # A framework conversion-integration signal (e.g. an nvflare.client.lightning
    # ``patch(trainer)`` call) is a definitive conversion signal even without an
    # explicit ``flare.send``, because the framework's callback performs the
    # result exchange. Detectors record these signals via on_call; do not
    # require static constructor evidence here (wrappers/factories can hide it).
    return bool(state.integration_signals and state.flare_imports)


def _target_type(path: Path, state: InspectState, detected_framework: Optional[str], conversion_state: str) -> str:
    if path.is_symlink():
        return "unknown_target"
    if path.is_file():
        return "single_training_script" if path.suffix == ".py" else "unknown_target"
    if conversion_state == "exported_job":
        return "exported_submit_ready_flare_job"
    if conversion_state == "flare_job":
        return "flare_job_source"
    if detected_framework and conversion_state in {"partial_client_api", "client_api_converted"}:
        return "mixed_workspace"
    if frameworks.family_base_has_member(detected_framework, state.framework_evidence):
        # A family base (e.g. PyTorch) detected alongside its superset member
        # (PyTorch Lightning). Distinct from the FLARE conversion
        # "mixed_workspace": two frameworks of the same family are present, not
        # a partial FLARE conversion.
        return "mixed_framework_workspace"
    if detected_framework:
        return "training_repository"
    return "unknown_target"


def _add_entry_point(path: Path, rel_path: str, tree: ast.Module, state: InspectState) -> None:
    functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    main_guard = any(_is_main_guard(node) for node in tree.body if isinstance(node, ast.If))
    conventional_name = path.name in {"client.py", "server.py", "train.py", "trainer.py", "main.py", "job.py"}
    has_main_function = "main" in functions
    if main_guard or conventional_name or has_main_function:
        if _append_bounded_list(
            state,
            "entry_points",
            state.entry_points,
            {
                "path": rel_path,
                "kind": "python_script",
                "functions": functions[:MAX_EVIDENCE_PER_BUCKET],
                "main_guard": main_guard,
            },
        ):
            if main_guard or conventional_name:
                priority = 3
            else:
                priority = 2
            state.entry_point_priorities[rel_path] = priority
            state.reachable_files_cache = None


def _is_main_guard(node: ast.If) -> bool:
    if not isinstance(node.test, ast.Compare) or len(node.test.ops) != 1 or not isinstance(node.test.ops[0], ast.Eq):
        return False
    if len(node.test.comparators) != 1:
        return False
    left = node.test.left
    right = node.test.comparators[0]
    return (_is_name_symbol(left) and _is_main_literal(right)) or (_is_main_literal(left) and _is_name_symbol(right))


def _is_name_symbol(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "__name__"


def _is_main_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "__main__"


def _skill_selection(detected_framework: Optional[str], conversion_state: str, state: InspectState) -> dict:
    recommended = []
    if state.incomplete_reasons:
        recommended.append("nvflare-orient")
    elif conversion_state == "exported_job":
        # Lifecycle skills are out of scope and not planned; exported jobs are
        # handled with product APIs directly, so no skill is recommended.
        pass
    elif detected_framework and conversion_state == "not_converted":
        skill = frameworks.recommended_skill_for(detected_framework)
        if skill:
            recommended.append(skill)
    if (state.findings or state.files_skipped) and "nvflare-orient" not in recommended:
        recommended.append("nvflare-orient")

    return {
        "detected_framework": detected_framework,
        "conversion_state": conversion_state,
        "exported_job": conversion_state == "exported_job",
        "recommended_skills": recommended,
        "safety_findings": [finding["code"] for finding in _findings_for_output(state)],
    }


def _recommended_next_commands(
    detected_framework: Optional[str], conversion_state: str, state: InspectState
) -> list[str]:
    commands = ["nvflare agent doctor --format json"]
    if state.incomplete_reasons:
        return commands
    if conversion_state == "exported_job":
        commands.append("nvflare job submit <job-folder> --format json")
    elif state.job_py and state.export_support:
        commands.append("python job.py --export --export-dir <job-dir>")
    elif detected_framework and conversion_state == "not_converted":
        skill = frameworks.recommended_skill_for(detected_framework)
        if skill:
            commands.append(f"Use the {skill} skill before editing.")
    return commands


def _record_symlink_skip(path: Path, state: InspectState) -> None:
    try:
        target = os.readlink(path)
    except OSError:
        target = ""
    _add_skip(
        state,
        {
            "code": "SYMLINK_SKIPPED",
            "path": _display_path(path, state.root, state.redact),
            "target": _redact_literal(target, state.redact),
            "message": "symlink was not followed during static inspection",
        },
    )


def _findings_for_output(state: InspectState) -> list[dict]:
    indexed = list(enumerate(state.findings))
    indexed.sort(
        key=lambda item: (
            0 if "LIMIT" in str(item[1].get("code", "")) else 1,
            item[0],
        )
    )
    return [finding for _index, finding in indexed[:MAX_EVIDENCE_PER_BUCKET]]


def _output_truncated_buckets(state: InspectState) -> list[str]:
    bucket_sizes = {
        "absolute_path_findings": len(state.absolute_path_findings),
        "distributed_patterns": len(state.distributed_patterns),
        "dynamic_patterns": len(state.dynamic_patterns),
        "entry_points": len(state.entry_points),
        "exported_job_markers": len(state.exported_job_markers),
        "findings": len(state.findings),
        "flare_calls": len(state.flare_calls),
        "flare_imports": len(state.flare_imports),
        "files_skipped": len(state.files_skipped),
    }
    for framework, evidence in state.framework_evidence.items():
        bucket_sizes[f"framework_evidence:{framework}"] = len(evidence)
    truncated = set(state.truncated_buckets)
    truncated.update(name for name, size in bucket_sizes.items() if size > MAX_EVIDENCE_PER_BUCKET)
    if state.files_skipped_count > len(state.files_skipped):
        truncated.add("files_skipped")
    return sorted(truncated)


def _mark_incomplete(state: InspectState, reason: str) -> None:
    state.incomplete_reasons.add(reason)


def _observe_bucket(state: InspectState, bucket_name: str, *, collected: bool) -> None:
    state.bucket_observed_counts[bucket_name] = state.bucket_observed_counts.get(bucket_name, 0) + 1
    if collected:
        state.bucket_collected_counts[bucket_name] = state.bucket_collected_counts.get(bucket_name, 0) + 1


def _mark_bucket_truncated(state: InspectState, bucket_name: str) -> bool:
    first = bucket_name not in state.truncated_buckets
    state.truncated_buckets.add(bucket_name)
    _mark_incomplete(state, f"EVIDENCE_BUCKET_LIMIT_REACHED:{bucket_name}")
    return first


def _append_bounded_list(
    state: InspectState,
    bucket_name: str,
    bucket: list,
    value,
    *,
    limit: Optional[int] = None,
) -> bool:
    limit = MAX_EVIDENCE_COLLECT if limit is None else limit
    if len(bucket) < limit:
        bucket.append(value)
        _observe_bucket(state, bucket_name, collected=True)
        return True
    _observe_bucket(state, bucket_name, collected=False)
    if _mark_bucket_truncated(state, bucket_name) and bucket_name != "findings":
        _add_limit_finding(
            state,
            "EVIDENCE_BUCKET_LIMIT_REACHED",
            None,
            f"Evidence collection was truncated for {bucket_name}.",
        )
    return False


def _add_bounded_set(
    state: InspectState,
    bucket_name: str,
    bucket: set,
    value,
    *,
    limit: Optional[int] = None,
) -> bool:
    limit = MAX_EVIDENCE_COLLECT if limit is None else limit
    if value in bucket:
        _observe_bucket(state, bucket_name, collected=False)
        return True
    if len(bucket) < limit:
        bucket.add(value)
        _observe_bucket(state, bucket_name, collected=True)
        return True
    _observe_bucket(state, bucket_name, collected=False)
    if _mark_bucket_truncated(state, bucket_name):
        _add_limit_finding(
            state,
            "EVIDENCE_BUCKET_LIMIT_REACHED",
            None,
            f"Evidence collection was truncated for {bucket_name}.",
        )
    return False


def _append_finding(state: InspectState, finding: dict) -> None:
    _append_bounded_list(state, "findings", state.findings, finding)


def _append_class_body_range(state: InspectState, file_path: str, value: tuple[int, int]) -> None:
    collected = state.bucket_collected_counts.get("class_body_ranges", 0)
    if collected >= MAX_EVIDENCE_COLLECT:
        _observe_bucket(state, "class_body_ranges", collected=False)
        if _mark_bucket_truncated(state, "class_body_ranges"):
            _add_limit_finding(
                state,
                "EVIDENCE_BUCKET_LIMIT_REACHED",
                file_path,
                "Class-body range collection was truncated.",
            )
        return
    state.class_body_ranges.setdefault(file_path, []).append(value)
    _observe_bucket(state, "class_body_ranges", collected=True)


def _add_skip(state: InspectState, entry: dict) -> None:
    state.files_skipped_count += 1
    code = entry.get("code")
    if code in _INCOMPLETE_SKIP_CODES:
        _mark_incomplete(state, code)
    if len(state.files_skipped) < MAX_EVIDENCE_PER_BUCKET:
        state.files_skipped.append(entry)
        _observe_bucket(state, "files_skipped", collected=True)
    else:
        _observe_bucket(state, "files_skipped", collected=False)


def _skip_entry(path: Path, state: InspectState, code: str, message: str, error: Exception = None) -> dict:
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


def _redact_literal(value: str, redact: bool) -> str:
    if not redact:
        return value
    if _looks_like_absolute_path(value):
        return "<REDACTED_PATH>"
    if SECRET_NAME_PATTERN.search(value):
        return "<REDACTED>"
    return value


def _looks_like_absolute_path(value: str) -> bool:
    return value.startswith(("/", "~")) or bool(re.match(r"^[A-Za-z]:[\\/]", value))


def _evidence(file_path: str, line: Optional[int], kind: str, value: str) -> dict:
    return {"file": file_path, "line": line, "kind": kind, "value": value}


def _append_framework_evidence(state: InspectState, framework: str, value: dict) -> None:
    bucket = state.framework_evidence.setdefault(framework, [])
    bucket_name = f"framework_evidence:{framework}"
    weights = frameworks.evidence_weights()
    incoming_weight = weights.get(value.get("kind"), 1)
    indices_by_weight = state.evidence_indices_by_weight.setdefault(framework, {})
    if len(bucket) < MAX_EVIDENCE_COLLECT:
        bucket.append(value)
        _observe_bucket(state, bucket_name, collected=True)
        indices_by_weight.setdefault(incoming_weight, deque()).append(len(bucket) - 1)
        return
    _observe_bucket(state, bucket_name, collected=False)
    lowest_weight = min(weight for weight, indices in indices_by_weight.items() if indices)
    if incoming_weight > lowest_weight:
        # Keep the strongest bounded evidence. In a pathological file with ten
        # thousand imports followed by a real model class, the cap must not erase
        # the active signal that determines routing.
        lowest_index = indices_by_weight[lowest_weight].popleft()
        bucket[lowest_index] = value
        indices_by_weight.setdefault(incoming_weight, deque()).append(lowest_index)
    if framework not in state.evidence_limit_reported:
        state.evidence_limit_reported.add(framework)
        _mark_bucket_truncated(state, bucket_name)
        _add_limit_finding(
            state,
            "FRAMEWORK_EVIDENCE_LIMIT_REACHED",
            value.get("file"),
            f"Framework evidence collection was truncated for {framework}.",
        )


def _add_limit_finding(state: InspectState, code: str, file_path: Optional[str], message: str) -> None:
    _mark_incomplete(state, code)
    finding = {
        "code": code,
        "severity": "warning",
        "message": message,
    }
    if file_path is not None:
        finding["file"] = file_path
    _append_finding(state, finding)


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _symbol_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Subscript):
        return _call_name(node.value)
    return _call_name(node)


def _target_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
