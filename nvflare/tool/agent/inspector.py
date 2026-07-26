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
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nvflare
from nvflare.tool.agent import dataset_inspect, frameworks
from nvflare.tool.agent.frameworks.base import DetectContext

DEFAULT_MAX_FILES = 250
DEFAULT_MAX_FILE_BYTES = 512 * 1024
MAX_EVIDENCE_PER_BUCKET = 12
# After max_files is reached, the inspector accounts for a bounded number of
# unvisited files/directories so callers can see that classification is
# incomplete without turning the cap into an unbounded full-tree walk.
MAX_FILE_LIMIT_ACCOUNTED_SKIPS = 10000
# Backstop for evidence collected per framework bucket. Far above the display
# cap so ranking/detection uses true counts; only a memory guard for pathological
# inputs, not a routing-relevant threshold.
MAX_EVIDENCE_COLLECT = 10000
# Packaging root dirs whose leading segment is not part of the import path
# (PyPA src-layout), so `src/pkg/mod.py` is importable as `pkg.mod`.
_PACKAGE_ROOT_DIR_NAMES = {"src"}
# These directory names commonly contain secondary or historical projects when
# inspection starts at a repository/monorepo root. Their entry points remain
# valid when inspected directly, but must not win an equal-depth authority tie
# against an independent framework project merely because they are entry points.
_SECONDARY_PROJECT_DIR_NAMES = {
    "archive",
    "archives",
    "archived",
    "fixture",
    "fixtures",
    "test",
    "tests",
    "vendor",
    "vendors",
    "vendored",
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
_INCOMPLETE_SCAN_SKIP_CODES = {
    "DIRECTORY_NOT_SCANNED_FILE_LIMIT",
    "FILE_LIMIT_REACHED",
    "FILE_TOO_LARGE",
    "NON_UTF8_FILE",
    "UNREADABLE_DIRECTORY",
    "UNREADABLE_FILE",
}

# Installed-skill discovery: read-only scan for <dir>/*/SKILL.md under known
# agent skill directories. Bounded so a pathological tree can't blow up the scan.
SKILL_FILE_NAME = "SKILL.md"
MAX_INSTALLED_SKILLS = 200
MAX_SKILL_FRONTMATTER_BYTES = 64 * 1024
# Project-scope skill dirs are resolved relative to the inspected path's project
# root (walked up to cwd); global-scope dirs live under the user home.
_PROJECT_SKILL_DIRS = (".claude/skills", ".agents/skills")
_GLOBAL_SKILL_DIRS = ("~/.claude/skills", "~/.codex/skills")

# Framework detection (import roots, symbols, evidence weights, recommended
# skills, and family/promotion rules) lives in nvflare.tool.agent.frameworks.
# This engine stays framework-agnostic; add a framework there, not here.


@dataclass
class _LocalImportGraph:
    """Directed local-import graph with one cached traversal per source file."""

    edges: dict[str, frozenset[str]]
    _reachable_cache: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    @property
    def files(self) -> frozenset[str]:
        return frozenset(self.edges)

    def reachable_from(self, source_file: str) -> frozenset[str]:
        cached = self._reachable_cache.get(source_file)
        if cached is not None:
            return cached

        pending = list(self.edges.get(source_file, ()))
        reachable = set()
        while pending:
            imported_file = pending.pop()
            if imported_file == source_file or imported_file in reachable:
                continue
            reachable.add(imported_file)
            pending.extend(self.edges.get(imported_file, ()))

        result = frozenset(reachable)
        self._reachable_cache[source_file] = result
        return result

    def component_from(self, anchor_file: str) -> frozenset[str]:
        """Return the anchor and every local dependency reachable from it."""
        return frozenset({anchor_file}) | self.reachable_from(anchor_file)


@dataclass(frozen=True)
class _ProjectComponent:
    """One project anchor and its directed local-dependency component."""

    anchor_file: str
    files: frozenset[str]

    @property
    def depth(self) -> int:
        return len(Path(self.anchor_file).parts)

    @property
    def directory(self) -> str:
        parent = Path(self.anchor_file).parent.as_posix()
        return "." if parent == "." else parent


def _component_groups(components: tuple[_ProjectComponent, ...]) -> tuple[tuple[_ProjectComponent, ...], ...]:
    """Group project anchors without merging unrelated importers of one dependency."""
    remaining = set(range(len(components)))
    groups = []
    while remaining:
        pending = [min(remaining)]
        remaining.remove(pending[0])
        group_indices = []
        while pending:
            current_index = pending.pop()
            group_indices.append(current_index)
            current = components[current_index]
            connected = sorted(
                candidate_index
                for candidate_index in remaining
                if _components_share_project(current, components[candidate_index])
            )
            remaining.difference_update(connected)
            pending.extend(connected)
        groups.append(tuple(components[index] for index in sorted(group_indices)))
    return tuple(groups)


def _components_share_project(left: _ProjectComponent, right: _ProjectComponent) -> bool:
    # Anchors in the same directory describe one project. Across directories,
    # require directed reachability between the anchors; merely sharing a common
    # helper dependency must not merge otherwise independent projects.
    return left.directory == right.directory or left.anchor_file in right.files or right.anchor_file in left.files


def _is_secondary_project_entry_point(file_path: str) -> bool:
    return any(part.lower() in _SECONDARY_PROJECT_DIR_NAMES for part in Path(file_path).parent.parts)


@dataclass(frozen=True)
class _ProjectScope:
    """Components authorized to describe the inspected target as a whole."""

    active_components: tuple[_ProjectComponent, ...]
    ambiguous: bool = False

    def component_for(self, file_path: str) -> Optional[_ProjectComponent]:
        if self.ambiguous:
            return None
        return next((component for component in self.active_components if file_path in component.files), None)

    def authorizes(self, file_path: str) -> bool:
        return self.component_for(file_path) is not None


def _authorizes_flare_evidence(project_scope: _ProjectScope, file_path: str) -> bool:
    # Secondary-tree source remains visible in inspection evidence, but it
    # cannot establish repository-level FLARE conversion authority. When the
    # user inspects that subtree directly its files are root-relative (for
    # example ``job.py``), so the same source is authoritative in that scope.
    return project_scope.authorizes(file_path) and not _is_secondary_project_entry_point(file_path)


@dataclass(frozen=True)
class _SourceJobCandidate:
    """A source-job signal and the provenance contained in its own component."""

    source_file: str
    flare_import_files: frozenset[str]
    project_component: Optional[_ProjectComponent]
    export_supported: bool

    @property
    def authoritative(self) -> bool:
        return self.project_component is not None and bool(self.flare_import_files)


@dataclass
class InspectState:
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
    # Built after the scan populates file_imports, then reused by framework
    # ranking, project-scope classification, and source-job provenance checks.
    local_import_graph_cache: Optional[_LocalImportGraph] = field(default=None, repr=False, compare=False)


def inspect_path(
    path: Path | str,
    *,
    redact: bool = True,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict:
    """Inspect a path without importing or executing user code."""
    target = _normalized_inspect_target(path)
    state = InspectState(root=target, redact=redact)

    if not target.exists() and not target.is_symlink():
        raise FileNotFoundError(f"inspect path does not exist: {path}")

    if target.is_symlink():
        _record_symlink_skip(target, state)
    elif target.is_file():
        _inspect_file(target, state, max_file_bytes)
    else:
        _inspect_dir(target, state, max_files=max_files, max_file_bytes=max_file_bytes)

    exported_job_info = _exported_job_info(state)
    project_scope = _build_project_scope(state)
    ranked_frameworks = _rank_frameworks(state)
    detected_framework = _detect_primary_framework(state, ranked_frameworks)
    ranked_frameworks = _order_frameworks_for_display(ranked_frameworks, detected_framework)
    source_job = _authoritative_source_job_candidate(state, project_scope)
    conversion_state = _conversion_state(state, detected_framework, exported_job_info, project_scope, source_job)
    target_type = _target_type(target, state, detected_framework, conversion_state)

    # Data-target classification runs when code classification found nothing,
    # OR when the only detected framework is a utility bucket (numpy): a
    # small helper script inside a data root is optional stats intent
    # evidence, not a training repo. Real code targets keep priority: FLARE
    # job states, exported jobs, and genuine training frameworks (with or
    # without a converter skill) are never overridden by a dataset.
    dataset = None
    utility_only_code = target_type == "training_repository" and detected_framework in frameworks.UTILITY_FRAMEWORKS
    if target.is_dir() and (target_type == "unknown_target" or utility_only_code):
        dataset = dataset_inspect.inspect_dataset(target, max_files=max_files, max_file_bytes=max_file_bytes)
        if dataset and dataset["modality"] in ("tabular", "image"):
            target_type = f"{dataset['modality']}_dataset"

    return {
        "schema_version": "1",
        "nvflare_version": nvflare.__version__,
        "path": _inspected_target_path(target),
        "target_type": target_type,
        "static_only": True,
        "redaction": "on" if redact else "off",
        "limits": {
            "max_files": max_files,
            "max_file_bytes": max_file_bytes,
            "max_evidence_per_bucket": MAX_EVIDENCE_PER_BUCKET,
        },
        "classification_incomplete": state.classification_incomplete,
        "scan": {
            "entries_visited": state.entries_visited,
            "files_considered": state.files_considered,
            "files_scanned": state.files_scanned,
            "bytes_scanned": state.bytes_scanned,
            "files_skipped_count": state.files_skipped_count,
            "files_skipped_count_approximate": state.file_limit_skip_accounting_truncated,
            "files_skipped_truncated": state.file_limit_reached,
            "files_skipped_evidence_truncated": state.files_skipped_count > len(state.files_skipped),
            "files_skipped": state.files_skipped,
        },
        "frameworks": ranked_frameworks,
        "entry_points": state.entry_points[:MAX_EVIDENCE_PER_BUCKET],
        "flare_integration": {
            "present": bool(state.flare_imports or state.flare_calls),
            "imports": state.flare_imports[:MAX_EVIDENCE_PER_BUCKET],
            "calls": sorted(state.flare_calls),
        },
        "conversion_state": conversion_state,
        "job": {
            "job_py": state.job_py,
            "sim_env_used": state.sim_env_used,
            "export_support": state.export_support,
            "exported_job_markers": state.exported_job_markers[:MAX_EVIDENCE_PER_BUCKET],
            "exported_job_candidates": exported_job_info["submit_ready_candidates"][:MAX_EVIDENCE_PER_BUCKET],
            "nested_candidates": exported_job_info["nested_candidates"][:MAX_EVIDENCE_PER_BUCKET],
        },
        "patterns": {
            "distributed": state.distributed_patterns[:MAX_EVIDENCE_PER_BUCKET],
            "dynamic": state.dynamic_patterns[:MAX_EVIDENCE_PER_BUCKET],
            "absolute_data_paths": state.absolute_path_findings[:MAX_EVIDENCE_PER_BUCKET],
        },
        "findings": state.findings[:MAX_EVIDENCE_PER_BUCKET],
        "dataset": dataset,
        "skill_selection": _skill_selection(detected_framework, conversion_state, state, dataset),
        "recommended_next_commands": _recommended_next_commands(detected_framework, conversion_state, source_job),
        "installed_skills": _installed_skills(target),
    }


def _inspect_dir(root: Path, state: InspectState, *, max_files: int, max_file_bytes: int) -> None:
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
    state: InspectState, root: Path, pending_stack: list[Path], remaining_children: list[Path]
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


def _record_unvisited_files_under_file_limit(directory: Path, state: InspectState, root: Path) -> None:
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


def _account_file_limit_skip(state: InspectState) -> bool:
    if state.file_limit_accounted_skips >= MAX_FILE_LIMIT_ACCOUNTED_SKIPS:
        state.file_limit_skip_accounting_truncated = True
        return False
    state.file_limit_accounted_skips += 1
    return True


def _add_file_limit_skip(state: InspectState, path: Path) -> bool:
    if not _account_file_limit_skip(state):
        return False
    state.files_considered += 1
    _add_skip(state, _skip_entry(path, state, "FILE_LIMIT_REACHED", "file scan limit reached"))
    return True


def _add_skip_after_file_limit(state: InspectState, entry: dict) -> bool:
    if not _account_file_limit_skip(state):
        return False
    _add_skip(state, entry)
    return True


def _add_directory_not_scanned_due_to_file_limit(state: InspectState, directory: Path) -> bool:
    return _add_skip_after_file_limit(
        state,
        _skip_entry(
            directory,
            state,
            "DIRECTORY_NOT_SCANNED_FILE_LIMIT",
            "directory not scanned because file scan limit was reached",
        ),
    )


def _inspect_file(path: Path, state: InspectState, max_file_bytes: int) -> None:
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
        if state.job_py is None or _is_root_level_file(rel_path):
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

    # Register every successfully parsed Python file, including leaf modules
    # with no imports, so local imports can resolve to the complete scanned graph.
    state.file_imports.setdefault(rel_path, set())
    visitor = _PythonInspector(path, rel_path, state)
    visitor.visit(tree)
    visitor.finalize()
    _add_entry_point(path, rel_path, tree, state)


class _PythonInspector(ast.NodeVisitor):
    def __init__(self, path: Path, rel_path: str, state: InspectState):
        self.path = path
        self.rel_path = rel_path
        self.state = state
        self._detectors = frameworks.detectors()
        self._detector_states = {detector.name: detector.new_file_state() for detector in self._detectors}
        self._scope_stack: list[str] = []
        self._ctx = DetectContext(
            self._emit_framework_evidence,
            self._add_flare_call,
            self._add_integration_signal,
        )

    def _emit_framework_evidence(self, framework: str, kind: str, value: str, lineno) -> None:
        _append_evidence(self.state.framework_evidence, framework, _evidence(self.rel_path, lineno, kind, value))

    def _add_integration_signal(self, framework: str, name: str) -> None:
        self.state.integration_signals.setdefault(framework, set()).add(name)
        self.state.integration_signal_files.add(self.rel_path)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record_import(alias.name, node.lineno)
            for detector in self._detectors:
                detector.on_import(
                    alias,
                    self._detector_states[detector.name],
                    self._ctx,
                    tuple(self._scope_stack),
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        self._record_import(_resolve_import_from_module(self.rel_path, module, node.level), node.lineno)
        self._record_import_from_modules(module, node.level, node.names)
        for detector in self._detectors:
            detector.on_import_from(
                module,
                node.names,
                self._detector_states[detector.name],
                self._ctx,
                tuple(self._scope_stack),
            )
        for alias in node.names:
            if alias.name in {"FedJob", "FLModel", "SimEnv"}:
                self.state.flare_imports.append(
                    _evidence(self.rel_path, node.lineno, "from_import", f"{module}.{alias.name}")
                )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        end_lineno = getattr(node, "end_lineno", None) or node.lineno
        self.state.class_body_ranges.setdefault(self.rel_path, []).append((node.lineno, end_lineno))
        base_names = [name for name in (_symbol_name(base) for base in node.bases) if name]
        for base_name in base_names:
            for detector in self._detectors:
                detector.on_class_base(
                    base_name,
                    node.lineno,
                    self._detector_states[detector.name],
                    self._ctx,
                    tuple(self._scope_stack),
                )
        for detector in self._detectors:
            detector.on_class_definition(
                node.name,
                base_names,
                node.lineno,
                self._detector_states[detector.name],
                self._ctx,
                tuple(self._scope_stack),
            )
        self._visit_scoped(node, "class")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_binding_names([node.name], node.lineno)
        self._visit_scoped(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_binding_names([node.name], node.lineno)
        self._visit_scoped(node, "async-function")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_scoped(node, "lambda")

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._record_binding_targets([node.target], getattr(node, "lineno", None))
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._record_binding_targets([node.target], getattr(node, "lineno", None))
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, "list-comprehension")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, "set-comprehension")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, "dict-comprehension")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, "generator-expression")

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        if call_name:
            self._record_call(call_name, node.lineno)
            for detector in self._detectors:
                detector.on_call(
                    call_name,
                    node.lineno,
                    self._detector_states[detector.name],
                    self._ctx,
                    tuple(self._scope_stack),
                )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._inspect_secret_assignment(node.targets, node.value, getattr(node, "lineno", None))
        self.visit(node.value)
        self._record_assignment(node.targets, node.value, getattr(node, "lineno", None))
        for target in node.targets:
            self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._inspect_secret_assignment([node.target], node.value, getattr(node, "lineno", None))
        if node.value:
            self.visit(node.value)
        self._record_assignment([node.target], node.value, getattr(node, "lineno", None))
        self.visit(node.target)
        self.visit(node.annotation)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        self._record_binding_targets([node.target], getattr(node, "lineno", None))

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._record_binding_targets([node.target], getattr(node, "lineno", None))

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._record_binding_targets([item.optional_vars], getattr(node, "lineno", None))
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type:
            self.visit(node.type)
        if node.name:
            self._record_binding_names([node.name], getattr(node, "lineno", None))
        for statement in node.body:
            self.visit(statement)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self._inspect_string_literal(node.value, getattr(node, "lineno", None))
        self.generic_visit(node)

    def _record_import(self, module: str, lineno: int) -> None:
        if not module:
            return
        self.state.file_imports.setdefault(self.rel_path, set()).add(module)
        framework = frameworks.framework_for_import(module)
        if framework:
            _append_evidence(
                self.state.framework_evidence,
                framework,
                _evidence(self.rel_path, lineno, "import", module),
            )
        if module == "nvflare" or module.startswith("nvflare."):
            self.state.flare_imports.append(_evidence(self.rel_path, lineno, "import", module))
        if module in {"hydra", "omegaconf"} or module.startswith(("hydra.", "omegaconf.")):
            self.state.dynamic_patterns.append(_evidence(self.rel_path, lineno, "dynamic_config", module))
        if module == "torch.distributed" or module.startswith("torch.distributed."):
            self.state.distributed_patterns.append(_evidence(self.rel_path, lineno, "distributed_import", module))
        if module == "accelerate" or module.startswith("accelerate."):
            self.state.distributed_patterns.append(_evidence(self.rel_path, lineno, "accelerate_import", module))

    def _record_import_from_modules(self, module: str, level: int, aliases: list[ast.alias]) -> None:
        resolved_module = _resolve_import_from_module(self.rel_path, module, level)
        imports = self.state.file_imports.setdefault(self.rel_path, set())
        if resolved_module:
            imports.add(resolved_module)
        for alias in aliases:
            if alias.name == "*":
                continue
            imports.add(f"{resolved_module}.{alias.name}" if resolved_module else alias.name)

    def _record_call(self, call_name: str, lineno: int) -> None:
        # Generic FLARE / distributed / dynamic-dispatch signals only. Ranked
        # framework activity (pytorch_call/pytorch_data_call,
        # lightning_trainer) and conversion signals (flare.patch) are recorded
        # by framework detectors via on_call.
        if call_name.startswith("flare.") or call_name.startswith("nvflare."):
            self._add_flare_call(call_name)
        if call_name in {"FedJob", "FLModel", "SimEnv"}:
            self._add_flare_call(call_name)
        if call_name == "SimEnv" or call_name.endswith(".SimEnv"):
            self.state.sim_env_used = True
            self.state.sim_env_files.add(self.rel_path)
        if call_name.endswith(".export"):
            self.state.export_support = True
            self.state.export_support_files.add(self.rel_path)
        if call_name in {"importlib.import_module", "__import__", "getattr"}:
            self.state.dynamic_patterns.append(_evidence(self.rel_path, lineno, "dynamic_dispatch", call_name))
        if call_name == "torch.compile":
            self.state.dynamic_patterns.append(_evidence(self.rel_path, lineno, "torch_compile", call_name))
        if call_name.endswith(("DataParallel", "FSDP", "Accelerator")):
            self.state.distributed_patterns.append(_evidence(self.rel_path, lineno, "distributed_call", call_name))

    def _add_flare_call(self, call_name: str) -> None:
        self.state.flare_calls.add(call_name)
        self.state.flare_calls_by_file.setdefault(self.rel_path, set()).add(call_name)

    def _record_assignment(self, targets: list[ast.AST], value: ast.AST, lineno: Optional[int]) -> None:
        call_name = _call_name(value.func) if isinstance(value, ast.Call) else None
        target_names = _assignment_target_names(targets)
        self._dispatch_assignment(target_names, call_name, lineno)

    def _record_binding_targets(self, targets: list[ast.AST], lineno: Optional[int]) -> None:
        self._record_binding_names(_assignment_target_names(targets), lineno)

    def _record_binding_names(self, target_names: list[str], lineno: Optional[int]) -> None:
        self._dispatch_assignment(target_names, None, lineno)

    def _dispatch_assignment(self, target_names: list[str], call_name: Optional[str], lineno: Optional[int]) -> None:
        if not target_names:
            return
        for detector in self._detectors:
            detector.on_assignment(
                target_names,
                call_name,
                lineno,
                self._detector_states[detector.name],
                self._ctx,
                tuple(self._scope_stack),
            )

    def _visit_comprehension(self, node: ast.AST, kind: str) -> None:
        generators = getattr(node, "generators", [])
        if not generators:
            return
        # Python evaluates the first iterable in the enclosing scope; loop
        # targets and all remaining expressions live in the comprehension's
        # implicit scope.
        self.visit(generators[0].iter)
        self._scope_stack.append(f"{kind}:<anonymous>:{getattr(node, 'lineno', 0)}")
        try:
            scope = tuple(self._scope_stack)
            target_names = {
                name
                for generator in generators
                for name in _assignment_target_names([generator.target])
                if "." not in name
            }
            for detector in self._detectors:
                detector.on_scope(
                    scope,
                    target_names,
                    set(),
                    set(),
                    self._detector_states[detector.name],
                    self._ctx,
                )
            for index, generator in enumerate(generators):
                if index:
                    self.visit(generator.iter)
                for condition in generator.ifs:
                    self.visit(condition)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)
        finally:
            self._scope_stack.pop()

    def finalize(self) -> None:
        for detector in self._detectors:
            detector.finalize_file(self._detector_states[detector.name], self._ctx)

    def _visit_scoped(self, node: ast.AST, kind: str) -> None:
        name = getattr(node, "name", "<anonymous>")
        self._scope_stack.append(f"{kind}:{name}:{getattr(node, 'lineno', 0)}")
        try:
            scope = tuple(self._scope_stack)
            local_names, global_names, nonlocal_names = _lexical_bindings(node)
            for detector in self._detectors:
                detector.on_scope(
                    scope,
                    local_names,
                    global_names,
                    nonlocal_names,
                    self._detector_states[detector.name],
                    self._ctx,
                )
            self.generic_visit(node)
        finally:
            self._scope_stack.pop()

    def _inspect_secret_assignment(self, targets: list[ast.AST], value: ast.AST, lineno: Optional[int]) -> None:
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            return
        for target in targets:
            name = _target_name(target)
            if name and SECRET_NAME_PATTERN.search(name):
                self.state.findings.append(
                    {
                        "code": "SECRET_LITERAL_REDACTED",
                        "severity": "warning",
                        "file": self.rel_path,
                        "line": lineno,
                        "name": name,
                        "value": "<REDACTED>" if self.state.redact else value.value,
                    }
                )

    def _inspect_string_literal(self, value: str, lineno: Optional[int]) -> None:
        if _looks_like_absolute_path(value):
            self.state.absolute_path_findings.append(
                {
                    "code": "ABSOLUTE_DATA_PATH",
                    "severity": "warning",
                    "file": self.rel_path,
                    "line": lineno,
                    "pattern_type": "absolute_path_literal",
                    "value": _redact_literal(value, self.state.redact),
                }
            )


class _LexicalBindingCollector(ast.NodeVisitor):
    """Collect names owned by one lexical scope without entering child scopes."""

    def __init__(self):
        self.local_names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.local_names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.local_names.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.local_names.add(alias.asname or alias.name)

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Defaults and decorators execute in the enclosing scope but do not
        # create bindings there; only the function name belongs to this scope.
        self.local_names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.local_names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.local_names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

    def visit_SetComp(self, node: ast.SetComp) -> None:
        return

    def visit_DictComp(self, node: ast.DictComp) -> None:
        return

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        return

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.local_names.add(node.name)
        for statement in node.body:
            self.visit(statement)


def _lexical_bindings(node: ast.AST) -> tuple[set[str], set[str], set[str]]:
    collector = _LexicalBindingCollector()
    arguments = getattr(node, "args", None)
    if isinstance(arguments, ast.arguments):
        for argument in arguments.posonlyargs + arguments.args + arguments.kwonlyargs:
            collector.local_names.add(argument.arg)
        if arguments.vararg:
            collector.local_names.add(arguments.vararg.arg)
        if arguments.kwarg:
            collector.local_names.add(arguments.kwarg.arg)

    if isinstance(node, ast.Lambda):
        collector.visit(node.body)
    else:
        for statement in getattr(node, "body", []):
            collector.visit(statement)

    collector.local_names.difference_update(collector.global_names | collector.nonlocal_names)
    return collector.local_names, collector.global_names, collector.nonlocal_names


def _rank_frameworks(state: InspectState) -> list[dict]:
    total = sum(len(evidence) for evidence in state.framework_evidence.values())
    ranked = []
    for framework, evidence in state.framework_evidence.items():
        count = len(evidence)
        confidence = 0.0
        if total:
            # Any static import evidence should register clearly, but cap below
            # certainty because import presence alone does not prove active use.
            confidence = min(0.99, 0.45 + (count / total) * 0.5)
        ranked.append(
            {
                "name": framework,
                "confidence": round(confidence, 2),
                "evidence": evidence[:MAX_EVIDENCE_PER_BUCKET],
                "contradicting_evidence": [],
            }
        )
    return sorted(ranked, key=lambda item: (-item["confidence"], item["name"]))


def _detect_primary_framework(state: InspectState, ranked: list[dict]) -> Optional[str]:
    if not ranked:
        return None
    primary = _primary_by_confidence_and_entry_context(state, ranked)
    return frameworks.resolve_primary_framework(primary, state.framework_evidence, _FamilyResolver(state))


def _primary_by_confidence_and_entry_context(state: InspectState, ranked: list[dict]) -> str:
    # Frameworks are ranked by (confidence, name), which is count-based and blind
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
    # DESIGN DECISION (do not revert to a pure entry-context or pure count rule):
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

    def training_owner_evidence(self, framework: str) -> list[dict]:
        return [item for item in self.evidence(framework) if frameworks.is_training_owner_evidence(framework, item)]

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
    entry_point_paths = {entry["path"] for entry in state.entry_points}
    return any(item["file"] in entry_point_paths for item in evidence)


def _entry_point_imports_file(state: InspectState, evidence_file: str) -> bool:
    if not _module_names_for_file(evidence_file):
        return False
    graph = _local_import_graph(state)
    return any(evidence_file in graph.reachable_from(entry_point["path"]) for entry_point in state.entry_points)


def _local_files_by_module(state: InspectState) -> dict[str, set[str]]:
    # Register every candidate module name for a file, including the src-layout
    # root-stripped name (mypkg.loop from src/mypkg/loop.py). When a name is
    # claimed by both a root-level file and a src/ copy, the collision is
    # resolved per-import by _prefer_shared_packaging_root using the importing
    # file's own packaging root, so neither a stale src/ copy nor a stale
    # root-level copy can steal the actively-imported module in either direction.
    #
    files_by_module: dict[str, set[str]] = {}
    for file_path in state.file_imports:
        for module_name in _module_names_for_file(file_path):
            files_by_module.setdefault(module_name, set()).add(file_path)
    return files_by_module


def _local_import_graph(state: InspectState) -> _LocalImportGraph:
    if state.local_import_graph_cache is not None:
        return state.local_import_graph_cache

    local_files_by_module = _local_files_by_module(state)
    edges = {}
    for source_file, imports in state.file_imports.items():
        imported_files = set()
        for import_name in imports:
            imported_files.update(_local_files_for_import(import_name, source_file, local_files_by_module))
        edges[source_file] = frozenset(imported_files)

    # Edges store resolved files rather than module names. A stale src-layout
    # copy (src/mypkg/loop.py) can share the stripped name "mypkg.loop" with a
    # root copy, but _local_files_for_import resolves that collision in the
    # importing file's packaging context before the graph is cached.
    graph = _LocalImportGraph(edges)
    state.local_import_graph_cache = graph
    return graph


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


def _is_root_level_file(path: str) -> bool:
    return len(Path(path).parts) == 1


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


def _exported_job_info(state: InspectState) -> dict:
    root = state.root if state.root.is_dir() and not state.root.is_symlink() else state.root.parent
    markers_by_dir: dict[Path, set[str]] = {}
    for path in state.exported_job_marker_paths:
        markers_by_dir.setdefault(path.parent, set()).add(path.name)

    valid_candidate_dirs = set()
    consumed_marker_dirs = set()
    for directory, names in markers_by_dir.items():
        if "meta.json" in names and names.intersection({"config_fed_server.json", "config_fed_client.json"}):
            valid_candidate_dirs.add(directory)

    meta_dirs = {directory for directory, names in markers_by_dir.items() if "meta.json" in names}
    config_paths = [
        path
        for path in state.exported_job_marker_paths
        if path.name in {"config_fed_server.json", "config_fed_client.json"}
    ]
    for meta_dir in meta_dirs:
        for config_path in config_paths:
            if config_path.parent.name == "config" and config_path.parent.parent.parent == meta_dir:
                valid_candidate_dirs.add(meta_dir)
                consumed_marker_dirs.add(config_path.parent)

    submit_ready = sorted(
        (_display_path(directory, root, state.redact) for directory in valid_candidate_dirs if directory == root)
    )
    nested = []
    for directory, names in sorted(markers_by_dir.items(), key=lambda item: _display_path(item[0], root, state.redact)):
        if directory in consumed_marker_dirs:
            continue
        rel_dir = _display_path(directory, root, state.redact)
        if directory in valid_candidate_dirs:
            if directory != root:
                nested.append(
                    {
                        "path": rel_dir,
                        "markers": sorted(names),
                        "reason": "nested_exported_job_candidate",
                    }
                )
        else:
            nested.append(
                {
                    "path": rel_dir,
                    "markers": sorted(names),
                    "reason": "incomplete_exported_job_marker_set",
                }
            )
    return {"submit_ready_candidates": submit_ready, "nested_candidates": nested}


def _build_project_scope(state: InspectState) -> _ProjectScope:
    graph = _local_import_graph(state)
    if state.root.is_file():
        inspected_file = _display_path(state.root, state.root, state.redact)
        component = _ProjectComponent(inspected_file, graph.component_from(inspected_file))
        return _ProjectScope(
            active_components=(component,),
        )

    entry_point_files = {entry["path"] for entry in state.entry_points}
    framework_files = {item["file"] for evidence in state.framework_evidence.values() for item in evidence}
    anchor_files = entry_point_files | framework_files
    components = tuple(
        _ProjectComponent(anchor_file, graph.component_from(anchor_file)) for anchor_file in sorted(anchor_files)
    )
    if not components:
        # With no project anchors, preserve whole-directory evidence rather than
        # hiding a standalone helper or dynamically-dispatched project.
        all_files = graph.files
        fallback = _ProjectComponent(".", all_files)
        return _ProjectScope(
            active_components=(fallback,),
        )

    # Group anchors into projects before applying root-distance preference. This
    # preserves a nested entry point that imports a shallower model: the entry
    # point and model are one project, and all of that project's components stay
    # authoritative. A separate deeper fixture remains a different project and
    # cannot promote the inspected root.
    project_groups = _component_groups(components)
    group_depths = tuple(min(component.depth for component in group) for group in project_groups)
    nearest_depth = min(group_depths)
    active_groups = tuple(group for group, depth in zip(project_groups, group_depths) if depth == nearest_depth)

    # A real training/job entry point can disambiguate an unrelated
    # framework-only helper at the same depth. Do not apply that preference to
    # entry points under tests, fixtures, archives, or vendored trees: those are
    # common secondary projects and must not suppress an equally near
    # independent framework project. If any tied entry point is secondary, keep
    # every tied group rather than allowing another entry point to suppress
    # framework-only groups. This intentionally biases uncertain repository
    # layouts toward ambiguity instead of false FLARE authority. Connected entry
    # points and framework files already share a project group through
    # directory/import connectivity.
    entry_point_groups = tuple(
        group for group in active_groups if any(component.anchor_file in entry_point_files for component in group)
    )
    has_secondary_entry_point = any(
        component.anchor_file in entry_point_files and _is_secondary_project_entry_point(component.anchor_file)
        for group in entry_point_groups
        for component in group
    )
    if entry_point_groups and not has_secondary_entry_point:
        active_groups = entry_point_groups

    active_components = tuple(component for group in active_groups for component in group)

    # Equally near independent project groups identify a multi-project workspace
    # with no single authority. Keep all remaining tied components for framework
    # reporting, but do not authorize one group's FLARE signals for the
    # workspace as a whole.
    ambiguous = len(active_groups) > 1
    return _ProjectScope(
        active_components=active_components,
        ambiguous=ambiguous,
    )


def _source_job_candidates(state: InspectState, project_scope: _ProjectScope) -> tuple[_SourceJobCandidate, ...]:
    graph = _local_import_graph(state)
    flare_import_files = frozenset(item["file"] for item in state.flare_imports)
    source_files = set(state.job_py_paths) | state.sim_env_files
    candidates = []
    for source_file in sorted(source_files):
        component_files = graph.component_from(source_file)
        candidates.append(
            _SourceJobCandidate(
                source_file=source_file,
                flare_import_files=flare_import_files & component_files,
                project_component=(
                    project_scope.component_for(source_file)
                    if _authorizes_flare_evidence(project_scope, source_file)
                    else None
                ),
                export_supported=source_file in state.export_support_files,
            )
        )
    return tuple(candidates)


def _authoritative_source_job_candidate(
    state: InspectState, project_scope: _ProjectScope
) -> Optional[_SourceJobCandidate]:
    candidates = [candidate for candidate in _source_job_candidates(state, project_scope) if candidate.authoritative]
    if not candidates:
        return None
    # Prefer the source closest to the inspected root when multiple candidates
    # live in the same active component.
    return min(candidates, key=lambda candidate: (len(Path(candidate.source_file).parts), candidate.source_file))


def _conversion_state(
    state: InspectState,
    detected_framework: Optional[str],
    exported_job_info: dict,
    project_scope: _ProjectScope,
    source_job: Optional[_SourceJobCandidate],
) -> str:
    if exported_job_info["submit_ready_candidates"]:
        return "exported_job"
    if project_scope.ambiguous:
        return "ambiguous"
    # job.py is a common filename (SLURM launchers) and SimEnv is a natural class
    # name in RL/robotics code, so neither is trustworthy on its own. A source
    # candidate must belong to an authoritative project component and carry
    # corroborating nvflare evidence in its own directed import component.
    if source_job:
        return "flare_job"
    if _has_conversion_integration(state, project_scope):
        return "client_api_converted"
    if all(_has_authoritative_flare_call(state, project_scope, name) for name in ("flare.receive", "flare.send")):
        return "client_api_converted"
    if _has_authoritative_flare_call(state, project_scope, "FLModel"):
        return "client_api_converted"
    if _has_authoritative_flare_evidence(state, project_scope):
        return "partial_client_api"
    if detected_framework:
        return "not_converted"
    return "unknown"


def _has_authoritative_flare_evidence(state: InspectState, project_scope: _ProjectScope) -> bool:
    evidence_files = {item["file"] for item in state.flare_imports} | set(state.flare_calls_by_file)
    return any(_authorizes_flare_evidence(project_scope, path) for path in evidence_files)


def _has_authoritative_flare_call(state: InspectState, project_scope: _ProjectScope, call_name: str) -> bool:
    return any(
        call_name in calls and _authorizes_flare_evidence(project_scope, path)
        for path, calls in state.flare_calls_by_file.items()
    )


def _has_conversion_integration(state: InspectState, project_scope: _ProjectScope) -> bool:
    # A framework conversion-integration signal (e.g. an nvflare.client.lightning
    # ``patch(trainer)`` call) is a definitive conversion signal even without an
    # explicit ``flare.send``, because the framework's callback performs the
    # result exchange. Detectors record these signals via on_call; do not
    # require static constructor evidence here (wrappers/factories can hide it).
    flare_import_files = {item["file"] for item in state.flare_imports}
    return any(
        path in flare_import_files and _authorizes_flare_evidence(project_scope, path)
        for path in state.integration_signal_files
    )


def _target_type(path: Path, state: InspectState, detected_framework: Optional[str], conversion_state: str) -> str:
    if path.is_symlink():
        return "unknown_target"
    if path.is_file():
        return "single_training_script" if path.suffix == ".py" else "unknown_target"
    if conversion_state == "exported_job":
        return "exported_submit_ready_flare_job"
    if conversion_state == "flare_job":
        return "flare_job_source"
    if conversion_state == "ambiguous":
        return "mixed_workspace"
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


def _installed_skills(target: Path) -> list[dict]:
    """Discover installed skills from known agent skill dirs (read-only).

    Scans ``<dir>/*/SKILL.md`` under project-scope dirs (relative to the inspected
    path's project root, walked up to cwd) and global-scope dirs (under the user
    home). Reads only the YAML frontmatter ``name``/``description`` with a small
    inline parser; no user code is imported or executed. Symlinked SKILL.md files
    are skipped and results are deduplicated by skill name and capped.
    """
    skills: list[dict] = []
    seen_names: set[str] = set()
    for base, scope in _installed_skill_search_roots(target):
        for skill_dir in _iter_skill_dirs(base):
            if len(skills) >= MAX_INSTALLED_SKILLS:
                return skills
            skill_file = skill_dir / SKILL_FILE_NAME
            if skill_file.is_symlink() or not skill_file.is_file():
                continue
            frontmatter = _read_skill_frontmatter(skill_file)
            if frontmatter is None:
                continue
            name = frontmatter.get("name") or skill_dir.name
            if name in seen_names:
                continue
            seen_names.add(name)
            skills.append(
                {
                    "name": name,
                    "description": frontmatter.get("description", ""),
                    "scope": scope,
                    "source": _installed_skill_source(skill_dir),
                }
            )
    return skills


def _installed_skill_search_roots(target: Path) -> list[tuple[Path, str]]:
    roots: list[tuple[Path, str]] = []
    project_root = _project_root_for(target)
    if project_root is not None:
        for rel in _PROJECT_SKILL_DIRS:
            roots.append((project_root / rel, "project"))
    home = Path.home()
    for rel in _GLOBAL_SKILL_DIRS:
        roots.append((Path(rel).expanduser() if rel.startswith("~") else home / rel, "global"))
    return roots


def _project_root_for(target: Path) -> Optional[Path]:
    # Walk up from the inspected path toward cwd looking for a directory that
    # holds a known project-scope skill dir. Fall back to cwd so a project with
    # no skills still reports an empty list rather than erroring.
    try:
        start = target if target.is_dir() and not target.is_symlink() else target.parent
        start = start.resolve()
        cwd = Path.cwd().resolve()
    except OSError:
        return None
    candidates = [start, *start.parents]
    for candidate in candidates:
        for rel in _PROJECT_SKILL_DIRS:
            if (candidate / rel).is_dir():
                return candidate
        if candidate == cwd:
            break
    return cwd


def _iter_skill_dirs(base: Path):
    if base.is_symlink() or not base.is_dir():
        return
    try:
        children = sorted(base.iterdir(), key=lambda p: p.name)
    except OSError:
        return
    for child in children:
        if child.is_symlink() or not child.is_dir():
            continue
        yield child


def _installed_skill_source(skill_dir: Path) -> str:
    try:
        return str(skill_dir.resolve(strict=False))
    except OSError:
        return str(skill_dir)


def _read_skill_frontmatter(skill_file: Path) -> Optional[dict]:
    """Parse the leading YAML frontmatter block for name/description only.

    Small inline parser (no PyYAML, no dev-tools import): reads the block between
    the leading ``---`` fences and extracts top-level ``name`` and ``description``
    scalars. Returns None on unreadable/oversized files or a missing block.
    """
    try:
        if skill_file.stat().st_size > MAX_SKILL_FRONTMATTER_BYTES:
            return None
        text = skill_file.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    result: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        # Only top-level keys (no leading indentation) so nested metadata is ignored.
        if line[:1] in (" ", "\t") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key in ("name", "description"):
            result[key] = _strip_scalar(value.strip())
    return result


def _strip_scalar(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _skill_selection(
    detected_framework: Optional[str], conversion_state: str, state: InspectState, dataset: Optional[dict] = None
) -> dict:
    recommended = []
    if conversion_state == "exported_job":
        # Lifecycle skills are out of scope and not planned; exported jobs are
        # handled with product APIs directly, so no skill is recommended.
        pass
    elif conversion_state == "flare_job":
        # An existing FLARE job source: optimization requests (improve a metric,
        # fix low accuracy, explore hyperparameters/algorithms) route to Auto-FL.
        recommended.append("nvflare-autofl")
    elif conversion_state == "ambiguous":
        recommended.append("nvflare-orient")
    elif dataset and dataset.get("modality") in ("tabular", "image"):
        # A dataset target routes to federated statistics; when a dataset
        # block exists, code classification found no converter route.
        recommended.append("nvflare-fed-stats")
    elif dataset and dataset.get("modality") == "mixed":
        # Mixed data is definitionally ambiguous, and routing ambiguity is
        # orient's job; an empty recommendation would strand the consumer.
        recommended.append("nvflare-orient")
    elif conversion_state in {"not_converted", "partial_client_api"} and frameworks.has_active_family_member_conflict(
        state.framework_evidence, _FamilyResolver(state)
    ):
        # Two specialized trainers in one family cannot both own one conversion.
        recommended.append("nvflare-orient")
    elif detected_framework and conversion_state in {"not_converted", "partial_client_api"}:
        evidence = state.framework_evidence.get(detected_framework, [])
        skill = frameworks.recommended_skill_for(detected_framework, evidence)
        if skill:
            recommended.append(skill)
        else:
            fallback = frameworks.fallback_skill_for(detected_framework, evidence)
            if fallback:
                recommended.append(fallback)
    if (
        (state.findings or _has_problematic_skips(state))
        and "nvflare-fed-stats" not in recommended
        and "nvflare-orient" not in recommended
    ):
        # Converter recommendations keep the historical orient companion on
        # findings; a classified dataset (or an already-routed mixed target)
        # keeps a single recommendation.
        recommended.append("nvflare-orient")

    return {
        "detected_framework": detected_framework,
        "conversion_state": conversion_state,
        "exported_job": conversion_state == "exported_job",
        "recommended_skills": recommended,
        "safety_findings": [finding["code"] for finding in state.findings[:MAX_EVIDENCE_PER_BUCKET]],
    }


def _has_problematic_skips(state: InspectState) -> bool:
    return state.classification_incomplete


def _recommended_next_commands(
    detected_framework: Optional[str],
    conversion_state: str,
    source_job: Optional[_SourceJobCandidate],
) -> list[str]:
    commands = []
    if conversion_state == "exported_job":
        commands.append("nvflare job submit <job-folder> --format json")
    elif source_job and source_job.export_supported:
        # Only suggest `job.py --export` for a genuine FLARE job.py: `.export`
        # calls (torch.onnx.export, YOLO model.export, ...) over-match, so without
        # corroborating nvflare evidence this would ship a command that fails with
        # an argparse error on an unrelated repo.
        commands.append(f"python {shlex.quote(source_job.source_file)} --export --export-dir <job-dir>")
    elif conversion_state in {"not_converted", "partial_client_api"} and frameworks.has_active_family_member_conflict(
        state.framework_evidence, _FamilyResolver(state)
    ):
        commands.append("Use the nvflare-orient skill before editing.")
    elif detected_framework and conversion_state in {"not_converted", "partial_client_api"}:
        evidence = state.framework_evidence.get(detected_framework, [])
        skill = frameworks.recommended_skill_for(detected_framework, evidence)
        if skill:
            commands.append(f"Use the {skill} skill before editing.")
        else:
            fallback = frameworks.fallback_skill_for(detected_framework, evidence)
            if fallback:
                commands.append(f"Use the {fallback} skill before editing.")
    return commands


def _record_symlink_skip(path: Path, state: InspectState) -> None:
    _add_skip(state, _symlink_skip_entry(path, state))


def _record_symlink_skip_after_file_limit(path: Path, state: InspectState) -> bool:
    return _add_skip_after_file_limit(state, _symlink_skip_entry(path, state))


def _symlink_skip_entry(path: Path, state: InspectState) -> dict:
    try:
        target = os.readlink(path)
    except OSError:
        target = ""
    return {
        "code": "SYMLINK_SKIPPED",
        "path": _display_path(path, state.root, state.redact),
        "target": _redact_literal(target, state.redact),
        "message": "symlink was not followed during static inspection",
    }


def _add_skip(state: InspectState, entry: dict) -> None:
    state.files_skipped_count += 1
    if entry.get("code") in _INCOMPLETE_SCAN_SKIP_CODES:
        state.classification_incomplete = True
    if len(state.files_skipped) < MAX_EVIDENCE_PER_BUCKET:
        state.files_skipped.append(entry)


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


def _normalized_inspect_target(path: Path | str) -> Path:
    return Path(_inspected_target_path(Path(path).expanduser()))


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


def _append_evidence(target: dict[str, list[dict]], key: str, value: dict) -> None:
    # Collect up to a generous backstop so framework ranking/detection sees the
    # true evidence counts. Display is truncated to MAX_EVIDENCE_PER_BUCKET
    # separately (see _rank_frameworks); capping at collection time would skew the
    # count-based confidence and let a file's first 12 imports decide routing.
    bucket = target.setdefault(key, [])
    if len(bucket) < MAX_EVIDENCE_COLLECT:
        bucket.append(value)


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


def _assignment_target_names(targets: list[ast.AST]) -> list[str]:
    names = []
    pending = list(targets)
    while pending:
        target = pending.pop()
        if isinstance(target, (ast.Tuple, ast.List)):
            pending.extend(target.elts)
            continue
        name = _call_name(target)
        if name:
            names.append(name)
    return names
