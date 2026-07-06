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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nvflare
from nvflare.tool.agent import frameworks
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
    # framework name -> FLARE conversion-integration call names (e.g. Lightning
    # flare.patch). Populated by framework detectors; used by _conversion_state.
    integration_signals: dict[str, set[str]] = field(default_factory=dict)
    file_imports: dict[str, set[str]] = field(default_factory=dict)
    entry_points: list[dict] = field(default_factory=list)
    job_py: Optional[str] = None
    sim_env_used: bool = False
    export_support: bool = False
    exported_job_markers: list[str] = field(default_factory=list)
    exported_job_marker_paths: list[Path] = field(default_factory=list)
    distributed_patterns: list[dict] = field(default_factory=list)
    dynamic_patterns: list[dict] = field(default_factory=list)
    absolute_path_findings: list[dict] = field(default_factory=list)
    # file -> list of (start_line, end_line) for every class definition. Used to
    # decide whether base-framework usage lives inside a superset model class
    # body (e.g. torch calls inside a LightningModule) versus standalone.
    class_body_ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # Cache for _local_files_by_module: built once after the scan populates
    # file_imports, reused across per-evidence entry-context reachability checks.
    local_files_by_module_cache: Optional[dict[str, set[str]]] = field(default=None, repr=False, compare=False)


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
    ranked_frameworks = _rank_frameworks(state)
    detected_framework = _detect_primary_framework(state, ranked_frameworks)
    ranked_frameworks = _order_frameworks_for_display(ranked_frameworks, detected_framework)
    conversion_state = _conversion_state(state, detected_framework, exported_job_info)
    target_type = _target_type(target, state, detected_framework, conversion_state)

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
        "skill_selection": _skill_selection(detected_framework, conversion_state, state),
        "recommended_next_commands": _recommended_next_commands(detected_framework, conversion_state, state),
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

    visitor = _PythonInspector(path, rel_path, state)
    visitor.visit(tree)
    _add_entry_point(path, rel_path, tree, state)


class _PythonInspector(ast.NodeVisitor):
    def __init__(self, path: Path, rel_path: str, state: InspectState):
        self.path = path
        self.rel_path = rel_path
        self.state = state
        self._detectors = frameworks.detectors()
        self._detector_states = {detector.name: detector.new_file_state() for detector in self._detectors}
        self._ctx = DetectContext(
            self._emit_framework_evidence,
            self.state.flare_calls.add,
            self._add_integration_signal,
        )

    def _emit_framework_evidence(self, framework: str, kind: str, value: str, lineno) -> None:
        _append_evidence(self.state.framework_evidence, framework, _evidence(self.rel_path, lineno, kind, value))

    def _add_integration_signal(self, framework: str, name: str) -> None:
        self.state.integration_signals.setdefault(framework, set()).add(name)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record_import(alias.name, node.lineno)
            for detector in self._detectors:
                detector.on_import(alias, self._detector_states[detector.name], self._ctx)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        self._record_import(_resolve_import_from_module(self.rel_path, module, node.level), node.lineno)
        self._record_import_from_modules(module, node.level, node.names)
        for detector in self._detectors:
            detector.on_import_from(module, node.names, self._detector_states[detector.name], self._ctx)
        for alias in node.names:
            if alias.name in {"FedJob", "FLModel", "SimEnv"}:
                self.state.flare_imports.append(
                    _evidence(self.rel_path, node.lineno, "from_import", f"{module}.{alias.name}")
                )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        end_lineno = getattr(node, "end_lineno", None) or node.lineno
        self.state.class_body_ranges.setdefault(self.rel_path, []).append((node.lineno, end_lineno))
        for base in node.bases:
            base_name = _symbol_name(base)
            if not base_name:
                continue
            for detector in self._detectors:
                detector.on_class_base(base_name, node.lineno, self._detector_states[detector.name], self._ctx)
        self.generic_visit(node)

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
        # framework activity (pytorch_call, lightning_trainer) and conversion
        # signals (flare.patch) are recorded by framework detectors via on_call.
        if call_name.startswith("flare.") or call_name.startswith("nvflare."):
            self.state.flare_calls.add(call_name)
        if call_name in {"FedJob", "FLModel", "SimEnv"}:
            self.state.flare_calls.add(call_name)
        if call_name == "SimEnv" or call_name.endswith(".SimEnv"):
            self.state.sim_env_used = True
        if call_name.endswith(".export"):
            self.state.export_support = True
        if call_name in {"importlib.import_module", "__import__", "getattr"}:
            self.state.dynamic_patterns.append(_evidence(self.rel_path, lineno, "dynamic_dispatch", call_name))
        if call_name == "torch.compile":
            self.state.dynamic_patterns.append(_evidence(self.rel_path, lineno, "torch_compile", call_name))
        if call_name.endswith(("DataParallel", "FSDP", "Accelerator")):
            self.state.distributed_patterns.append(_evidence(self.rel_path, lineno, "distributed_call", call_name))

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
    local_files_by_module = _local_files_by_module(state)
    for entry_point in state.entry_points:
        if _imports_reach_file(
            state,
            entry_point["path"],
            state.file_imports.get(entry_point["path"], set()),
            evidence_file,
            local_files_by_module,
        ):
            return True
    return False


def _imports_reach_file(
    state: InspectState,
    importing_file: str,
    imports: set[str],
    target_file: str,
    local_files_by_module: dict[str, set[str]],
) -> bool:
    # Match on the resolved file, not on module names: a stale src-layout copy
    # (src/mypkg/loop.py) shares the stripped module name "mypkg.loop" with a
    # root-level mypkg/loop.py, so name-based matching would let the entry point
    # "reach" the never-imported copy. _local_files_by_module resolves the shared
    # name to the root file only, and comparing files keeps them distinct.
    pending_imports = [(importing_file, import_name) for import_name in imports]
    seen_imports = set()
    seen_files = set()
    while pending_imports:
        source_file, import_name = pending_imports.pop()
        import_key = (source_file, import_name)
        if import_key in seen_imports:
            continue
        seen_imports.add(import_key)
        for imported_file in _local_files_for_import(import_name, source_file, local_files_by_module):
            if imported_file in seen_files:
                continue
            seen_files.add(imported_file)
            if imported_file == target_file:
                return True
            pending_imports.extend(
                (imported_file, nested_import) for nested_import in state.file_imports.get(imported_file, set())
            )
    return False


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


def _conversion_state(state: InspectState, detected_framework: Optional[str], exported_job_info: dict) -> str:
    if exported_job_info["submit_ready_candidates"]:
        return "exported_job"
    # job.py is a common filename (SLURM launchers) and SimEnv is a natural class
    # name in RL/robotics code, so neither is trustworthy on its own. Require
    # corroborating nvflare evidence (an nvflare-rooted import) before treating a
    # name-only signal as a FLARE job, mirroring the exported-job marker grouping.
    if (state.job_py or state.sim_env_used) and state.flare_imports:
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


def _skill_selection(detected_framework: Optional[str], conversion_state: str, state: InspectState) -> dict:
    recommended = []
    if conversion_state == "exported_job":
        # Lifecycle skills are out of scope and not planned; exported jobs are
        # handled with product APIs directly, so no skill is recommended.
        pass
    elif detected_framework and conversion_state == "not_converted":
        skill = frameworks.recommended_skill_for(detected_framework)
        if skill:
            recommended.append(skill)
    if state.findings or _has_problematic_skips(state):
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
    detected_framework: Optional[str], conversion_state: str, state: InspectState
) -> list[str]:
    commands = []
    if conversion_state == "exported_job":
        commands.append("nvflare job submit <job-folder> --format json")
    elif state.job_py and state.export_support and state.flare_imports:
        # Only suggest `job.py --export` for a genuine FLARE job.py: `.export`
        # calls (torch.onnx.export, YOLO model.export, ...) over-match, so without
        # corroborating nvflare evidence this would ship a command that fails with
        # an argparse error on an unrelated repo.
        commands.append("python job.py --export --export-dir <job-dir>")
    elif detected_framework and conversion_state == "not_converted":
        skill = frameworks.recommended_skill_for(detected_framework)
        if skill:
            commands.append(f"Use the {skill} skill before editing.")
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
