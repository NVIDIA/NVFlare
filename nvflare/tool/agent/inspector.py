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
    distributed_patterns: list[dict] = field(default_factory=list)
    dynamic_patterns: list[dict] = field(default_factory=list)
    absolute_path_findings: list[dict] = field(default_factory=list)


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
        "scan": {
            "entries_visited": state.entries_visited,
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
            "calls": sorted(state.flare_calls),
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
        "findings": state.findings[:MAX_EVIDENCE_PER_BUCKET],
        "skill_selection": _skill_selection(detected_framework, conversion_state, state),
        "recommended_next_commands": _recommended_next_commands(detected_framework, conversion_state, state),
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
                _add_skip(state, _skip_entry(child, state, "FILE_LIMIT_REACHED", "file scan limit reached"))
                _record_unvisited_directories_due_to_file_limit(state, root, stack, children[index + 1 :])
                return
            state.entries_visited += 1
            if not child.is_file():
                continue
            _inspect_file(child, state, max_file_bytes)
            if state.entries_visited >= max_files:
                _record_next_file_due_to_file_limit(state, children[index + 1 :])
                _record_unvisited_directories_due_to_file_limit(state, root, stack, children[index + 1 :])
                return


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


def _inspect_file(path: Path, state: InspectState, max_file_bytes: int) -> None:
    state.files_considered += 1
    rel_path = _display_path(path, state.root, state.redact)
    if _is_sensitive_file(path):
        _add_skip(state, _skip_entry(path, state, "SENSITIVE_FILE_SKIPPED", "sensitive file skipped"))
        return
    if _is_exported_job_marker(path):
        state.exported_job_markers.append(rel_path)
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
        text = path.read_text(encoding="utf-8")
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
            rel_path,
            self._emit_framework_evidence,
            self.state.flare_calls.add,
            self._add_integration_signal,
        )

    def _emit_framework_evidence(self, framework: str, kind: str, value: str, lineno) -> None:
        _append_capped(self.state.framework_evidence, framework, _evidence(self.rel_path, lineno, kind, value))

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
            _append_capped(
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
        if call_name.endswith("DistributedDataParallel") or call_name.endswith("DataParallel"):
            self.state.distributed_patterns.append(_evidence(self.rel_path, lineno, "distributed_call", call_name))
        if call_name.endswith("FSDP") or call_name.endswith("Accelerator"):
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
    primary = ranked[0]["name"]
    return frameworks.resolve_primary_framework(primary, state.framework_evidence, _FamilyResolver(state))


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
        return _has_inspected_file_or_entry_point(self._state)

    def has_evidence_outside_files(self, evidence: list[dict], reference_evidence: list[dict]) -> bool:
        reference_files = {item["file"] for item in reference_evidence}
        return any(item["file"] not in reference_files for item in evidence)


def _framework_evidence_tied_to_entry_context(state: InspectState, evidence: list[dict]) -> bool:
    if _framework_evidence_tied_to_inspected_file_or_entry_point(state, evidence):
        return True
    if state.root.is_file():
        return False
    return any(_entry_point_imports_file(state, item["file"]) for item in evidence)


def _has_inspected_file_or_entry_point(state: InspectState) -> bool:
    return state.root.is_file() or bool(state.entry_points)


def _framework_evidence_tied_to_inspected_file_or_entry_point(state: InspectState, evidence: list[dict]) -> bool:
    if state.root.is_file():
        inspected_file = _display_path(state.root, state.root, state.redact)
        return any(item["file"] == inspected_file for item in evidence)
    entry_point_paths = {entry["path"] for entry in state.entry_points}
    return any(item["file"] in entry_point_paths for item in evidence)


def _entry_point_imports_file(state: InspectState, evidence_file: str) -> bool:
    evidence_modules = _module_names_for_file(evidence_file)
    if not evidence_modules:
        return False
    local_files_by_module = _local_files_by_module(state)
    for entry_point in state.entry_points:
        if _imports_reach_modules(
            state,
            entry_point["path"],
            state.file_imports.get(entry_point["path"], set()),
            evidence_modules,
            local_files_by_module,
        ):
            return True
    return False


def _imports_reach_modules(
    state: InspectState,
    importing_file: str,
    imports: set[str],
    target_modules: set[str],
    local_files_by_module: dict[str, set[str]],
) -> bool:
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
            if target_modules.intersection(_module_names_for_file(imported_file)):
                return True
            pending_imports.extend(
                (imported_file, nested_import) for nested_import in state.file_imports.get(imported_file, set())
            )
    return False


def _local_files_by_module(state: InspectState) -> dict[str, set[str]]:
    files_by_module: dict[str, set[str]] = {}
    for file_path in state.file_imports:
        for module_name in _module_names_for_file(file_path):
            files_by_module.setdefault(module_name, set()).add(file_path)
    return files_by_module


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
    return files


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


def _module_names_for_file(file_path: str) -> set[str]:
    if not file_path.endswith(".py"):
        return set()
    path = Path(file_path)
    parts = path.parent.parts if path.name == "__init__.py" else path.with_suffix("").parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
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
    path = Path(importing_file)
    package_parts = path.parent.parts if path.name != "__init__.py" else path.parent.parts
    keep = max(0, len(package_parts) - level + 1)
    parts = list(package_parts[:keep])
    if module:
        parts.extend(module.split("."))
    return ".".join(part for part in parts if part)


def _evidence_score(evidence: list[dict]) -> int:
    weights = frameworks.evidence_weights()
    return sum(weights.get(item["kind"], 1) for item in evidence)


def _order_frameworks_for_display(ranked: list[dict], detected_framework: Optional[str]) -> list[dict]:
    family_member = frameworks.family_member_of_base(detected_framework)
    if family_member:
        return _order_family_base_before_member(ranked, detected_framework, family_member)

    # Keep the confidence-ranked order but surface the detected primary framework
    # first so callers reading frameworks[0] stay aligned with the routing
    # decision. sorted() is stable, so non-primary frameworks keep their order.
    return sorted(ranked, key=lambda item: item["name"] != detected_framework)


def _order_family_base_before_member(ranked: list[dict], base: str, member: str) -> list[dict]:
    # When the detected primary is a family base (e.g. PyTorch) that also has a
    # superset member present (PyTorch Lightning), keep the base ahead of the
    # member so frameworks[0] matches the routing decision.
    names = [item["name"] for item in ranked]
    try:
        base_index = names.index(base)
        member_index = names.index(member)
    except ValueError:
        return ranked

    if base_index < member_index:
        return ranked

    ordered = list(ranked)
    base_item = ordered.pop(base_index)
    member_index = next(index for index, item in enumerate(ordered) if item["name"] == member)
    ordered.insert(member_index, base_item)
    return ordered


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
    if _is_mixed_family_workspace(state, detected_framework):
        return "mixed_workspace"
    if detected_framework:
        return "training_repository"
    return "unknown_target"


def _is_mixed_family_workspace(state: InspectState, detected_framework: Optional[str]) -> bool:
    # A family base (e.g. PyTorch) detected alongside its superset member
    # (PyTorch Lightning) in the evidence is a mixed workspace.
    return bool(frameworks.family_base_has_member(detected_framework, state.framework_evidence))


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
    if state.findings or state.files_skipped:
        recommended.append("nvflare-orient")

    return {
        "detected_framework": detected_framework,
        "conversion_state": conversion_state,
        "exported_job": conversion_state == "exported_job",
        "recommended_skills": recommended,
        "safety_findings": [finding["code"] for finding in state.findings[:MAX_EVIDENCE_PER_BUCKET]],
    }


def _recommended_next_commands(
    detected_framework: Optional[str], conversion_state: str, state: InspectState
) -> list[str]:
    commands = ["nvflare agent doctor --format json"]
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


def _add_skip(state: InspectState, entry: dict) -> None:
    state.files_skipped_count += 1
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


def _append_capped(target: dict[str, list[dict]], key: str, value: dict) -> None:
    bucket = target.setdefault(key, [])
    if len(bucket) < MAX_EVIDENCE_PER_BUCKET:
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
