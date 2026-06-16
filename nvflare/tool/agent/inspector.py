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

DEFAULT_MAX_FILES = 250
DEFAULT_MAX_FILE_BYTES = 512 * 1024
MAX_EVIDENCE_PER_BUCKET = 12

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
LIGHTNING_FRAMEWORK = "pytorch_lightning"
LIGHTNING_MODULES = {"pytorch_lightning", "lightning", "lightning.pytorch"}
LIGHTNING_CLASS_SYMBOLS = {"LightningModule", "LightningDataModule"}
LIGHTNING_TRAINER_SYMBOLS = {"Trainer"}
LIGHTNING_SYMBOLS = LIGHTNING_CLASS_SYMBOLS | LIGHTNING_TRAINER_SYMBOLS
LIGHTNING_PATCH_MODULE = "nvflare.client.lightning"

FRAMEWORK_IMPORTS = {
    "torch": "pytorch",
    "torchvision": "pytorch",
    "torchaudio": "pytorch",
    "pytorch_lightning": LIGHTNING_FRAMEWORK,
    "lightning": LIGHTNING_FRAMEWORK,
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
    "xgboost": "xgboost",
    "sklearn": "sklearn",
    "jax": "jax",
    "flax": "jax",
    "optax": "jax",
    "numpy": "numpy",
}
FRAMEWORK_SKILLS = {
    "pytorch": "nvflare-convert-pytorch",
}
# Future candidate mappings. Keep these inactive until the matching skill
# directories are implemented, packaged, and covered by admission tests.
# "pytorch_lightning": "nvflare-convert-lightning",
# "tensorflow": "nvflare-convert-tensorflow",
# "xgboost": "nvflare-convert-xgboost",
# "sklearn": "nvflare-convert-sklearn",
# "jax": "nvflare-convert-jax",
# "numpy": "nvflare-convert-numpy",


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
    lightning_patch_calls: set[str] = field(default_factory=set)
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

    frameworks = _rank_frameworks(state)
    detected_framework = frameworks[0]["name"] if frameworks else None
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
        "frameworks": frameworks,
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
        self.lightning_aliases: set[str] = set()
        self.lightning_symbols: dict[str, str] = {}
        self.lightning_patch_symbols: set[str] = set()
        self.lightning_patch_modules: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record_import(alias.name, node.lineno)
            self._record_lightning_import_alias(alias)
            self._record_lightning_patch_module_alias(alias)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        self._record_import(module, node.lineno)
        self._record_lightning_from_imports(module, node.names)
        self._record_lightning_patch_imports(module, node.names)
        for alias in node.names:
            if alias.name in {"FedJob", "FLModel", "SimEnv"}:
                self.state.flare_imports.append(
                    _evidence(self.rel_path, node.lineno, "from_import", f"{module}.{alias.name}")
                )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            base_name = _symbol_name(base)
            if base_name and self._is_lightning_class_base(base_name):
                _append_capped(
                    self.state.framework_evidence,
                    LIGHTNING_FRAMEWORK,
                    _evidence(self.rel_path, node.lineno, "lightning_class", base_name),
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        if call_name:
            self._record_call(call_name, node.lineno)
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
        framework = _framework_for_import(module)
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

    def _record_lightning_import_alias(self, alias: ast.alias) -> None:
        if alias.name in LIGHTNING_MODULES:
            self.lightning_aliases.add(alias.asname or alias.name.split(".")[0])

    def _record_lightning_patch_module_alias(self, alias: ast.alias) -> None:
        if alias.name == LIGHTNING_PATCH_MODULE:
            # ``import nvflare.client.lightning as flare`` -> ``flare.patch`` is
            # the canonical conversion call; a plain import keeps the full path.
            self.lightning_patch_modules.add(alias.asname or alias.name)

    def _record_call(self, call_name: str, lineno: int) -> None:
        if call_name.startswith("flare.") or call_name.startswith("nvflare."):
            self.state.flare_calls.add(call_name)
        if call_name in self.lightning_patch_symbols or self._is_lightning_patch_call(call_name):
            self.state.flare_calls.add(call_name)
            self.state.lightning_patch_calls.add(call_name)
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
        if self._is_lightning_trainer_call(call_name):
            _append_capped(
                self.state.framework_evidence,
                LIGHTNING_FRAMEWORK,
                _evidence(self.rel_path, lineno, "lightning_trainer", call_name),
            )

    def _record_lightning_from_imports(self, module: str, aliases: list[ast.alias]) -> None:
        if module not in LIGHTNING_MODULES and not any(module.startswith(f"{prefix}.") for prefix in LIGHTNING_MODULES):
            return
        for alias in aliases:
            if alias.name in LIGHTNING_SYMBOLS:
                self.lightning_symbols[alias.asname or alias.name] = alias.name

    def _record_lightning_patch_imports(self, module: str, aliases: list[ast.alias]) -> None:
        if module != LIGHTNING_PATCH_MODULE:
            return
        for alias in aliases:
            if alias.name == "patch":
                self.lightning_patch_symbols.add(alias.asname or alias.name)

    def _is_lightning_class_base(self, base_name: str) -> bool:
        if self.lightning_symbols.get(base_name) in LIGHTNING_CLASS_SYMBOLS:
            return True
        if "." not in base_name:
            return False
        prefix, _, symbol = base_name.rpartition(".")
        return symbol in LIGHTNING_CLASS_SYMBOLS and (
            prefix in self.lightning_aliases or prefix in LIGHTNING_MODULES or prefix.startswith("lightning.pytorch")
        )

    def _is_lightning_patch_call(self, call_name: str) -> bool:
        prefix, _, symbol = call_name.rpartition(".")
        if symbol != "patch":
            return False
        return prefix in self.lightning_patch_modules or prefix == LIGHTNING_PATCH_MODULE

    def _is_lightning_trainer_call(self, call_name: str) -> bool:
        if self.lightning_symbols.get(call_name) in LIGHTNING_TRAINER_SYMBOLS:
            return True
        if "." not in call_name:
            return False
        prefix, _, symbol = call_name.rpartition(".")
        return symbol in LIGHTNING_TRAINER_SYMBOLS and (
            prefix in self.lightning_aliases or prefix in LIGHTNING_MODULES or prefix.startswith("lightning.pytorch")
        )

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


def _framework_for_import(module: str) -> Optional[str]:
    parts = module.split(".")
    if not parts:
        return None
    first = parts[0]
    if first == "lightning" and len(parts) > 1 and parts[1] == "pytorch":
        return "pytorch_lightning"
    if first == "sklearn":
        return "sklearn"
    return FRAMEWORK_IMPORTS.get(first)


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
    ranked = sorted(ranked, key=lambda item: (-item["confidence"], item["name"]))
    return _prefer_lightning_over_pytorch(ranked)


def _prefer_lightning_over_pytorch(ranked: list[dict]) -> list[dict]:
    names = [item["name"] for item in ranked]
    if "pytorch" not in names or LIGHTNING_FRAMEWORK not in names:
        return ranked

    lightning = ranked.pop(names.index(LIGHTNING_FRAMEWORK))
    pytorch_index = next(index for index, item in enumerate(ranked) if item["name"] == "pytorch")
    ranked.insert(pytorch_index, lightning)
    return ranked


def _conversion_state(state: InspectState, detected_framework: Optional[str]) -> str:
    if state.exported_job_markers:
        return "exported_job"
    if state.job_py or state.sim_env_used:
        return "flare_job"
    if state.lightning_patch_calls and state.flare_imports:
        # An nvflare.client.lightning ``patch(trainer)`` call is the definitive
        # Lightning conversion signal regardless of how the trainer was built
        # (e.g. ``nl.Trainer`` from a wrapper such as nemo.lightning).
        return "client_api_converted"
    if {"flare.receive", "flare.send"} <= state.flare_calls or "FLModel" in state.flare_calls:
        return "client_api_converted"
    if state.flare_imports or state.flare_calls:
        return "partial_client_api"
    if detected_framework:
        return "not_converted"
    return "unknown"


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


def _skill_selection(detected_framework: Optional[str], conversion_state: str, state: InspectState) -> dict:
    recommended = []
    if conversion_state == "exported_job":
        recommended.append("nvflare-job-lifecycle")
    elif detected_framework and conversion_state == "not_converted":
        skill = FRAMEWORK_SKILLS.get(detected_framework)
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
        skill = FRAMEWORK_SKILLS.get(detected_framework)
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
