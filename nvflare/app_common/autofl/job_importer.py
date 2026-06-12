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

"""Deterministically import supported NVFlare job scripts into ``autofl.yaml``.

The Auto-FL skill uses this module as its trust layer.  The importer parses
Python source with ``ast``; it never imports or executes the user's ``job.py``.
Supported Recipe/FedJob patterns are converted into a reviewable config, while
dynamic or unsupported fields are surfaced under ``unresolved``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

AUTOFL_CONFIG_SCHEMA_VERSION = "nvflare.autofl.config.v1"
IMPORTER_VERSION = "nvflare-autofl-job-importer/v1"

SUPPORTED_ENV_NAMES = {"PocEnv", "ProdEnv", "SimEnv"}
TUNABLE_ARG_NAMES = {
    "aggregation_epochs",
    "alpha",
    "batch_size",
    "cosine_lr_eta_min_factor",
    "eval_batch_size",
    "epochs",
    "fedopt_beta1",
    "fedopt_beta2",
    "fedopt_tau",
    "fedproxloss_mu",
    "local_epochs",
    "local_train_steps",
    "lr",
    "max_model_params",
    "model_arch",
    "momentum",
    "num_workers",
    "server_lr",
    "server_momentum",
    "weight_decay",
}


@dataclass(frozen=True)
class ArgSpec:
    """Static argparse field extracted from source."""

    name: str
    flags: Tuple[str, ...]
    default: Any = None
    default_source: str = "argparse_default"
    default_unresolved: bool = False
    value_type: Optional[str] = None
    choices: Optional[List[Any]] = None
    action: Optional[str] = None


@dataclass(frozen=True)
class ResolvedValue:
    """Resolved expression value plus provenance and confidence."""

    value: Any
    source: str
    confidence: str = "high"
    unresolved: bool = False


@dataclass(frozen=True)
class CallInfo:
    """Supported call found in ``job.py`` with source-local resolution context."""

    name: str
    full_name: str
    keywords: Dict[str, ast.AST]
    assignments: Dict[str, ast.AST]
    source: str
    function_name: Optional[str] = None


class JobImportError(ValueError):
    """Raised when the importer cannot read or parse a job file."""


class DeterministicJobImporter:
    """Rule-based importer for supported NVFlare Recipe and FedJob scripts."""

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = Path(workspace_root or ".").resolve()

    def import_job(
        self,
        job_path: str,
        *,
        metric: Optional[str] = None,
        mode: str = "max",
        target_env: Optional[str] = None,
        max_candidates: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return an ``autofl.yaml``-shaped config for ``job_path``.

        Args:
            job_path: Path to ``job.py`` or a directory containing ``job.py``.
            metric: Optional optimization metric requested by the user.
            mode: ``max`` or ``min`` objective direction.
            target_env: Optional target environment, such as ``sim`` or ``prod``.
            max_candidates: Optional fixed candidate budget.

        Returns:
            A deterministic, YAML-serializable mapping.
        """

        if mode not in {"max", "min"}:
            raise JobImportError("mode must be 'max' or 'min'")

        source_path = self._resolve_job_path(job_path)
        source_text = source_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source_text, filename=str(source_path))
        except SyntaxError as e:
            raise JobImportError(f"failed to parse {source_path}: {e}") from e

        index = _ImportIndex.from_tree(tree, source_text)
        job_call = index.first_job_call()
        env_call = index.first_env_call()
        train_script = self._resolve_train_script(source_path, job_call, index.parser_args, source_text)
        train_args = _collect_argparse_args_from_file(train_script) if train_script else {}
        unresolved: List[Dict[str, str]] = []

        if not job_call:
            unresolved.append(_unresolved("job.surface", "no supported Recipe or FedJob constructor was found"))
        if not train_script:
            unresolved.append(_unresolved("job.train_script", "no train_script was found or resolved"))

        metric_name, metric_source, metric_issue = self._resolve_metric(metric, job_call, index.parser_args)
        if metric_issue:
            unresolved.append(metric_issue)

        budget, budget_issues = self._resolve_budget(max_candidates, job_call, env_call, index.parser_args, source_text)
        unresolved.extend(budget_issues)

        allowed_edit_paths = self._allowed_edit_paths(source_path, train_script)
        search_space, search_issues = self._suggest_search_space(index.parser_args, train_args)
        unresolved.extend(search_issues)

        job_payload = {
            "source": self._display_path(source_path),
            "surface": _surface_name(job_call),
            "entrypoint": "main" if _has_main_entrypoint(tree) else "unresolved",
            "allowed_edit_paths": allowed_edit_paths,
        }
        if job_call:
            call_args, call_issues = self._resolved_call_keywords(job_call, index.parser_args, source_text)
            unresolved.extend(call_issues)
            if _is_recipe_call(job_call):
                job_payload.update(
                    {
                        "recipe": job_call.name,
                        "recipe_class": index.imports.get(job_call.name, job_call.full_name),
                        "recipe_args": call_args,
                    }
                )
            else:
                job_payload.update(
                    {
                        "fed_job": job_call.name,
                        "fed_job_class": index.imports.get(job_call.name, job_call.full_name),
                        "fed_job_args": call_args,
                    }
                )
        if train_script:
            job_payload["train_script"] = self._display_path(train_script)

        environment = self._environment_profile(target_env, env_call, index.parser_args, source_text)
        if env_call:
            env_args, env_issues = self._resolved_call_keywords(env_call, index.parser_args, source_text)
            unresolved.extend(env_issues)
            environment["discovered"] = {"name": env_call.name, "args": env_args}

        config = {
            "schema_version": AUTOFL_CONFIG_SCHEMA_VERSION,
            "import": {
                "importer_version": IMPORTER_VERSION,
                "source": self._display_path(source_path),
                "source_sha256": _sha256_text(source_text),
                "confidence": _overall_confidence(unresolved, job_call),
                "support": {
                    "status": "supported" if job_call else "partial",
                    "patterns": _support_patterns(job_call, env_call),
                },
            },
            "job": job_payload,
            "objective": {"metric": metric_name, "mode": mode, "source": metric_source},
            "budget": budget,
            "environment": environment,
            "search_space": {"suggested": search_space},
            "artifacts": {
                "collect": ["logs", "metrics", "job_config", "candidate_diff", "candidate_manifest"],
                "result_root": "autofl_runs",
            },
            "trust_contract": {
                "extracted": _trust_extracted(job_call, env_call, train_script, budget, metric_name, search_space),
                "unresolved": list(unresolved),
                "allowed_edit_paths": allowed_edit_paths,
                "agent_controls": {
                    "must_not_edit_outside_allowed_paths": True,
                    "must_preserve_fixed_training_budget": bool(budget.get("fixed_training_budget")),
                    "must_report_candidate_diffs": True,
                },
            },
            "unresolved": list(unresolved),
        }
        return config

    def dump_yaml(self, config: Dict[str, Any]) -> str:
        """Return deterministic YAML for an imported Auto-FL config."""

        return dump_autofl_yaml(config)

    def _resolve_job_path(self, job_path: str) -> Path:
        path = Path(job_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        if path.is_dir():
            path = path / "job.py"
        if not path.exists():
            raise JobImportError(f"job.py not found: {job_path}")
        if not path.is_file():
            raise JobImportError(f"job path must be a file or directory containing job.py: {job_path}")
        return path.resolve()

    def _display_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.workspace_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _resolve_train_script(
        self,
        source_path: Path,
        job_call: Optional[CallInfo],
        parser_args: Dict[str, ArgSpec],
        source_text: str,
    ) -> Optional[Path]:
        if not job_call:
            return _existing_path(source_path.parent / "client.py")

        train_script_node = job_call.keywords.get("train_script")
        if not train_script_node:
            return _existing_path(source_path.parent / "client.py")

        resolved = _resolve_value(train_script_node, job_call.assignments, parser_args, source_text)
        value = resolved.value
        if isinstance(value, str) and _is_resolved_path_string(resolved):
            return _existing_path((source_path.parent / value).resolve())
        return None

    def _resolve_metric(
        self,
        requested_metric: Optional[str],
        job_call: Optional[CallInfo],
        parser_args: Dict[str, ArgSpec],
    ) -> Tuple[str, str, Optional[Dict[str, str]]]:
        if requested_metric:
            return requested_metric, "user_request", None
        if job_call and "key_metric" in job_call.keywords:
            resolved = _resolve_value(job_call.keywords["key_metric"], job_call.assignments, parser_args, "")
            if isinstance(resolved.value, str) and not resolved.unresolved:
                return resolved.value, resolved.source, None
            return "accuracy", "default", _unresolved("objective.metric", resolved.source)
        if "key_metric" in parser_args and isinstance(parser_args["key_metric"].default, str):
            return parser_args["key_metric"].default, "arg:key_metric", None
        return (
            "accuracy",
            "default",
            _unresolved("objective.metric", "metric defaulted to accuracy; validation metric source is unknown"),
        )

    def _resolve_budget(
        self,
        max_candidates: Optional[int],
        job_call: Optional[CallInfo],
        env_call: Optional[CallInfo],
        parser_args: Dict[str, ArgSpec],
        source_text: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        budget: Dict[str, Any] = {}
        fixed_training_budget: Dict[str, Any] = {}
        unresolved = []
        if max_candidates is not None:
            budget["max_candidates"] = max_candidates

        if job_call:
            for output_key, job_key in (("num_rounds", "num_rounds"), ("min_clients", "min_clients")):
                if job_key in job_call.keywords:
                    resolved = _resolve_value(
                        job_call.keywords[job_key], job_call.assignments, parser_args, source_text
                    )
                    if resolved.unresolved:
                        unresolved.append(_unresolved(f"budget.fixed_training_budget.{output_key}", resolved.source))
                    else:
                        fixed_training_budget[output_key] = resolved.value

        if env_call and env_call.name == "SimEnv" and "num_clients" in env_call.keywords:
            resolved = _resolve_value(env_call.keywords["num_clients"], env_call.assignments, parser_args, source_text)
            if resolved.unresolved:
                unresolved.append(_unresolved("budget.fixed_training_budget.num_clients", resolved.source))
            else:
                fixed_training_budget["num_clients"] = resolved.value

        if fixed_training_budget:
            budget["fixed_training_budget"] = fixed_training_budget
        else:
            unresolved.append(_unresolved("budget.fixed_training_budget", "no fixed training budget was resolved"))
        return budget, unresolved

    def _environment_profile(
        self,
        target_env: Optional[str],
        env_call: Optional[CallInfo],
        parser_args: Dict[str, ArgSpec],
        source_text: str,
    ) -> Dict[str, Any]:
        requested = target_env or (_env_name_to_profile(env_call.name) if env_call else "sim")
        environment: Dict[str, Any] = {"requested": requested, "profiles": {}}
        if env_call and env_call.name == "SimEnv":
            sim_profile: Dict[str, Any] = {}
            if "num_clients" in env_call.keywords:
                resolved = _resolve_value(
                    env_call.keywords["num_clients"], env_call.assignments, parser_args, source_text
                )
                if not resolved.unresolved:
                    sim_profile["num_clients"] = resolved.value
            environment["profiles"]["sim"] = sim_profile
        return environment

    def _allowed_edit_paths(self, source_path: Path, train_script: Optional[Path]) -> List[str]:
        candidates = [source_path]
        if train_script:
            candidates.append(train_script)
        for filename in ("model.py", "mutation_schema.yaml", "requirements.txt"):
            path = source_path.parent / filename
            if path.exists():
                candidates.append(path.resolve())
        return list(dict.fromkeys(self._display_path(path) for path in candidates))

    def _suggest_search_space(
        self,
        job_args: Dict[str, ArgSpec],
        train_args: Dict[str, ArgSpec],
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        suggested = {}
        unresolved = []
        for arg_name, arg_spec in sorted({**job_args, **train_args}.items()):
            if arg_name not in TUNABLE_ARG_NAMES:
                continue
            item = {
                "type": arg_spec.value_type or _type_name(arg_spec.default),
                "default": arg_spec.default,
                "source": f"argparse:{arg_name}",
                "confidence": "low" if arg_spec.default_unresolved else "high",
            }
            if arg_spec.choices:
                item["values"] = arg_spec.choices
            if arg_spec.default_unresolved:
                item["default_source"] = arg_spec.default_source
                item["unresolved"] = True
                unresolved.append(
                    _unresolved(
                        f"search_space.suggested.{arg_name}.default",
                        f"default is dynamic expression: {arg_spec.default}",
                    )
                )
            suggested[arg_name] = item

        if not suggested:
            unresolved.append(_unresolved("search_space.suggested", "no supported tunable argparse fields were found"))
        return suggested, unresolved

    def _resolved_call_keywords(
        self,
        call_info: CallInfo,
        parser_args: Dict[str, ArgSpec],
        source_text: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        resolved = {}
        unresolved = []
        for key, value_node in sorted(call_info.keywords.items()):
            value = _resolve_value(value_node, call_info.assignments, parser_args, source_text)
            resolved[key] = {
                "value": value.value,
                "source": value.source,
                "confidence": value.confidence,
            }
            if value.unresolved:
                unresolved.append(_unresolved(f"job.{call_info.name}.{key}", value.source))
        return resolved, unresolved


def import_job_to_autofl_config(
    job_path: str,
    *,
    workspace_root: Optional[str] = None,
    metric: Optional[str] = None,
    mode: str = "max",
    target_env: Optional[str] = None,
    max_candidates: Optional[int] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for deterministic job import."""

    importer = DeterministicJobImporter(workspace_root=workspace_root)
    return importer.import_job(
        job_path,
        metric=metric,
        mode=mode,
        target_env=target_env,
        max_candidates=max_candidates,
    )


def dump_autofl_yaml(config: Dict[str, Any]) -> str:
    """Return deterministic YAML for an imported Auto-FL config."""

    return yaml.dump(config, Dumper=_NoAliasSafeDumper, sort_keys=False)


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


class _ImportIndex(ast.NodeVisitor):
    def __init__(self, source_text: str):
        self.source_text = source_text
        self.imports: Dict[str, str] = {}
        self.parser_args: Dict[str, ArgSpec] = {}
        self.module_assignments: Dict[str, ast.AST] = {}
        self._local_assignments_stack: List[Dict[str, ast.AST]] = []
        self._function_stack: List[str] = []
        self.job_calls: List[CallInfo] = []
        self.env_calls: List[CallInfo] = []

    @classmethod
    def from_tree(cls, tree: ast.AST, source_text: str) -> "_ImportIndex":
        index = cls(source_text)
        index.visit(tree)
        return index

    def first_job_call(self) -> Optional[CallInfo]:
        return self.job_calls[0] if self.job_calls else None

    def first_env_call(self) -> Optional[CallInfo]:
        return self.env_calls[0] if self.env_calls else None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            for alias in node.names:
                self.imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self._local_assignments_stack.append({})
        self.generic_visit(node)
        self._local_assignments_stack.pop()
        self._function_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._current_assignments()[target.id] = node.value
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value:
            self._current_assignments()[node.target.id] = node.value
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        if _is_argparse_add_argument_call(call_name):
            arg_spec = _arg_spec_from_call(node)
            if arg_spec:
                self.parser_args[arg_spec.name] = arg_spec

        short_name = call_name.split(".")[-1]
        if _is_supported_job_call_name(short_name) or short_name in SUPPORTED_ENV_NAMES:
            call_info = CallInfo(
                name=short_name,
                full_name=call_name,
                keywords={keyword.arg: keyword.value for keyword in node.keywords if keyword.arg},
                assignments=self._resolution_assignments(),
                source=_source_segment(self.source_text, node),
                function_name=self._function_stack[-1] if self._function_stack else None,
            )
            if short_name in SUPPORTED_ENV_NAMES:
                self.env_calls.append(call_info)
            else:
                self.job_calls.append(call_info)
        self.generic_visit(node)

    def _current_assignments(self) -> Dict[str, ast.AST]:
        if self._local_assignments_stack:
            return self._local_assignments_stack[-1]
        return self.module_assignments

    def _resolution_assignments(self) -> Dict[str, ast.AST]:
        assignments = dict(self.module_assignments)
        if self._local_assignments_stack:
            assignments.update(self._local_assignments_stack[-1])
        return assignments


def _collect_argparse_args_from_file(path: Optional[Path]) -> Dict[str, ArgSpec]:
    if not path or not path.exists():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return {}
    return _ImportIndex.from_tree(tree, "").parser_args


def _arg_spec_from_call(node: ast.Call) -> Optional[ArgSpec]:
    flags = []
    for arg in node.args:
        is_literal, value = _literal_value(arg)
        if is_literal and isinstance(value, str) and value.startswith("-"):
            flags.append(value)
    if not flags:
        return None

    keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
    name = _literal_keyword_value(keywords.get("dest")) or _name_from_flags(flags)
    if not name:
        return None

    action = _literal_keyword_value(keywords.get("action"))
    default, default_source, default_unresolved = _arg_default_from_keywords(keywords)
    if not default_unresolved and default is None and action == "store_true":
        default = False
        default_source = "argparse_action"
    elif not default_unresolved and default is None and action == "store_false":
        default = True
        default_source = "argparse_action"

    return ArgSpec(
        name=name,
        flags=tuple(flags),
        default=default,
        default_source=default_source,
        default_unresolved=default_unresolved,
        value_type=_call_name(keywords["type"]) if "type" in keywords else None,
        choices=_literal_sequence(keywords.get("choices")),
        action=action,
    )


def _arg_default_from_keywords(keywords: Dict[str, ast.AST]) -> Tuple[Any, str, bool]:
    if "default" not in keywords:
        return None, "argparse_default", False

    node = keywords["default"]
    is_literal, literal = _literal_value(node)
    if is_literal:
        return literal, "literal", False
    return _unparse(node), "expression", True


def _name_from_flags(flags: Iterable[str]) -> Optional[str]:
    long_flags = [flag for flag in flags if flag.startswith("--")]
    selected = long_flags[0] if long_flags else next(iter(flags), None)
    if not selected:
        return None
    return selected.lstrip("-").replace("-", "_")


def _resolve_value(
    node: ast.AST,
    assignments: Dict[str, ast.AST],
    parser_args: Dict[str, ArgSpec],
    source_text: str,
    seen: Optional[set[str]] = None,
) -> ResolvedValue:
    seen = seen or set()
    is_literal, literal = _literal_value(node)
    if is_literal:
        return ResolvedValue(literal, "literal")

    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args":
        arg_spec = parser_args.get(node.attr)
        if arg_spec:
            return _resolve_arg_default(node.attr, arg_spec)
        return ResolvedValue(None, f"unresolved arg:{node.attr}", "low", True)

    if isinstance(node, ast.Name):
        if node.id in seen:
            return ResolvedValue(None, f"recursive reference:{node.id}", "low", True)
        if node.id in parser_args:
            return _resolve_arg_default(node.id, parser_args[node.id])
        if node.id in assignments:
            return _resolve_value(assignments[node.id], assignments, parser_args, source_text, seen | {node.id})
        return ResolvedValue(node.id, f"name:{node.id}", "low", True)

    if isinstance(node, ast.Call):
        call_name = _call_name(node.func)
        if call_name in {"Path", "pathlib.Path", "os.path.join"}:
            arg_value = _first_resolved_argparse_string(node, assignments, parser_args, source_text)
            if arg_value is not None:
                return arg_value
        return ResolvedValue(_source_segment(source_text, node) or call_name, f"call:{call_name}", "medium")

    return ResolvedValue(_source_segment(source_text, node) or type(node).__name__, "expression", "low", True)


def _resolve_arg_default(name: str, arg_spec: ArgSpec) -> ResolvedValue:
    if arg_spec.default_unresolved:
        return ResolvedValue(arg_spec.default, f"arg:{name}:{arg_spec.default_source}", "low", True)
    return ResolvedValue(arg_spec.default, f"arg:{name}")


def _first_resolved_argparse_string(
    node: ast.Call,
    assignments: Dict[str, ast.AST],
    parser_args: Dict[str, ArgSpec],
    source_text: str,
) -> Optional[ResolvedValue]:
    for arg in node.args:
        resolved = _resolve_value(arg, assignments, parser_args, source_text)
        if resolved.source.startswith("arg:") and isinstance(resolved.value, str):
            return resolved
    return None


def _call_name(node: Optional[ast.AST]) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _literal_value(node: Optional[ast.AST]) -> Tuple[bool, Any]:
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for item in node.elts:
            is_literal, value = _literal_value(item)
            if not is_literal:
                return False, None
            values.append(value)
        return True, values
    return False, None


def _literal_keyword_value(node: Optional[ast.AST]) -> Any:
    is_literal, value = _literal_value(node)
    return value if is_literal else None


def _literal_sequence(node: Optional[ast.AST]) -> Optional[List[Any]]:
    is_literal, value = _literal_value(node)
    return value if is_literal and isinstance(value, list) else None


def _source_segment(source_text: str, node: ast.AST) -> str:
    if not source_text:
        return ""
    return ast.get_source_segment(source_text, node) or ""


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _is_argparse_add_argument_call(call_name: str) -> bool:
    return call_name.endswith(".add_argument")


def _is_supported_job_call_name(name: str) -> bool:
    return name.endswith("Recipe") or name in {"BaseFedJob", "FedJob"}


def _is_recipe_call(call_info: CallInfo) -> bool:
    return call_info.name.endswith("Recipe")


def _surface_name(call_info: Optional[CallInfo]) -> str:
    if not call_info:
        return "unknown"
    return "recipe" if _is_recipe_call(call_info) else "fed_job"


def _env_name_to_profile(env_name: str) -> str:
    return env_name.removesuffix("Env").lower()


def _support_patterns(job_call: Optional[CallInfo], env_call: Optional[CallInfo]) -> List[str]:
    patterns = []
    if job_call:
        patterns.append(f"{_surface_name(job_call)}:{job_call.name}")
    if env_call:
        patterns.append(f"env:{env_call.name}")
    return patterns


def _trust_extracted(
    job_call: Optional[CallInfo],
    env_call: Optional[CallInfo],
    train_script: Optional[Path],
    budget: Dict[str, Any],
    metric_name: str,
    search_space: Dict[str, Any],
) -> List[Dict[str, Any]]:
    extracted = []
    if job_call:
        extracted.append({"field": "job.surface", "value": _surface_name(job_call)})
        extracted.append({"field": f"job.{_surface_name(job_call)}", "value": job_call.name})
    if env_call:
        extracted.append({"field": "environment.discovered", "value": env_call.name})
    if train_script:
        extracted.append({"field": "job.train_script", "value": train_script.name})
    extracted.append({"field": "objective.metric", "value": metric_name})
    if "fixed_training_budget" in budget:
        extracted.append({"field": "budget.fixed_training_budget", "value": budget["fixed_training_budget"]})
    if search_space:
        extracted.append({"field": "search_space.suggested", "value": sorted(search_space)})
    return extracted


def _has_main_entrypoint(tree: ast.AST) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in ast.walk(tree))


def _existing_path(path: Path) -> Optional[Path]:
    return path.resolve() if path.exists() else None


def _is_resolved_path_string(value: ResolvedValue) -> bool:
    return not value.unresolved and (value.source == "literal" or value.source.startswith("arg:"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        return "unknown"
    return type(value).__name__


def _overall_confidence(unresolved: List[Dict[str, str]], job_call: Optional[CallInfo]) -> str:
    if not job_call:
        return "low"
    return "medium" if unresolved else "high"


def _unresolved(field: str, reason: str) -> Dict[str, str]:
    return {"field": field, "reason": reason}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministically import an NVFlare job.py into autofl.yaml")
    parser.add_argument("job", help="NVFlare job.py file or directory containing job.py")
    parser.add_argument("--output", default="autofl.yaml", help="output path for generated autofl.yaml")
    parser.add_argument("--metric", help="requested optimization metric")
    parser.add_argument("--mode", default="max", choices=["max", "min"])
    parser.add_argument("--env", dest="target_env", choices=["sim", "poc", "prod"], help="target environment")
    parser.add_argument("--max-candidates", type=int, help="candidate budget")
    args = parser.parse_args(argv)

    importer = DeterministicJobImporter()
    try:
        config = importer.import_job(
            args.job,
            metric=args.metric,
            mode=args.mode,
            target_env=args.target_env,
            max_candidates=args.max_candidates,
        )
    except JobImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(importer.dump_yaml(config), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
