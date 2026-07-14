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

import ast
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

AUTOFL_CONFIG_SCHEMA_VERSION = "nvflare.autofl.config.v1"
IMPORTER_VERSION = "nvflare-autofl-job-importer/v1"
ALLOWED_CREATE_PATTERNS = ["**/*.py"]

SUPPORTED_ENV_NAMES = {"PocEnv", "ProdEnv", "SimEnv"}
NON_OPTIMIZATION_RECIPE_NAMES = {"FedEvalRecipe", "FedStatsRecipe", "NumpyCrossSiteEvalRecipe"}
UNSUPPORTED_NESTED_RECIPE_NAMES = {"FlowerRecipe"}
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
    function_name: Optional[str] = None


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
        job_args: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Return an ``autofl.yaml``-shaped config for ``job_path``."""

        try:
            return self._import_job(
                job_path,
                metric=metric,
                mode=mode,
                target_env=target_env,
                max_candidates=max_candidates,
                job_args=job_args,
            )
        except JobImportError:
            raise
        except (OSError, UnicodeError, SyntaxError, RecursionError) as e:
            raise JobImportError(f"failed to parse {job_path}: {e}") from e

    def _import_job(
        self,
        job_path: str,
        *,
        metric: Optional[str] = None,
        mode: str = "max",
        target_env: Optional[str] = None,
        max_candidates: Optional[int] = None,
        job_args: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Return an ``autofl.yaml``-shaped config for ``job_path``.

        Args:
            job_path: Path to ``job.py`` or a directory containing ``job.py``.
            metric: Optional optimization metric requested by the user.
            mode: ``max`` or ``min`` objective direction.
            target_env: Optional target environment, such as ``sim`` or ``prod``.
            max_candidates: Optional fixed candidate budget.
            job_args: Optional job CLI arguments used to resolve argparse defaults
                and simple conditional recipe branches.

        Returns:
            A deterministic, YAML-serializable mapping.
        """

        if mode not in {"max", "min"}:
            raise JobImportError("mode must be 'max' or 'min'")

        source_path = self._resolve_job_path(job_path)
        try:
            source_text = source_path.read_text(encoding="utf-8")
            tree = ast.parse(source_text, filename=str(source_path))
            index = _ImportIndex.from_tree(tree, source_text)
        except (OSError, UnicodeError, SyntaxError, RecursionError) as e:
            raise JobImportError(f"failed to parse {source_path}: {e}") from e

        parser_args, reachable_functions = _reachable_parser_args(tree, index, job_args or [])
        job_call = index.first_job_call(reachable_functions)
        env_call = index.first_env_call(reachable_functions)
        train_script = self._resolve_train_script(
            source_path, job_call, index, parser_args, source_text, reachable_functions
        )
        train_args = _collect_argparse_args_from_file(train_script) if train_script else {}
        unresolved: List[Dict[str, str]] = []

        if not job_call:
            unsupported = index.first_unsupported_job_call()
            reason = "no supported Recipe or NVFlare FedJob constructor was found"
            if unsupported:
                reason = (
                    f"constructor {unsupported.full_name} is a local or non-NVFlare Job subclass; "
                    "its contract cannot be imported deterministically"
                )
            unresolved.append(_unresolved("job.surface", reason))
        support_status, support_reason = _support_status(job_call)
        if support_reason and job_call:
            unresolved.append(_unresolved("job.surface", support_reason))
        if not train_script:
            unresolved.append(_unresolved("job.train_script", "no train_script was found or resolved"))

        metric_name, metric_source, metric_issue = self._resolve_metric(metric, job_call, parser_args)
        objective = _objective_contract(metric_name, mode, metric_source)
        if metric_issue:
            unresolved.append(metric_issue)

        budget, budget_issues = self._resolve_budget(max_candidates, job_call, env_call, parser_args, source_text)
        unresolved.extend(budget_issues)

        allowed_edit_paths = self._allowed_edit_paths(source_path, train_script)
        search_space, search_issues = self._suggest_search_space(parser_args, train_args)
        unresolved.extend(search_issues)

        job_payload = {
            "source": self._display_path(source_path),
            "surface": _surface_name(job_call),
            "entrypoint": "main" if _has_main_entrypoint(tree) else "unresolved",
        }
        if job_call:
            call_args, call_issues = self._resolved_call_keywords(job_call, parser_args, source_text)
            unresolved.extend(call_issues)
            if _is_recipe_call(job_call):
                job_payload.update(
                    {
                        "recipe": job_call.name,
                        "recipe_class": job_call.full_name,
                        "recipe_args": call_args,
                    }
                )
            else:
                job_payload.update(
                    {
                        "fed_job": job_call.name,
                        "fed_job_class": job_call.full_name,
                        "fed_job_args": call_args,
                    }
                )
        if train_script:
            job_payload["train_script"] = self._display_path(train_script)

        environment = self._environment_profile(target_env, env_call, parser_args, source_text)
        if env_call:
            env_args, env_issues = self._resolved_call_keywords(env_call, parser_args, source_text)
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
                    "status": support_status,
                    "patterns": _support_patterns(job_call, env_call),
                    **({"reason": support_reason} if support_reason else {}),
                },
            },
            "job": job_payload,
            "objective": objective,
            "budget": budget,
            "environment": environment,
            "search_space": {"suggested": search_space},
            "artifacts": {
                "collect": ["logs", "metrics", "job_config", "candidate_diff", "candidate_manifest"],
                "result_root": "autofl_runs",
            },
            "trust_contract": {
                "extracted": _trust_extracted(
                    job_call,
                    env_call,
                    train_script,
                    budget,
                    objective,
                    search_space,
                ),
                "unresolved": list(unresolved),
                "allowed_edit_paths": allowed_edit_paths,
                "allowed_create_patterns": list(ALLOWED_CREATE_PATTERNS),
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
        index: "_ImportIndex",
        parser_args: Dict[str, ArgSpec],
        source_text: str,
        reachable_functions: Set[str],
    ) -> Optional[Path]:
        if not job_call:
            return None

        train_script_node = job_call.keywords.get("train_script")
        if train_script_node:
            resolved = _resolve_value(train_script_node, job_call.assignments, parser_args, source_text)
            value = resolved.value
            if isinstance(value, str) and _is_resolved_path_string(resolved):
                return self._existing_train_script(source_path, value)
            return None

        script_paths = set()
        for runner_call in index.script_runner_calls:
            if runner_call.function_name is not None and runner_call.function_name not in reachable_functions:
                continue
            script_node = runner_call.keywords.get("script")
            if not script_node:
                continue
            resolved = _resolve_value(script_node, runner_call.assignments, parser_args, source_text)
            if isinstance(resolved.value, str) and _is_resolved_path_string(resolved):
                path = self._existing_train_script(source_path, resolved.value)
                if path:
                    script_paths.add(path)
        return next(iter(script_paths)) if len(script_paths) == 1 else None

    def _existing_train_script(self, source_path: Path, value: str) -> Optional[Path]:
        path = (source_path.parent / value).resolve()
        try:
            path.relative_to(self.workspace_root)
        except ValueError:
            return None
        return _existing_path(path)

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
        if "key_metric" in parser_args:
            metric_arg = parser_args["key_metric"]
            if isinstance(metric_arg.default, str) and not metric_arg.default_unresolved:
                return metric_arg.default, "arg:key_metric", None
            if metric_arg.default_unresolved:
                return "accuracy", "default", _unresolved("objective.metric", metric_arg.default_source)
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
            if "n_clients" in job_call.keywords:
                resolved = _resolve_value(
                    job_call.keywords["n_clients"], job_call.assignments, parser_args, source_text
                )
                if resolved.unresolved:
                    unresolved.append(_unresolved("budget.fixed_training_budget.num_clients", resolved.source))
                else:
                    fixed_training_budget["num_clients"] = resolved.value

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
                reason = f"default is dynamic expression: {arg_spec.default}"
                if arg_spec.default_source == "required_positional":
                    reason = "required positional argument has no deterministic default"
                unresolved.append(
                    _unresolved(
                        f"search_space.suggested.{arg_name}.default",
                        reason,
                    )
                )
            if not arg_spec.flags:
                item["confidence"] = "low"
                item["mutable_via_run_args"] = False
                item["unresolved"] = True
                unresolved.append(
                    _unresolved(
                        f"search_space.suggested.{arg_name}.interface",
                        "positional argparse fields require source edits; candidate run_args support long options only",
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
            if value.unresolved and key != "model":
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
    job_args: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for deterministic job import."""

    importer = DeterministicJobImporter(workspace_root=workspace_root)
    return importer.import_job(
        job_path,
        metric=metric,
        mode=mode,
        target_env=target_env,
        max_candidates=max_candidates,
        job_args=job_args,
    )


def inspect_job_cli_flags(job_path: str, job_args: Optional[Sequence[str]] = None) -> List[str]:
    """Return long argparse flags reachable from a job's deterministic entry path."""

    path = Path(job_path).resolve()
    try:
        source_text = path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(path))
        index = _ImportIndex.from_tree(tree, source_text)
    except (OSError, UnicodeError, SyntaxError, RecursionError) as e:
        raise JobImportError(f"failed to parse {path}: {e}") from e
    parser_args, _ = _reachable_parser_args(tree, index, job_args or [])
    return sorted({flag for spec in parser_args.values() for flag in spec.flags if flag.startswith("--")})


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
        self.parser_arg_definitions: Dict[str, List[ArgSpec]] = {}
        self.module_assignments: Dict[str, ast.AST] = {}
        self._local_assignments_stack: List[Dict[str, ast.AST]] = []
        self._function_stack: List[str] = []
        self._argparse_parser_names: Set[Tuple[Optional[str], str]] = set()
        self._argparse_subparser_names: Set[Tuple[Optional[str], str]] = set()
        self.job_calls: List[CallInfo] = []
        self.unsupported_job_calls: List[CallInfo] = []
        self.script_runner_calls: List[CallInfo] = []
        self.env_calls: List[CallInfo] = []

    @classmethod
    def from_tree(cls, tree: ast.AST, source_text: str) -> "_ImportIndex":
        index = cls(source_text)
        index.visit(tree)
        return index

    def first_job_call(self, reachable_functions: Optional[Set[str]] = None) -> Optional[CallInfo]:
        return _first_reachable_call(self.job_calls, reachable_functions)

    def first_env_call(self, reachable_functions: Optional[Set[str]] = None) -> Optional[CallInfo]:
        return _first_reachable_call(self.env_calls, reachable_functions)

    def first_unsupported_job_call(self) -> Optional[CallInfo]:
        return self.unsupported_job_calls[0] if self.unsupported_job_calls else None

    def parser_args(self, reachable_functions: Optional[Set[str]] = None) -> Dict[str, ArgSpec]:
        definitions = {}
        for name, specs in self.parser_arg_definitions.items():
            selected = [
                spec
                for spec in specs
                if reachable_functions is None
                or spec.function_name is None
                or spec.function_name in reachable_functions
            ]
            if selected:
                definitions[name] = selected
        return _collapse_arg_specs(definitions)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            for alias in node.names:
                self.imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self._local_assignments_stack.append({})
        self.generic_visit(node)
        self._local_assignments_stack.pop()
        self._function_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._current_assignments()[target.id] = node.value
                self._track_argparse_assignment(target.id, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value:
            self._current_assignments()[node.target.id] = node.value
            self._track_argparse_assignment(node.target.id, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            # The result depends on runtime state. Preserve the expression so
            # downstream resolution marks the value as unresolved instead of
            # reusing an earlier static assignment.
            self._current_assignments()[node.target.id] = node
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        if self._is_argparse_add_argument_call(call_name):
            arg_spec = _arg_spec_from_call(node, self._function_stack[-1] if self._function_stack else None)
            if arg_spec:
                self.parser_arg_definitions.setdefault(arg_spec.name, []).append(arg_spec)

        resolved_name = self._resolve_import_path(call_name)
        short_name = resolved_name.split(".")[-1]
        is_supported_job = _is_supported_job_call(short_name, resolved_name)
        is_environment = short_name in SUPPORTED_ENV_NAMES
        is_script_runner = short_name == "ScriptRunner" and resolved_name.startswith("nvflare.")
        is_unsupported_job = short_name.endswith("Job") and not is_supported_job and short_name != "Job"
        if is_supported_job or is_environment or is_script_runner or is_unsupported_job:
            call_info = CallInfo(
                name=short_name,
                full_name=resolved_name,
                keywords={keyword.arg: keyword.value for keyword in node.keywords if keyword.arg},
                assignments=self._resolution_assignments(),
                source=_source_segment(self.source_text, node),
                function_name=self._function_stack[-1] if self._function_stack else None,
            )
            if is_environment:
                self.env_calls.append(call_info)
            elif is_script_runner:
                self.script_runner_calls.append(call_info)
            elif is_supported_job:
                self.job_calls.append(call_info)
            else:
                self.unsupported_job_calls.append(call_info)
        self.generic_visit(node)

    def _track_argparse_assignment(self, target: str, value: ast.AST) -> None:
        if not isinstance(value, ast.Call):
            return
        call_name = self._resolve_import_path(_call_name(value.func))
        scope = self._function_stack[-1] if self._function_stack else None
        if call_name in {"argparse.ArgumentParser", "argparse._ArgumentGroup"}:
            self._argparse_parser_names.add((scope, target))
            return
        owner, _, method = call_name.rpartition(".")
        if method == "add_subparsers" and self._is_known_argparse_name(owner, self._argparse_parser_names):
            self._argparse_subparser_names.add((scope, target))
        elif method == "add_parser" and self._is_known_argparse_name(owner, self._argparse_subparser_names):
            self._argparse_parser_names.add((scope, target))

    def _is_argparse_add_argument_call(self, call_name: str) -> bool:
        owner, _, method = call_name.rpartition(".")
        return method == "add_argument" and self._is_known_argparse_name(owner, self._argparse_parser_names)

    def _is_known_argparse_name(self, name: str, names: Set[Tuple[Optional[str], str]]) -> bool:
        scope = self._function_stack[-1] if self._function_stack else None
        return (scope, name) in names or (None, name) in names

    def _resolve_import_path(self, call_name: str) -> str:
        if not call_name:
            return call_name
        root, separator, remainder = call_name.partition(".")
        imported = self.imports.get(root)
        if not imported:
            return call_name
        return f"{imported}.{remainder}" if separator else imported

    def _current_assignments(self) -> Dict[str, ast.AST]:
        if self._local_assignments_stack:
            return self._local_assignments_stack[-1]
        return self.module_assignments

    def _resolution_assignments(self) -> Dict[str, ast.AST]:
        assignments = dict(self.module_assignments)
        if self._local_assignments_stack:
            assignments.update(self._local_assignments_stack[-1])
        return assignments


def _first_reachable_call(calls: Sequence[CallInfo], reachable_functions: Optional[Set[str]]) -> Optional[CallInfo]:
    if not calls:
        return None
    if reachable_functions is None:
        return calls[0]
    reachable_calls = [
        call for call in calls if call.function_name is None or call.function_name in reachable_functions
    ]
    return reachable_calls[0] if reachable_calls else None


def _arg_spec_signature(spec: ArgSpec) -> Tuple[Any, ...]:
    return (
        spec.default,
        spec.default_source,
        spec.default_unresolved,
        spec.value_type,
        tuple(spec.choices) if spec.choices is not None else None,
        spec.action,
    )


def _collapse_arg_specs(definitions: Dict[str, List[ArgSpec]]) -> Dict[str, ArgSpec]:
    collapsed = {}
    for name, specs in definitions.items():
        flags = tuple(dict.fromkeys(flag for spec in specs for flag in spec.flags))
        first = specs[0]
        if all(_arg_spec_signature(spec) == _arg_spec_signature(first) for spec in specs[1:]):
            collapsed[name] = ArgSpec(
                name=name,
                flags=flags,
                default=first.default,
                default_source=first.default_source,
                default_unresolved=first.default_unresolved,
                value_type=first.value_type,
                choices=first.choices,
                action=first.action,
                function_name=first.function_name,
            )
            continue
        same_value_type = all(spec.value_type == first.value_type for spec in specs[1:])
        same_action = all(spec.action == first.action for spec in specs[1:])
        same_choices = all(spec.choices == first.choices for spec in specs[1:])
        collapsed[name] = ArgSpec(
            name=name,
            flags=flags,
            default=None,
            default_source="conflicting argparse definitions",
            default_unresolved=True,
            value_type=first.value_type if same_value_type else None,
            choices=first.choices if same_choices else None,
            action=first.action if same_action else None,
        )
    return collapsed


def _reachable_parser_args(
    tree: ast.AST, index: _ImportIndex, job_args: Sequence[str]
) -> Tuple[Dict[str, ArgSpec], Set[str]]:
    initial = _parser_args_with_cli_overrides(index.parser_args(), job_args)
    reachable = _reachable_function_names(tree, initial)
    parser_args = _parser_args_with_cli_overrides(index.parser_args(reachable), job_args)
    reachable = _reachable_function_names(tree, parser_args)
    return _parser_args_with_cli_overrides(index.parser_args(reachable), job_args), reachable


def _parser_args_with_cli_overrides(parser_args: Dict[str, ArgSpec], job_args: Sequence[str]) -> Dict[str, ArgSpec]:
    resolved = dict(parser_args)
    flags = {flag: spec for spec in parser_args.values() for flag in spec.flags}
    index = 0
    while index < len(job_args):
        token = job_args[index]
        flag, separator, inline_value = token.partition("=")
        spec = flags.get(flag)
        if spec is None:
            index += 1
            continue

        if spec.action in {"store_true", "store_false"}:
            value = spec.action == "store_true"
            consumed = 1
        elif separator:
            value = _coerce_cli_value(inline_value, spec)
            consumed = 1
        elif index + 1 < len(job_args):
            value = _coerce_cli_value(job_args[index + 1], spec)
            consumed = 2
        else:
            index += 1
            continue

        resolved[spec.name] = ArgSpec(
            name=spec.name,
            flags=spec.flags,
            default=value,
            default_source="job_cli_arg",
            value_type=spec.value_type,
            choices=spec.choices,
            action=spec.action,
        )
        index += consumed
    return resolved


def _coerce_cli_value(value: str, spec: ArgSpec) -> Any:
    value_type = (spec.value_type or "").split(".")[-1]
    try:
        if value_type == "int":
            return int(value)
        if value_type == "float":
            return float(value)
    except ValueError:
        return value
    return value


def _reachable_function_names(tree: ast.AST, parser_args: Dict[str, ArgSpec]) -> Set[str]:
    functions = {
        node.name: node
        for node in getattr(tree, "body", [])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "main" not in functions:
        return set(functions)

    arg_values = {name: spec.default for name, spec in parser_args.items() if not spec.default_unresolved}
    reachable = set()
    pending = ["main"]
    while pending:
        name = pending.pop()
        if name in reachable or name not in functions:
            continue
        reachable.add(name)
        called = _called_functions(functions[name].body, arg_values)
        pending.extend(sorted((called & functions.keys()) - reachable, reverse=True))
    return reachable


def _called_functions(statements: Sequence[ast.stmt], arg_values: Dict[str, Any]) -> Set[str]:
    calls: Set[str] = set()
    for statement in statements:
        if isinstance(statement, ast.If):
            condition = _static_condition_value(statement.test, arg_values)
            if condition is True:
                calls.update(_called_functions(statement.body, arg_values))
            elif condition is False:
                calls.update(_called_functions(statement.orelse, arg_values))
            else:
                calls.update(_called_functions(statement.body, arg_values))
                calls.update(_called_functions(statement.orelse, arg_values))
            continue
        collector = _DirectCallCollector()
        collector.visit(statement)
        calls.update(collector.names)
    return calls


class _DirectCallCollector(ast.NodeVisitor):
    def __init__(self):
        self.names: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.names.add(node.func.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _static_condition_value(node: ast.AST, arg_values: Dict[str, Any]) -> Optional[bool]:
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        left_known, left = _static_condition_operand(node.left, arg_values)
        right_known, right = _static_condition_operand(node.comparators[0], arg_values)
        if not left_known or not right_known:
            return None
        if isinstance(node.ops[0], ast.Eq):
            return left == right
        if isinstance(node.ops[0], ast.NotEq):
            return left != right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _static_condition_value(node.operand, arg_values)
        return None if value is None else not value
    return None


def _static_condition_operand(node: ast.AST, arg_values: Dict[str, Any]) -> Tuple[bool, Any]:
    is_literal, value = _literal_value(node)
    if is_literal:
        return True, value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args":
        if node.attr in arg_values:
            return True, arg_values[node.attr]
    return False, None


def _collect_argparse_args_from_file(path: Optional[Path]) -> Dict[str, ArgSpec]:
    if not path or not path.exists():
        return {}
    try:
        source_text = path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(path))
        index = _ImportIndex.from_tree(tree, source_text)
    except (OSError, UnicodeError, SyntaxError, RecursionError):
        return {}
    parser_args, _ = _reachable_parser_args(tree, index, ())
    return parser_args


def _arg_spec_from_call(node: ast.Call, function_name: Optional[str]) -> Optional[ArgSpec]:
    flags = []
    positional_names = []
    for arg in node.args:
        is_literal, value = _literal_value(arg)
        if not is_literal or not isinstance(value, str):
            continue
        if value.startswith("-"):
            flags.append(value)
        else:
            positional_names.append(value)
    if not flags and len(positional_names) != 1:
        return None

    keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
    name = _literal_keyword_value(keywords.get("dest")) or (_name_from_flags(flags) if flags else positional_names[0])
    if not name:
        return None

    action = _literal_keyword_value(keywords.get("action"))
    default, default_source, default_unresolved = _arg_default_from_keywords(keywords)
    if not flags and "default" not in keywords:
        default_source = "required_positional"
        default_unresolved = True
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
        function_name=function_name,
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
            path_value = _resolve_path_call(node, call_name, assignments, parser_args, source_text)
            if path_value is not None:
                return path_value
        return ResolvedValue(_source_segment(source_text, node) or call_name, f"call:{call_name}", "low", True)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _resolve_value(node.left, assignments, parser_args, source_text)
        right = _resolve_value(node.right, assignments, parser_args, source_text)
        if all(isinstance(value.value, str) and not value.unresolved for value in (left, right)):
            confidence = "low" if "low" in {left.confidence, right.confidence} else "high"
            return ResolvedValue(
                str(Path(left.value) / right.value),
                f"path:{left.source}+{right.source}",
                confidence,
            )

    return ResolvedValue(_source_segment(source_text, node) or type(node).__name__, "expression", "low", True)


def _resolve_arg_default(name: str, arg_spec: ArgSpec) -> ResolvedValue:
    if arg_spec.default_unresolved:
        return ResolvedValue(arg_spec.default, f"arg:{name}:{arg_spec.default_source}", "low", True)
    return ResolvedValue(arg_spec.default, f"arg:{name}")


def _resolve_path_call(
    node: ast.Call,
    call_name: str,
    assignments: Dict[str, ast.AST],
    parser_args: Dict[str, ArgSpec],
    source_text: str,
) -> Optional[ResolvedValue]:
    resolved_parts = []
    for arg in node.args:
        resolved = _resolve_value(arg, assignments, parser_args, source_text)
        if resolved.unresolved or not isinstance(resolved.value, str):
            return None
        resolved_parts.append(resolved)
    if not resolved_parts:
        return None

    values = [part.value for part in resolved_parts]
    value = os.path.join(*values) if call_name == "os.path.join" else str(Path(*values))
    confidence = "low" if any(part.confidence == "low" for part in resolved_parts) else "high"
    sources = "+".join(part.source for part in resolved_parts)
    return ResolvedValue(value, f"path:{sources}", confidence)


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


def _is_supported_job_call(name: str, resolved_name: str) -> bool:
    if not resolved_name.startswith("nvflare."):
        return False
    if name.endswith("Recipe"):
        return True
    return name.endswith("Job") and name != "Job"


def _is_recipe_call(call_info: CallInfo) -> bool:
    return call_info.name.endswith("Recipe")


def _surface_name(call_info: Optional[CallInfo]) -> str:
    if not call_info:
        return "unknown"
    return "recipe" if _is_recipe_call(call_info) else "fed_job"


def _support_status(job_call: Optional[CallInfo]) -> Tuple[str, Optional[str]]:
    if not job_call:
        return "partial", "no supported Recipe or NVFlare FedJob constructor was found"
    if job_call.name in NON_OPTIMIZATION_RECIPE_NAMES:
        return (
            "partial",
            f"{job_call.name} is evaluation/statistics-only and has no training loop for Auto-FL optimization",
        )
    if job_call.name in UNSUPPORTED_NESTED_RECIPE_NAMES:
        return (
            "partial",
            f"{job_call.name} wraps a nested application whose training source cannot be imported deterministically",
        )
    return "supported", None


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
    objective: Dict[str, Any],
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
    extracted.append({"field": "objective.metric", "value": objective["metric"]})
    extracted.append({"field": "objective.optimization_metric", "value": objective["optimization_metric"]})
    if "fixed_training_budget" in budget:
        extracted.append({"field": "budget.fixed_training_budget", "value": budget["fixed_training_budget"]})
    if search_space:
        extracted.append({"field": "search_space.suggested", "value": sorted(search_space)})
    return extracted


def _objective_contract(metric_name: str, mode: str, source: str) -> Dict[str, Any]:
    return {
        "metric": metric_name,
        "requested_metric": metric_name,
        "optimization_metric": metric_name,
        "metric_extraction_order": [metric_name],
        "mode": mode,
        "metric_contract_source": source,
    }


def _has_main_entrypoint(tree: ast.AST) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main" for node in ast.walk(tree)
    )


def _existing_path(path: Path) -> Optional[Path]:
    return path.resolve() if path.exists() else None


def _is_resolved_path_string(value: ResolvedValue) -> bool:
    return not value.unresolved and isinstance(value.value, str)


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
