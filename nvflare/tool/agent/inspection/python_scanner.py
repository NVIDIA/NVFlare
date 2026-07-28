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

"""Framework-neutral Python AST traversal for agent inspection."""

import ast
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from nvflare.tool.agent import frameworks
from nvflare.tool.agent.frameworks.base import DetectContext
from nvflare.tool.agent.inspection.models import SECRET_NAME_PATTERN, looks_like_absolute_path, redact_literal
from nvflare.tool.agent.inspection.project import _resolve_import_from_module

MAX_EVIDENCE_COLLECT = 10000


class _PythonInspector(ast.NodeVisitor):
    def __init__(self, path: Path, rel_path: str, state):
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

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        end_lineno = getattr(node, "end_lineno", None) or node.lineno
        self.state.class_body_ranges.setdefault(self.rel_path, []).append((node.lineno, end_lineno))
        for decorator in node.decorator_list:
            self.visit(decorator)
        for type_param in getattr(node, "type_params", []):
            self.visit(type_param)

        base_names = []
        for base in node.bases:
            base_name = _symbol_name(base)
            if base_name:
                base_names.append(base_name)
                for detector in self._detectors:
                    detector.on_class_base(
                        base_name,
                        node.lineno,
                        self._detector_states[detector.name],
                        self._ctx,
                        tuple(self._scope_stack),
                    )
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

        self._visit_body_in_scope(node, "class", predeclare_locals=False)
        for detector in self._detectors:
            detector.on_class_definition(
                node.name,
                base_names,
                node.lineno,
                self._detector_states[detector.name],
                self._ctx,
                tuple(self._scope_stack),
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, "async-function")

    def _visit_function(self, node: ast.AST, kind: str) -> None:
        self._visit_function_definition_expressions(node)
        self._record_binding_names([node.name], node.lineno)
        self._visit_body_in_scope(node, kind)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_argument_defaults(node.args)
        self._visit_expression_in_scope(node, "lambda", node.body)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._visit_assignment_target(node.target, getattr(node, "lineno", None))
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, "list-comprehension")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, "set-comprehension")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, "dict-comprehension")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, "generator-expression")

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for case in node.cases:
            self.visit(case.pattern)
            collector = _LexicalBindingCollector()
            collector.visit(case.pattern)
            self._record_binding_names(sorted(collector.local_names), getattr(case.pattern, "lineno", None))
            if case.guard:
                self.visit(case.guard)
            for statement in case.body:
                self.visit(statement)

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
        call_name, value_info = self._assignment_value_info(node.value)
        self.visit(node.value)
        for target in node.targets:
            self._visit_assignment_target(
                target,
                getattr(node, "lineno", None),
                call_name,
                value_info,
            )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._inspect_secret_assignment([node.target], node.value, getattr(node, "lineno", None))
        call_name, value_info = self._assignment_value_info(node.value)
        if node.value:
            self.visit(node.value)
        self._visit_assignment_target(
            node.target,
            getattr(node, "lineno", None),
            call_name,
            value_info,
        )
        self.visit(node.annotation)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        self._record_binding_names(_assignment_target_names([node.target]), getattr(node, "lineno", None))

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._visit_assignment_target(
            node.target,
            getattr(node, "lineno", None),
        )

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._visit_assignment_target(item.optional_vars, getattr(node, "lineno", None))
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

    def _visit_assignment_target(
        self,
        target: ast.AST,
        lineno: Optional[int],
        call_name: Optional[str] = None,
        value_info: Optional[dict[str, object]] = None,
    ) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._visit_assignment_target(item, lineno, call_name, value_info)
            return
        if isinstance(target, ast.Starred):
            self._visit_assignment_target(target.value, lineno, call_name, value_info)
            return
        self.visit(target)
        target_names = _assignment_target_names([target])
        self._dispatch_assignment(target_names, call_name, lineno, value_info)

    def _record_binding_names(self, target_names: list[str], lineno: Optional[int]) -> None:
        self._dispatch_assignment(target_names, None, lineno)

    def _classify_assignment_value(self, call_name: Optional[str]) -> dict[str, object]:
        scope = tuple(self._scope_stack)
        return {
            detector.name: detector.classify_assignment_value(
                call_name,
                self._detector_states[detector.name],
                scope,
            )
            for detector in self._detectors
        }

    def _assignment_value_info(self, value: Optional[ast.AST]) -> tuple[Optional[str], dict[str, object]]:
        call_name = _call_name(value.func) if isinstance(value, ast.Call) else None
        return call_name, self._classify_assignment_value(call_name)

    def _dispatch_assignment(
        self,
        target_names: list[str],
        call_name: Optional[str],
        lineno: Optional[int],
        value_info: Optional[dict[str, object]] = None,
    ) -> None:
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
                value_info.get(detector.name) if value_info is not None else None,
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

    def _visit_function_definition_expressions(self, node: ast.AST) -> None:
        for decorator in getattr(node, "decorator_list", []):
            self.visit(decorator)
        for type_param in getattr(node, "type_params", []):
            self.visit(type_param)
        arguments = getattr(node, "args", None)
        if isinstance(arguments, ast.arguments):
            self._visit_argument_defaults(arguments)
            self._visit_argument_annotations(arguments)
        returns = getattr(node, "returns", None)
        if returns:
            self.visit(returns)

    def _visit_argument_defaults(self, arguments: ast.arguments) -> None:
        for default in arguments.defaults:
            self.visit(default)
        for default in arguments.kw_defaults:
            if default:
                self.visit(default)

    def _visit_argument_annotations(self, arguments: ast.arguments) -> None:
        parameters = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
        if arguments.vararg:
            parameters.append(arguments.vararg)
        if arguments.kwarg:
            parameters.append(arguments.kwarg)
        for parameter in parameters:
            if parameter.annotation:
                self.visit(parameter.annotation)

    def _visit_body_in_scope(self, node: ast.AST, kind: str, *, predeclare_locals: bool = True) -> None:
        with self._inspection_scope(node, kind, predeclare_locals=predeclare_locals):
            for statement in getattr(node, "body", []):
                self.visit(statement)

    def _visit_expression_in_scope(self, node: ast.AST, kind: str, expression: ast.AST) -> None:
        with self._inspection_scope(node, kind, name="<anonymous>"):
            self.visit(expression)

    @contextmanager
    def _inspection_scope(
        self,
        node: ast.AST,
        kind: str,
        *,
        name: Optional[str] = None,
        predeclare_locals: bool = True,
    ) -> Iterator[None]:
        scope_name = name if name is not None else getattr(node, "name", "<anonymous>")
        self._scope_stack.append(f"{kind}:{scope_name}:{getattr(node, 'lineno', 0)}")
        try:
            scope = tuple(self._scope_stack)
            local_names, global_names, nonlocal_names = _lexical_bindings(node)
            if not predeclare_locals:
                local_names = set()
            for detector in self._detectors:
                detector.on_scope(
                    scope,
                    local_names,
                    global_names,
                    nonlocal_names,
                    self._detector_states[detector.name],
                    self._ctx,
                )
            yield
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
        if looks_like_absolute_path(value):
            self.state.absolute_path_findings.append(
                {
                    "code": "ABSOLUTE_DATA_PATH",
                    "severity": "warning",
                    "file": self.rel_path,
                    "line": lineno,
                    "pattern_type": "absolute_path_literal",
                    "value": redact_literal(value, self.state.redact),
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

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.pattern:
            self.visit(node.pattern)
        if node.name:
            self.local_names.add(node.name)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.local_names.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        for pattern in node.patterns:
            self.visit(pattern)
        if node.rest:
            self.local_names.add(node.rest)


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
