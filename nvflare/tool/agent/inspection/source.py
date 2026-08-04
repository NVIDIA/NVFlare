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

"""One-pass recognition of the closed V3 training-owner forms."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from nvflare.tool.agent.inspection.types import Dependency, FactRecord, FileFacts, SourceScan

OPTIMIZERS = {"Adagrad", "Adam", "AdamW", "RMSprop", "SGD"}
HF_CONFIGS = {
    "transformers.Seq2SeqTrainingArguments",
    "transformers.TrainingArguments",
    "trl.SFTConfig",
}
OWNER_METHODS = {"huggingface": {"train"}, "lightning": {"fit", "test", "validate"}}
SUPPORTING_METHODS = {
    "huggingface": {"evaluate", "predict"},
    "lightning": {"predict"},
    "pytorch": {"backward"},
}
SECRET_NAME = re.compile(r"api[_-]?key|secret|token|password|passwd|credential|access[_-]?key", re.I)
TRY_NODES = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())
CLASS_BODY_COMPOUND_NODES = (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Match) + TRY_NODES
EXCLUDED_EXPRESSION_SCOPES = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


@dataclass
class Bindings:
    path: str = "."
    redact: bool = True
    findings: list[dict] = field(default_factory=list)
    symbols: dict[str, str] = field(default_factory=dict)
    subclasses: dict[str, str] = field(default_factory=dict)
    instances: dict[str, str] = field(default_factory=dict)
    optimizers: set[str] = field(default_factory=set)
    scalers: set[str] = field(default_factory=set)
    dependencies: dict[str, tuple[Dependency, ...]] = field(default_factory=dict)
    inherited_names: set[str] = field(default_factory=set)
    client_names_seen: dict[str, str | None] = field(default_factory=dict)
    uncertain_client_names: set[str] = field(default_factory=set)
    hf_config_constructed: bool = False
    framework_context_seen: set[str] = field(default_factory=set)
    outer_symbols: dict[str, str] = field(default_factory=dict)
    outer_symbol_kills: set[str] = field(default_factory=set)
    outer_names: set[str] = field(default_factory=set)

    def child_scope(self) -> Bindings:
        return Bindings(
            path=self.path,
            redact=self.redact,
            findings=self.findings,
            symbols=dict(self.symbols),
            subclasses=dict(self.subclasses),
            client_names_seen=dict(self.client_names_seen),
            uncertain_client_names=set(self.uncertain_client_names),
            inherited_names=set(self.symbols) | set(self.subclasses),
        )

    def nested_scope(self) -> Bindings:
        outer_symbols = dict(self.outer_symbols)
        for name in self.outer_symbol_kills:
            outer_symbols.pop(name, None)
        outer_symbols.update(self.symbols)
        return Bindings(
            path=self.path,
            redact=self.redact,
            findings=self.findings,
            outer_symbols=outer_symbols,
            client_names_seen=dict(self.client_names_seen),
            uncertain_client_names=set(self.uncertain_client_names),
        )

    def branch_copy(self) -> Bindings:
        return Bindings(
            path=self.path,
            redact=self.redact,
            findings=self.findings,
            symbols=dict(self.symbols),
            subclasses=dict(self.subclasses),
            instances=dict(self.instances),
            optimizers=set(self.optimizers),
            scalers=set(self.scalers),
            dependencies=dict(self.dependencies),
            inherited_names=set(self.inherited_names),
            client_names_seen=dict(self.client_names_seen),
            uncertain_client_names=set(self.uncertain_client_names),
            hf_config_constructed=self.hf_config_constructed,
            framework_context_seen=set(self.framework_context_seen),
            outer_symbols=dict(self.outer_symbols),
            outer_symbol_kills=set(self.outer_symbol_kills),
            outer_names=self.outer_names,
        )

    def rebind(self, name: str) -> None:
        if name in self.outer_symbols:
            self.outer_symbol_kills.add(name)
        self.symbols.pop(name, None)
        self.subclasses.pop(name, None)
        self.instances.pop(name, None)
        self.optimizers.discard(name)
        self.scalers.discard(name)
        self.dependencies.pop(name, None)
        self.inherited_names.discard(name)
        self.uncertain_client_names.discard(name)


def analyze_tree(
    tree: ast.Module,
    path: str,
    *,
    is_job_py: bool,
    findings: list[dict],
    redact: bool = True,
) -> FileFacts:
    facts = FileFacts(path=path, is_job_py=is_job_py)
    bindings = Bindings(path=path, redact=redact, findings=findings)
    _analyze_body(tree.body, bindings, facts, function_depth=0)
    _remove_changed_module_records(facts, bindings)
    return facts


def ownership(scan: SourceScan) -> dict:
    owners = [(path, framework, line) for path, facts in scan.facts.items() for framework, line, _ in facts.owners]
    unresolved = [
        (path, framework, line) for path, facts in scan.facts.items() for framework, line, _ in facts.unresolved
    ]
    supporting = [
        (path, framework, line) for path, facts in scan.facts.items() for framework, line, _ in facts.supporting
    ]
    owner_frameworks = {item[1] for item in owners}
    unresolved_frameworks = {item[1] for item in unresolved}
    supporting_frameworks = {item[1] for item in supporting}

    if len(owner_frameworks) > 1:
        state, framework, reason = "conflicting", None, "multiple_direct_owners"
    elif owner_frameworks and unresolved_frameworks - owner_frameworks:
        state, framework, reason = "unresolved", None, "unsupported_indirection"
    elif not scan.complete:
        state, framework, reason = "unresolved", None, "incomplete_scan"
    elif len(owner_frameworks) == 1:
        state, framework = "clear", next(iter(owner_frameworks))
        reason = "multiple_direct_owners" if len(owners) > 1 else "direct_owner"
    elif unresolved:
        state, framework, reason = "unresolved", None, "unsupported_indirection"
    elif supporting:
        state, framework, reason = "unresolved", None, "supporting_only"
    else:
        state, framework, reason = "none", None, "no_training_lifecycle"

    owner_files = sorted({path for path, _, _ in owners})
    evidence = [
        {"file": path, "line": line, "kind": kind, "framework": candidate}
        for kind, records in (
            ("direct_owner", owners),
            ("unresolved_owner_attempt", unresolved),
            ("supporting_lifecycle", supporting),
        )
        for path, candidate, line in records
    ]
    return {
        "state": state,
        "complete": scan.complete,
        "framework": framework,
        "candidate_frameworks": sorted(owner_frameworks | unresolved_frameworks | supporting_frameworks),
        "owner_file": owner_files[0] if state == "clear" and len(owner_files) == 1 else None,
        "candidate_files": sorted({item[0] for item in (*owners, *unresolved, *supporting)}),
        "reason": reason,
        "evidence": evidence,
    }


def _analyze_body(
    body: list[ast.stmt],
    bindings: Bindings,
    facts: FileFacts,
    *,
    function_depth: int,
    module_bindings: Bindings | None = None,
    conditional_imports: bool = False,
) -> tuple[bool, set[str]]:
    if module_bindings is None:
        module_bindings = bindings
    contains_yield = False
    bound_names: set[str] = set()
    for statement in body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            _record_import(statement, bindings, facts, uncertain=conditional_imports)
            bound_names.update(_import_names(statement))
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner_start = len(facts.owners)
            supporting_start = len(facts.supporting)
            unresolved_start = len(facts.unresolved)
            for expression in (*statement.decorator_list, *statement.args.defaults, *statement.args.kw_defaults):
                if expression is not None:
                    contains_yield |= _record_expression_calls(expression, bindings, facts, bound_names)
            bindings.rebind(statement.name)
            bound_names.add(statement.name)
            child = bindings.child_scope() if function_depth == 0 else bindings.nested_scope()
            _invalidate(child, _argument_names(statement.args), remember_framework=False)
            _record_parameter_bindings(statement.args, child)
            child_yield, child_bound_names = _analyze_body(
                statement.body,
                child,
                facts,
                function_depth=function_depth + 1,
                module_bindings=module_bindings,
                conditional_imports=False,
            )
            _remove_shadowed_records(
                facts,
                _shadowed_names(bindings, child, child_bound_names),
                owner_start=owner_start,
                supporting_start=supporting_start,
                unresolved_start=unresolved_start,
            )
            if statement.decorator_list or child_yield:
                _move_owners_to_unresolved(facts, owner_start)
        elif isinstance(statement, (ast.Global, ast.Nonlocal)):
            bindings.outer_names.update(statement.names)
        elif isinstance(statement, ast.ClassDef):
            owner_start = len(facts.owners)
            for decorator in statement.decorator_list:
                contains_yield |= _record_expression_calls(decorator, bindings, facts, bound_names)
            _record_class(statement, bindings, facts)
            bound_names.add(statement.name)
            static_bindings = bindings.branch_copy()
            for item in statement.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _analyze_direct_method(item, module_bindings, facts)
                else:
                    _record_class_body_facts(item, static_bindings, facts)
            if statement.decorator_list:
                _move_owners_to_unresolved(facts, owner_start)
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            contains_yield |= _record_assignment(statement, bindings, facts, bound_names)
        elif isinstance(statement, ast.AugAssign):
            contains_yield |= _record_expression_calls(statement.value, bindings, facts, bound_names)
            names = _target_names(statement.target)
            _invalidate(bindings, names, remember_framework=False)
            bound_names.update(names)
        elif isinstance(statement, ast.Delete):
            names = set().union(*(_target_names(target) for target in statement.targets))
            _invalidate(bindings, names, remember_framework=False)
            bound_names.update(names)
        elif isinstance(statement, ast.Expr):
            contains_yield |= _record_expression_calls(statement.value, bindings, facts, bound_names)
        elif isinstance(statement, ast.Return) and statement.value is not None:
            contains_yield |= _record_expression_calls(statement.value, bindings, facts, bound_names)
        elif isinstance(statement, ast.Assert):
            contains_yield |= _record_expression_calls(statement.test, bindings, facts, bound_names)
            if statement.msg is not None:
                contains_yield |= _record_expression_calls(statement.msg, bindings, facts, bound_names)
        elif isinstance(statement, ast.Raise):
            for expression in (statement.exc, statement.cause):
                if expression is not None:
                    contains_yield |= _record_expression_calls(expression, bindings, facts, bound_names)
        elif isinstance(statement, (ast.If, ast.While)):
            kills: set[str] = set()
            contains_yield |= _record_expression_calls(statement.test, bindings, facts, kills)
            for branch in (statement.body, statement.orelse):
                branch_bindings = bindings.branch_copy()
                branch_yield, branch_kills = _analyze_body(
                    branch,
                    branch_bindings,
                    facts,
                    function_depth=function_depth,
                    module_bindings=module_bindings,
                    conditional_imports=True,
                )
                _merge_framework_context(bindings, branch_bindings)
                contains_yield |= branch_yield
                kills.update(branch_kills)
            _invalidate(bindings, kills)
            bound_names.update(kills)
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            kills = set()
            contains_yield |= _record_expression_calls(statement.iter, bindings, facts, kills)
            loop_names = _target_names(statement.target)
            loop_bindings = bindings.branch_copy()
            _invalidate(loop_bindings, loop_names)
            body_yield, body_kills = _analyze_body(
                statement.body,
                loop_bindings,
                facts,
                function_depth=function_depth,
                module_bindings=module_bindings,
                conditional_imports=True,
            )
            _merge_framework_context(bindings, loop_bindings)
            else_bindings = bindings.branch_copy()
            _invalidate(else_bindings, loop_names)
            else_yield, else_kills = _analyze_body(
                statement.orelse,
                else_bindings,
                facts,
                function_depth=function_depth,
                module_bindings=module_bindings,
                conditional_imports=True,
            )
            _merge_framework_context(bindings, else_bindings)
            contains_yield |= body_yield or else_yield
            kills.update(loop_names | body_kills | else_kills)
            _invalidate(bindings, kills)
            bound_names.update(kills)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            kills = set()
            branch = bindings.branch_copy()
            for item in statement.items:
                contains_yield |= _record_expression_calls(item.context_expr, branch, facts, kills)
                names = _target_names(item.optional_vars)
                _invalidate(branch, names)
                kills.update(names)
            body_yield, body_kills = _analyze_body(
                statement.body,
                branch,
                facts,
                function_depth=function_depth,
                module_bindings=module_bindings,
                conditional_imports=True,
            )
            _merge_framework_context(bindings, branch)
            contains_yield |= body_yield
            kills.update(body_kills)
            _invalidate(bindings, kills)
            bound_names.update(kills)
        elif isinstance(statement, TRY_NODES):
            kills = set()
            body_bindings = bindings.branch_copy()
            branch_yield, branch_kills = _analyze_body(
                statement.body,
                body_bindings,
                facts,
                function_depth=function_depth,
                module_bindings=module_bindings,
                conditional_imports=True,
            )
            _merge_framework_context(bindings, body_bindings)
            contains_yield |= branch_yield
            kills.update(branch_kills)
            for handler in statement.handlers:
                branch = bindings.branch_copy()
                _invalidate(branch, branch_kills)
                handler_names = {handler.name} if handler.name else set()
                _invalidate(branch, handler_names)
                handler_yield, handler_kills = _analyze_body(
                    handler.body,
                    branch,
                    facts,
                    function_depth=function_depth,
                    module_bindings=module_bindings,
                    conditional_imports=True,
                )
                _merge_framework_context(bindings, branch)
                contains_yield |= handler_yield
                kills.update(handler_names | handler_kills)
            for branch_body in (statement.orelse, statement.finalbody):
                branch = bindings.branch_copy()
                _invalidate(branch, kills)
                branch_yield, branch_kills = _analyze_body(
                    branch_body,
                    branch,
                    facts,
                    function_depth=function_depth,
                    module_bindings=module_bindings,
                    conditional_imports=True,
                )
                _merge_framework_context(bindings, branch)
                contains_yield |= branch_yield
                kills.update(branch_kills)
            _invalidate(bindings, kills)
            bound_names.update(kills)
        elif isinstance(statement, ast.Match):
            kills = set()
            contains_yield |= _record_expression_calls(statement.subject, bindings, facts, kills)
            for case in statement.cases:
                branch = bindings.branch_copy()
                captures = _pattern_names(case.pattern)
                _invalidate(branch, captures)
                kills.update(captures)
                if case.guard is not None:
                    contains_yield |= _record_expression_calls(case.guard, branch, facts, kills)
                branch_yield, branch_kills = _analyze_body(
                    case.body,
                    branch,
                    facts,
                    function_depth=function_depth,
                    module_bindings=module_bindings,
                    conditional_imports=True,
                )
                _merge_framework_context(bindings, branch)
                contains_yield |= branch_yield
                kills.update(branch_kills)
            _invalidate(bindings, kills)
            bound_names.update(kills)
    return contains_yield, bound_names


def _analyze_direct_method(
    method: ast.FunctionDef | ast.AsyncFunctionDef, module_bindings: Bindings, facts: FileFacts
) -> None:
    scratch: set[str] = set()
    definition_bindings = module_bindings.branch_copy()
    owner_start = len(facts.owners)
    supporting_start = len(facts.supporting)
    unresolved_start = len(facts.unresolved)
    for expression in (*method.decorator_list, *method.args.defaults, *method.args.kw_defaults):
        if expression is not None:
            _record_expression_calls(expression, definition_bindings, facts, scratch)
    child = module_bindings.child_scope()
    _invalidate(child, _argument_names(method.args), remember_framework=False)
    _record_parameter_bindings(method.args, child)
    contains_yield, bound_names = _analyze_body(
        method.body,
        child,
        facts,
        function_depth=1,
        module_bindings=module_bindings,
        conditional_imports=False,
    )
    _remove_shadowed_records(
        facts,
        _shadowed_names(module_bindings, child, bound_names),
        owner_start=owner_start,
        supporting_start=supporting_start,
        unresolved_start=unresolved_start,
    )
    if method.decorator_list or contains_yield:
        _move_owners_to_unresolved(facts, owner_start)


def _record_expression_calls(
    expression: ast.expr, bindings: Bindings, facts: FileFacts, bound_names: set[str] | None = None
) -> bool:
    if isinstance(expression, EXCLUDED_EXPRESSION_SCOPES):
        _record_unsupported_expression_calls(expression, bindings, facts)
        return False
    contains_yield = isinstance(expression, (ast.Yield, ast.YieldFrom))
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        _record_absolute_path(expression.value, expression.lineno, bindings)
    for child in ast.iter_child_nodes(expression):
        if isinstance(child, ast.expr):
            contains_yield |= _record_expression_calls(child, bindings, facts, bound_names)
    if isinstance(expression, ast.Call):
        _record_call(expression, bindings, facts)
    elif isinstance(expression, ast.NamedExpr):
        names = _target_names(expression.target)
        _invalidate(bindings, names)
        if bound_names is not None:
            bound_names.update(names)
    return contains_yield


def _record_unsupported_expression_calls(expression: ast.expr, bindings: Bindings, facts: FileFacts) -> None:
    owner_start = len(facts.owners)
    client_start = len(facts.client_calls)
    scratch = bindings.branch_copy()
    for node in ast.walk(expression):
        if isinstance(node, ast.Call):
            _record_call(node, scratch, facts)
    _move_owners_to_unresolved(facts, owner_start)
    facts.possible_client_calls.extend(line for _, line in facts.client_calls[client_start:])
    del facts.client_calls[client_start:]


def _record_import(
    node: ast.Import | ast.ImportFrom, bindings: Bindings, facts: FileFacts, *, uncertain: bool = False
) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            canonical = alias.name if alias.asname else alias.name.split(".")[0]
            bindings.rebind(local)
            bindings.symbols[local] = canonical
            _remember_client_name(local, alias.name, bindings)
            if uncertain and _is_client_import(alias.name):
                bindings.uncertain_client_names.add(local)
            facts.nvflare_import |= alias.name == "nvflare" or alias.name.startswith("nvflare.")
            facts.local_imports.append((alias.name, 0, ()))
        return
    module = node.module
    names = tuple(alias.name for alias in node.names if alias.name != "*")
    facts.local_imports.append((module, node.level, names))
    full_module = "." * node.level + (module or "")
    facts.nvflare_import |= bool(node.level == 0 and module and (module == "nvflare" or module.startswith("nvflare.")))
    for alias in node.names:
        if alias.name == "*":
            continue
        local = alias.asname or alias.name
        canonical = f"{full_module}.{alias.name}"
        bindings.rebind(local)
        bindings.symbols[local] = canonical
        _remember_client_name(local, canonical, bindings)
        if uncertain and _is_client_import(canonical):
            bindings.uncertain_client_names.add(local)


def _record_class(node: ast.ClassDef, bindings: Bindings, facts: FileFacts) -> None:
    resolved_bases = [_resolve_expr(base, bindings) for base in node.bases]
    frameworks = {framework for base in resolved_bases if (framework := _trainer_framework(base))}
    has_flmodel_base = any(_is_flmodel(base) for base in resolved_bases)
    possible_flmodel_base = any(_is_possible_flmodel_base(base, bindings) for base in node.bases)
    uncertain_flmodel_base = any((_root_name(base) or "") in bindings.uncertain_client_names for base in node.bases)
    bindings.rebind(node.name)
    if has_flmodel_base and not uncertain_flmodel_base:
        facts.client_calls.append(("FLModel", node.lineno))
    elif has_flmodel_base or possible_flmodel_base:
        facts.possible_client_calls.append(node.lineno)
    if len(frameworks) == 1 and not node.decorator_list:
        bindings.subclasses[node.name] = next(iter(frameworks))
    else:
        bindings.framework_context_seen.update(frameworks)
        if len(frameworks) > 1:
            facts.unresolved.extend((framework, node.lineno, ()) for framework in sorted(frameworks))


def _record_assignment(
    node: ast.Assign | ast.AnnAssign, bindings: Bindings, facts: FileFacts, bound_names: set[str]
) -> bool:
    value = node.value
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = set().union(*(_target_names(target) for target in targets)) if targets else set()
    simple_name = targets[0].id if len(targets) == 1 and isinstance(targets[0], ast.Name) else None
    _record_secret_assignment(node, names, bindings)
    canonical = (
        _resolve_expr(value.func, bindings, allow_outer=bool(names & bindings.outer_names))
        if isinstance(value, ast.Call)
        else None
    )
    dependencies = _inherited_dependencies(value.func, bindings) if isinstance(value, ast.Call) else ()
    contains_yield = _record_expression_calls(value, bindings, facts, bound_names) if value is not None else False
    _invalidate(bindings, names, remember_framework=False)
    bound_names.update(names)
    if not isinstance(value, ast.Call) or simple_name is None:
        return contains_yield
    framework = bindings.subclasses.get(canonical or "") or _trainer_framework(canonical)
    optimizer = _is_optimizer(canonical)
    scaler = _is_scaler(canonical)
    if simple_name in bindings.outer_names:
        candidate = framework or ("pytorch" if optimizer or scaler else None)
        if candidate:
            facts.unresolved.append((candidate, node.lineno, dependencies))
        return contains_yield
    if framework:
        bindings.instances[simple_name] = framework
        bindings.dependencies[simple_name] = dependencies
    elif optimizer:
        bindings.optimizers.add(simple_name)
        bindings.dependencies[simple_name] = dependencies
    elif scaler:
        bindings.scalers.add(simple_name)
        bindings.dependencies[simple_name] = dependencies
    return contains_yield


def _record_call(call: ast.Call, bindings: Bindings, facts: FileFacts) -> None:
    _record_integration_call(call, bindings, facts)
    canonical = _resolve_expr(call.func, bindings)
    if canonical in HF_CONFIGS:
        bindings.hf_config_constructed = True
    if not isinstance(call.func, ast.Attribute):
        return
    method = call.func.attr
    receiver = call.func.value
    if isinstance(receiver, ast.Name) and receiver.id in bindings.instances:
        framework = bindings.instances[receiver.id]
        dependencies = bindings.dependencies.get(receiver.id, ())
        if method in OWNER_METHODS[framework]:
            facts.owners.append((framework, call.lineno, dependencies))
        elif method in SUPPORTING_METHODS[framework]:
            facts.supporting.append((framework, call.lineno, dependencies))
        return
    if isinstance(receiver, ast.Call) and method in OWNER_METHODS["lightning"]:
        if _trainer_framework(_resolve_expr(receiver.func, bindings)) == "lightning":
            facts.owners.append(("lightning", call.lineno, _inherited_dependencies(receiver.func, bindings)))
            return
    if isinstance(receiver, ast.Name) and receiver.id in bindings.optimizers and method == "step":
        facts.owners.append(("pytorch", call.lineno, bindings.dependencies.get(receiver.id, ())))
        return
    if isinstance(receiver, ast.Name) and receiver.id in bindings.scalers and method == "step":
        optimizer_name = call.args[0].id if call.args and isinstance(call.args[0], ast.Name) else None
        for keyword in call.keywords:
            if keyword.arg == "optimizer" and isinstance(keyword.value, ast.Name):
                optimizer_name = keyword.value.id
        record = facts.owners if optimizer_name in bindings.optimizers else facts.unresolved
        dependencies = set(bindings.dependencies.get(receiver.id, ()))
        dependencies.update(bindings.dependencies.get(optimizer_name or "", ()))
        record.append(("pytorch", call.lineno, tuple(sorted(dependencies))))
        return
    candidate = _method_candidate(method, bindings)
    if candidate:
        record = facts.supporting if method in SUPPORTING_METHODS.get(candidate, set()) else facts.unresolved
        record.append((candidate, call.lineno, ()))


def _record_integration_call(call: ast.Call, bindings: Bindings, facts: FileFacts) -> None:
    root = _root_name(call.func)
    if root in bindings.uncertain_client_names:
        facts.possible_client_calls.append(call.lineno)
        return
    canonical = _resolve_expr(call.func, bindings)
    if _is_flmodel(canonical):
        facts.client_calls.append(("FLModel", call.lineno))
        return
    if canonical and (canonical == "nvflare.client" or canonical.startswith("nvflare.client.")):
        terminal = canonical.rsplit(".", 1)[-1]
        if terminal in {"FLModel", "patch", "receive", "send"}:
            facts.client_calls.append((terminal, call.lineno))
            return
    terminal = _terminal_name(call.func)
    remembered = bindings.client_names_seen.get(root or "", "missing")
    client_calls = {"FLModel", "patch", "receive", "send"}
    if remembered is None and isinstance(call.func, ast.Attribute) and terminal in client_calls:
        facts.possible_client_calls.append(call.lineno)
    elif terminal == root and remembered in client_calls:
        facts.possible_client_calls.append(call.lineno)


def _invalidate(bindings: Bindings, names: set[str], *, remember_framework: bool = True) -> None:
    for name in names:
        if remember_framework:
            canonical = bindings.symbols.get(name)
            framework = bindings.instances.get(name) or bindings.subclasses.get(name)
            framework = framework or _trainer_framework(canonical) or _framework_family(canonical)
            if (
                name in bindings.optimizers
                or name in bindings.scalers
                or _is_optimizer(canonical)
                or _is_scaler(canonical)
            ):
                framework = "pytorch"
            if framework:
                bindings.framework_context_seen.add(framework)
        bindings.rebind(name)


def _resolve_expr(node: ast.AST, bindings: Bindings, *, allow_outer: bool = False) -> str | None:
    if isinstance(node, ast.Name):
        if node.id in bindings.symbols:
            return bindings.symbols[node.id]
        if node.id in bindings.subclasses:
            return node.id
        if allow_outer and node.id not in bindings.outer_symbol_kills:
            return bindings.outer_symbols.get(node.id)
    elif isinstance(node, ast.Attribute):
        base = _resolve_expr(node.value, bindings, allow_outer=allow_outer)
        return f"{base}.{node.attr}" if base else None
    return None


def _trainer_framework(canonical: str | None) -> str | None:
    if not canonical:
        return None
    terminal = canonical.rsplit(".", 1)[-1]
    if (canonical.startswith("transformers.") or canonical.startswith("trl.")) and terminal.endswith("Trainer"):
        return "huggingface"
    if canonical in {"lightning.Trainer", "lightning.pytorch.Trainer", "pytorch_lightning.Trainer"}:
        return "lightning"
    return None


def _framework_family(canonical: str | None) -> str | None:
    if canonical in {"transformers", "trl"}:
        return "huggingface"
    if canonical in {"lightning", "lightning.pytorch", "pytorch_lightning"}:
        return "lightning"
    if canonical in {"torch", "torch.optim"}:
        return "pytorch"
    return None


def _is_optimizer(canonical: str | None) -> bool:
    return bool(
        canonical
        and canonical.count(".") == 2
        and canonical.startswith("torch.optim.")
        and canonical.rsplit(".", 1)[-1] in OPTIMIZERS
    )


def _is_scaler(canonical: str | None) -> bool:
    return canonical in {"torch.amp.GradScaler", "torch.cuda.amp.GradScaler"}


def _is_flmodel(canonical: str | None) -> bool:
    return canonical in {
        "nvflare.app_common.abstract.fl_model.FLModel",
        "nvflare.client.FLModel",
        "nvflare.client.api.FLModel",
    }


def _method_candidate(method: str, bindings: Bindings) -> str | None:
    frameworks = {_trainer_framework(value) for value in bindings.symbols.values()}
    frameworks.discard(None)
    frameworks |= set(bindings.subclasses.values()) | bindings.framework_context_seen
    if method == "train" and ("huggingface" in frameworks or bindings.hf_config_constructed):
        return "huggingface"
    if method == "evaluate" and "huggingface" in frameworks:
        return "huggingface"
    if method in {"fit", "predict", "test", "validate"} and "lightning" in frameworks:
        return "lightning"
    if method == "step" and (
        "pytorch" in frameworks or any(_is_optimizer(value) for value in bindings.symbols.values())
    ):
        return "pytorch"
    if method == "backward" and (
        "pytorch" in frameworks or any(_framework_family(value) == "pytorch" for value in bindings.symbols.values())
    ):
        return "pytorch"
    return None


def _record_class_body_facts(statement: ast.stmt, bindings: Bindings, facts: FileFacts) -> None:
    rebound_names: set[str] = set()
    client_start = len(facts.client_calls)
    ambiguous_client_scope = isinstance(statement, CLASS_BODY_COMPOUND_NODES)
    for node in ast.walk(statement):
        if not isinstance(statement, ast.ClassDef) and isinstance(node, EXCLUDED_EXPRESSION_SCOPES):
            ambiguous_client_scope = True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _record_import(node, bindings, facts)
            if node is not statement:
                rebound_names.update(_import_names(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = set().union(*(_target_names(target) for target in targets)) if targets else set()
            _record_secret_assignment(node, names, bindings)
            rebound_names.update(names)
        elif isinstance(node, ast.ClassDef):
            resolved_bases = [_resolve_expr(base, bindings) for base in node.bases]
            if any(_is_flmodel(base) for base in resolved_bases):
                facts.client_calls.append(("FLModel", node.lineno))
            elif any(_is_possible_flmodel_base(base, bindings) for base in node.bases):
                facts.possible_client_calls.append(node.lineno)
        elif isinstance(node, ast.Call):
            _record_integration_call(node, bindings, facts)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            _record_absolute_path(node.value, node.lineno, bindings)
    _invalidate(bindings, rebound_names, remember_framework=False)
    if ambiguous_client_scope:
        facts.possible_client_calls.extend(line for _, line in facts.client_calls[client_start:])
        del facts.client_calls[client_start:]


def _record_secret_assignment(node: ast.Assign | ast.AnnAssign, names: set[str], bindings: Bindings) -> None:
    value = node.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return
    for name in sorted(names):
        if SECRET_NAME.search(name):
            bindings.findings.append(
                {
                    "file": bindings.path,
                    "line": node.lineno,
                    "code": "SECRET_LITERAL_REDACTED",
                    "name": name,
                    "value": "<REDACTED>" if bindings.redact else value.value,
                }
            )


def _merge_framework_context(target: Bindings, source: Bindings) -> None:
    frameworks = set(source.framework_context_seen)
    frameworks.update(filter(None, (_trainer_framework(value) for value in source.symbols.values())))
    frameworks.update(filter(None, (_framework_family(value) for value in source.symbols.values())))
    frameworks.update(source.subclasses.values())
    frameworks.update(source.instances.values())
    if source.optimizers or source.scalers:
        frameworks.add("pytorch")
    if source.hf_config_constructed:
        frameworks.add("huggingface")
    target.framework_context_seen.update(frameworks)
    target.hf_config_constructed |= source.hf_config_constructed
    for name, kind in source.client_names_seen.items():
        if name not in target.client_names_seen:
            target.client_names_seen[name] = kind
        elif target.client_names_seen[name] != kind:
            target.client_names_seen[name] = None


def _record_absolute_path(value: str, line: int, bindings: Bindings) -> None:
    if not (value.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", value)):
        return
    bindings.findings.append(
        {
            "file": bindings.path,
            "line": line,
            "code": "ABSOLUTE_DATA_PATH",
            "value": "<REDACTED_PATH>" if bindings.redact else value,
        }
    )


def _remember_client_name(local: str, canonical: str, bindings: Bindings) -> None:
    if canonical == "nvflare.client" or canonical.startswith("nvflare.client."):
        terminal = canonical.rsplit(".", 1)[-1]
        bindings.client_names_seen[local] = terminal if terminal in {"FLModel", "patch", "receive", "send"} else None


def _is_client_import(canonical: str) -> bool:
    return canonical == "nvflare.client" or canonical.startswith("nvflare.client.")


def _is_possible_flmodel_base(base: ast.AST, bindings: Bindings) -> bool:
    root = _root_name(base)
    remembered = bindings.client_names_seen.get(root or "", "missing")
    if isinstance(base, ast.Name):
        return remembered == "FLModel"
    return isinstance(base, ast.Attribute) and base.attr == "FLModel" and remembered is None


def _move_owners_to_unresolved(facts: FileFacts, start: int) -> None:
    facts.unresolved.extend(facts.owners[start:])
    del facts.owners[start:]


def _argument_names(arguments: ast.arguments) -> set[str]:
    names = {argument.arg for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
    if arguments.vararg:
        names.add(arguments.vararg.arg)
    if arguments.kwarg:
        names.add(arguments.kwarg.arg)
    return names


def _record_parameter_bindings(arguments: ast.arguments, bindings: Bindings) -> None:
    parameters = (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    for parameter in parameters:
        if _resolve_expr(parameter.annotation, bindings) == "torch.optim.Optimizer":
            bindings.optimizers.add(parameter.arg)


def _inherited_dependencies(node: ast.AST, bindings: Bindings) -> tuple[Dependency, ...]:
    root = _root_name(node)
    suffix = _attribute_suffix(node)
    framework = _constructor_framework(node, bindings)
    return ((root, suffix),) if root in bindings.inherited_names and framework else ()


def _shadowed_names(parent: Bindings, child: Bindings, bound_names: set[str]) -> set[str]:
    inherited = set(parent.symbols) | set(parent.subclasses)
    return (bound_names - child.outer_names) & inherited


def _remove_shadowed_records(
    facts: FileFacts,
    names: set[str],
    *,
    owner_start: int,
    supporting_start: int,
    unresolved_start: int,
) -> None:
    if not names:
        return
    facts.owners[owner_start:] = [
        item for item in facts.owners[owner_start:] if not names.intersection(name for name, _ in item[2])
    ]
    facts.supporting[supporting_start:] = [
        item for item in facts.supporting[supporting_start:] if not names.intersection(name for name, _ in item[2])
    ]
    facts.unresolved[unresolved_start:] = [
        item for item in facts.unresolved[unresolved_start:] if not names.intersection(name for name, _ in item[2])
    ]


def _remove_changed_module_records(facts: FileFacts, bindings: Bindings) -> None:
    def unchanged(record: FactRecord) -> bool:
        return all(_dependency_framework(name, suffix, bindings) == record[0] for name, suffix in record[2])

    facts.owners[:] = filter(unchanged, facts.owners)
    facts.supporting[:] = filter(unchanged, facts.supporting)
    facts.unresolved[:] = filter(unchanged, facts.unresolved)


def _constructor_framework(node: ast.AST, bindings: Bindings) -> str | None:
    canonical = _resolve_expr(node, bindings)
    if isinstance(node, ast.Name) and node.id in bindings.subclasses:
        return bindings.subclasses[node.id]
    if _is_optimizer(canonical) or _is_scaler(canonical):
        return "pytorch"
    return _trainer_framework(canonical)


def _dependency_framework(name: str, suffix: tuple[str, ...], bindings: Bindings) -> str | None:
    if not suffix and name in bindings.subclasses:
        return bindings.subclasses[name]
    canonical = bindings.symbols.get(name)
    if not canonical:
        return None
    if suffix:
        canonical = ".".join((canonical, *suffix))
    if _is_optimizer(canonical) or _is_scaler(canonical):
        return "pytorch"
    return _trainer_framework(canonical)


def _attribute_suffix(node: ast.AST) -> tuple[str, ...]:
    suffix: list[str] = []
    while isinstance(node, ast.Attribute):
        suffix.append(node.attr)
        node = node.value
    return tuple(reversed(suffix))


def _import_names(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.split(".")[0] for alias in node.names}
    return {alias.asname or alias.name for alias in node.names if alias.name != "*"}


def _target_names(target: ast.AST | None) -> set[str]:
    if target is None:
        return set()
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in target.elts)) if target.elts else set()
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return set()


def _pattern_names(pattern: ast.pattern) -> set[str]:
    names: set[str] = set()
    if isinstance(pattern, ast.MatchAs) and pattern.name:
        names.add(pattern.name)
    elif isinstance(pattern, ast.MatchStar) and pattern.name:
        names.add(pattern.name)
    elif isinstance(pattern, ast.MatchMapping) and pattern.rest:
        names.add(pattern.rest)
    for child in ast.iter_child_nodes(pattern):
        if isinstance(child, ast.pattern):
            names.update(_pattern_names(child))
    return names


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else None


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None
