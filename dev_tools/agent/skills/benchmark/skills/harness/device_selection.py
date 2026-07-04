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

"""Deterministic checks for conversion skill device-selection behavior."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Literal, Optional

DEVICE_SELECTION_BEHAVIOR_ID = "device-selection-respects-availability"

DeviceSelectionStatus = Literal["pass", "fail", "missing", "not_applicable"]

_DEVICE_TARGET_NAMES = {"accelerator", "device"}
_GPU_DEVICE_VALUES = {"cuda", "gpu"}
_CPU_DEVICE_VALUES = {"cpu"}


@dataclass(frozen=True)
class DeviceSelectionResult:
    """Scored device-selection behavior for one conversion run."""

    status: DeviceSelectionStatus
    evidence: str

    def as_behavior_record(self) -> dict[str, dict[str, dict[str, str]]]:
        """Return the run-record shape expected by skill eval reports."""

        return {
            "mandatory_behavior": {
                DEVICE_SELECTION_BEHAVIOR_ID: {
                    "status": self.status,
                    "evidence": self.evidence,
                }
            }
        }


@dataclass(frozen=True)
class _StaticDeviceEvidence:
    conditional_selection: bool
    hardcoded_devices: frozenset[str]
    has_device_logic: bool


@dataclass
class _ScopeState:
    """Names whose meaning has been established in one lexical scope."""

    torch_aliases: set[str] = field(default_factory=set)
    cuda_aliases: set[str] = field(default_factory=set)
    availability_call_names: set[str] = field(default_factory=set)
    availability_vars: set[str] = field(default_factory=set)

    def branch_copy(self) -> "_ScopeState":
        return _ScopeState(
            torch_aliases=set(self.torch_aliases),
            cuda_aliases=set(self.cuda_aliases),
            availability_call_names=set(self.availability_call_names),
            availability_vars=set(self.availability_vars),
        )

    def child_scope(self, local_names: set[str]) -> "_ScopeState":
        child = self.branch_copy()
        child.availability_vars.clear()
        for name in local_names:
            child.invalidate(name)
        return child

    def invalidate(self, name: str) -> None:
        self.torch_aliases.discard(name)
        self.cuda_aliases.discard(name)
        self.availability_call_names.discard(name)
        self.availability_vars.discard(name)


class _DeviceSelectionAnalyzer:
    """Recognize narrow, scope-aware device-selection patterns without execution.

    This intentionally does not attempt general Python control-flow analysis. A
    conversion passes only when a reachable canonical assignment/call directly
    links ``torch.cuda.is_available()`` (or a still-valid local alias) to GPU and
    CPU values for the same target.
    """

    def __init__(self, tree: ast.AST):
        self.tree = tree
        self._module_conditional_targets: set[str] = set()
        self._module_plain_targets: set[str] = set()
        self._nested_conditional_targets: set[str] = set()
        self.hardcoded_devices: set[str] = set()
        self.has_device_logic = False

    def evidence(self) -> _StaticDeviceEvidence:
        if isinstance(self.tree, ast.Module):
            self._analyze_sequence(self.tree.body, _ScopeState(), scope_kind="module")
        module_evidence = self._module_conditional_targets - self._module_plain_targets
        nested_evidence = self._nested_conditional_targets - self._module_plain_targets
        conditional_selection = bool(module_evidence or nested_evidence)
        return _StaticDeviceEvidence(
            conditional_selection=conditional_selection,
            hardcoded_devices=frozenset(self.hardcoded_devices),
            has_device_logic=self.has_device_logic,
        )

    def _analyze_sequence(
        self,
        statements: list[ast.stmt],
        state: _ScopeState,
        *,
        record_scope_evidence: bool = True,
        scope_kind: str = "nested",
    ) -> None:
        default_devices: dict[str, str] = {}
        conditional_targets: dict[str, bool] = {}
        plain_targets: dict[str, str] = {}
        for statement in statements:
            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                self._inspect_statement(statement, state)
                break

            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                self._apply_import(statement, state)
                for name in _bound_names_without_nested_scopes(statement):
                    target = ast.Name(id=name, ctx=ast.Store())
                    _discard_target_selections(default_devices, target)
                    _discard_target_selections(conditional_targets, target)
                    _discard_target_selections(plain_targets, target)
                continue

            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._analyze_function(statement, state)
                state.invalidate(statement.name)
                target = ast.Name(id=statement.name, ctx=ast.Store())
                _discard_target_selections(default_devices, target)
                _discard_target_selections(conditional_targets, target)
                _discard_target_selections(plain_targets, target)
                continue

            if isinstance(statement, ast.ClassDef):
                self._analyze_class(statement, state)
                state.invalidate(statement.name)
                target = ast.Name(id=statement.name, ctx=ast.Store())
                _discard_target_selections(default_devices, target)
                _discard_target_selections(conditional_targets, target)
                _discard_target_selections(plain_targets, target)
                continue

            self._inspect_statement(statement, state)

            if isinstance(statement, ast.If):
                selected_targets = self._inspect_if(statement, state, default_devices)
                constant = _constant_truth(statement.test)
                if constant is not False:
                    self._analyze_sequence(statement.body, state.branch_copy(), record_scope_evidence=False)
                if constant is not True:
                    self._analyze_sequence(statement.orelse, state.branch_copy(), record_scope_evidence=False)
                self._invalidate_after_compound(
                    statement,
                    state,
                    default_devices,
                    conditional_targets,
                    plain_targets,
                )
                for target in selected_targets:
                    conditional_targets[target] = True
                    plain_targets.pop(target, None)
                continue

            if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                if not (isinstance(statement, ast.While) and _constant_truth(statement.test) is False):
                    self._analyze_sequence(statement.body, state.branch_copy(), record_scope_evidence=False)
                self._analyze_sequence(statement.orelse, state.branch_copy(), record_scope_evidence=False)
                self._invalidate_after_compound(
                    statement,
                    state,
                    default_devices,
                    conditional_targets,
                    plain_targets,
                )
                continue

            if isinstance(statement, (ast.With, ast.AsyncWith)):
                self._analyze_sequence(statement.body, state.branch_copy(), record_scope_evidence=False)
                self._invalidate_after_compound(
                    statement,
                    state,
                    default_devices,
                    conditional_targets,
                    plain_targets,
                )
                continue

            if isinstance(statement, ast.Try):
                self._analyze_sequence(statement.body, state.branch_copy(), record_scope_evidence=False)
                for handler in statement.handlers:
                    self._analyze_sequence(handler.body, state.branch_copy(), record_scope_evidence=False)
                self._analyze_sequence(statement.orelse, state.branch_copy(), record_scope_evidence=False)
                self._analyze_sequence(statement.finalbody, state.branch_copy(), record_scope_evidence=False)
                self._invalidate_after_compound(
                    statement,
                    state,
                    default_devices,
                    conditional_targets,
                    plain_targets,
                )
                continue

            assignments = _direct_device_selections(statement, state)
            for target in _direct_mutated_targets(statement):
                _discard_target_selections(default_devices, target)
                _discard_target_selections(conditional_targets, target)
                _discard_target_selections(plain_targets, target)
            for key, kind in assignments.items():
                default_devices[key] = kind
                plain_targets[key] = kind
            for key in _conditional_selection_targets(statement, state):
                conditional_targets[key] = True
                plain_targets.pop(key, None)
            self._update_state_for_simple_statement(statement, state)

        if record_scope_evidence:
            if scope_kind == "module":
                self._module_conditional_targets.update(conditional_targets)
                self._module_plain_targets.update(plain_targets)
            else:
                self._nested_conditional_targets.update(set(conditional_targets) - set(plain_targets))

    def _analyze_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, outer: _ScopeState) -> None:
        local_names = _function_local_names(node)
        child = outer.child_scope(local_names)
        self._analyze_sequence(node.body, child)

    def _analyze_class(self, node: ast.ClassDef, outer: _ScopeState) -> None:
        # Method globals resolve against the enclosing module/function, not the
        # class namespace. Analyze methods with the outer aliases and keep class
        # assignments from shadowing those globals.
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._analyze_function(statement, outer)
            elif isinstance(statement, ast.ClassDef):
                self._analyze_class(statement, outer)

    def _apply_import(self, node: ast.Import | ast.ImportFrom, state: _ScopeState) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                state.invalidate(bound_name)
                if alias.name == "torch":
                    state.torch_aliases.add(bound_name)
                elif alias.name == "torch.cuda" and alias.asname:
                    state.cuda_aliases.add(bound_name)
            return

        for alias in node.names:
            bound_name = alias.asname or alias.name
            state.invalidate(bound_name)
            if node.module == "torch" and alias.name == "cuda":
                state.cuda_aliases.add(bound_name)
            elif node.module == "torch.cuda" and alias.name == "is_available":
                state.availability_call_names.add(bound_name)

    def _inspect_statement(self, statement: ast.stmt, state: _ScopeState) -> None:
        selections = _direct_device_selections(statement, state)
        if selections:
            self.has_device_logic = True
            self.hardcoded_devices.update(selections.values())
        if _statement_has_device_logic(statement, state):
            self.has_device_logic = True

    def _inspect_if(self, node: ast.If, state: _ScopeState, defaults: dict[str, str]) -> set[str]:
        polarity = _availability_polarity(node.test, state)
        if not polarity:
            return set()
        self.has_device_logic = True

        true_selections = _direct_branch_selections(node.body, state)
        false_selections = _direct_branch_selections(node.orelse, state)
        matched_targets = _same_target_gpu_cpu_branches(polarity, true_selections, false_selections)
        if matched_targets:
            return matched_targets

        # Canonical default/override form: ``device = "cpu"`` followed by
        # ``if is_available(): device = "cuda"`` with no else branch.
        if polarity == 1 and not node.orelse:
            return {
                target for target, kind in true_selections.items() if defaults.get(target) == "cpu" and kind == "gpu"
            }
        return set()

    def _invalidate_after_compound(
        self,
        node: ast.AST,
        state: _ScopeState,
        default_devices: dict[str, str],
        conditional_targets: dict[str, bool],
        plain_targets: dict[str, str],
    ) -> None:
        for name in _bound_names_without_nested_scopes(node):
            state.invalidate(name)
            target = ast.Name(id=name, ctx=ast.Store())
            _discard_target_selections(default_devices, target)
            _discard_target_selections(conditional_targets, target)
            _discard_target_selections(plain_targets, target)
        for target in _device_target_keys_without_nested_scopes(node):
            default_devices.pop(target, None)
            conditional_targets.pop(target, None)
            plain_targets.pop(target, None)

    def _update_state_for_simple_statement(self, statement: ast.stmt, state: _ScopeState) -> None:
        bound_names = _bound_names_without_nested_scopes(statement)
        for name in bound_names:
            state.invalidate(name)
        for target in _direct_mutated_targets(statement):
            if isinstance(target, ast.Attribute):
                root_name = _attribute_root_name(target)
                if root_name:
                    state.invalidate(root_name)

        target, value = _simple_name_assignment(statement)
        if target and value is not None and _availability_polarity(value, state) == 1:
            state.availability_vars.add(target)


def check_device_selection(
    source_text: str,
    generated_text: str,
    *,
    runtime_log: str | None = None,
    gpu_available: bool | None = None,
) -> DeviceSelectionResult:
    """Score whether a conversion preserves GPU-when-available selection.

    Raw runtime logs are intentionally non-authoritative: user code can print a
    forged ``device=cuda`` or ``device=cpu`` line. The checker uses Python AST
    evidence from source and generated code. ``runtime_log`` remains in the
    signature for report-harness compatibility but cannot change the result.
    """

    if not source_uses_gpu_when_available(source_text):
        return DeviceSelectionResult(
            "not_applicable",
            "source does not select CUDA/GPU conditionally with torch.cuda.is_available() in its Python AST; "
            "device rule not scored",
        )

    evidence = _static_device_evidence(generated_text)
    if evidence.conditional_selection:
        if gpu_available is True:
            detail = "static GPU-available branch selects CUDA/GPU"
        elif gpu_available is False:
            detail = "static GPU-unavailable branch selects the CPU fallback"
        else:
            detail = "static GPU-available and GPU-unavailable branches are both present"
        return DeviceSelectionResult(
            "pass",
            f"generated Python AST keeps torch.cuda.is_available() conditional selection; {detail}",
        )

    ignored_log = " Raw runtime log text was not trusted as device evidence." if runtime_log else ""
    if evidence.hardcoded_devices == {"cpu"}:
        return DeviceSelectionResult(
            "fail",
            "source used GPU-when-available but generated code hard-codes CPU device selection." + ignored_log,
        )
    if evidence.hardcoded_devices == {"gpu"}:
        return DeviceSelectionResult(
            "fail",
            "source used GPU-when-available but generated code hard-codes GPU without a verified CPU fallback."
            + ignored_log,
        )
    if not evidence.has_device_logic:
        return DeviceSelectionResult(
            "missing",
            "source used GPU-when-available but generated code has no AST-detectable device-selection logic."
            + ignored_log,
        )
    return DeviceSelectionResult(
        "missing",
        "source used GPU-when-available but generated code does not show a reachable, same-target conditional "
        "CUDA/GPU selection with a CPU fallback." + ignored_log,
    )


def source_uses_gpu_when_available(text: str) -> bool:
    """Return True for a reachable canonical CUDA-availability selection."""

    return _static_device_evidence(text).conditional_selection


def _static_device_evidence(text: str) -> _StaticDeviceEvidence:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return _StaticDeviceEvidence(False, frozenset(), False)
    return _DeviceSelectionAnalyzer(tree).evidence()


def _availability_polarity(node: ast.AST, state: _ScopeState) -> int:
    """Return 1 for available, -1 for unavailable, and 0 for unknown."""

    if _is_availability_call(node, state):
        return 1
    if isinstance(node, ast.Name) and node.id in state.availability_vars:
        return 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return -_availability_polarity(node.operand, state)
    return 0


def _is_availability_call(node: ast.AST, state: _ScopeState) -> bool:
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return False
    function = node.func
    if isinstance(function, ast.Name):
        return function.id in state.availability_call_names
    if not isinstance(function, ast.Attribute) or function.attr != "is_available":
        return False
    owner = function.value
    if isinstance(owner, ast.Name):
        return owner.id in state.cuda_aliases
    return (
        isinstance(owner, ast.Attribute)
        and owner.attr == "cuda"
        and isinstance(owner.value, ast.Name)
        and owner.value.id in state.torch_aliases
    )


def _conditional_selection_targets(statement: ast.stmt, state: _ScopeState) -> set[str]:
    selections: set[str] = set()
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        target, value = _assignment_target_and_value(statement)
        if target is None or value is None:
            return selections
        target_key = _target_key(target)
        if _is_device_target(target) and _conditional_device_expr(value, state):
            selections.add(target_key)
        if isinstance(value, ast.Call):
            for keyword in value.keywords:
                if (
                    keyword.arg
                    and keyword.arg.lower() in _DEVICE_TARGET_NAMES
                    and _conditional_device_expr(keyword.value, state)
                ):
                    selections.add(f"{target_key}:keyword:{keyword.arg.lower()}")
            if (
                isinstance(value.func, ast.Attribute)
                and value.func.attr == "to"
                and value.args
                and _conditional_device_expr(value.args[0], state)
            ):
                selections.add(f"receiver:{_target_key(value.func.value)}:to")
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        call = statement.value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "to"
            and call.args
            and _conditional_device_expr(call.args[0], state)
        ):
            selections.add(f"receiver:{_target_key(call.func.value)}:to")
    return selections


def _conditional_device_expr(node: ast.AST, state: _ScopeState) -> bool:
    if isinstance(node, ast.IfExp):
        polarity = _availability_polarity(node.test, state)
        true_kind = _device_kind_from_expr(node.body, state)
        false_kind = _device_kind_from_expr(node.orelse, state)
        return _branches_select_gpu_and_cpu(polarity, true_kind, false_kind)
    if isinstance(node, ast.Call) and _is_torch_device_call(node, state) and node.args:
        return _conditional_device_expr(node.args[0], state)
    return False


def _branches_select_gpu_and_cpu(polarity: int, true_kind: Optional[str], false_kind: Optional[str]) -> bool:
    if polarity == 1:
        return true_kind == "gpu" and false_kind == "cpu"
    if polarity == -1:
        return true_kind == "cpu" and false_kind == "gpu"
    return False


def _same_target_gpu_cpu_branches(polarity: int, true_values: dict[str, str], false_values: dict[str, str]) -> set[str]:
    return {
        target
        for target in true_values.keys() & false_values.keys()
        if _branches_select_gpu_and_cpu(polarity, true_values[target], false_values[target])
    }


def _device_kind_from_expr(node: ast.AST, state: _ScopeState) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value.strip().lower()
        if value in _CPU_DEVICE_VALUES:
            return "cpu"
        if value in _GPU_DEVICE_VALUES or value.startswith("cuda:"):
            return "gpu"
    if isinstance(node, ast.Call) and _is_torch_device_call(node, state) and node.args:
        return _device_kind_from_expr(node.args[0], state)
    return None


def _is_torch_device_call(node: ast.Call, state: _ScopeState) -> bool:
    function = node.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "device"
        and isinstance(function.value, ast.Name)
        and function.value.id in state.torch_aliases
    )


def _direct_device_selections(statement: ast.stmt, state: _ScopeState) -> dict[str, str]:
    """Return direct same-statement target -> constant device selections."""

    selections: dict[str, str] = {}
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        target, value = _assignment_target_and_value(statement)
        if target is None or value is None:
            return selections
        target_key = _target_key(target)
        if _is_device_target(target):
            kind = _device_kind_from_expr(value, state)
            if kind:
                selections[target_key] = kind
        if isinstance(value, ast.Call):
            _add_call_selections(selections, value, state, target_key)
        return selections

    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        call = statement.value
        # A discarded constructor/function result is not a stable target for a
        # later default/override pair. Expression calls only count when ``.to``
        # updates a concrete receiver.
        if isinstance(call.func, ast.Attribute) and call.func.attr == "to" and call.args:
            kind = _device_kind_from_expr(call.args[0], state)
            if kind:
                selections[f"receiver:{_target_key(call.func.value)}:to"] = kind
    return selections


def _add_call_selections(selections: dict[str, str], call: ast.Call, state: _ScopeState, target_prefix: str) -> None:
    for keyword in call.keywords:
        if keyword.arg and keyword.arg.lower() in _DEVICE_TARGET_NAMES:
            kind = _device_kind_from_expr(keyword.value, state)
            if kind:
                selections[f"{target_prefix}:keyword:{keyword.arg.lower()}"] = kind
    if isinstance(call.func, ast.Attribute) and call.func.attr == "to" and call.args:
        kind = _device_kind_from_expr(call.args[0], state)
        if kind:
            selections[f"receiver:{_target_key(call.func.value)}:to"] = kind


def _direct_branch_selections(statements: list[ast.stmt], state: _ScopeState) -> dict[str, str]:
    selections: dict[str, str] = {}
    ambiguous: set[str] = set()
    for statement in statements:
        if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            break
        # Do not descend into nested control flow. A selection inside ``if
        # False`` or a different nested condition is not evidence for this
        # branch's direct outcome.
        if isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
            continue
        for target in _direct_mutated_targets(statement):
            _discard_target_selections(selections, target)
        for target, kind in _direct_device_selections(statement, state).items():
            if target in selections and selections[target] != kind:
                ambiguous.add(target)
            selections[target] = kind
    for target in ambiguous:
        selections.pop(target, None)
    return selections


def _statement_has_device_logic(statement: ast.stmt, state: _ScopeState) -> bool:
    for node in ast.walk(statement):
        if _is_availability_call(node, state):
            return True
        if isinstance(node, ast.Name) and node.id.lower() in _DEVICE_TARGET_NAMES | {"devices"}:
            return True
        if isinstance(node, ast.Attribute) and node.attr.lower() in _DEVICE_TARGET_NAMES | {"devices"}:
            return True
        if isinstance(node, ast.Call):
            if _is_torch_device_call(node, state) or (isinstance(node.func, ast.Attribute) and node.func.attr == "to"):
                return True
            if any(
                keyword.arg and keyword.arg.lower() in _DEVICE_TARGET_NAMES | {"devices"} for keyword in node.keywords
            ):
                return True
    return False


def _assignment_target_and_value(node: ast.Assign | ast.AnnAssign) -> tuple[Optional[ast.AST], Optional[ast.AST]]:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            return None, None
        return node.targets[0], node.value
    return node.target, node.value


def _simple_name_assignment(node: ast.stmt) -> tuple[Optional[str], Optional[ast.AST]]:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return None, None
    target, value = _assignment_target_and_value(node)
    if isinstance(target, ast.Name):
        return target.id, value
    return None, None


def _is_device_target(target: ast.AST) -> bool:
    return (isinstance(target, ast.Name) and target.id.lower() in _DEVICE_TARGET_NAMES) or (
        isinstance(target, ast.Attribute) and target.attr.lower() in _DEVICE_TARGET_NAMES
    )


def _target_key(target: ast.AST) -> str:
    # Load/Store/Del are evaluation context, not target identity. Normalizing
    # them lets ``model = replacement`` invalidate an earlier
    # ``model.to("cpu")`` selection, whose receiver is represented with Load.
    key = ast.dump(target, annotate_fields=True, include_attributes=False)
    for context in ("Load", "Store", "Del"):
        key = key.replace(f"ctx={context}()", "ctx=Context()")
    return key


def _direct_mutated_targets(statement: ast.stmt) -> list[ast.AST]:
    targets: list[ast.AST] = []
    if isinstance(statement, ast.Assign):
        raw_targets = statement.targets
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        raw_targets = [statement.target]
    elif isinstance(statement, ast.Delete):
        raw_targets = statement.targets
    else:
        raw_targets = []

    def add_target(target: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                add_target(element)
        elif isinstance(target, ast.Starred):
            add_target(target.value)
        else:
            targets.append(target)

    for raw_target in raw_targets:
        add_target(raw_target)
    return targets


def _discard_target_selections(selections: dict[str, str], target: ast.AST) -> None:
    target_key = _target_key(target)
    for selection_key in list(selections):
        if (
            selection_key == target_key
            or selection_key.startswith(f"{target_key}:")
            or selection_key.startswith(f"receiver:{target_key}:")
        ):
            selections.pop(selection_key, None)


def _attribute_root_name(node: ast.Attribute) -> Optional[str]:
    root: ast.AST = node
    while isinstance(root, ast.Attribute):
        root = root.value
    return root.id if isinstance(root, ast.Name) else None


def _function_local_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = {argument.arg for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
    if node.args.vararg:
        names.add(node.args.vararg.arg)
    if node.args.kwarg:
        names.add(node.args.kwarg.arg)
    names.update(_bound_names_without_nested_scopes_from_statements(node.body))
    return names


def _bound_names_without_nested_scopes(node: ast.AST) -> set[str]:
    collector = _BoundNameCollector()
    collector.visit(node)
    return collector.names


def _bound_names_without_nested_scopes_from_statements(statements: list[ast.stmt]) -> set[str]:
    collector = _BoundNameCollector()
    for statement in statements:
        collector.visit(statement)
    return collector.names


class _BoundNameCollector(ast.NodeVisitor):
    def __init__(self):
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802 - ast visitor API
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast visitor API
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".", maxsplit=1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - ast visitor API
        for alias in node.names:
            self.names.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast visitor API
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802 - ast visitor API
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 - ast visitor API
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802 - ast visitor API
        return


def _device_target_keys_without_nested_scopes(node: ast.AST) -> set[str]:
    collector = _DeviceTargetCollector()
    collector.visit(node)
    return collector.targets


class _DeviceTargetCollector(ast.NodeVisitor):
    def __init__(self):
        self.targets: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 - ast visitor API
        for target in node.targets:
            if _is_device_target(target):
                self.targets.add(_target_key(target))
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802 - ast visitor API
        if _is_device_target(node.target):
            self.targets.add(_target_key(node.target))
        if node.value:
            self.visit(node.value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast visitor API
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802 - ast visitor API
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 - ast visitor API
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802 - ast visitor API
        return


def _constant_truth(node: ast.AST) -> Optional[bool]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float, type(None))):
        return bool(node.value)
    return None
