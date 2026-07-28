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

"""Base contract for inspector framework detectors.

A ``FrameworkDetector`` encapsulates everything the inspector needs to know
about one framework. The engine calls the ``on_*`` hooks once per relevant AST
node, passing a per-file ``DetectContext`` the detector uses to record evidence
and FLARE-integration signals. Detectors keep their own per-file scratch state
(import aliases, imported symbols) via ``new_file_state``; the engine treats it
as opaque.
"""

import ast
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional


class EvidenceStrength(str, Enum):
    """Framework-neutral meaning of one static evidence item."""

    IMPORT = "import"
    CANDIDATE = "candidate"
    TRAINING_OWNER = "training_owner"


@dataclass
class LexicalScopeBindings:
    """Resolve scoped identities while honoring Python binding declarations."""

    bindings: dict = field(default_factory=dict)
    scope_locals: dict = field(default_factory=dict)
    scope_globals: dict = field(default_factory=dict)
    scope_nonlocals: dict = field(default_factory=dict)

    def declare_scope(
        self,
        scope: tuple[str, ...],
        local_names: set[str],
        global_names: set[str],
        nonlocal_names: set[str],
    ) -> None:
        self.scope_locals[scope] = local_names
        self.scope_globals[scope] = global_names
        self.scope_nonlocals[scope] = nonlocal_names

    def bind(
        self,
        name: str,
        scope: tuple[str, ...],
        *identity_collections,
    ) -> tuple[str, ...]:
        binding_scope = self.binding_scope(name, scope)
        self.bindings.setdefault(binding_scope, set()).add(name)
        key = (binding_scope, name)
        for identities in identity_collections:
            if isinstance(identities, dict):
                identities.pop(key, None)
            else:
                identities.discard(key)
        return binding_scope

    def has_identity(self, name: str, scope: tuple[str, ...], identities) -> bool:
        return self.lookup_identity_key(name, scope, identities) is not None

    def lookup_mapping(self, name: str, scope: tuple[str, ...], identities: dict):
        key = self.lookup_identity_key(name, scope, identities)
        return identities.get(key) if key else None

    def lookup_identity_key(self, name: str, scope: tuple[str, ...], identities):
        for candidate in self.candidate_scopes(name, scope):
            key = (candidate, name)
            if key in identities:
                return key
            if name in self.scope_locals.get(candidate, set()) or name in self.bindings.get(candidate, set()):
                return None
        return None

    def can_resolve_from_enclosing_scope(self, name: str, scope: tuple[str, ...]) -> bool:
        if not scope or "." in name:
            return False
        # A class body executes immediately and resolves names sequentially.
        # Deferred finalization is only valid for function-like scopes whose
        # bodies run after the enclosing scope has finished binding names.
        if scope[-1].startswith("class:"):
            return False
        if name in self.scope_globals.get(scope, set()):
            return True
        if name in self.scope_nonlocals.get(scope, set()):
            return True
        return name not in self.scope_locals.get(scope, set())

    @staticmethod
    def has_deferred_function_scope(scope: tuple[str, ...]) -> bool:
        """Whether an enclosing function-like body executes after surrounding bindings settle."""
        return any(part.startswith(("function:", "async-function:", "lambda:")) for part in scope)

    def binding_scope(self, name: str, scope: tuple[str, ...]) -> tuple[str, ...]:
        if "." in name:
            return scope
        if name in self.scope_globals.get(scope, set()):
            return ()
        if name in self.scope_nonlocals.get(scope, set()):
            candidates = self.candidate_scopes(name, scope, include_current=False, include_module=False)
            for candidate in candidates:
                if name in self.scope_locals.get(candidate, set()) or name in self.bindings.get(candidate, set()):
                    return candidate
        return scope

    def candidate_scopes(
        self,
        name: str,
        scope: tuple[str, ...],
        *,
        include_current: bool = True,
        include_module: bool = True,
    ) -> list[tuple[str, ...]]:
        if "." in name:
            return [scope] if include_current else []
        if name in self.scope_globals.get(scope, set()):
            return [()]

        candidates = []
        start = len(scope) if include_current else len(scope) - 1
        nonlocal_only = name in self.scope_nonlocals.get(scope, set())
        for length in range(start, -1, -1):
            candidate = scope[:length]
            if not candidate and (nonlocal_only or not include_module):
                continue
            if (
                candidate
                and candidate[-1].startswith("class:")
                and any(
                    part.startswith(
                        (
                            "function:",
                            "async-function:",
                            "lambda:",
                            "list-comprehension:",
                            "set-comprehension:",
                            "dict-comprehension:",
                            "generator-expression:",
                        )
                    )
                    for part in scope[length:]
                )
            ):
                continue
            candidates.append(candidate)
        return candidates


class DetectContext:
    """Per-file sink the engine hands to detector hooks.

    The detector records framework evidence and FLARE-integration signals
    through this context instead of touching inspector internals directly, so
    the engine owns how those signals are stored and ranked.
    """

    def __init__(self, emit_evidence, add_flare_call, add_integration_signal):
        self._emit_evidence = emit_evidence
        self._add_flare_call = add_flare_call
        self._add_integration_signal = add_integration_signal

    def evidence(self, framework: str, kind: str, value: str, lineno: Optional[int]) -> None:
        """Record ranked framework evidence (import, class base, activity call)."""
        self._emit_evidence(framework, kind, value, lineno)

    def flare_call(self, call_name: str) -> None:
        """Record a FLARE-integration call such as ``flare.patch``."""
        self._add_flare_call(call_name)

    def integration_signal(self, framework: str, name: str) -> None:
        """Record a framework-specific FLARE conversion signal.

        Used by ``conversion_state`` to tell a converted job apart from raw
        training code (for example, a Lightning ``flare.patch(trainer)`` call).
        """
        self._add_integration_signal(framework, name)


class FrameworkDetector:
    """Static-detection plugin for a single framework.

    Subclasses set the class attributes and override the ``on_*`` hooks they
    need. Every hook is optional; the base implementations do nothing.
    """

    #: Canonical framework name reported in inspector output (e.g. ``"pytorch"``).
    name: str = ""
    #: Top-level import module names that map to this framework's evidence
    #: bucket, e.g. ``{"torch": "pytorch"}``. Used for ranked import evidence.
    import_roots: Mapping[str, str] = MappingProxyType({})
    #: Evidence-kind -> ranking weight contributed by this framework.
    evidence_weights: Mapping[str, int] = MappingProxyType({})
    #: Evidence-kind -> framework-neutral routing strength.
    evidence_strengths: Mapping[str, EvidenceStrength] = MappingProxyType({})
    #: Conversion skill recommended when this framework is primary, or ``None``.
    recommended_skill: Optional[str] = None
    #: Whether inspector recommendations require active evidence rather than
    #: imports alone. Useful when a package also has non-training uses.
    recommendation_requires_active_evidence: bool = False
    #: Family this framework belongs to for cross-framework disambiguation
    #: (e.g. Lightning declares ``"pytorch"``). ``None`` means standalone.
    family: Optional[str] = None

    def new_file_state(self) -> Any:
        """Return fresh per-file scratch state (import aliases, symbols)."""
        return None

    def on_import(
        self,
        alias: ast.alias,
        file_state: Any,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        """Handle ``import x`` / ``import x as y`` aliases."""

    def on_import_from(
        self,
        module: str,
        aliases: list,
        file_state: Any,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        """Handle ``from module import ...`` symbols."""

    def on_scope(
        self,
        scope: tuple[str, ...],
        local_names: set[str],
        global_names: set[str],
        nonlocal_names: set[str],
        file_state: Any,
        ctx: DetectContext,
    ) -> None:
        """Declare lexical bindings before a function, class, or lambda body is visited."""

    def on_class_definition(
        self,
        class_name: str,
        base_names: list[str],
        lineno: Optional[int],
        file_state: Any,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        """Handle a class definition after its expressions and body are visited."""

    def on_class_base(
        self,
        base_name: str,
        lineno: Optional[int],
        file_state: Any,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        """Handle a base class name in a ``class X(Base):`` definition."""

    def on_call(
        self,
        call_name: str,
        lineno: Optional[int],
        file_state: Any,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        """Handle a called name such as ``torch.optim.SGD`` or ``flare.patch``."""

    def classify_assignment_value(
        self,
        call_name: Optional[str],
        file_state: Any,
        scope: tuple[str, ...] = (),
    ) -> Any:
        """Snapshot detector-specific RHS provenance before assignment targets bind."""
        return None

    def on_assignment(
        self,
        target_names: list[str],
        call_name: Optional[str],
        lineno: Optional[int],
        file_state: Any,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
        value_info: Any = None,
    ) -> None:
        """Handle assignment and optional call construction in a lexical scope."""

    def finalize_file(self, file_state: Any, ctx: DetectContext) -> None:
        """Resolve evidence that depends on bindings collected later in the file."""

    # --- cross-framework family resolution -------------------------------

    def evidence_strength(self, evidence: dict) -> EvidenceStrength:
        """Return the routing meaning of one evidence item."""
        kind = evidence.get("kind")
        if kind == "import":
            return EvidenceStrength.IMPORT
        return self.evidence_strengths.get(kind, EvidenceStrength.CANDIDATE)

    def is_active_evidence(self, evidence: dict) -> bool:
        """Whether an evidence item counts as active (in-use) for this framework.

        Used by family disambiguation to distinguish active use from an
        incidental import. Defaults to non-import evidence.
        """
        return evidence.get("kind") != "import"

    def is_training_owner_evidence(self, evidence: dict) -> bool:
        """Whether evidence claims ownership of the training lifecycle."""
        return self.evidence_strength(evidence) == EvidenceStrength.TRAINING_OWNER

    def is_candidate_evidence(self, evidence: dict) -> bool:
        """Whether evidence identifies a framework object without loop ownership."""
        return self.evidence_strength(evidence) == EvidenceStrength.CANDIDATE

    def promote_over_family(self, family_base: str, resolver) -> bool:
        """For a family member, decide whether to win over the family base.

        Only called for detectors that declare a ``family``. ``resolver`` gives
        access to the collected evidence and entry-context helpers so the
        decision stays in the framework module. Default: never promote.
        """
        return False

    def fallback_skill_for(self, evidence: list[dict]) -> Optional[str]:
        """Return a safe fallback when evidence cannot justify the conversion skill."""
        return None
