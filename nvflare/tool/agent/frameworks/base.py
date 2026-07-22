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
from typing import Any, Optional


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
    import_roots: dict[str, str] = {}
    #: Evidence-kind -> ranking weight contributed by this framework.
    evidence_weights: dict[str, int] = {}
    #: Conversion skill recommended when this framework is primary, or ``None``.
    recommended_skill: Optional[str] = None
    #: Family this framework belongs to for cross-framework disambiguation
    #: (e.g. Lightning declares ``"pytorch"``). ``None`` means standalone.
    family: Optional[str] = None

    def new_file_state(self) -> Any:
        """Return fresh per-file scratch state (import aliases, symbols)."""
        return None

    def on_import(self, alias: ast.alias, file_state: Any, ctx: DetectContext) -> None:
        """Handle ``import x`` / ``import x as y`` aliases."""

    def on_import_from(self, module: str, aliases: list, file_state: Any, ctx: DetectContext) -> None:
        """Handle ``from module import ...`` symbols."""

    def on_class_base(self, base_name: str, lineno: Optional[int], file_state: Any, ctx: DetectContext) -> None:
        """Handle a base class name in a ``class X(Base):`` definition."""

    def on_call(self, call_name: str, lineno: Optional[int], file_state: Any, ctx: DetectContext) -> None:
        """Handle a called name such as ``torch.optim.SGD`` or ``flare.patch``."""

    # --- cross-framework family resolution -------------------------------

    def is_active_evidence(self, evidence: dict) -> bool:
        """Whether an evidence item counts as active (in-use) for this framework.

        Used by family disambiguation to distinguish active use from an
        incidental import. Defaults to non-import evidence.
        """
        return evidence.get("kind") != "import"

    def promote_over_family(self, family_base: str, resolver) -> bool:
        """For a family member, decide whether to win over the family base.

        Only called for detectors that declare a ``family``. ``resolver`` gives
        access to the collected evidence and entry-context helpers so the
        decision stays in the framework module. Default: never promote.
        """
        return False
