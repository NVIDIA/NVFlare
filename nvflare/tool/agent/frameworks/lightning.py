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

"""PyTorch Lightning framework detector.

Lightning is a member of the PyTorch family: it declares ``family = "pytorch"``
and owns the base/superset promotion decision (``promote_over_family``) so the
inspector engine stays framework-agnostic.
"""

import ast
from dataclasses import dataclass, field
from typing import Optional

from .base import DetectContext, FrameworkDetector, LexicalScopeBindings

FRAMEWORK = "pytorch_lightning"
BASE_FAMILY = "pytorch"

LIGHTNING_MODULES = {"pytorch_lightning", "lightning", "lightning.pytorch"}
LIGHTNING_CLASS_SYMBOLS = {"LightningModule", "LightningDataModule"}
LIGHTNING_TRAINER_SYMBOLS = {"Trainer"}
LIGHTNING_SYMBOLS = LIGHTNING_CLASS_SYMBOLS | LIGHTNING_TRAINER_SYMBOLS
LIGHTNING_PATCH_PARENT_MODULE = "nvflare.client"
LIGHTNING_PATCH_SUBMODULE = "lightning"
LIGHTNING_PATCH_MODULE = f"{LIGHTNING_PATCH_PARENT_MODULE}.{LIGHTNING_PATCH_SUBMODULE}"


@dataclass
class _LightningFileState:
    # alias name -> resolved Lightning module it points at ("lightning",
    # "lightning.pytorch", or "pytorch_lightning"). "lightning" is the bare
    # package whose symbols live under ``.pytorch``; the others expose symbols
    # directly.
    aliases: dict = field(default_factory=dict)
    symbols: dict = field(default_factory=dict)
    scopes: LexicalScopeBindings = field(default_factory=LexicalScopeBindings)
    patch_symbols: set = field(default_factory=set)
    patch_modules: set = field(default_factory=set)
    pending_patch_calls: list = field(default_factory=list)


class LightningDetector(FrameworkDetector):
    name = FRAMEWORK
    import_roots = {"pytorch_lightning": FRAMEWORK, "lightning": FRAMEWORK}
    evidence_weights = {"lightning_class": 3, "lightning_trainer": 3}
    recommended_skill = "nvflare-convert-lightning"
    family = BASE_FAMILY

    def new_file_state(self) -> _LightningFileState:
        return _LightningFileState()

    def on_import(
        self,
        alias: ast.alias,
        file_state: _LightningFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        alias_name = alias.asname or alias.name.split(".", 1)[0]
        binding_scope = self._bind_name(alias_name, scope, file_state)
        if self._is_lightning_import(alias.name):
            # ``import lightning.pytorch as pl`` -> pl points at lightning.pytorch;
            # ``import lightning as L`` -> L points at the bare lightning package.
            resolved_module = alias.name if alias.asname else alias_name
            file_state.aliases[(binding_scope, alias_name)] = resolved_module
        if alias.name == LIGHTNING_PATCH_MODULE:
            # ``import nvflare.client.lightning as flare`` -> ``flare.patch`` is
            # the canonical conversion call; a plain import keeps the full path.
            file_state.patch_modules.add((binding_scope, alias_name))

    def on_import_from(
        self,
        module: str,
        aliases: list,
        file_state: _LightningFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        bound_names = {}
        for alias in aliases:
            if alias.name == "*":
                continue
            alias_name = alias.asname or alias.name
            bound_names[alias_name] = self._bind_name(alias_name, scope, file_state)

        if module in LIGHTNING_MODULES or any(module.startswith(f"{prefix}.") for prefix in LIGHTNING_MODULES):
            for alias in aliases:
                alias_name = alias.asname or alias.name
                binding_scope = bound_names.get(alias_name, scope)
                if alias.name in LIGHTNING_SYMBOLS:
                    file_state.symbols[(binding_scope, alias_name)] = alias.name
                elif f"{module}.{alias.name}" in LIGHTNING_MODULES:
                    # ``from lightning import pytorch as pl`` -> pl points at the
                    # lightning.pytorch module that exposes LightningModule/Trainer.
                    file_state.aliases[(binding_scope, alias_name)] = f"{module}.{alias.name}"
        if module == LIGHTNING_PATCH_MODULE:
            for alias in aliases:
                if alias.name == "patch":
                    alias_name = alias.asname or alias.name
                    file_state.patch_symbols.add((bound_names[alias_name], alias_name))
        # ``from nvflare.client import lightning as flare`` -> ``flare.patch`` is
        # the module-alias form of the canonical conversion call.
        elif module == LIGHTNING_PATCH_PARENT_MODULE:
            for alias in aliases:
                if alias.name == LIGHTNING_PATCH_SUBMODULE:
                    alias_name = alias.asname or alias.name
                    file_state.patch_modules.add((bound_names[alias_name], alias_name))

    def on_scope(
        self,
        scope: tuple[str, ...],
        local_names: set[str],
        global_names: set[str],
        nonlocal_names: set[str],
        file_state: _LightningFileState,
        ctx: DetectContext,
    ) -> None:
        file_state.scopes.declare_scope(scope, local_names, global_names, nonlocal_names)

    def on_class_definition(
        self,
        class_name: str,
        base_names: list[str],
        lineno: Optional[int],
        file_state: _LightningFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        self._bind_name(class_name, scope, file_state)

    def on_class_base(
        self,
        base_name: str,
        lineno: Optional[int],
        file_state: _LightningFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        if self._is_lightning_class_base(base_name, file_state, scope):
            ctx.evidence(FRAMEWORK, "lightning_class", base_name, lineno)

    def on_call(
        self,
        call_name: str,
        lineno: Optional[int],
        file_state: _LightningFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        is_patch_call = file_state.scopes.has_identity(
            call_name, scope, file_state.patch_symbols
        ) or self._is_lightning_patch_call(call_name, file_state, scope)
        if is_patch_call:
            ctx.flare_call(call_name)
            ctx.integration_signal(FRAMEWORK, call_name)
        else:
            identity_name = self._patch_identity_name(call_name)
            if identity_name and file_state.scopes.can_resolve_from_enclosing_scope(identity_name, scope):
                file_state.pending_patch_calls.append((call_name, scope))
        if self._is_lightning_trainer_call(call_name, file_state, scope):
            ctx.evidence(FRAMEWORK, "lightning_trainer", call_name, lineno)

    def on_assignment(
        self,
        target_names: list[str],
        call_name: Optional[str],
        lineno: Optional[int],
        file_state: _LightningFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
        value_info=None,
    ) -> None:
        for target_name in target_names:
            self._bind_name(target_name, scope, file_state)

    def finalize_file(self, file_state: _LightningFileState, ctx: DetectContext) -> None:
        for call_name, scope in file_state.pending_patch_calls:
            if file_state.scopes.has_identity(
                call_name, scope, file_state.patch_symbols
            ) or self._is_lightning_patch_call(call_name, file_state, scope):
                ctx.flare_call(call_name)
                ctx.integration_signal(FRAMEWORK, call_name)

    def is_active_evidence(self, evidence: dict) -> bool:
        return evidence.get("kind") in {"lightning_class", "lightning_trainer"}

    def is_training_owner_evidence(self, evidence: dict) -> bool:
        # A LightningModule can be trained by another framework; Trainer owns
        # the loop when resolving mixed Lightning/Hugging Face projects.
        return evidence.get("kind") == "lightning_trainer"

    @staticmethod
    def _is_lightning_import(module: str) -> bool:
        return (
            module == "pytorch_lightning"
            or module.startswith("pytorch_lightning.")
            or module == "lightning"
            or module == "lightning.pytorch"
            or module.startswith("lightning.pytorch.")
        )

    @staticmethod
    def _prefix_exposes_lightning_symbols(
        prefix: str,
        file_state: _LightningFileState,
        scope: tuple[str, ...],
    ) -> bool:
        # A module prefix exposes LightningModule/Trainer when it is (an alias of)
        # a Lightning module -- ``lightning`` re-exports them at the top level, so
        # a bare ``lightning`` alias counts too -- or the ``.pytorch`` submodule of
        # a bare ``lightning`` alias (``import lightning as L`` -> ``L.pytorch``).
        # Require an actual import binding instead of treating arbitrary local
        # names as Lightning symbols.
        if file_state.scopes.lookup_mapping(prefix, scope, file_state.aliases) in LIGHTNING_MODULES:
            return True
        head, _, rest = prefix.partition(".")
        return rest == "pytorch" and file_state.scopes.lookup_mapping(head, scope, file_state.aliases) == "lightning"

    @classmethod
    def _is_lightning_class_base(
        cls,
        base_name: str,
        file_state: _LightningFileState,
        scope: tuple[str, ...],
    ) -> bool:
        if file_state.scopes.lookup_mapping(base_name, scope, file_state.symbols) in LIGHTNING_CLASS_SYMBOLS:
            return True
        if "." not in base_name:
            return False
        prefix, _, symbol = base_name.rpartition(".")
        return symbol in LIGHTNING_CLASS_SYMBOLS and cls._prefix_exposes_lightning_symbols(prefix, file_state, scope)

    @staticmethod
    def _is_lightning_patch_call(
        call_name: str,
        file_state: _LightningFileState,
        scope: tuple[str, ...],
    ) -> bool:
        _, _, symbol = call_name.rpartition(".")
        if symbol != "patch":
            return False
        identity_name = LightningDetector._patch_identity_name(call_name)
        return bool(identity_name and file_state.scopes.has_identity(identity_name, scope, file_state.patch_modules))

    @classmethod
    def _is_lightning_trainer_call(
        cls,
        call_name: str,
        file_state: _LightningFileState,
        scope: tuple[str, ...],
    ) -> bool:
        if file_state.scopes.lookup_mapping(call_name, scope, file_state.symbols) in LIGHTNING_TRAINER_SYMBOLS:
            return True
        if "." not in call_name:
            return False
        prefix, _, symbol = call_name.rpartition(".")
        return symbol in LIGHTNING_TRAINER_SYMBOLS and cls._prefix_exposes_lightning_symbols(prefix, file_state, scope)

    @staticmethod
    def _patch_identity_name(call_name: str) -> Optional[str]:
        prefix, separator, symbol = call_name.rpartition(".")
        if not separator:
            return call_name
        if symbol != "patch":
            return None
        if prefix == LIGHTNING_PATCH_MODULE:
            return prefix.split(".", 1)[0]
        return prefix

    @staticmethod
    def _bind_name(
        name: str,
        scope: tuple[str, ...],
        file_state: _LightningFileState,
    ) -> tuple[str, ...]:
        return file_state.scopes.bind(
            name,
            scope,
            file_state.aliases,
            file_state.symbols,
            file_state.patch_symbols,
            file_state.patch_modules,
        )

    def promote_over_family(self, family_base: str, resolver) -> bool:
        # PyTorch Lightning is a PyTorch superset. Resolve the conflict whenever
        # both buckets exist, but do not let incidental Lightning imports in a
        # plain PyTorch workspace hide the PyTorch entry point from routing.
        lightning_evidence = resolver.evidence(self.name)
        pytorch_evidence = resolver.evidence(family_base)
        if not lightning_evidence or not pytorch_evidence:
            return False

        # Base/superset precedence:
        #   1. Active superset evidence tied to the entry context -> superset.
        #   2. Any base evidence tied to the entry context -> base.
        #   3. Entry point/inspected file exists and there is *standalone* active
        #      base usage (active PyTorch evidence in a file that has no active
        #      Lightning evidence) but neither is tied -> base. torch.optim/losses/
        #      dataloaders inside a LightningModule file are not standalone use,
        #      so an unrelated entry point cannot hide a dominant superset.
        #   4. Model-only/no-entry contexts use weighted evidence fallback.
        active_lightning_evidence = resolver.active_evidence(self.name)
        active_pytorch_evidence = resolver.active_evidence(family_base)

        # An active Lightning model reachable from the entry context means the
        # project is a Lightning project -> route to the superset. torch used to
        # build or train that model (optimizers, losses, dataloaders, and helper
        # nn.Module submodules composed by the LightningModule) does not demote
        # it, so a realistic Lightning repo is not mis-scored as PyTorch-dominant.
        #
        # DESIGN DECISION (do not re-add a "standalone torch dominates -> base"
        # guard here): a reachable LightningModule wins even when co-located with
        # dominant plain-torch code. This deliberately favors the common case
        # (real Lightning projects that compose torch submodules and heavy torch
        # usage) over the rare edge of a stray/empty leftover LightningModule in a
        # torch-only repo. Routing that rare edge to the Lightning skill is
        # low-harm (the conversion still works); mis-routing real Lightning repos
        # to the PyTorch skill is not. The stray-leftover case is intentionally
        # accepted, not a bug to "fix" by demoting reachable Lightning.
        if resolver.tied_to_entry_context(active_lightning_evidence):
            return True
        # Any PyTorch evidence (not only active) tied to the entry context keeps
        # the base framework primary when Lightning is not reachable there.
        if resolver.tied_to_entry_context(pytorch_evidence):
            return False

        # Neither bucket is tied to the entry context: fall back to weighted
        # evidence. No active Lightning model at all means the base framework
        # stays primary.
        if not active_lightning_evidence:
            return False

        # torch used *inside* a Lightning model class body is Lightning's, so
        # only active PyTorch outside those bodies competes as standalone base
        # usage; a sibling plain-torch model or module-level training code
        # co-located with a stray Lightning class still counts as genuine base use.
        active_lightning_class_evidence = [
            item for item in active_lightning_evidence if item.get("kind") == "lightning_class"
        ]
        standalone_active_pytorch_evidence = resolver.evidence_outside_class_bodies(
            active_pytorch_evidence, active_lightning_class_evidence
        )
        if standalone_active_pytorch_evidence:
            if resolver.has_inspected_file_or_entry_point():
                return False
            active_lightning_score = resolver.score(active_lightning_evidence)
            standalone_active_pytorch_score = resolver.score(standalone_active_pytorch_evidence)
            if active_lightning_score != standalone_active_pytorch_score:
                return active_lightning_score > standalone_active_pytorch_score

        # Active scores tied (or no standalone base usage): compare the full
        # Lightning bucket against PyTorch evidence outside active-Lightning files.
        pytorch_evidence_outside_active_lightning_files = resolver.evidence_outside_files(
            pytorch_evidence, active_lightning_evidence
        )
        return resolver.score(lightning_evidence) > resolver.score(pytorch_evidence_outside_active_lightning_files)
