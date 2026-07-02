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

from .base import DetectContext, FrameworkDetector

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
    aliases: set = field(default_factory=set)
    symbols: dict = field(default_factory=dict)
    patch_symbols: set = field(default_factory=set)
    patch_modules: set = field(default_factory=set)


class LightningDetector(FrameworkDetector):
    name = FRAMEWORK
    import_roots = {"pytorch_lightning": FRAMEWORK, "lightning": FRAMEWORK}
    evidence_weights = {"lightning_class": 3, "lightning_trainer": 3}
    recommended_skill = "nvflare-convert-lightning"
    family = BASE_FAMILY

    def new_file_state(self) -> _LightningFileState:
        return _LightningFileState()

    def on_import(self, alias: ast.alias, file_state: _LightningFileState, ctx: DetectContext) -> None:
        if alias.name in LIGHTNING_MODULES:
            file_state.aliases.add(alias.asname or alias.name.split(".")[0])
        if alias.name == LIGHTNING_PATCH_MODULE:
            # ``import nvflare.client.lightning as flare`` -> ``flare.patch`` is
            # the canonical conversion call; a plain import keeps the full path.
            file_state.patch_modules.add(alias.asname or alias.name)

    def on_import_from(self, module: str, aliases: list, file_state: _LightningFileState, ctx: DetectContext) -> None:
        if module in LIGHTNING_MODULES or any(module.startswith(f"{prefix}.") for prefix in LIGHTNING_MODULES):
            for alias in aliases:
                if alias.name in LIGHTNING_SYMBOLS:
                    file_state.symbols[alias.asname or alias.name] = alias.name
        if module == LIGHTNING_PATCH_MODULE:
            for alias in aliases:
                if alias.name == "patch":
                    file_state.patch_symbols.add(alias.asname or alias.name)
        # ``from nvflare.client import lightning as flare`` -> ``flare.patch`` is
        # the module-alias form of the canonical conversion call.
        elif module == LIGHTNING_PATCH_PARENT_MODULE:
            for alias in aliases:
                if alias.name == LIGHTNING_PATCH_SUBMODULE:
                    file_state.patch_modules.add(alias.asname or alias.name)

    def on_class_base(
        self, base_name: str, lineno: Optional[int], file_state: _LightningFileState, ctx: DetectContext
    ) -> None:
        if self._is_lightning_class_base(base_name, file_state):
            ctx.evidence(FRAMEWORK, "lightning_class", base_name, lineno)

    def on_call(
        self, call_name: str, lineno: Optional[int], file_state: _LightningFileState, ctx: DetectContext
    ) -> None:
        if call_name in file_state.patch_symbols or self._is_lightning_patch_call(call_name, file_state):
            ctx.flare_call(call_name)
            ctx.integration_signal(FRAMEWORK, call_name)
        if self._is_lightning_trainer_call(call_name, file_state):
            ctx.evidence(FRAMEWORK, "lightning_trainer", call_name, lineno)

    def is_active_evidence(self, evidence: dict) -> bool:
        return evidence.get("kind") in {"lightning_class", "lightning_trainer"}

    @staticmethod
    def _is_lightning_class_base(base_name: str, file_state: _LightningFileState) -> bool:
        if file_state.symbols.get(base_name) in LIGHTNING_CLASS_SYMBOLS:
            return True
        if "." not in base_name:
            return False
        prefix, _, symbol = base_name.rpartition(".")
        return symbol in LIGHTNING_CLASS_SYMBOLS and (
            prefix in file_state.aliases or prefix in LIGHTNING_MODULES or prefix.startswith("lightning.pytorch")
        )

    @staticmethod
    def _is_lightning_patch_call(call_name: str, file_state: _LightningFileState) -> bool:
        prefix, _, symbol = call_name.rpartition(".")
        if symbol != "patch":
            return False
        return prefix in file_state.patch_modules or prefix == LIGHTNING_PATCH_MODULE

    @staticmethod
    def _is_lightning_trainer_call(call_name: str, file_state: _LightningFileState) -> bool:
        if file_state.symbols.get(call_name) in LIGHTNING_TRAINER_SYMBOLS:
            return True
        if "." not in call_name:
            return False
        prefix, _, symbol = call_name.rpartition(".")
        return symbol in LIGHTNING_TRAINER_SYMBOLS and (
            prefix in file_state.aliases or prefix in LIGHTNING_MODULES or prefix.startswith("lightning.pytorch")
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
        #   3. Entry point or inspected file exists but neither is tied -> base.
        #   4. Model-only/no-entry contexts use weighted evidence fallback.
        active_lightning_evidence = resolver.active_evidence(self.name)
        active_pytorch_evidence = resolver.active_evidence(family_base)
        if resolver.tied_to_entry_context(active_lightning_evidence):
            return True
        # Any PyTorch evidence (not only active) tied to the entry context keeps
        # the base framework primary, matching the pre-refactor behavior.
        if resolver.tied_to_entry_context(pytorch_evidence):
            return False
        if resolver.has_inspected_file_or_entry_point():
            return False

        active_lightning_score = resolver.score(active_lightning_evidence)
        if active_lightning_score == 0:
            return False
        active_pytorch_score = resolver.score(active_pytorch_evidence)
        if active_pytorch_score == 0 and resolver.has_evidence_outside_files(
            pytorch_evidence, active_lightning_evidence
        ):
            if resolver.score(pytorch_evidence) >= resolver.score(lightning_evidence):
                return False
        if active_lightning_score != active_pytorch_score:
            return active_lightning_score > active_pytorch_score
        return resolver.score(lightning_evidence) > resolver.score(pytorch_evidence)
