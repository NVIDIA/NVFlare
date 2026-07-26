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

"""Hugging Face Trainer framework detector."""

import ast
from dataclasses import dataclass, field
from typing import Optional

from .base import DetectContext, FrameworkDetector, LexicalScopeBindings

FRAMEWORK = "huggingface"

HUGGINGFACE_MODULES = {"transformers", "trl"}
HUGGINGFACE_TRAINER_SUBMODULES = {"trainer", "trainer_seq2seq"}
TRAINING_CONFIG_SYMBOLS = {"TrainingArguments", "Seq2SeqTrainingArguments", "SFTConfig"}
TRAINER_CANDIDATE_EVIDENCE = {
    "huggingface_trainer",
    "huggingface_trainer_class",
    "huggingface_training_config",
}
PATCH_PARENT_MODULE = "nvflare.client"
PATCH_SUBMODULE = "hf"
PATCH_MODULE = f"{PATCH_PARENT_MODULE}.{PATCH_SUBMODULE}"


@dataclass
class _HuggingFaceFileState:
    module_aliases: dict = field(default_factory=dict)
    config_symbols: dict = field(default_factory=dict)
    trainer_symbols: dict = field(default_factory=dict)
    trainer_classes: set = field(default_factory=set)
    trainer_instances: set = field(default_factory=set)
    scopes: LexicalScopeBindings = field(default_factory=LexicalScopeBindings)
    pending_train_calls: list = field(default_factory=list)
    patch_symbols: set = field(default_factory=set)
    patch_modules: set = field(default_factory=set)
    pending_patch_calls: list = field(default_factory=list)


class HuggingFaceDetector(FrameworkDetector):
    name = FRAMEWORK
    import_roots = {"transformers": FRAMEWORK, "trl": FRAMEWORK}
    evidence_weights = {
        "huggingface_train": 4,
        "huggingface_trainer": 2,
        "huggingface_trainer_class": 2,
        "huggingface_training_config": 1,
    }
    recommended_skill = "nvflare-convert-huggingface"
    recommendation_requires_active_evidence = True
    family = "pytorch"

    def new_file_state(self) -> _HuggingFaceFileState:
        return _HuggingFaceFileState()

    def on_import(
        self,
        alias: ast.alias,
        file_state: _HuggingFaceFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        alias_name = alias.asname or alias.name.split(".", 1)[0]
        binding_scope = self._bind_name(alias_name, scope, file_state)
        if self._is_huggingface_module(alias.name):
            file_state.module_aliases[(binding_scope, alias_name)] = alias.name
        if alias.name == PATCH_MODULE:
            file_state.patch_modules.add((binding_scope, alias_name))

    def on_import_from(
        self,
        module: str,
        aliases: list,
        file_state: _HuggingFaceFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        bound_names = {}
        for alias in aliases:
            if alias.name == "*":
                continue
            alias_name = alias.asname or alias.name
            bound_names[alias_name] = self._bind_name(alias_name, scope, file_state)

        if self._is_huggingface_module(module):
            for alias in aliases:
                alias_name = alias.asname or alias.name
                binding_scope = bound_names.get(alias_name, scope)
                if self._is_trainer_symbol(alias.name):
                    file_state.trainer_symbols[(binding_scope, alias_name)] = alias.name
                elif alias.name in TRAINING_CONFIG_SYMBOLS:
                    file_state.config_symbols[(binding_scope, alias_name)] = alias.name
                elif self._is_trainer_submodule(module, alias.name):
                    file_state.module_aliases[(binding_scope, alias_name)] = f"{module}.{alias.name}"
        if module == PATCH_MODULE:
            for alias in aliases:
                if alias.name == "patch":
                    alias_name = alias.asname or alias.name
                    file_state.patch_symbols.add((bound_names[alias_name], alias_name))
        elif module == PATCH_PARENT_MODULE:
            for alias in aliases:
                if alias.name == PATCH_SUBMODULE:
                    alias_name = alias.asname or alias.name
                    file_state.patch_modules.add((bound_names[alias_name], alias_name))

    def on_scope(
        self,
        scope: tuple[str, ...],
        local_names: set[str],
        global_names: set[str],
        nonlocal_names: set[str],
        file_state: _HuggingFaceFileState,
        ctx: DetectContext,
    ) -> None:
        file_state.scopes.declare_scope(scope, local_names, global_names, nonlocal_names)

    def on_class_definition(
        self,
        class_name: str,
        base_names: list[str],
        lineno: Optional[int],
        file_state: _HuggingFaceFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        is_trainer_class = any(self._is_trainer_name(base_name, file_state, scope) for base_name in base_names)
        binding_scope = self._bind_name(class_name, scope, file_state)
        if is_trainer_class:
            file_state.trainer_classes.add((binding_scope, class_name))

    def on_class_base(
        self,
        base_name: str,
        lineno: Optional[int],
        file_state: _HuggingFaceFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        if self._is_trainer_name(base_name, file_state, scope):
            ctx.evidence(FRAMEWORK, "huggingface_trainer_class", base_name, lineno)

    def on_call(
        self,
        call_name: str,
        lineno: Optional[int],
        file_state: _HuggingFaceFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        is_patch_call = file_state.scopes.has_identity(
            call_name, scope, file_state.patch_symbols
        ) or self._is_patch_call(call_name, file_state, scope)
        if is_patch_call:
            ctx.flare_call(call_name)
            ctx.integration_signal(FRAMEWORK, call_name)
        else:
            identity_name = self._patch_identity_name(call_name)
            if identity_name and file_state.scopes.can_resolve_from_enclosing_scope(identity_name, scope):
                file_state.pending_patch_calls.append((call_name, scope))
        if self._is_trainer_name(call_name, file_state, scope):
            ctx.evidence(FRAMEWORK, "huggingface_trainer", call_name, lineno)
        elif self._is_training_config_name(call_name, file_state, scope):
            ctx.evidence(FRAMEWORK, "huggingface_training_config", call_name, lineno)
        else:
            receiver = self._train_call_receiver(call_name)
            if receiver:
                identity_key = self._lookup_identity_key(
                    receiver,
                    scope,
                    file_state.trainer_instances,
                    file_state,
                )
                if identity_key and identity_key[0] == scope:
                    ctx.evidence(FRAMEWORK, "huggingface_train", call_name, lineno)
                elif self._can_resolve_from_enclosing_scope(receiver, scope, file_state):
                    # Calls resolved through an enclosing scope are finalized
                    # after the full file is visited so later lexical bindings
                    # cannot change the result based on source order.
                    file_state.pending_train_calls.append((receiver, scope, call_name, lineno))

    def classify_assignment_value(
        self,
        call_name: Optional[str],
        file_state: _HuggingFaceFileState,
        scope: tuple[str, ...] = (),
    ) -> bool:
        return bool(call_name and self._is_trainer_name(call_name, file_state, scope))

    def on_assignment(
        self,
        target_names: list[str],
        call_name: Optional[str],
        lineno: Optional[int],
        file_state: _HuggingFaceFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
        value_info: Optional[bool] = None,
    ) -> None:
        is_trainer = (
            bool(call_name and self._is_trainer_name(call_name, file_state, scope))
            if value_info is None
            else value_info
        )
        for target_name in target_names:
            binding_scope = self._bind_name(target_name, scope, file_state)
            if is_trainer:
                file_state.trainer_instances.add((binding_scope, target_name))

    def finalize_file(self, file_state: _HuggingFaceFileState, ctx: DetectContext) -> None:
        for receiver, scope, call_name, lineno in file_state.pending_train_calls:
            if self._has_identity(receiver, scope, file_state.trainer_instances, file_state):
                ctx.evidence(FRAMEWORK, "huggingface_train", call_name, lineno)
        for call_name, scope in file_state.pending_patch_calls:
            if file_state.scopes.has_identity(call_name, scope, file_state.patch_symbols) or self._is_patch_call(
                call_name, file_state, scope
            ):
                ctx.flare_call(call_name)
                ctx.integration_signal(FRAMEWORK, call_name)

    def is_active_evidence(self, evidence: dict) -> bool:
        return evidence.get("kind") == "huggingface_train"

    def promote_over_family(self, family_base: str, resolver) -> bool:
        active_huggingface = resolver.active_evidence(self.name)
        if not active_huggingface:
            # Import-only PyTorch evidence cannot claim training ownership.
            # A Trainer/config candidate outranks shared data plumbing or a
            # model class and routes safely to nvflare-orient. Without a
            # Trainer candidate, active base evidence keeps the PyTorch
            # converter while pure inference remains recommendation-free.
            if resolver.training_owner_evidence(family_base):
                return False
            candidate_evidence = [
                item for item in resolver.evidence(self.name) if item.get("kind") in TRAINER_CANDIDATE_EVIDENCE
            ]
            return bool(candidate_evidence) or not resolver.active_evidence(family_base)
        if resolver.tied_to_entry_context(active_huggingface):
            return True
        pytorch_outside_trainer_files = resolver.evidence_outside_files(
            resolver.evidence(family_base), active_huggingface
        )
        return resolver.score(active_huggingface) > resolver.score(pytorch_outside_trainer_files)

    def fallback_skill_for(self, evidence: list[dict]) -> Optional[str]:
        if any(item.get("kind") in TRAINER_CANDIDATE_EVIDENCE for item in evidence):
            return "nvflare-orient"
        return None

    @staticmethod
    def _is_huggingface_module(module: str) -> bool:
        return any(module == root or module.startswith(f"{root}.") for root in HUGGINGFACE_MODULES)

    @staticmethod
    def _is_trainer_submodule(module: str, symbol: str) -> bool:
        if symbol in HUGGINGFACE_TRAINER_SUBMODULES:
            return True
        return (module == "trl.trainer" or module.startswith("trl.trainer.")) and symbol.endswith("_trainer")

    @staticmethod
    def _is_patch_call(
        call_name: str,
        file_state: _HuggingFaceFileState,
        scope: tuple[str, ...],
    ) -> bool:
        _, _, symbol = call_name.rpartition(".")
        if symbol != "patch":
            return False
        identity_name = HuggingFaceDetector._patch_identity_name(call_name)
        return bool(identity_name and file_state.scopes.has_identity(identity_name, scope, file_state.patch_modules))

    @staticmethod
    def _patch_identity_name(call_name: str) -> Optional[str]:
        prefix, separator, symbol = call_name.rpartition(".")
        if not separator:
            return call_name
        if symbol != "patch":
            return None
        if prefix == PATCH_MODULE:
            return prefix.split(".", 1)[0]
        return prefix

    @classmethod
    def _is_trainer_name(cls, name: str, file_state: _HuggingFaceFileState, scope: tuple[str, ...] = ()) -> bool:
        if cls._has_identity(name, scope, file_state.trainer_classes, file_state):
            return True
        symbol = cls._lookup_mapping(name, scope, file_state.trainer_symbols, file_state)
        if cls._is_trainer_symbol(symbol or ""):
            return True
        prefix, separator, symbol = name.rpartition(".")
        if not separator or not cls._is_trainer_symbol(symbol):
            return False
        module = cls._lookup_mapping(prefix, scope, file_state.module_aliases, file_state) or prefix
        return cls._is_huggingface_module(module)

    @classmethod
    def _is_training_config_name(
        cls, name: str, file_state: _HuggingFaceFileState, scope: tuple[str, ...] = ()
    ) -> bool:
        if cls._lookup_mapping(name, scope, file_state.config_symbols, file_state) in TRAINING_CONFIG_SYMBOLS:
            return True
        prefix, separator, symbol = name.rpartition(".")
        if not separator or symbol not in TRAINING_CONFIG_SYMBOLS:
            return False
        module = cls._lookup_mapping(prefix, scope, file_state.module_aliases, file_state) or prefix
        return cls._is_huggingface_module(module)

    @classmethod
    def _bind_name(cls, name: str, scope: tuple[str, ...], file_state: _HuggingFaceFileState) -> tuple[str, ...]:
        return file_state.scopes.bind(
            name,
            scope,
            file_state.module_aliases,
            file_state.config_symbols,
            file_state.trainer_symbols,
            file_state.trainer_classes,
            file_state.trainer_instances,
            file_state.patch_symbols,
            file_state.patch_modules,
        )

    @classmethod
    def _has_identity(
        cls,
        name: str,
        scope: tuple[str, ...],
        identities: set,
        file_state: _HuggingFaceFileState,
    ) -> bool:
        return cls._lookup_identity_key(name, scope, identities, file_state) is not None

    @staticmethod
    def _can_resolve_from_enclosing_scope(
        name: str,
        scope: tuple[str, ...],
        file_state: _HuggingFaceFileState,
    ) -> bool:
        return file_state.scopes.can_resolve_from_enclosing_scope(name, scope)

    @classmethod
    def _lookup_mapping(
        cls,
        name: str,
        scope: tuple[str, ...],
        identities: dict,
        file_state: _HuggingFaceFileState,
    ):
        return file_state.scopes.lookup_mapping(name, scope, identities)

    @classmethod
    def _lookup_identity_key(
        cls,
        name: str,
        scope: tuple[str, ...],
        identities: set,
        file_state: _HuggingFaceFileState,
    ):
        return file_state.scopes.lookup_identity_key(name, scope, identities)

    @staticmethod
    def _train_call_receiver(call_name: str) -> Optional[str]:
        receiver, separator, method = call_name.rpartition(".")
        return receiver if separator and method == "train" else None

    @staticmethod
    def _is_trainer_symbol(symbol: str) -> bool:
        return symbol.endswith("Trainer")
