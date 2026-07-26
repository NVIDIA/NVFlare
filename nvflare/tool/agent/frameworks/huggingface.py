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

from .base import DetectContext, FrameworkDetector

FRAMEWORK = "huggingface"

HUGGINGFACE_MODULES = {"transformers", "trl"}
PATCH_PARENT_MODULE = "nvflare.client"
PATCH_SUBMODULE = "hf"
PATCH_MODULE = f"{PATCH_PARENT_MODULE}.{PATCH_SUBMODULE}"


@dataclass
class _HuggingFaceFileState:
    module_aliases: dict = field(default_factory=dict)
    trainer_symbols: dict = field(default_factory=dict)
    patch_symbols: set = field(default_factory=set)
    patch_modules: set = field(default_factory=set)


class HuggingFaceDetector(FrameworkDetector):
    name = FRAMEWORK
    import_roots = {"transformers": FRAMEWORK, "trl": FRAMEWORK}
    evidence_weights = {"huggingface_trainer": 4, "huggingface_trainer_class": 4}
    recommended_skill = "nvflare-convert-huggingface"
    recommendation_requires_active_evidence = True

    def new_file_state(self) -> _HuggingFaceFileState:
        return _HuggingFaceFileState()

    def on_import(self, alias: ast.alias, file_state: _HuggingFaceFileState, ctx: DetectContext) -> None:
        if self._is_huggingface_module(alias.name):
            file_state.module_aliases[alias.asname or alias.name] = alias.name
        if alias.name == PATCH_MODULE:
            file_state.patch_modules.add(alias.asname or alias.name)

    def on_import_from(self, module: str, aliases: list, file_state: _HuggingFaceFileState, ctx: DetectContext) -> None:
        if self._is_huggingface_module(module):
            for alias in aliases:
                if self._is_trainer_symbol(alias.name):
                    file_state.trainer_symbols[alias.asname or alias.name] = alias.name
        if module == PATCH_MODULE:
            for alias in aliases:
                if alias.name == "patch":
                    file_state.patch_symbols.add(alias.asname or alias.name)
        elif module == PATCH_PARENT_MODULE:
            for alias in aliases:
                if alias.name == PATCH_SUBMODULE:
                    file_state.patch_modules.add(alias.asname or alias.name)

    def on_class_base(
        self, base_name: str, lineno: Optional[int], file_state: _HuggingFaceFileState, ctx: DetectContext
    ) -> None:
        if self._is_trainer_name(base_name, file_state):
            ctx.evidence(FRAMEWORK, "huggingface_trainer_class", base_name, lineno)

    def on_call(
        self, call_name: str, lineno: Optional[int], file_state: _HuggingFaceFileState, ctx: DetectContext
    ) -> None:
        if call_name in file_state.patch_symbols or self._is_patch_call(call_name, file_state):
            ctx.flare_call(call_name)
            ctx.integration_signal(FRAMEWORK, call_name)
        if self._is_trainer_name(call_name, file_state):
            ctx.evidence(FRAMEWORK, "huggingface_trainer", call_name, lineno)

    def is_active_evidence(self, evidence: dict) -> bool:
        return evidence.get("kind") in {"huggingface_trainer", "huggingface_trainer_class"}

    @staticmethod
    def _is_huggingface_module(module: str) -> bool:
        return any(module == root or module.startswith(f"{root}.") for root in HUGGINGFACE_MODULES)

    @staticmethod
    def _is_patch_call(call_name: str, file_state: _HuggingFaceFileState) -> bool:
        prefix, _, symbol = call_name.rpartition(".")
        return symbol == "patch" and (prefix in file_state.patch_modules or prefix == PATCH_MODULE)

    @classmethod
    def _is_trainer_name(cls, name: str, file_state: _HuggingFaceFileState) -> bool:
        if cls._is_trainer_symbol(file_state.trainer_symbols.get(name, "")):
            return True
        prefix, separator, symbol = name.rpartition(".")
        if not separator or not cls._is_trainer_symbol(symbol):
            return False
        module = file_state.module_aliases.get(prefix, prefix)
        return cls._is_huggingface_module(module)

    @staticmethod
    def _is_trainer_symbol(symbol: str) -> bool:
        return symbol == "Trainer" or symbol.endswith("Trainer")
