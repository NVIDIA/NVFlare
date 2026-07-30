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

"""PyTorch framework detector."""

import ast
from dataclasses import dataclass, field
from typing import Optional

from .base import DetectContext, FrameworkDetector, LexicalScopeBindings

FRAMEWORK = "pytorch"

PYTORCH_MODULE_SYMBOLS = {"Module"}
PYTORCH_DATA_SYMBOLS = {"DataLoader", "DistributedSampler", "TensorDataset"}
PYTORCH_OPTIM_LOSS_SYMBOLS = {
    "Adagrad",
    "Adam",
    "AdamW",
    "BCELoss",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "MSELoss",
    "NLLLoss",
    "RMSprop",
    "SGD",
}
PYTORCH_TRAINING_SYMBOLS = PYTORCH_DATA_SYMBOLS | PYTORCH_OPTIM_LOSS_SYMBOLS


@dataclass
class _PyTorchFileState:
    torch_aliases: set = field(default_factory=set)
    torch_nn_aliases: set = field(default_factory=set)
    torch_optim_aliases: set = field(default_factory=set)
    torch_data_aliases: set = field(default_factory=set)
    module_symbols: set = field(default_factory=set)
    training_symbols: dict = field(default_factory=dict)
    scopes: LexicalScopeBindings = field(default_factory=LexicalScopeBindings)


class PyTorchDetector(FrameworkDetector):
    name = FRAMEWORK
    import_roots = {"torch": FRAMEWORK, "torchvision": FRAMEWORK, "torchaudio": FRAMEWORK}
    evidence_weights = {"import": 1, "pytorch_call": 2, "pytorch_data_call": 2, "pytorch_class": 3}
    recommended_skill = "nvflare-convert-pytorch"

    def new_file_state(self) -> _PyTorchFileState:
        return _PyTorchFileState()

    def on_import(
        self,
        alias: ast.alias,
        file_state: _PyTorchFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        name = alias.name
        alias_name = alias.asname or name.split(".", 1)[0]
        binding_scope = self._bind_name(alias_name, scope, file_state)
        key = (binding_scope, alias_name)
        if name == "torch" or (name.startswith("torch.") and not alias.asname):
            file_state.torch_aliases.add(key)
        elif name == "torch.nn":
            file_state.torch_nn_aliases.add(key)
        elif name == "torch.optim":
            file_state.torch_optim_aliases.add(key)
        elif name == "torch.utils.data":
            file_state.torch_data_aliases.add(key)

    def on_import_from(
        self,
        module: str,
        aliases: list,
        file_state: _PyTorchFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        bound_names = {}
        for alias in aliases:
            if alias.name == "*":
                continue
            alias_name = alias.asname or alias.name
            bound_names[alias_name] = self._bind_name(alias_name, scope, file_state)

        if module == "torch":
            for alias in aliases:
                alias_name = alias.asname or alias.name
                if alias.name == "nn":
                    file_state.torch_nn_aliases.add((bound_names.get(alias_name, scope), alias_name))
                elif alias.name == "optim":
                    file_state.torch_optim_aliases.add((bound_names.get(alias_name, scope), alias_name))
        elif module == "torch.nn":
            for alias in aliases:
                alias_name = alias.asname or alias.name
                if alias.name in PYTORCH_MODULE_SYMBOLS:
                    file_state.module_symbols.add((bound_names.get(alias_name, scope), alias_name))
                elif alias.name in PYTORCH_TRAINING_SYMBOLS:
                    file_state.training_symbols[(bound_names.get(alias_name, scope), alias_name)] = alias.name
        elif module == "torch.optim":
            for alias in aliases:
                if alias.name in PYTORCH_TRAINING_SYMBOLS:
                    alias_name = alias.asname or alias.name
                    file_state.training_symbols[(bound_names.get(alias_name, scope), alias_name)] = alias.name
        elif module == "torch.utils.data":
            for alias in aliases:
                if alias.name in PYTORCH_TRAINING_SYMBOLS:
                    alias_name = alias.asname or alias.name
                    file_state.training_symbols[(bound_names.get(alias_name, scope), alias_name)] = alias.name

    def on_scope(
        self,
        scope: tuple[str, ...],
        local_names: set[str],
        global_names: set[str],
        nonlocal_names: set[str],
        file_state: _PyTorchFileState,
        ctx: DetectContext,
    ) -> None:
        file_state.scopes.declare_scope(scope, local_names, global_names, nonlocal_names)

    def on_class_definition(
        self,
        class_name: str,
        base_names: list[str],
        lineno: Optional[int],
        file_state: _PyTorchFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        self._bind_name(class_name, scope, file_state)

    def on_class_base(
        self,
        base_name: str,
        lineno: Optional[int],
        file_state: _PyTorchFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        if self._is_pytorch_class_base(base_name, file_state, scope):
            ctx.evidence(FRAMEWORK, "pytorch_class", base_name, lineno)

    def on_call(
        self,
        call_name: str,
        lineno: Optional[int],
        file_state: _PyTorchFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        symbol = self._pytorch_activity_symbol(call_name, file_state, scope)
        if symbol:
            kind = "pytorch_data_call" if symbol in PYTORCH_DATA_SYMBOLS else "pytorch_call"
            ctx.evidence(FRAMEWORK, kind, call_name, lineno)

    def on_assignment(
        self,
        target_names: list[str],
        call_name: Optional[str],
        lineno: Optional[int],
        file_state: _PyTorchFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
        value_info=None,
    ) -> None:
        for target_name in target_names:
            self._bind_name(target_name, scope, file_state)

    def is_active_evidence(self, evidence: dict) -> bool:
        return evidence.get("kind") in {"pytorch_class", "pytorch_call", "pytorch_data_call"}

    def is_training_owner_evidence(self, evidence: dict) -> bool:
        return evidence.get("kind") == "pytorch_call"

    @staticmethod
    def _is_pytorch_class_base(
        base_name: str,
        file_state: _PyTorchFileState,
        scope: tuple[str, ...],
    ) -> bool:
        if file_state.scopes.has_identity(base_name, scope, file_state.module_symbols):
            return True
        if "." not in base_name:
            return False
        prefix, _, symbol = base_name.rpartition(".")
        if symbol not in PYTORCH_MODULE_SYMBOLS:
            return False
        if file_state.scopes.has_identity(prefix, scope, file_state.torch_nn_aliases):
            return True
        root, separator, rest = prefix.partition(".")
        return bool(
            separator and rest == "nn" and file_state.scopes.has_identity(root, scope, file_state.torch_aliases)
        )

    @staticmethod
    def _pytorch_activity_symbol(
        call_name: str,
        file_state: _PyTorchFileState,
        scope: tuple[str, ...],
    ) -> Optional[str]:
        imported_symbol = file_state.scopes.lookup_mapping(call_name, scope, file_state.training_symbols)
        if imported_symbol:
            return imported_symbol
        if "." not in call_name:
            return None
        prefix, _, symbol = call_name.rpartition(".")
        if symbol not in PYTORCH_TRAINING_SYMBOLS:
            return None
        if (
            file_state.scopes.has_identity(prefix, scope, file_state.torch_nn_aliases)
            or file_state.scopes.has_identity(prefix, scope, file_state.torch_optim_aliases)
            or file_state.scopes.has_identity(prefix, scope, file_state.torch_data_aliases)
        ):
            return symbol
        root, separator, rest = prefix.partition(".")
        if (
            separator
            and rest in {"nn", "optim", "utils.data"}
            and file_state.scopes.has_identity(root, scope, file_state.torch_aliases)
        ):
            return symbol
        return None

    @staticmethod
    def _bind_name(
        name: str,
        scope: tuple[str, ...],
        file_state: _PyTorchFileState,
    ) -> tuple[str, ...]:
        return file_state.scopes.bind(
            name,
            scope,
            file_state.torch_aliases,
            file_state.torch_nn_aliases,
            file_state.torch_optim_aliases,
            file_state.torch_data_aliases,
            file_state.module_symbols,
            file_state.training_symbols,
        )
