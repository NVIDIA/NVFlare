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

from .base import DetectContext, FrameworkDetector

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
        alias_name = alias.asname or name
        if name == "torch":
            file_state.torch_aliases.add(alias_name)
        elif name == "torch.nn":
            file_state.torch_nn_aliases.add(alias_name)
        elif name == "torch.optim":
            file_state.torch_optim_aliases.add(alias_name)
        elif name == "torch.utils.data":
            file_state.torch_data_aliases.add(alias_name)

    def on_import_from(
        self,
        module: str,
        aliases: list,
        file_state: _PyTorchFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        if module == "torch":
            for alias in aliases:
                alias_name = alias.asname or alias.name
                if alias.name == "nn":
                    file_state.torch_nn_aliases.add(alias_name)
                elif alias.name == "optim":
                    file_state.torch_optim_aliases.add(alias_name)
        elif module == "torch.nn":
            for alias in aliases:
                alias_name = alias.asname or alias.name
                if alias.name in PYTORCH_MODULE_SYMBOLS:
                    file_state.module_symbols.add(alias_name)
                elif alias.name in PYTORCH_TRAINING_SYMBOLS:
                    file_state.training_symbols[alias_name] = alias.name
        elif module == "torch.optim":
            for alias in aliases:
                if alias.name in PYTORCH_TRAINING_SYMBOLS:
                    file_state.training_symbols[alias.asname or alias.name] = alias.name
        elif module == "torch.utils.data":
            for alias in aliases:
                if alias.name in PYTORCH_TRAINING_SYMBOLS:
                    file_state.training_symbols[alias.asname or alias.name] = alias.name

    def on_class_base(
        self,
        base_name: str,
        lineno: Optional[int],
        file_state: _PyTorchFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        if self._is_pytorch_class_base(base_name, file_state):
            ctx.evidence(FRAMEWORK, "pytorch_class", base_name, lineno)

    def on_call(
        self,
        call_name: str,
        lineno: Optional[int],
        file_state: _PyTorchFileState,
        ctx: DetectContext,
        scope: tuple[str, ...] = (),
    ) -> None:
        symbol = self._pytorch_activity_symbol(call_name, file_state)
        if symbol:
            kind = "pytorch_data_call" if symbol in PYTORCH_DATA_SYMBOLS else "pytorch_call"
            ctx.evidence(FRAMEWORK, kind, call_name, lineno)

    def is_active_evidence(self, evidence: dict) -> bool:
        return evidence.get("kind") in {"pytorch_class", "pytorch_call", "pytorch_data_call"}

    def is_training_owner_evidence(self, evidence: dict) -> bool:
        return evidence.get("kind") == "pytorch_call"

    @staticmethod
    def _is_pytorch_class_base(base_name: str, file_state: _PyTorchFileState) -> bool:
        if base_name in file_state.module_symbols:
            return True
        if "." not in base_name:
            return False
        prefix, _, symbol = base_name.rpartition(".")
        if symbol not in PYTORCH_MODULE_SYMBOLS:
            return False
        if prefix in file_state.torch_nn_aliases:
            return True
        for alias in file_state.torch_aliases:
            if prefix == f"{alias}.nn":
                return True
        return False

    @staticmethod
    def _pytorch_activity_symbol(call_name: str, file_state: _PyTorchFileState) -> Optional[str]:
        if call_name in file_state.training_symbols:
            return file_state.training_symbols[call_name]
        if "." not in call_name:
            return None
        prefix, _, symbol = call_name.rpartition(".")
        if symbol not in PYTORCH_TRAINING_SYMBOLS:
            return None
        if (
            prefix in file_state.torch_nn_aliases
            or prefix in file_state.torch_optim_aliases
            or prefix in file_state.torch_data_aliases
        ):
            return symbol
        for alias in file_state.torch_aliases:
            if prefix in {f"{alias}.nn", f"{alias}.optim", f"{alias}.utils.data"}:
                return symbol
        return None
