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

from dataclasses import dataclass, field
from typing import Optional

import pytorch_lightning as pl

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AlgorithmConstants


@dataclass
class _AlgorithmResult:
    metadata: dict = field(default_factory=dict)
    num_steps: Optional[int] = None


def _create_scaffold_handler():
    # Keep the algorithm implementation out of the generic Lightning import path.
    from .scaffold import _ScaffoldHandler

    return _ScaffoldHandler()


class _AlgorithmHandlerManager:
    """Lazily select and forward Lightning hooks to a client algorithm handler."""

    def __init__(self):
        self._handler = None

    def start_round(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", input_model: FLModel):
        if self._handler is None:
            meta = input_model.meta or {}
            if AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL not in meta:
                return
            self._handler = _create_scaffold_handler()

        self._handler.start_round(trainer=trainer, pl_module=pl_module, input_model=input_model)

    def before_optimizer_step(self, optimizer):
        if self._handler is not None:
            self._handler.before_optimizer_step(optimizer)

    def after_train_batch(self, pl_module: "pl.LightningModule"):
        if self._handler is not None:
            self._handler.after_train_batch(pl_module)

    def finish_round(self, pl_module: "pl.LightningModule") -> _AlgorithmResult:
        if self._handler is None or not self._handler.active:
            return _AlgorithmResult()

        num_steps = self._handler.num_steps
        return _AlgorithmResult(metadata=self._handler.finish_round(pl_module), num_steps=num_steps)
