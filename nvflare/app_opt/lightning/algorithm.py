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

import logging
from dataclasses import dataclass, field
from typing import Optional

import pytorch_lightning as pl

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AlgorithmConstants

logger = logging.getLogger(__name__)


@dataclass
class _AlgorithmResult:
    metadata: dict = field(default_factory=dict)
    num_steps: Optional[int] = None


def _create_scaffold_handler():
    # Keep the algorithm implementation out of the generic Lightning import path.
    from .scaffold import _ScaffoldHandler

    return _ScaffoldHandler()


def _create_fedprox_handler():
    # Keep the algorithm implementation out of the generic Lightning import path.
    from .fedprox import _FedProxHandler

    return _FedProxHandler()


def _get_handler_factories():
    # Resolve factory names at round start so tests and integrations can replace a private factory safely.
    return (
        (AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL, _create_scaffold_handler),
        (AlgorithmConstants.FEDPROX_MU, _create_fedprox_handler),
    )


class _AlgorithmHandlerManager:
    """Lazily compose and forward Lightning hooks to client algorithm handlers."""

    def __init__(self):
        self._handlers = {}

    def start_round(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", input_model: FLModel):
        meta = input_model.meta or {}
        for metadata_key, factory in _get_handler_factories():
            if metadata_key in meta and metadata_key not in self._handlers:
                self._handlers[metadata_key] = factory()

        for handler in self._handlers.values():
            handler.start_round(trainer=trainer, pl_module=pl_module, input_model=input_model)

    def before_optimizer_step(self, optimizer):
        for handler in self._handlers.values():
            if handler.active:
                handler.before_optimizer_step(optimizer)

    def after_train_batch(self, pl_module: "pl.LightningModule"):
        for handler in self._handlers.values():
            if handler.active:
                handler.after_train_batch(pl_module)

    def finish_round(self, pl_module: "pl.LightningModule") -> _AlgorithmResult:
        active_handlers = [handler for handler in self._handlers.values() if handler.active]
        if not active_handlers:
            return _AlgorithmResult()

        step_counts = [handler.num_steps for handler in active_handlers]
        handler_errors = []
        if any(num_steps != step_counts[0] for num_steps in step_counts[1:]):
            handler_errors.append(
                RuntimeError(
                    "Lightning automatic algorithm handlers reported inconsistent completed optimizer step counts: "
                    f"{step_counts}."
                )
            )

        metadata = {}
        for handler in active_handlers:
            try:
                result = handler.finish_round(pl_module)
            except Exception as e:
                handler_errors.append(e)
                continue
            collisions = sorted(set(metadata).intersection(result))
            if collisions:
                handler_errors.append(
                    RuntimeError(
                        "Lightning automatic algorithm handlers returned conflicting metadata keys: " f"{collisions}."
                    )
                )
                continue
            metadata.update(result)
        if handler_errors:
            for error in handler_errors[1:]:
                logger.error(
                    "An additional Lightning automatic algorithm handler failed while finishing the round.",
                    exc_info=(type(error), error, error.__traceback__),
                )
            raise handler_errors[0]
        return _AlgorithmResult(metadata=metadata, num_steps=step_counts[0])
