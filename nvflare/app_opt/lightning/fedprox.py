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

import math
from numbers import Real

import pytorch_lightning as pl
import torch

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AlgorithmConstants


class _FedProxHandler:
    """Inject the exact FedProx proximal gradient into Lightning optimization."""

    def __init__(self):
        self._metadata_seen = False
        self._active = False
        self._mu = None
        self._module = None
        self._optimizer = None
        self._parameters = None
        self._global_parameters = None
        self._trainability = None
        self._pending_step = False
        self._num_steps = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def start_round(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", input_model: FLModel):
        meta = input_model.meta or {}
        mu_key = AlgorithmConstants.FEDPROX_MU
        if mu_key not in meta:
            if self._metadata_seen:
                raise RuntimeError(
                    "FedProx metadata was received in an earlier training round, but the received FLModel is "
                    f"missing meta['{mu_key}']. Custom schedules must send an explicit 0.0 to disable FedProx."
                )
            self._clear_round_state()
            return

        mu = self._validate_mu(meta[mu_key])
        self._metadata_seen = True
        if mu == 0.0:
            self._clear_round_state()
            return

        if not getattr(pl_module, "automatic_optimization", True):
            raise RuntimeError(
                "Automatic Lightning FedProx support requires automatic optimization. For manual optimization, "
                "use an explicit receive/train/send loop without flare.patch() and add the proximal term directly."
            )

        precision = str(getattr(trainer, "precision", ""))
        if precision not in {"32-true", "bf16-mixed"}:
            raise RuntimeError(
                "Automatic Lightning FedProx support requires trainer.precision to be '32-true' or "
                f"'bf16-mixed', but received {precision!r}."
            )

        scaler = getattr(trainer, "scaler", None)
        if scaler is None:
            scaler = getattr(getattr(trainer, "precision_plugin", None), "scaler", None)
        if scaler is not None:
            raise RuntimeError(
                "Automatic Lightning FedProx support does not support scaler-backed precision because the "
                "scaler can skip optimizer steps. Use precision='32-true' or precision='bf16-mixed'."
            )

        optimizers = getattr(trainer, "optimizers", None) or []
        if len(optimizers) != 1:
            raise RuntimeError(
                "Automatic Lightning FedProx support requires exactly one optimizer, "
                f"but the Trainer has {len(optimizers)}."
            )
        optimizer = optimizers[0]
        if isinstance(optimizer, torch.optim.LBFGS):
            raise RuntimeError("Automatic Lightning FedProx support does not support closure-based LBFGS.")
        if isinstance(optimizer, torch.optim.SparseAdam):
            raise RuntimeError("Automatic Lightning FedProx support does not support SparseAdam.")

        named_parameters = dict(pl_module.named_parameters())
        if not named_parameters:
            raise RuntimeError("Automatic Lightning FedProx support requires a model with parameters.")
        parameter_names = {id(parameter): name for name, parameter in named_parameters.items()}
        optimizer_parameters = []
        seen = set()
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter_id = id(parameter)
                if parameter_id not in parameter_names:
                    raise RuntimeError(
                        "Automatic Lightning FedProx support found an optimizer parameter that is not owned by "
                        "the LightningModule."
                    )
                if parameter_id in seen:
                    raise RuntimeError(
                        f"Automatic Lightning FedProx support found duplicate optimizer parameter "
                        f"'{parameter_names[parameter_id]}'."
                    )
                seen.add(parameter_id)
                if parameter.requires_grad:
                    optimizer_parameters.append((parameter_names[parameter_id], parameter))

        if not optimizer_parameters:
            raise RuntimeError(
                "Automatic Lightning FedProx support requires at least one optimizer-owned trainable parameter."
            )

        self._mu = mu
        self._module = pl_module
        self._optimizer = optimizer
        self._parameters = optimizer_parameters
        self._global_parameters = {name: parameter.detach().clone() for name, parameter in self._parameters}
        self._trainability = {
            name: (id(parameter), parameter.requires_grad) for name, parameter in named_parameters.items()
        }
        self._pending_step = False
        self._num_steps = 0
        self._active = True

    @staticmethod
    def _validate_mu(value) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeError(
                f"FLModel.meta['{AlgorithmConstants.FEDPROX_MU}'] must be a finite non-negative number, "
                f"but received {value!r}."
            )
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(
                f"FLModel.meta['{AlgorithmConstants.FEDPROX_MU}'] must be a finite non-negative number, "
                f"but received {value!r}."
            )
        return value

    def _validate_trainability(self, pl_module: "pl.LightningModule") -> None:
        current = {name: (id(parameter), parameter.requires_grad) for name, parameter in pl_module.named_parameters()}
        if current != self._trainability:
            raise RuntimeError(
                "Automatic Lightning FedProx support detected a model parameter or trainability change during "
                "the training round. Parameters may be frozen or unfrozen only between rounds."
            )

    def before_optimizer_step(self, optimizer):
        if not self._active:
            return
        if optimizer is not self._optimizer:
            raise RuntimeError("Automatic Lightning FedProx support received an unexpected optimizer.")
        if self._pending_step:
            raise RuntimeError(
                "Automatic Lightning FedProx support observed multiple optimizer steps in one training batch. "
                "Manual or multi-step optimization is not supported."
            )

        self._validate_trainability(self._module)
        for name, parameter in self._parameters:
            gradient = parameter.grad
            if gradient is not None and gradient.layout != torch.strided:
                raise RuntimeError(
                    f"Automatic Lightning FedProx support requires dense gradients, but parameter '{name}' "
                    f"has gradient layout {gradient.layout}."
                )

        with torch.no_grad():
            for name, parameter in self._parameters:
                proximal_gradient = parameter.detach() - self._global_parameters[name]
                if parameter.grad is None:
                    parameter.grad = proximal_gradient.mul(self._mu)
                else:
                    parameter.grad.add_(proximal_gradient, alpha=self._mu)
        self._pending_step = True

    def after_train_batch(self, pl_module: "pl.LightningModule"):
        if not self._active or not self._pending_step:
            return
        self._validate_trainability(pl_module)
        self._pending_step = False
        self._num_steps += 1

    def finish_round(self, pl_module: "pl.LightningModule") -> dict:
        if not self._active:
            return {}
        try:
            self._validate_trainability(pl_module)
            if self._pending_step:
                raise RuntimeError(
                    "Automatic Lightning FedProx support observed an optimizer step, but Lightning did not call "
                    "on_train_batch_end to complete the hook sequence. Training may have been interrupted."
                )
            if self._num_steps == 0:
                raise RuntimeError(
                    "Automatic Lightning FedProx support requires at least one completed optimizer step per round."
                )
            return {}
        finally:
            self._clear_round_state()

    def abort_round(self) -> None:
        """Release state after this or another handler fails to start the round."""
        self._clear_round_state()

    def _clear_round_state(self) -> None:
        self._active = False
        self._mu = None
        self._module = None
        self._optimizer = None
        self._parameters = None
        self._global_parameters = None
        self._trainability = None
        self._pending_step = False
        self._num_steps = 0
