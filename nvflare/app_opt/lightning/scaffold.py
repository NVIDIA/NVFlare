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
from collections.abc import Mapping

import pytorch_lightning as pl
import torch

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AlgorithmConstants
from nvflare.app_opt.pt.scaffold import PTScaffoldHelper
from nvflare.app_opt.pt.utils import inspect_model_params


class _StateDictSnapshot:
    """Keep an immutable model-state snapshot in CPU memory."""

    def __init__(self, state_dict: Mapping):
        self._state_dict = {key: value.detach().cpu().clone() for key, value in state_dict.items()}

    def state_dict(self) -> dict:
        return self._state_dict


class _ScaffoldHandler:
    """Apply SCAFFOLD client updates around Lightning's automatic optimization loop."""

    def __init__(self):
        self._helper = None
        self._ever_activated = False
        self._active = False
        self._model_global = None
        self._c_global_para = None
        self._c_local_para = None
        self._pending_lr = None
        self._lr_sum = 0.0
        self._num_steps = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def start_round(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", input_model: FLModel):
        meta = input_model.meta or {}
        control_key = AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL
        if control_key not in meta:
            if self._ever_activated:
                raise RuntimeError(
                    "SCAFFOLD was active in an earlier training round, but the received FLModel is missing "
                    f"meta['{control_key}']."
                )
            self._active = False
            return
        global_controls = meta[control_key]

        if not getattr(pl_module, "automatic_optimization", True):
            raise RuntimeError(
                "Automatic Lightning SCAFFOLD support requires automatic optimization. "
                "For manual optimization, use an explicit receive/train/send loop without flare.patch() and "
                "integrate PTScaffoldHelper directly."
            )

        scaler = getattr(trainer, "scaler", None)
        if scaler is None:
            scaler = getattr(getattr(trainer, "precision_plugin", None), "scaler", None)
        if scaler is not None:
            raise RuntimeError(
                "Automatic Lightning SCAFFOLD support does not support mixed precision that uses a gradient "
                "scaler because the scaler can skip optimizer steps. Use precision='32-true' or "
                "precision='bf16-mixed', or use an explicit receive/train/send loop without flare.patch() and "
                "integrate PTScaffoldHelper directly."
            )

        optimizers = getattr(trainer, "optimizers", None) or []
        if len(optimizers) != 1:
            raise RuntimeError(
                "Automatic Lightning SCAFFOLD support requires exactly one optimizer, "
                f"but the Trainer has {len(optimizers)}."
            )

        if not isinstance(global_controls, Mapping) or not global_controls:
            raise RuntimeError(f"SCAFFOLD requires non-empty mapping metadata at FLModel.meta['{control_key}'].")

        local_state = pl_module.state_dict()
        self._validate_controls(local_state=local_state, global_controls=global_controls)
        controls_on_device = self._controls_to_device(local_state=local_state, global_controls=global_controls)

        try:
            device = next(pl_module.parameters()).device
        except StopIteration as e:
            raise RuntimeError("Automatic Lightning SCAFFOLD support requires a model with parameters.") from e

        if self._helper is None:
            self._helper = PTScaffoldHelper()
            self._helper.init(pl_module)
        else:
            self._helper.c_global.to(device)
            self._helper.c_local.to(device)

        self._model_global = _StateDictSnapshot(local_state)
        self._helper.load_global_controls(controls_on_device)
        self._c_global_para, self._c_local_para = self._helper.get_params()
        self._pending_lr = None
        self._lr_sum = 0.0
        self._num_steps = 0
        self._active = True
        self._ever_activated = True

    def before_optimizer_step(self, optimizer):
        if not self._active:
            return
        if self._pending_lr is not None:
            raise RuntimeError(
                "Automatic Lightning SCAFFOLD support observed multiple optimizer steps in one training batch. "
                "Manual or multi-step optimization is not supported."
            )

        lr_values = [float(group["lr"]) for group in optimizer.param_groups]
        if not lr_values or any(not math.isfinite(lr) or lr < 0.0 for lr in lr_values):
            raise RuntimeError(
                "Automatic Lightning SCAFFOLD support requires every optimizer parameter group "
                f"to have a finite non-negative learning rate, but received {lr_values}."
            )

        reference_lr = lr_values[0]
        if any(not math.isclose(lr, reference_lr, rel_tol=1e-12, abs_tol=0.0) for lr in lr_values[1:]):
            raise RuntimeError(
                "Automatic Lightning SCAFFOLD support requires all optimizer parameter groups "
                f"to use the same learning rate, but received {lr_values}."
            )
        self._pending_lr = reference_lr

    def after_train_batch(self, pl_module: "pl.LightningModule"):
        if not self._active or self._pending_lr is None:
            return

        curr_lr = self._pending_lr
        self._helper.model_update(
            model=pl_module,
            curr_lr=curr_lr,
            c_global_para=self._c_global_para,
            c_local_para=self._c_local_para,
        )
        self._pending_lr = None
        self._lr_sum += curr_lr
        self._num_steps += 1

    def finish_round(self, pl_module: "pl.LightningModule") -> dict:
        if not self._active:
            return {}
        if self._pending_lr is not None:
            raise RuntimeError(
                "Automatic Lightning SCAFFOLD support observed an optimizer step, but Lightning did not call "
                "on_train_batch_end to apply the final model correction. Training may have been interrupted "
                "between optimizer hooks, or a Lightning plugin may have changed the documented hook sequence."
            )
        if self._num_steps == 0:
            raise RuntimeError(
                "Automatic Lightning SCAFFOLD support requires at least one completed optimizer step per round."
            )
        if not math.isfinite(self._lr_sum) or self._lr_sum <= 0.0:
            raise RuntimeError(
                "Automatic Lightning SCAFFOLD support requires positive total learning-rate exposure per round, "
                f"but the {self._num_steps} completed optimizer steps summed to {self._lr_sum}."
            )

        average_lr = self._lr_sum / self._num_steps
        self._helper.terms_update(
            model=pl_module,
            curr_lr=average_lr,
            c_global_para=self._c_global_para,
            c_local_para=self._c_local_para,
            model_global=self._model_global,
        )
        result = {AlgorithmConstants.SCAFFOLD_CTRL_DIFF: self._helper.get_delta_controls()}

        self._active = False
        self._model_global = None
        self._c_global_para = None
        self._c_local_para = None
        self._pending_lr = None
        return result

    @staticmethod
    def _validate_controls(local_state: Mapping, global_controls: Mapping):
        local_keys = set(local_state)
        control_keys = set(global_controls)
        missing_keys = sorted(local_keys - control_keys)
        unexpected_keys = sorted(control_keys - local_keys)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "SCAFFOLD global controls must exactly match the Lightning module state dict. "
                f"Missing keys: {missing_keys[:5]}; unexpected keys: {unexpected_keys[:5]}."
            )

        report = inspect_model_params(local_state, global_controls)
        if report.shape_mismatches:
            raise RuntimeError(f"Invalid SCAFFOLD global controls. {report.format_shape_mismatch_error()}")

    @staticmethod
    def _controls_to_device(local_state: Mapping, global_controls: Mapping) -> dict:
        result = {}
        for key, value in global_controls.items():
            expected = local_state[key]
            try:
                result[key] = torch.as_tensor(value, dtype=expected.dtype, device=expected.device)
            except (TypeError, ValueError, RuntimeError) as e:
                raise RuntimeError(f"Failed to convert SCAFFOLD global control '{key}' to a tensor: {e}") from e
        return result
