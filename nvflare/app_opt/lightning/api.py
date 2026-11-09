# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import logging
from typing import Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import TrainerFn
from torch import Tensor

from nvflare.app_common.abstract.fl_model import FLModel, MetaKey
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.app_opt.pt.utils import inspect_model_params
from nvflare.client.api import (
    clear,
    get_config,
    get_context,
    init,
    is_evaluate,
    is_submit_model,
    is_train,
    receive,
    send,
    system_info,
)
from nvflare.client.config import ConfigKey, TransferType
from nvflare.fuel.utils import fobs

from .algorithm import _AlgorithmHandlerManager
from .callbacks import RestoreState

FL_META_KEY = "__fl_meta__"


def patch(
    trainer: pl.Trainer,
    restore_state: bool = True,
    load_state_dict_strict: bool = True,
    update_fit_loop: bool = True,
    sync_every_n_steps: Optional[int] = None,
):
    """Patches the PyTorch Lightning Trainer for usage with NVFlare.

    Args:
        trainer: the PyTorch Lightning trainer.
        restore_state: whether to restore optimizer and learning rate scheduler states.
            Defaults to `True`. Task lifetime requires `False` and explicit native checkpoint
            restoration because the in-memory callback state cannot survive fresh workers.
        load_state_dict_strict: exposes `strict` argument of `torch.nn.Module.load_state_dict()`
            used to load the received model. Defaults to `True`.
            See https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict for details.
            NVFlare still validates incoming keys and shapes before calling ``load_state_dict()``.
            With ``True``, any incoming key that does not exist in the local Lightning
            module is rejected before loading. With ``False``, NVFlare warns and
            filters the payload down to matching keys, which is useful for partial
            model updates where the client only keeps part of the server keyspace.
        update_fit_loop: whether to increase `trainer.fit_loop.max_epochs` and `trainer.fit_loop.epoch_loop.max_steps` each FL round.
            Defaults to `True` which is suitable for most PyTorch Lightning applications.
        sync_every_n_steps: when set, one ``trainer.fit()`` exchanges the model with the server every
            ``sync_every_n_steps`` optimizer steps instead of once per ``fit()``. The received model is loaded
            in place, so the optimizer, LR scheduler and data position carry over. The server's ``num_rounds``
            decides how long training runs: the trainer's ``max_epochs``, ``max_steps``, ``min_epochs`` and
            ``min_steps`` are lifted, and ``fit()``
            ends when the server stops sending training tasks. An LR scheduler sized from the trainer's limits
            (``OneCycleLR``, cosine with ``T_max`` from ``estimated_stepping_batches``) must be sized for
            ``num_rounds * sync_every_n_steps`` instead; step-counting schedulers work unchanged. ``restore_state``
            and ``update_fit_loop`` have no effect in this mode, and ``train_with_evaluation`` is not supported.
            Defaults to `None` (one FL round per ``fit()``).

    Hooks:
        Any callback in ``trainer.callbacks`` may define ``on_fl_receive(trainer, pl_module, fl_model)``
        (before the received model is loaded), ``on_fl_model_loaded(trainer, pl_module, fl_model)`` and
        ``on_fl_before_send(trainer, pl_module, fl_model)``. They are called on every rank;
        ``on_fl_before_send`` gets ``None`` on ranks other than 0.

    SCAFFOLD:
        When the received model contains SCAFFOLD global controls, ``patch`` automatically
        applies ``PTScaffoldHelper`` around Lightning's optimizer steps and returns the
        required control difference. This supports automatic optimization with one optimizer
        and ``precision="32-true"`` or ``precision="bf16-mixed"``. Manual optimization and
        other precision modes are not supported by the patched path. Those clients must use
        an explicit receive/train/send loop and integrate ``PTScaffoldHelper`` directly.

    FedProx:
        When the received model contains a positive FedProx coefficient, ``patch`` automatically
        adds the exact proximal gradient to optimizer-owned trainable parameters. FedProx composes
        with automatic SCAFFOLD support. The proximal gradient is applied after gradient accumulation
        and AMP unscaling and before gradient clipping. Consequently, a loss logged from
        ``training_step`` excludes the injected proximal term even though optimization includes its
        exact gradient. Automatic FedProx has the same one-optimizer, automatic-optimization, and
        precision restrictions as automatic SCAFFOLD; closure-based LBFGS and sparse gradients are
        also unsupported.

    Example:

        Normal usage:

        .. code-block:: python

            trainer = Trainer(max_epochs=1)
            flare.patch(trainer)


        Advanced usage:

        If users want to pass additional information to FLARE server side via the lightning API,
        they will need to set the information inside the attributes called ``__fl_meta__`` in their LightningModule.

        .. code-block:: python

            class LitNet(LightningModule):
                def __init__(self):
                    super().__init__()
                    self.save_hyperparameters()
                    self.model = Net()
                    self.train_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
                    self.valid_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
                    self.__fl_meta__ = {"CUSTOM_VAR": "VALUE_OF_THE_VAR"}

    """
    if sync_every_n_steps is not None and (
        isinstance(sync_every_n_steps, bool) or not isinstance(sync_every_n_steps, int) or sync_every_n_steps < 1
    ):
        raise ValueError(f"sync_every_n_steps must be a positive int or None, got {sync_every_n_steps!r}")
    fobs.register(TensorDecomposer)
    callbacks = trainer.callbacks
    if isinstance(callbacks, Callback):
        callbacks = [callbacks]
    elif isinstance(callbacks, list):
        callbacks = list(callbacks)
    else:
        callbacks = []

    fl_callback = next((cb for cb in callbacks if isinstance(cb, _FLCallbackBase)), None)
    if fl_callback is None:
        if sync_every_n_steps is None:
            fl_callback = FLCallback(
                rank=trainer.global_rank, load_state_dict_strict=load_state_dict_strict, update_fit_loop=update_fit_loop
            )
        else:
            fl_callback = _StepSyncFLCallback(
                sync_every_n_steps, rank=trainer.global_rank, load_state_dict_strict=load_state_dict_strict
            )
        callbacks.append(fl_callback)
    elif sync_every_n_steps is not None and getattr(fl_callback, "sync_every_n_steps", None) != sync_every_n_steps:
        raise ValueError(
            f"sync_every_n_steps={sync_every_n_steps} conflicts with the {type(fl_callback).__name__} already on the "
            "trainer; remove it or let patch() create the callback."
        )
    step_mode = isinstance(fl_callback, _StepSyncFLCallback)

    # A caller can supply an existing callback without an active API context.
    # Patching remains valid before initialization; inspect only a live context.
    try:
        context = get_context()
    except RuntimeError:
        context = None
    if restore_state and not step_mode and context is not None and "task_checkpoint" in system_info(ctx=context):
        raise ValueError(
            "Lightning restore_state=True keeps optimizer and scheduler state only in memory and cannot "
            "restore it across task workers. Use job lifetime, or set restore_state=False and handle "
            "native checkpoint restoration explicitly."
        )

    if restore_state and not step_mode and not any(isinstance(cb, RestoreState) for cb in callbacks):
        callbacks.append(RestoreState())

    trainer.callbacks = callbacks


class _FLCallbackBase(Callback):
    """Receive, load and send logic shared by the fit-per-round and step-sync callbacks."""

    def __init__(self, rank: int = 0, load_state_dict_strict: bool = True):
        super().__init__()
        init(rank=str(rank))
        task_exchange = get_config().get(ConfigKey.TASK_EXCHANGE, {})
        self.train_with_evaluation = task_exchange.get(ConfigKey.TRAIN_WITH_EVAL, False)
        self._diff_transfer = task_exchange.get(ConfigKey.TRANSFER_TYPE) == TransferType.DIFF
        self.current_round = None
        self.metrics = None
        self.rank = rank
        self._is_training = False
        self._is_evaluation = False
        self._is_submit_model = False
        self._load_state_dict_strict = load_state_dict_strict
        self._algorithm_handler_manager = _AlgorithmHandlerManager()
        self._pending_train_model = None
        self._training_round_started = False
        self._round_start_global_step = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def reset_state(self, trainer):
        """Resets the state after a round has been sent."""
        self._clear_round_state()

    def _clear_round_state(self):
        self.metrics = None
        self._pending_train_model = None
        self._training_round_started = False
        self._round_start_global_step = None
        clear()

    def _start_round(self, trainer, pl_module, input_model):
        self._round_start_global_step = trainer.global_step
        self._algorithm_handler_manager.start_round(trainer=trainer, pl_module=pl_module, input_model=input_model)
        self._training_round_started = True

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, *args):
        self._algorithm_handler_manager.before_optimizer_step(optimizer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._algorithm_handler_manager.after_train_batch(pl_module)

    def _finish_round_and_send(self, trainer, pl_module):
        if hasattr(pl_module, FL_META_KEY):
            fl_meta = getattr(pl_module, FL_META_KEY)
            if not isinstance(fl_meta, dict):
                raise RuntimeError(f"The {FL_META_KEY} needs to be a dictionary")
            fl_meta = dict(fl_meta)
        else:
            fl_meta = {}
        if self._is_training:
            algorithm_result = self._algorithm_handler_manager.finish_round(pl_module)
            metadata_conflicts = sorted(set(fl_meta).intersection(algorithm_result.metadata))
            if metadata_conflicts:
                raise RuntimeError(
                    "Lightning automatic algorithm metadata conflicts with user-provided "
                    f"{FL_META_KEY} keys: {metadata_conflicts}. Remove these reserved keys from {FL_META_KEY}; "
                    "manual algorithm integration must use an explicit receive/train/send loop without "
                    "flare.patch()."
                )
            fl_meta.update(algorithm_result.metadata)
            if MetaKey.NUM_STEPS_CURRENT_ROUND not in fl_meta:
                fl_meta[MetaKey.NUM_STEPS_CURRENT_ROUND] = (
                    algorithm_result.num_steps
                    if algorithm_result.num_steps is not None
                    else self._get_round_num_steps(trainer)
                )
            model = FLModel(params=self._params_to_send(trainer, pl_module), meta=fl_meta)
            if self.train_with_evaluation and self.metrics is None:
                raise RuntimeError("train with evaluation requires validation metrics; call validate before fit.")
            if self.metrics is not None:
                model.metrics = self.metrics
            self._send(trainer, pl_module, model)
            self.reset_state(trainer)

    def _params_to_send(self, trainer, pl_module):
        raise NotImplementedError

    def on_validation_start(self, trainer, pl_module):
        # Only an explicit validation before fit evaluates the received global model for model selection.
        # Lightning keeps trainer.state.fn at FITTING for both fit sanity checks and in-fit validation,
        # so neither is eligible to become INITIAL_METRICS on the server.
        if (
            pl_module
            and self.metrics is None
            and not self._training_round_started
            and trainer.state.fn == TrainerFn.VALIDATING
        ):
            input_model = self._receive_and_update_model(trainer, pl_module)
            if input_model and self._is_training:
                self._pending_train_model = input_model

    def _get_round_num_steps(self, trainer) -> int:
        if self._round_start_global_step is None:
            raise RuntimeError(
                "Cannot determine round steps because on_train_start did not record trainer.global_step."
            )
        completed_steps = trainer.global_step - self._round_start_global_step
        if completed_steps < 0:
            raise RuntimeError(
                "Cannot determine round steps because trainer.global_step moved backwards from "
                f"{self._round_start_global_step} to {trainer.global_step}."
            )
        return completed_steps

    def on_validation_end(self, trainer, pl_module):
        if (
            pl_module
            and self.metrics is None
            and not self._training_round_started
            and trainer.state.fn == TrainerFn.VALIDATING
        ):
            self.metrics = _extract_metrics(trainer.callback_metrics)
            if self._is_evaluation:
                self._send(trainer, pl_module, FLModel(metrics=self.metrics))
                # An evaluate task trains nothing, so the training budget stays as the last
                # training round left it. Recomputing it here would load a train dataloader the
                # validate() call may not have, and would extend max_steps by another round.
                if self.current_round is not None:
                    trainer.num_sanity_val_steps = 0
                self._clear_round_state()

    def _receive_and_update_model(self, trainer, pl_module):
        """Receive a global model and apply the compatible portion locally.

        The incoming payload is validated before ``load_state_dict()`` so that
        wrapper-induced key drift and shape mismatches fail with actionable
        diagnostics instead of being silently skipped. In non-strict mode,
        incoming keys that are not present locally are filtered out after a
        warning, which allows partial model updates as long as some keys match.
        """

        model = self._receive_model(trainer)
        if model:
            return self._load_received_model(trainer, pl_module, model)
        return model

    def _load_received_model(self, trainer, pl_module, model: FLModel):
        editable = copy.copy(model)
        editable.params = dict(model.params) if model.params is not None else None
        has_hooks = _has_fl_hooks(trainer, "on_fl_receive", "on_fl_model_loaded")
        if editable.params and self._diff_transfer and trainer.global_rank == 0 and has_hooks:
            editable.params = {k: v.clone() if isinstance(v, Tensor) else v for k, v in editable.params.items()}
        if model.meta is not None:
            editable.meta = copy.deepcopy(model.meta) if has_hooks else dict(model.meta)
        _call_fl_hooks(trainer, pl_module, "on_fl_receive", editable)
        self._update_model(pl_module, editable)
        _call_fl_hooks(trainer, pl_module, "on_fl_model_loaded", editable)
        return editable

    def _send(self, trainer, pl_module, model: FLModel):
        outgoing = model if trainer.global_rank == 0 else None
        _call_fl_hooks(trainer, pl_module, "on_fl_before_send", outgoing)
        if outgoing is not None:
            self._send_model(outgoing)

    def _update_model(self, pl_module, model: FLModel):
        """Apply a previously received FLModel to a Lightning module."""
        if model.params:
            try:
                report = inspect_model_params(pl_module.state_dict(), model.params)
                if report.shape_mismatches:
                    raise RuntimeError(report.format_shape_mismatch_error())

                if not report.matched_keys:
                    raise RuntimeError(report.format_zero_match_error())

                params_to_load = model.params
                if report.unexpected_keys:
                    if self._load_state_dict_strict:
                        raise RuntimeError(report.format_unexpected_keys_error())

                    self.logger.warning(report.format_unexpected_keys_warning())
                    params_to_load = {key: model.params[key] for key in report.matched_keys}

                result = pl_module.load_state_dict(params_to_load, strict=self._load_state_dict_strict)
                if result is not None:
                    missing_keys, unexpected_keys = result
                    if len(missing_keys) > 0:
                        self.logger.warning(
                            f"There were missing keys when loading the global state_dict: {missing_keys}"
                        )
                    if len(unexpected_keys) > 0:
                        self.logger.warning(
                            f"There were unexpected keys when loading the global state_dict: {unexpected_keys}"
                        )
            except Exception as e:
                self.logger.error(f"Failed to load state dict: {str(e)}")
                raise RuntimeError(f"Failed to load model state dict: {str(e)}")
        if model.current_round is not None:
            self.current_round = model.current_round

    def _receive_model(self, trainer) -> FLModel:
        """Receives model from NVFlare."""
        model = None
        _is_training = False
        _is_evaluation = False
        _is_submit_model = False
        if self.rank == 0:
            model = receive()
            _is_training = is_train()
            _is_evaluation = is_evaluate()
            _is_submit_model = is_submit_model()

        model = trainer.strategy.broadcast(model, src=0)
        self._is_training = trainer.strategy.broadcast(_is_training, src=0)
        self._is_evaluation = trainer.strategy.broadcast(_is_evaluation, src=0)
        self._is_submit_model = trainer.strategy.broadcast(_is_submit_model, src=0)
        return model

    def _send_model(self, output_model: FLModel):
        try:
            send(output_model, clear_cache=False)
        except Exception as e:
            raise RuntimeError(f"failed to send FL model: {e}")


class FLCallback(_FLCallbackBase):
    def __init__(self, rank: int = 0, load_state_dict_strict: bool = True, update_fit_loop: bool = True):
        """FL callback for lightning API: one FL round per ``trainer.fit()`` call.

        Args:
            rank: global rank of the PyTorch Lightning trainer.
            load_state_dict_strict: exposes `strict` argument of `torch.nn.Module.load_state_dict()`
                used to load the received model. Defaults to `True`.
                See https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict for details.
                NVFlare still validates incoming keys and shapes before calling ``load_state_dict()``.
                With ``True``, unexpected incoming keys are treated as contract
                drift and fail fast. With ``False``, unexpected keys are logged
                and ignored, while compatible keys are still loaded.
            update_fit_loop: whether to increase `trainer.fit_loop.max_epochs` and `trainer.fit_loop.epoch_loop.max_steps` each FL round.
                Defaults to `True` which is suitable for most PyTorch Lightning applications.
        """
        super().__init__(rank=rank, load_state_dict_strict=load_state_dict_strict)
        self.total_local_epochs = 0
        self.total_local_steps = 0
        self.max_epochs_per_round = None
        self.max_steps_per_round = None
        self._update_fit_loop = update_fit_loop

    def reset_state(self, trainer):
        """Resets the state.

        If the next round of federated training needs to reuse the same callback
        instance, the reset_state() needs to be called first
        Not only resets the states, also sets states for next round
        """
        # set states for next round
        if self.current_round is not None:
            self._set_next_round_budget(trainer)
        self._clear_round_state()

    def _set_next_round_budget(self, trainer):
        """Record the per-round budget and extend the fit loop by one round past the trained progress."""
        if self.max_epochs_per_round is None and trainer.max_epochs and trainer.max_epochs > 0:
            self.max_epochs_per_round = trainer.max_epochs
        if self.max_steps_per_round is None and trainer.max_steps and trainer.max_steps > 0:
            self.max_steps_per_round = trainer.max_steps

        # record total local epochs/steps
        self.total_local_epochs = trainer.current_epoch
        self.total_local_steps = trainer.estimated_stepping_batches

        # for next round
        trainer.num_sanity_val_steps = 0  # Turn off sanity validation steps in following rounds of FL

        if self._update_fit_loop:
            if self.total_local_epochs and self.max_epochs_per_round is not None:
                trainer.fit_loop.max_epochs = self.max_epochs_per_round + self.total_local_epochs
            if self.total_local_steps and self.max_steps_per_round is not None:
                trainer.fit_loop.epoch_loop.max_steps = self.max_steps_per_round + self.total_local_steps

    def on_train_start(self, trainer, pl_module):
        input_model = self._pending_train_model
        self._pending_train_model = None
        if input_model is None:
            input_model = self._receive_and_update_model(trainer, pl_module)
        else:
            self._update_model(pl_module, input_model)
        if input_model and self._is_training:
            self._start_round(trainer, pl_module, input_model)

    def on_train_end(self, trainer, pl_module):
        self._finish_round_and_send(trainer, pl_module)

    def _params_to_send(self, trainer, pl_module):
        return pl_module.cpu().state_dict()


class _StepSyncFLCallback(_FLCallbackBase):
    """FL callback that exchanges the model every N optimizer steps within one ``trainer.fit()``."""

    def __init__(self, sync_every_n_steps: int, rank: int = 0, load_state_dict_strict: bool = True):
        super().__init__(rank=rank, load_state_dict_strict=load_state_dict_strict)
        if self.train_with_evaluation:
            raise ValueError(
                "sync_every_n_steps does not support train_with_evaluation: no global-model validation runs "
                "between syncs. Disable train_with_evaluation or sync once per fit()."
            )
        self.sync_every_n_steps = sync_every_n_steps
        self._trainer_step_budget = None
        self._trainer_limits = None
        self._received_model = None

    def on_train_start(self, trainer, pl_module):
        budget = trainer.estimated_stepping_batches
        self._trainer_step_budget = int(budget) if 0 < budget < float("inf") else None
        fit_loop = trainer.fit_loop
        self._trainer_limits = (
            fit_loop.max_epochs,
            fit_loop.epoch_loop.max_steps,
            fit_loop.min_epochs,
            fit_loop.epoch_loop.min_steps,
        )
        trainer.fit_loop.max_epochs = -1
        trainer.fit_loop.epoch_loop.max_steps = -1
        trainer.fit_loop.min_epochs = 0
        trainer.fit_loop.epoch_loop.min_steps = None
        if self._pending_train_model is None:
            self._receive_next_round(trainer)

    def _receive_next_round(self, trainer):
        model = self._receive_model(trainer)
        if model and self._is_training:
            self._received_model = model
            return
        trainer.should_stop = True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._training_round_started:
            return
        input_model, self._pending_train_model = self._pending_train_model, None
        if input_model is not None:
            self._update_model(pl_module, input_model)
        else:
            received, self._received_model = self._received_model, None
            input_model = self._load_received_model(trainer, pl_module, received)
        self._start_round(trainer, pl_module, input_model)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self._trainer_step_budget is not None and trainer.global_step > self._trainer_step_budget:
            self.logger.warning(
                f"Training passed the trainer's own limit of {self._trainer_step_budget} optimizer steps "
                "(max_epochs / max_steps) because the server's num_rounds decides how long step sync runs. "
                "An LR scheduler sized from that limit (e.g. OneCycleLR, or CosineAnnealingLR with T_max from "
                "estimated_stepping_batches) is now past its end; size it for num_rounds * sync_every_n_steps."
            )
            self._trainer_step_budget = None
        if not self._training_round_started:
            return
        round_steps = self._get_round_num_steps(trainer)
        if round_steps > self.sync_every_n_steps:
            raise RuntimeError(
                f"Step sync checks the sync point once per batch, but this batch took the round to {round_steps} "
                f"optimizer steps, past sync_every_n_steps={self.sync_every_n_steps}. With manual optimization, "
                "take one optimizer step per batch or set sync_every_n_steps to a multiple of the steps per batch."
            )
        if round_steps == self.sync_every_n_steps:
            self._finish_round_and_send(trainer, pl_module)
            if not trainer.should_stop:
                self._receive_next_round(trainer)

    def on_train_end(self, trainer, pl_module):
        self._restore_trainer_limits(trainer)
        self._received_model = None
        if not self._training_round_started:
            return
        if self._get_round_num_steps(trainer) == 0:
            self.logger.warning(
                f"Training stopped before the first optimizer step of round {self.current_round}. No result is sent "
                "for it, so the server handles this site as one that stopped responding."
            )
            self._training_round_started = False
            self._round_start_global_step = None
            return
        self._finish_round_and_send(trainer, pl_module)

    def on_exception(self, trainer, pl_module, exception):
        self._restore_trainer_limits(trainer)

    def _restore_trainer_limits(self, trainer):
        if self._trainer_limits is None:
            return
        fit_loop = trainer.fit_loop
        fit_loop.max_epochs, fit_loop.epoch_loop.max_steps, fit_loop.min_epochs, fit_loop.epoch_loop.min_steps = (
            self._trainer_limits
        )
        self._trainer_limits = None

    def _params_to_send(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return None
        return {
            key: value.detach().to("cpu", copy=True) if isinstance(value, Tensor) else copy.deepcopy(value)
            for key, value in pl_module.state_dict().items()
        }


def _has_fl_hooks(trainer, *names: str) -> bool:
    return any(callable(getattr(cb, name, None)) for cb in getattr(trainer, "callbacks", []) for name in names)


def _call_fl_hooks(trainer, pl_module, name: str, fl_model: Optional[FLModel]):
    for callback in getattr(trainer, "callbacks", []):
        hook = getattr(callback, name, None)
        if callable(hook):
            hook(trainer, pl_module, fl_model)


def _extract_metrics(metrics: Dict[str, Tensor]):
    result_metrics = {}
    for key, t in metrics.items():
        result_metrics[key] = t.item()
    return result_metrics
