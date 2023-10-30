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

from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import Tensor

from nvflare.app_common.abstract.fl_model import FLModel, MetaKey
from nvflare.client.api import _get_model_registry, clear, get_config, init, receive, send
from nvflare.client.config import ConfigKey

from .callbacks import RestoreState

FL_META_KEY = "__fl_meta__"


def patch(trainer: pl.Trainer, restore_optimizers: bool = True):
    fl_callback = FLCallback(rank=trainer.global_rank)
    callbacks = trainer.callbacks
    if isinstance(callbacks, list):
        callbacks.append(fl_callback)
    elif isinstance(callbacks, Callback):
        callbacks = [callbacks, fl_callback]
    else:
        callbacks = [fl_callback]

    if restore_optimizers:
        callbacks.append(RestoreState())

    trainer.callbacks = callbacks


class FLCallback(Callback):
    def __init__(self, rank: int = 0):
        super(FLCallback, self).__init__()
        init(rank=str(rank))
        self.train_with_evaluation = get_config().get(ConfigKey.TRAIN_WITH_EVAL, False)
        self.current_round = None
        self.metrics = None
        self.total_local_epochs = 0
        self.total_local_steps = 0
        self.max_epochs_per_round = None
        self.max_steps_per_round = None

    def reset_state(self, trainer):
        """Resets the state.

        If the next round of federated training needs to reuse the same callback
        instance, the reset_state() needs to be called first
        Not only resets the states, also sets states for next round
        """
        # set states for next round
        if self.current_round is not None:
            if self.max_epochs_per_round is None:
                if trainer.max_epochs and trainer.max_epochs > 0:
                    self.max_epochs_per_round = trainer.max_epochs
                if trainer.max_steps and trainer.max_steps > 0:
                    self.max_steps_per_round = trainer.max_steps

            # record total local epochs/steps
            self.total_local_epochs = trainer.current_epoch
            self.total_local_steps = trainer.estimated_stepping_batches

            # for next round
            trainer.num_sanity_val_steps = 0  # Turn off sanity validation steps in following rounds of FL
            if self.total_local_epochs and self.max_epochs_per_round is not None:
                trainer.fit_loop.max_epochs = self.max_epochs_per_round + self.total_local_epochs
            if self.total_local_steps and self.max_steps_per_round is not None:
                trainer.fit_loop.epoch_loop.max_steps = self.max_steps_per_round + self.total_local_steps

        # resets attributes
        self.metrics = None
        clear()

    def on_train_start(self, trainer, pl_module):
        # receive the global model and update the local model with global model
        self._receive_and_update_model(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if hasattr(pl_module, FL_META_KEY):
            fl_meta = getattr(pl_module, FL_META_KEY)
            if not isinstance(fl_meta, dict):
                raise RuntimeError(f"The {FL_META_KEY} needs to be a dictionary")
        else:
            fl_meta = {}
        if MetaKey.NUM_STEPS_CURRENT_ROUND not in fl_meta:
            fl_meta[MetaKey.NUM_STEPS_CURRENT_ROUND] = trainer.estimated_stepping_batches
        self._send_model(FLModel(params=pl_module.cpu().state_dict(), meta=fl_meta))
        self.reset_state(trainer)

    def on_validation_start(self, trainer, pl_module):
        # receive the global model and update the local model with global model
        # the 1st time validate() or train() is called.
        # expect user will validate the global model first (i.e. validate()), once that's done.
        # the metrics will be set.
        # The subsequent validate() calls will not trigger the receive update model.
        # Hence the validate() will be validating the local model.
        if pl_module and self.metrics is None:
            self._receive_and_update_model(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if pl_module and self.metrics is None:
            self.metrics = _extract_metrics(trainer.callback_metrics)
            self._send_model(FLModel(metrics=self.metrics))

    def _receive_and_update_model(self, trainer, pl_module):
        model = self._receive_model(trainer)
        if model and model.params:
            pl_module.load_state_dict(model.params)
        if model and model.current_round is not None:
            self.current_round = model.current_round

    def _receive_model(self, trainer) -> FLModel:
        """Receives model from NVFlare."""
        model = receive()
        registry = _get_model_registry()
        model = trainer.strategy.broadcast(model, src=0)
        task_name = trainer.strategy.broadcast(registry.task_name, src=0)
        registry.set_task_name(task_name)
        return model

    def _send_model(self, output_model: FLModel):
        try:
            send(output_model, clear_registry=False)
        except Exception as e:
            raise RuntimeError("failed to send FL model", e)


def _extract_metrics(metrics: Dict[str, Tensor]):
    result_metrics = {}
    for key, t in metrics.items():
        result_metrics[key] = t.item()
    return result_metrics
