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

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.client.api import clear, get_config, init, receive, send
from nvflare.client.config import ConfigKey


def patch(trainer: pl.Trainer):
    fl_callback = FLCallback()
    callbacks = trainer.callbacks
    if isinstance(callbacks, list):
        callbacks.append(fl_callback)
    elif isinstance(callbacks, Callback):
        callbacks = [callbacks, fl_callback]
    else:
        callbacks = [fl_callback]
    trainer.callbacks = callbacks


class FLCallback(Callback):
    def __init__(self):
        super(FLCallback, self).__init__()
        init()
        self.has_global_eval = get_config().get(ConfigKey.GLOBAL_EVAL, False)
        self.has_training = get_config().get(ConfigKey.TRAINING, False)
        self.input_fl_model = receive(sys_info_receive=True)
        self.metrics = None

    def reset_state(self):
        # If the next round of federated training needs to reuse the same callback
        # instance, the reset_state() needs to be called first
        self.input_fl_model = None
        self.metrics = None

    def on_train_start(self, trainer, pl_module):
        # receive the global model and update the local model with global model
        if self.has_training:
            self._receive_update_model(pl_module)

    def on_train_end(self, trainer, pl_module):
        if self.has_training:
            self._send_model(FLModel(params=pl_module.cpu().state_dict()))
            self.reset_state()

    def on_validation_start(self, trainer, pl_module):
        # receive the global model and update the local model with global model
        # the 1st time validate() or train() is called.
        # expect user will validate the global model first (i.e. validate()), once that's done.
        # the metrics will be set.
        # The subsequence validate() calls will not trigger the receive update model.
        # Hence the validate() will be validating the local model.
        if pl_module and self.has_global_eval and self.metrics is None:
            self._receive_update_model(pl_module)

    def on_validation_end(self, trainer, pl_module):
        if pl_module and self.has_global_eval and self.metrics is None:
            self.metrics = _extract_metrics(trainer.callback_metrics)
            self._send_model(FLModel(metrics=self.metrics))

    def _receive_update_model(self, pl_module):
        if not self.input_fl_model:
            model = self._receive_model()
            if model and model.params:
                pl_module.load_state_dict(model.params)

    def _receive_model(self) -> FLModel:
        model = receive()
        if model:
            self.input_fl_model = model
        return model

    def _send_model(self, output_model: FLModel):
        try:
            send(output_model, clear=False)
        except Exception as e:
            raise RuntimeError("failed to send FL model", e)

    def __del__(self):
        clear()


def _extract_metrics(metrics: Dict[str, Tensor]):
    result_metrics = {}
    for key, t in metrics.items():
        result_metrics[key] = t.item()
    return result_metrics
