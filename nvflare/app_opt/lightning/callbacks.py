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

import logging
from copy import deepcopy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class RestoreState(Callback):
    """Callback to restore the local optimizer and learning rate scheduler states at each round of FL"""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.optimizer_states = []
        self.scaler_states = []
        self.lr_scheduler_states = []

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if len(self.optimizer_states) > 0:
            trainer.strategy.load_optimizer_state_dict({"optimizer_states": self.optimizer_states})
            self.logger.info("optimizer states restored.")
        else:
            return

        if len(self.scaler_states) > 0:
            trainer.scaler.load_state_dict(self.scaler_states[0])
            self.logger.info("scaler states restored.")

        if len(self.lr_scheduler_states) > 0:
            for config, lr_scheduler_state in zip(trainer.lr_scheduler_configs, self.lr_scheduler_states):
                config.scheduler.load_state_dict(lr_scheduler_state)
            self.logger.info("LR scheduler states restored.")

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.optimizer_states = [deepcopy(opt.state_dict()) for opt in trainer.optimizers]
        if trainer.scaler:
            self.scaler_states = [deepcopy(trainer.scaler.state_dict())]
        self.lr_scheduler_states = [deepcopy(config.scheduler.state_dict()) for config in trainer.lr_scheduler_configs]
