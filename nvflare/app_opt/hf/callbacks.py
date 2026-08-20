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
import math
from numbers import Number

try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object


class FLCallback(TrainerCallback):
    def __init__(self, task_state=None):
        super().__init__()
        self.task_state = task_state

    def on_train_begin(self, args, state, control, **kwargs):
        if self.task_state:
            self.task_state.on_train_begin(state)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.task_state:
            return self.task_state.on_budget_boundary(state, control)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.task_state:
            return self.task_state.on_budget_boundary(state, control)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.task_state:
            self.task_state.on_evaluate(metrics or {})
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.task_state:
            self.task_state.on_train_end(state)
        return control


class FLMetricsCallback(TrainerCallback):
    def __init__(self, task_state=None):
        super().__init__()
        self.task_state = task_state
        self.logger = logging.getLogger(self.__class__.__name__)
        self._writer = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.task_state or self.task_state.rank != 0:
            return control
        logs = logs or {}
        scalars = {}
        for key, value in logs.items():
            scalar = _to_finite_scalar(value)
            if scalar is not None:
                scalars[key] = scalar
            else:
                self.logger.debug("Skipping non-finite or non-scalar HF metric '%s'", key)

        if not scalars:
            return control

        writer = self._get_writer()
        step = self.task_state.metric_step(getattr(state, "global_step", None))
        for key, value in scalars.items():
            writer.add_scalar(key, value, global_step=step)
        return control

    def _get_writer(self):
        if self._writer is None:
            from nvflare.client.tracking import SummaryWriter

            self._writer = SummaryWriter()
        return self._writer


def _to_finite_scalar(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return None

    if not isinstance(value, Number):
        return None

    value = float(value)
    if math.isfinite(value):
        return value
    return None
