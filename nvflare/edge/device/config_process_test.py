# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import json
from typing import Any

from nvflare.edge.device.config import ComponentResolver, process_train_config
from nvflare.edge.device.defs import Batch, Context, EventHandler, EventType, Executor, Filter, Signal, Transform


class DLTrainer(Executor):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def execute(self, model, ctx: Context, abort_signal: Signal):
        pass


class SGDOptimizer:

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def bce_loss(pred, label):
    pass


class Rotate(Transform):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def transform(self, batch: Batch, ctx: Context, abort_signal: Signal) -> Batch:
        pass


class DPFilter(Filter):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def filter(self, model: Any, ctx: Context, abort_signal: Signal) -> Any:
        pass


class StatsKeeper(Filter, EventHandler):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def filter(self, model, ctx: Context, abort_signal: Signal):
        # add stats data (saved in ctx) to the "model" to be sent to host
        # do not keep stats data in self.
        pass

    def handle_event(self, event_type: str, event_data, ctx: Context, abort_signal: Signal):
        if event_type == EventType.BEFORE_TRAIN:
            ctx["train_start_time"] = event_data
        elif event_type == EventType.AFTER_TRAIN:
            ctx["train_end_time"] = event_data[0]
        elif event_type == EventType.LOSS_GENERATED:
            ctx["train_loss"] = event_data


class BCELossResolver(ComponentResolver):

    def __init__(self, t, name, args):
        super().__init__(t, name, args)

    def resolve(self):
        return bce_loss


CONFIG_DATA = """
{
  "components": [
    {
      "type": "Trainer.DLTrainer",
      "name": "trainer",
      "args": {
        "epoch": 5,
        "lr": 0.0001,
        "optimizer": "@opt",
        "loss": "@loss",
        "transforms": {
            "pre": ["@t1", "@t2"],
            "post": ["@t2", "@t1"]
        }
      }
    },
    {
      "type": "Optimizer.SGD",
      "name": "opt",
      "args": {}
    },
    {
      "type": "Loss.BCELoss",
      "name": "loss",
      "args": {}
    },
    {
      "type": "Transform.rotate",
      "name": "t1",
      "args": {
        "angle": 90
      }
    },
    {
      "type": "Transform.rotate",
      "name": "t2",
      "args": {
        "angle": -60
      }
    },
    {
      "type": "Filter.DP",
      "name": "dp",
      "args": {}
    },
    {
      "type": "Handler.StatsKeeper",
      "name": "stats"
    }
  ],
  "out_filters": ["@dp", "@stats"],
  "handlers": ["@stats"],
  "executors": {
    "*": "@trainer"
  }
}
"""


reg = {
    "Trainer.DLTrainer": DLTrainer,
    "Optimizer.SGD": SGDOptimizer,
    "Loss.BCELoss": BCELossResolver,
    "Transform.rotate": Rotate,
    "Filter.DP": DPFilter,
    "Handler.StatsKeeper": StatsKeeper,
}

config = json.loads(CONFIG_DATA)
train_config = process_train_config(config, reg)

print("DONE")
