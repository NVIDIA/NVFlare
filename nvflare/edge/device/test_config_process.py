import json

from nvflare.edge.device.config import ComponentCreator, process_train_config
from nvflare.edge.device.sdk_spec import (
    Batch,
    Context,
    DataSource,
    EventHandler,
    EventType,
    Filter,
    Model,
    Signal,
    Trainer,
    Transform,
)


class DLTrainer(Trainer):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def train(self, data_source: DataSource, model: Model, ctx: Context, abort_signal: Signal) -> Model:
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

    def filter(self, model: Model, ctx: Context, abort_signal: Signal) -> Model:
        pass


class StatsKeeper(Filter, EventHandler):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def filter(self, model: Model, ctx: Context, abort_signal: Signal) -> Model:
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


class DLTrainerCreator(ComponentCreator):

    def __init__(self, t, name, args):
        super().__init__(t, name, args)

    def create(self) -> DLTrainer:
        return DLTrainer(**self.args)


class SGDCreator(ComponentCreator):

    def __init__(self, t, name, args):
        super().__init__(t, name, args)

    def create(self) -> SGDOptimizer:
        return SGDOptimizer(**self.args)


class BCELossCreator(ComponentCreator):

    def __init__(self, t, name, args):
        super().__init__(t, name, args)

    def create(self):
        return bce_loss


class RotateCreator(ComponentCreator):

    def __init__(self, t, name, args):
        super().__init__(t, name, args)

    def create(self) -> Rotate:
        return Rotate(**self.args)


class DPCreator(ComponentCreator):

    def __init__(self, t, name, args):
        super().__init__(t, name, args)

    def create(self) -> DPFilter:
        return DPFilter(**self.args)


class StatsKeeperCreator(ComponentCreator):

    def __init__(self, t, name, args):
        super().__init__(t, name, args)

    def create(self) -> StatsKeeper:
        return StatsKeeper(**self.args)


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
  "filters": ["@dp", "@stats"],
  "handlers": ["@stats"]
}
"""


reg = {
    "Trainer.DLTrainer": DLTrainerCreator,
    "Optimizer.SGD": SGDCreator,
    "Loss.BCELoss": BCELossCreator,
    "Transform.rotate": RotateCreator,
    "Filter.DP": DPCreator,
    "Handler.StatsKeeper": StatsKeeperCreator,
}

config = json.loads(CONFIG_DATA)
obj_table, filters, handlers = process_train_config(config, reg)

print("DONE")
