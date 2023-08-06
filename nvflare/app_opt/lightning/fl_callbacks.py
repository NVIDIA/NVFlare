from enum import IntEnum

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.client.config import ConfigKey
from nvflare.client.constants import ModelExchangeFormat


class SendTrigger(IntEnum):
    AFTER_TRAIN_AND_TEST = 1
    AFTER_TRAIN = 2
    AFTER_TEST = 3


class FLCallback(pl.callbacks.Callback):
    def __init__(self, send_trigger: SendTrigger = SendTrigger.AFTER_TRAIN_AND_TEST):
        super(FLCallback, self).__init__()
        flare.init(
            config={
                ConfigKey.EXCHANGE_PATH: "./",
                ConfigKey.EXCHANGE_FORMAT: ModelExchangeFormat.PYTORCH,
                ConfigKey.TRANSFER_TYPE: "FULL",
            }
        )
        self.send_mode = send_trigger
        self.input_fl_model = None
        self.output_fl_model = None
        self.model_delivered = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        loop = trainer.test_loop
        loop.run = test_loop_run_decorator(loop, self)

    def on_fit_start(self, trainer, pl_module):
        self._receive_update_model(pl_module)

    def on_train_end(self, trainer, pl_module):
        if self.output_fl_model:
            self.output_fl_model.params = pl_module.cpu().state_dict()
        else:
            self.output_fl_model = flare.FLModel(params=pl_module.cpu().state_dict())

    def on_test_start(self, trainer, pl_module):
        if pl_module:
            self._receive_update_model(pl_module)

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self._check_and_send()

    def _receive_update_model(self, pl_module):
        if not self.input_fl_model:
            model = self._receive_model()
            if model and model.params:
                pl_module.load_state_dict(model.params)

    def _receive_model(self) -> FLModel:
        model = flare.receive()
        if model:
            self.input_fl_model = model
        return model

    def _check_and_send(self):
        if self.output_fl_model:
            if self.send_mode == SendTrigger.AFTER_TRAIN_AND_TEST:
                if self.output_fl_model.metrics and self.output_fl_model.params:
                    self.delivery()
            elif self.send_mode == SendTrigger.AFTER_TRAIN and self.output_fl_model.params:
                self.delivery()
            elif self.send_mode == SendTrigger.AFTER_TEST and self.output_fl_model.metrics:
                self.delivery()

    def delivery(self):
        if not self.model_delivered:
            try:
                flare.send(self.output_fl_model)
                self.model_delivered = True
            except Exception as e:
                self.model_delivered = False


def test_loop_run_decorator(loop, cb):
    func = loop.run

    def wrapper(*args, **kwargs):
        if cb.model_delivered:
            return func(*args, **kwargs)
        else:
            metrics = func(*args, **kwargs)
            if loop.trainer.state.fn == TrainerFn.TESTING and metrics:
                if cb.output_fl_model is None:
                    cb.output_fl_model = flare.FLModel(metrics=metrics[0])
                elif not cb.output_fl_model.metrics:
                    cb.output_fl_model.metrics = metrics[0]
            return metrics

    return wrapper
