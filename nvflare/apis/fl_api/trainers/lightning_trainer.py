import pytorch_lightning as pl
from typing import Optional, Any, Dict
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from nvflare.apis.fl_api.trainers.base.fed_trainer import FedTrainer
from nvflare.apis.fl_api.trainers.base.trainer_config import TrainerConfig


class LightingFedTrainer(FedTrainer):
    def __init__(
        self,
        trainer: Trainer,
        lightning_module: LightningModule,
        local_trainer: Any,
        datamodule: Optional[LightningDataModule] = None,
        config: Optional[TrainerConfig] = None,
    ):
        super().__init__(local_trainer, config)
        self.lightning_module = lightning_module
        self.trainer = trainer or pl.Trainer()
        self.config = config
        self.datamodule = datamodule

    # Define get_state_fn from lightning model
    def get_state(self):
        return self.lightning_module.state_dict()

    # Define set_state_fn for lightning model
    def set_state(self, state: Dict[str, Any]):
        self.lightning_module.load_state_dict(state)

    # Define evaluate function (optional, here using Lightning's validate)
    def evaluate_fn(self, eval_args=None):
        eval_args = eval_args or {}
        # Run validation or testing and return metrics dictionary
        ckpt_path: Optional[str] = self.config.metadata.get("ckpt_path", None)
        verbose: bool = self.config.metadata.get("verbose", True)
        results = self.trainer.validate(
            model=self.lightning_module, datamodule=self.datamodule, ckpt_path=ckpt_path, verbose=verbose
        )
        # results is a list of dicts, merge or return first for simplicity
        return results[0] if results else {}

    def fit(self):
        self.trainer.fit(model=self.lightning_module, datamodule=self.datamodule)

        super().__init__(
            local_trainer=trainer,
            config=config,
            get_state_fn=get_state,
            set_state_fn=set_state,
            evaluate_fn=evaluate_fn,
        )
