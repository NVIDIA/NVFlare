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
            datamodule: Optional[LightningDataModule] = None,
            config: Optional[TrainerConfig] = None,
    ):
        self.lightning_module = lightning_module
        self.trainer = trainer or pl.Trainer()


        # Define get_state_fn from lightning model
        def get_state():
            return lightning_module.state_dict()

        # Define set_state_fn for lightning model
        def set_state(state: Dict[str, Any]):
            lightning_module.load_state_dict(state)

        # Define evaluate function (optional, here using Lightning's validate)
        def evaluate_fn(eval_args=None):
            eval_args = eval_args or {}
            # Run validation or testing and return metrics dictionary
            results = trainer.validate(
                model=lightning_module,
                datamodule=datamodule,
                **eval_args,
            )
            # results is a list of dicts, merge or return first for simplicity
            return results[0] if results else {}

        super().__init__(
            local_trainer=trainer,
            config=config,
            get_state_fn=get_state,
            set_state_fn=set_state,
            evaluate_fn=evaluate_fn,
            config=config,
        )
