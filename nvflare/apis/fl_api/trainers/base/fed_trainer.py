from typing import Any, Callable, Optional

from nvflare.apis.fl_api.trainer.trainer_config import TrainerConfig


class FedTrainer:
    def __init__(
            self,
            local_trainer: Any,  # User's native trainer/model/pipeline (e.g., PyTorch Lightning, sklearn)
            fit_args: Optional[dict] = None,  # Optional arguments for fit/train method
            eval_args: Optional[dict] = None,  # Optional arguments for evaluate/score method

            # Custom function to extract model state to be sent to server or peers
            get_state_fn: Optional[Callable[[], Any]] = None,

            # Custom function to apply received model state from server or peers
            set_state_fn: Optional[Callable[[Any], None]] = None,

            # Custom evaluation function (optional)
            evaluate_fn: Optional[Callable[[Any], Any]] = None,

            config: Optional[TrainerConfig] = None,
    ):
        self.local_trainer = local_trainer
        self.fit_args = fit_args or {}
        self.eval_args = eval_args or {}
        self.get_state_fn = get_state_fn or self._default_get_state
        self.set_state_fn = set_state_fn or self._default_set_state
        self.evaluate_fn = evaluate_fn or self._default_evaluate
        self.config = config

    def fit(self):
        if hasattr(self.local_trainer, "fit"):
            return self.local_trainer.fit(**self.fit_args)
        elif hasattr(self.local_trainer, "train"):
            return self.local_trainer.train(**self.fit_args)
        else:
            raise NotImplementedError("No fit/train method found on local_trainer")

    def evaluate(self):
        return self.evaluate_fn(self.eval_args)

    def get_state(self) -> Any:
        return self.get_state_fn()

    def set_state(self, state: Any) -> None:
        self.set_state_fn(state)


    @classmethod
    def from_function(
            cls,
            train_fn: Callable[[Any], Tuple[Any, Dict]],
            get_state_fn: Callable[[], Any],
            set_state_fn: Callable[[Any], None],
            evaluate_fn: Optional[Callable[[Any], Dict]] = None,
            config: Optional[TrainerConfig] = None,
    ) -> "FedTrainer":
        return cls(
            local_trainer=train_fn,
            fit_args={},  # Provided in the function signature
            get_state_fn=get_state_fn,
            set_state_fn=set_state_fn,
            evaluate_fn=evaluate_fn,
            config=config,
        )

    @classmethod
    def from_preset(cls, name: str, model: Any, config: Optional[TrainerConfig] = None) -> "FedTrainer":
        """
        Load a pre-defined trainer setup.
        Supported: 'torch', 'lightning', 'sklearn'
        """
        if name == "torch":
            import torch

            def get_state():
                return model.state_dict()

            def set_state(state):
                model.load_state_dict(state)

            def train_fn(_):
                model.train()
                # Placeholder: use config to simulate training
                return get_state(), {"loss": 0.1}

            def eval_fn(state):
                model.eval()
                return {"accuracy": 0.9}

            return cls(
                local_trainer=train_fn,
                get_state_fn=get_state,
                set_state_fn=set_state,
                evaluate_fn=eval_fn,
                config=config,
            )

        elif name == "sklearn":
            def train_fn(_):
                model.fit(config.other["X"], config.other["y"])
                return model, {"score": model.score(config.other["X"], config.other["y"])}

            return cls(
                local_trainer=train_fn,
                get_state_fn=lambda: model,
                set_state_fn=lambda x: model.__dict__.update(x.__dict__),
                evaluate_fn=lambda _: {"score": model.score(config.other["X"], config.other["y"])},
                config=config,
            )

        else:
            raise ValueError(f"Unknown preset: {name}")

    def _default_get_state(self):
        if hasattr(self.local_trainer, "get_params"):
            return self.local_trainer.get_params()
        elif hasattr(self.local_trainer, "state_dict"):
            return self.local_trainer.state_dict()
        else:
            raise NotImplementedError("No method found to get model state")

    def _default_set_state(self, state):
        if hasattr(self.local_trainer, "set_params"):
            return self.local_trainer.set_params(state)
        elif hasattr(self.local_trainer, "load_state_dict"):
            return self.local_trainer.load_state_dict(state)
        else:
            raise NotImplementedError("No method found to set model state")

    def _default_evaluate(self, eval_args):
        if hasattr(self.local_trainer, "evaluate"):
            return self.local_trainer.evaluate(**eval_args)
        elif hasattr(self.local_trainer, "score"):
            return self.local_trainer.score(**eval_args)
        else:
            raise NotImplementedError("No evaluate/score method found on local_trainer")
