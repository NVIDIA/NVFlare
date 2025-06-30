from typing import Any, Callable, Optional

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
    ):
        self.local_trainer = local_trainer
        self.fit_args = fit_args or {}
        self.eval_args = eval_args or {}
        self.get_state_fn = get_state_fn or self._default_get_state
        self.set_state_fn = set_state_fn or self._default_set_state
        self.evaluate_fn = evaluate_fn or self._default_evaluate

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
