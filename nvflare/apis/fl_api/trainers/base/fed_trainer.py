from typing import Any, Callable, Optional, Tuple, Dict

from nvflare.apis.fl_api import TrainerConfig, get_trainer_registry
from nvflare.apis.fl_api.registry.trainer_registry import _TRAINER_REGISTRY

class FedTrainer:
    def __init__(
            self,
            local_trainer: Any,  # User's native trainer/model/pipeline (e.g., PyTorch Lightning, sklearn)

            config: Optional[TrainerConfig] = None,

            # Custom function to extract model state to be sent to server or peers
            get_state_fn: Optional[Callable[[], Any]] = None,

            # Custom function to apply received model state from server or peers
            set_state_fn: Optional[Callable[[Any], None]] = None,

            # Custom evaluation function (optional)
            evaluate_fn: Optional[Callable[[Any], Any]] = None,
    ):
        self.local_trainer = local_trainer
        self.get_state_fn = get_state_fn or self._default_get_state
        self.set_state_fn = set_state_fn or self._default_set_state
        self.evaluate_fn = evaluate_fn or self._default_evaluate
        self.config = config

    def fit(self):
        pass

    def evaluate(self):
        pass

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
            config=config,
            get_state_fn=get_state_fn,
            set_state_fn=set_state_fn,
            evaluate_fn=evaluate_fn,
        )

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "FedTrainer":
        """
        Load a pre-defined trainer setup.
        Supported: 'torch', 'lightning', 'sklearn'
        """
        trainer_cls = get_trainer_registry().get(name, None)
        if trainer_cls:
            return trainer_cls(**kwargs)
        else:
            raise ValueError(f"Unknown preset: {name}")
