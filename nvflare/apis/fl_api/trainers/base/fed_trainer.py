from abc import ABC
from typing import Any, Callable, Optional, Tuple, Dict

from nvflare.apis.fl_api import TrainerConfig, get_trainer_registry
from nvflare.apis.fl_api.communication.comm_layer import CommunicationLayer


class FedTrainer(ABC):
    def __init__(
            self,
            local_trainer: Any,  # User's native trainer/model/pipeline (e.g., PyTorch Lightning, sklearn)

            communication: CommunicationLayer = None,

            config: Optional[TrainerConfig] = None,

            # Custom function to extract model state to be sent to server or peers
            get_state_fn: Optional[Callable[[], Any]] = None,

            # Custom function to apply received model state from server or peers
            set_state_fn: Optional[Callable[[Any], None]] = None,

            # Custom evaluation function (optional)
            evaluate_fn: Optional[Callable[[Any], Any]] = None,
    ):
        self.local_trainer = local_trainer
        self.get_state_fn = get_state_fn
        self.set_state_fn = set_state_fn
        self.evaluate_fn = evaluate_fn
        self.config = config
        self.communication = communication or self._load_communication()

    def fit(self) -> Dict:
        global_state = self.communication.receive_all_states()
        # 1. Apply global model
        self.set_state(global_state)

        # 2. Run local training
        local_update = self.local_trainer.fit()

        # 3. Evaluate model locally
        metrics = {}
        if self.evaluate_fn:
            metrics = self.evaluate_fn()

        # 4. Send update to server
        self.communication.send_update(
            client_id=self.client_id,
            update=local_update,
            metrics=metrics
        )

        return metrics


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

    def _load_communication(self) -> CommunicationLayer:
        # add method to communication layer to set up communication
        # based on network configuration
        pass
