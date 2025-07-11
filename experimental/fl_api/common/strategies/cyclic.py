from typing import List, Any, Optional, Dict
from pydantic import Field

from experimental.fl_api.nvflare.communication.wf_comm_client_layers import MessageType
from experimental.fl_api.common.interfaces.strategy import Strategy, StrategyConfig
from nvflare.app_common.abstract.fl_model import FLModel


class CyclicConfig(StrategyConfig):
    schedule: List[str] = Field(..., description="List of client names in the cyclic schedule.")
    num_rounds: int = Field(10, description="Number of rounds to run the cyclic strategy.")
    start_round: int = Field(1, description="Starting round index.")


class Cyclic(Strategy):
    def __init__(self, strategy_config: Optional[CyclicConfig] = None, **kwargs):
        config = strategy_config or CyclicConfig(**kwargs)
        super().__init__(strategy_config=config)
        self.schedule = config.schedule

    def coordinate(
        self,
        available_clients: List[str],
        **kwargs,
    ):
        print("Start Cyclic.")

        model: FLModel = self.load_model()
        start_round = self.strategy_config.start_round
        total_rounds = self.strategy_config.num_rounds

        for r in range(start_round, start_round + total_rounds):
            print(f"Round {r} started.")
            model.current_round = r

            clients = self.sample_clients(available_clients)
            result: Optional[FLModel] = None
            for client in clients:
                meta = {"round": r}
                result: FLModel = self.send_model_and_wait(targets=[client], fl_model=model, meta=meta)[0]

            if result:
                self.save_model(result)
            else:
                raise RuntimeError("result model is None")

        print("Finished Cyclic.")

    def sample_clients(self, clients: List[str]) -> List[str]:
        if self.strategy_config.sample_clients_fn:
            return self.strategy_config.sample_clients_fn(clients)
        else:
            return clients

    def load_model(self) -> Any:
        """Load initial model. Should be implemented by user or subclass."""
        if self.strategy_config.load_model_fn:
            return self.strategy_config.load_model_fn()

    def save_model(self, model: FLModel) -> None:
        if self.strategy_config.save_model_fn:
            self.strategy_config.save_model_fn(model)

    def send_model_and_wait(self, targets: List[str], fl_model: FLModel, meta: Dict) -> Dict[str, FLModel]:
        if not meta:
            meta = {}
        """Send model to clients and wait for responses."""
        fl_model.context = self.strategy_config.dict()
        fl_model.meta.update(meta)
        message: Dict[str, MessageType] = self.communicator.broadcast_and_wait(sites=targets, message=fl_model)
        results = {}
        for site, response in message.items():
            if not isinstance(response, FLModel):
                raise RuntimeError(f"Expected FLModel, got {type(response)}")
            results[site] = response
        return results
