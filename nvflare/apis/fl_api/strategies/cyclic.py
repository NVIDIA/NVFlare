from typing import List, Any, Optional
from pydantic import Field
from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer
from nvflare.apis.fl_api.interfaces.strategy import Strategy, StrategyConfig

class CyclicConfig(StrategyConfig):
    schedule: List[str] = Field(..., description="List of client names in the cyclic schedule.")
    num_rounds: int = Field(10, description="Number of rounds to run the cyclic strategy.")
    start_round: int = Field(1, description="Starting round index.")

class Cyclic(Strategy):
    def __init__(self, strategy_config: Optional[CyclicConfig] = None, **kwargs):
        config = strategy_config or CyclicConfig(**kwargs)
        super().__init__(strategy_config=config)
        self.schedule = config.schedule
        self.num_rounds = config.num_rounds

    def coordinate(
            self,
            selected_clients: List[str],
            **kwargs,
    ):
        current_round = self.strategy_config.start_round
        load_model = self.load_model()

        

        current_index = current_round % len(self.schedule)
        current_client = self.schedule[current_index]

        next_index = (current_round + 1) % len(self.schedule)
        next_client = self.schedule[next_index]

        self.communicator.broadcast_and_wait(sites=[next_client], message=global_state)

        # Receive updated state from next client
        update = communicator.collect_from_queue(next_client)
        return update
