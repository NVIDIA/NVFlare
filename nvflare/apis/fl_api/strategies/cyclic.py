

from typing import List, Any

from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer
from nvflare.apis.fl_api.registry.strategy_registry import register_strategy
from nvflare.apis.fl_api.interfaces.strategy import Strategy


@register_strategy("cyclic")
class CyclicStrategy(Strategy):
    def __init__(self, schedule: List[str], **kwargs):
        super().__init__(**kwargs)
        self.schedule = schedule

    def coordinate(
            self,
            selected_clients: List[str],
            global_state: Any,
            round_number: int,
            communication: CommunicationLayer,
            **kwargs,
    ) -> Any:
        current_client = self.schedule[round_number % len(self.schedule)]
        next_index = (round_number + 1) % len(self.schedule)
        next_client = self.schedule[next_index]

        # Send global_state to next client (not to current client)
        communication.broadcast_state([next_client], global_state, exclude=[current_client])

        # Receive updated state from next client
        update = communication.receive_state(next_client)
        return update
