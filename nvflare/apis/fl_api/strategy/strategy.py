from abc import ABC
from typing import List, Any, Optional, Callable
from nvflare.apis.fl_api.communication.communication_layer import CommunicationLayer

class Strategy(ABC):

    def coordinate(
            self,
            selected_clients: List[str],
            global_state: Any,
            round_number: int,
            communication: "CommunicationLayer",
            **kwargs,
    ) -> Any:
        pass

    async def coordinate_async(
            self,
            selected_clients: List[str],
            global_state: Any,
            round_number: int,
            communication: "CommunicationLayer",
            **kwargs,
    ) -> Any:
        """
        Async coordination for streaming updates.
        """
    pass