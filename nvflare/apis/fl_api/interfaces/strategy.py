from abc import ABC
from typing import List, Any

from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer
from nvflare.apis.fl_api.registry.strategy_registry import _STRATEGY_REGISTRY


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
