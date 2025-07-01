from abc import ABC
from typing import List, Any

from nvflare.apis.fl_api.communication.comm_layer import CommunicationLayer
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

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "Strategy":
        strategy_cls = _STRATEGY_REGISTRY.get(name)
        if not strategy_cls:
            raise ValueError(f"Unknown strategy preset: {name}")
        return strategy_cls(**kwargs)
