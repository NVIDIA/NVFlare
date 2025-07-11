from enum import Enum
from abc import ABC
from typing import List, Any, Optional, Dict, Callable
from pydantic import BaseModel, Field, PositiveInt

from experimental.fl_api.nvflare.communication.wf_comm_client_layers import MessageType
from experimental.fl_api.common.interfaces.comm_layer import CommunicationLayer

#
# class FrameworkType(str, Enum):
#     PYTORCH = "pytorch"
#     LIGHTNING = "lightning"
#     TENSORFLOW = "tensorflow"
#     XGBOOST = "xgboost"
#     SCIKIT_LEARN = "scikit-learn"
#     NONE = "none"


class StrategyConfig(BaseModel):
    """
    Base config for strategies. Extend this in specific strategies for more fields.
    """
    num_clients: PositiveInt = 3
    sample_clients_fn: Optional[Callable[[int], List[str]]] = None
    extra: Dict[str, Any] = dict()

    class Config:
        extra = "allow"


class Strategy(ABC):
    def __init__(self, strategy_config: Optional[StrategyConfig] = None, **kwargs):
        self.strategy_config = strategy_config or StrategyConfig(**kwargs)
        self.communicator: Optional[CommunicationLayer] = None

        # Only put unknown/extra fields into extra
        if kwargs:
            for k, v in kwargs.items():
                if not hasattr(self.strategy_config, k):
                    self.strategy_config.extra[k] = v

    def initialize(self, communicator: Optional[CommunicationLayer] = None, **kwargs):
        self.communicator = communicator

    def coordinate(
        self,
        available_clients: List[str],
        **kwargs,
    ) -> Optional[MessageType]:
        pass

    def finalize(self):
        self.communicator = None
