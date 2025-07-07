from enum import Enum
from abc import ABC
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field

from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer, SimulatedCommLayer


class FrameworkType(str, Enum):
    PYTORCH = "pytorch"
    LIGHTNING = "lightning"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    SCIKIT_LEARN = "scikit-learn"
    NONE = "none"


class StrategyConfig(BaseModel):
    """
    Base config for strategies. Extend this in specific strategies for more fields.
    """
    framework: FrameworkType = Field(None, description="ML framework to use.")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Extra/experimental parameters.")

    class Config:
        extra = "allow"


class Strategy(ABC):
    def __init__(self,
                 strategy_config: Optional[StrategyConfig] = None,
                 **kwargs):
        self.strategy_config = strategy_config or StrategyConfig(**kwargs)
        self.communicator: Optional[CommunicationLayer] = SimulatedCommLayer()

    # Only put unknown/extra fields into extra
        for k, v in kwargs.items():
            if not hasattr(self.strategy_config, k):
                self.strategy_config.extra[k] = v

    def initialize(self, communicator: Optional[CommunicationLayer] = None, **kwargs):
        if communicator is not None:
            self.communicator = communicator


    def coordinate(
            self,
            selected_clients: List[str],
            **kwargs,
    ):
        pass


    def finalize(self):
        self.communicator = None
