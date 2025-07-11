from abc import ABC, abstractmethod
from typing import List
from experimental.fl_api.common.interfaces import MessageType


class FedOptBase(ABC):
    @abstractmethod
    def __init__(self, global_model_state, optimizer, **kwargs):
        pass

    @abstractmethod
    def average(self, updates: List[MessageType]) -> MessageType:
        """Aggregate and update global model using the optimizer."""
        pass
