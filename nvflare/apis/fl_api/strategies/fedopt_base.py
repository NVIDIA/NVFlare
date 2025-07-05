from abc import ABC, abstractmethod
from typing import List
from nvflare.apis.fl_api.message.fl_message import MessageType

class FedOptBase(ABC):
    @abstractmethod
    def __init__(self, global_model_state, optimizer, **kwargs):
        pass

    @abstractmethod
    def average(self, updates: List[MessageType]) -> MessageType:
        """Aggregate and update global model using the optimizer."""
        pass
