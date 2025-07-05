from typing import List, Any, Tuple, Union, Optional

from nvflare.apis.fl_api.aggregator.handlers.tensor_handler import TensorHandler
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.apis.fl_api.message.fl_message import FLMessage, MessageEnvelope
from nvflare.apis.fl_api.interfaces.comm_layer import MessageType

np, _ = optional_import("numpy")
torch, _ = optional_import("torch")
tf, _ = optional_import("tensorflow")


#TODO: focused on sync aggregate for now, handle async incremental aggregate later.

class Aggregator:
    def aggregate(self, updates: List[MessageType]) -> MessageType:
        raise NotImplementedError


class FuncAggregatorWrapper(Aggregator):
    def __init__(self, fn):
        self.fn = fn

    def aggregate(self, updates: List[MessageType]) -> MessageType:
        return self.fn(updates)

class DefaultAggregator(Aggregator):
    def aggregate(self, updates: List[MessageType]) -> MessageType:
        # Weighted average aggregation using TensorHandler
        total_weight = sum(update.meta.get("weight", 1) for update in updates)
        if total_weight == 0:
            raise ValueError("Total weight is zero. Cannot aggregate.")
        # Weighted scaling and summing
        result = None
        for update in updates:
            scale = update.meta.get("weight", 1) / total_weight
            scaled = TensorHandler.scale(update, scale)
            if result is None:
                result = scaled
            else:
                result = TensorHandler.add(result, scaled)
        return result
