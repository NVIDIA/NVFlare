from typing import Any, Optional

from nvflare.apis.fl_api.message.fl_message import FLMessage
from nvflare.apis.fl_api.aggregator.handlers.agg_utils import scale_dict, add_dict, average_dict


# --- Handler Interface ---
class TensorHandler:
    @staticmethod
    def scale(message: FLMessage, factor: float) -> Optional[FLMessage]:
        """Scale all relevant states (model_state, optimizer_state) in the message if present."""
        msg = message.copy(deep=True)
        msg.model_state = scale_dict(msg.model_state, factor)
        msg.optimizer_state = scale_dict(msg.optimizer_state, factor)
        return msg

    @staticmethod
    def add(a: FLMessage, b: FLMessage) -> Optional[FLMessage]:
        """Add all relevant states in the messages if present."""
        msg = a.copy(deep=True)
        msg.model_state = add_dict(a.model_state, b.model_state)
        msg.optimizer_state = add_dict(a.optimizer_state, b.optimizer_state)
        return msg

    @staticmethod
    def average_pair(prev: FLMessage, new: FLMessage) -> Optional[FLMessage]:
        """Average all relevant states in the messages if present."""
        msg = prev.copy(deep=True)
        msg.model_state = average_dict(prev.model_state, new.model_state)
        msg.optimizer_state = average_dict(prev.optimizer_state, new.optimizer_state)
        return msg
