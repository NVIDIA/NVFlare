from typing import Optional

from experimental.fl_api.common.interfaces.message_type import FedModel
from experimental.fl_api.common.aggregator.handlers.agg_utils import scale_dict, add_dict, average_dict


# --- Handler Interface ---
class TensorHandler:
    @staticmethod
    def scale(message: FedModel, factor: float) -> Optional[FedModel]:
        """Scale all relevant states (model_state, optimizer_state) in the message if present."""
        msg = message.copy(deep=True)
        msg.model_state = scale_dict(msg.model_state, factor)
        msg.optimizer_state = scale_dict(msg.optimizer_state, factor)
        return msg

    @staticmethod
    def add(a: FedModel, b: FedModel) -> Optional[FedModel]:
        """Add all relevant states in the messages if present."""
        msg = a.copy(deep=True)
        msg.params = add_dict(a.params, b.params)
        msg.optimizer_params = add_dict(a.optimizer_params, b.optimizer_params)
        return msg

    @staticmethod
    def average_pair(prev: FedModel, new: FedModel) -> Optional[FedModel]:
        """Average all relevant states in the messages if present."""
        msg = prev.copy(deep=True)
        msg.model_state = average_dict(prev.params, new.params)
        msg.optimizer_state = average_dict(prev.optimizer_params, new.optimizer_params)
        return msg
