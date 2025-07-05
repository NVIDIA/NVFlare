from typing import List
from nvflare.apis.fl_api.strategies.fedopt_base import FedOptBase
from nvflare.apis.fl_api.registry.strategy_registry import register_strategy
from nvflare.apis.fl_api.message.fl_message import MessageType

@register_strategy("pt_fedopt")
class PTFedOpt(FedOptBase):
    def __init__(self, global_model_state, optimizer, aggregator, **kwargs):
        self.global_model_state = global_model_state
        self.optimizer = optimizer
        self.aggregator = aggregator

    def average(self, updates: List[MessageType]) -> MessageType:
        agg_result = self.aggregator.aggregate(updates)
        # Set gradients for each parameter
        for name, param in self.global_model_state.items():
            param.grad = agg_result.model_state[name]
        self.optimizer.step()
        agg_result.model_state = {name: param.data.clone() for name, param in self.global_model_state.items()}
        if hasattr(self.optimizer, 'state_dict'):
            agg_result.optimizer_state = self.optimizer.state_dict()
        return agg_result
