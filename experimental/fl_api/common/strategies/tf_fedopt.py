from typing import List
from experimental.fl_api.common.strategies.fedopt_base import FedOptBase
from experimental.fl_api.registry.strategy_registry import register_strategy
from experimental.fl_api.common.interfaces import MessageType


@register_strategy("tf_fedopt")
class TFFedOpt(FedOptBase):
    def __init__(self, global_model_state, optimizer, aggregator, **kwargs):
        self.global_model_state = global_model_state
        self.optimizer = optimizer
        self.aggregator = aggregator

    def average(self, updates: List[MessageType]) -> MessageType:
        agg_result = self.aggregator.aggregate(updates)
        # TensorFlow-style optimizer step (pseudocode)
        grads_and_vars = [(agg_result.model_state[name], var) for name, var in self.global_model_state.items()]
        self.optimizer.apply_gradients(grads_and_vars)
        agg_result.model_state = {name: var.numpy() for name, var in self.global_model_state.items()}
        # TensorFlow optimizer state handling as needed (not shown)
        return agg_result
