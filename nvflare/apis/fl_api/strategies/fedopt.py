from typing import List

from nvflare.apis.fl_api.message.fl_message import MessageType
from nvflare.apis.fl_api.registry.strategy_registry import register_strategy
from nvflare.apis.fl_api.strategies.fedavg import FedAvg
from nvflare.apis.fl_api.aggregator.handlers.tensor_handler import TensorHandler

# Pytorch only


# 2. FedOpt (e.g., FedAdam, FedYogi)
@register_strategy("fedopt")
class FedOpt(FedAvg):
    def __init__(self, global_model_state, optimizer, **kwargs):
        super().__init__(**kwargs)
        self.global_model_state = global_model_state  # dict of model parameters (e.g. torch.nn.Parameter or tensors)
        self.optimizer = optimizer  # e.g., a torch.optim.Optimizer instance

    def average(self, updates: List[MessageType]):
        agg_result = self.aggregator.aggregate(updates)
        # Assume agg_result.model_state contains aggregated gradients (same keys as global_model_state)
        # Set gradients for each parameter
        for name, param in self.global_model_state.items():
            if hasattr(param, 'grad'):
                param.grad = agg_result.model_state[name]
            else:
                # For raw tensors, assign .grad attribute dynamically (PyTorch only)
                try:
                    param.grad = agg_result.model_state[name]
                except Exception:
                    pass
        # Optimizer step
        self.optimizer.step()

        # Update FLMessage with new weights and optimizer state
        agg_result.model_state = {name: param.data.clone() for name, param in self.global_model_state.items()}
        # Save optimizer state (PyTorch: state_dict)
        if hasattr(self.optimizer, 'state_dict'):
            agg_result.optimizer_state = self.optimizer.state_dict()
        return agg_result