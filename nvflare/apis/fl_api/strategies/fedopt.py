from nvflare.apis.fl_api.registry.strategy_registry import register_strategy
from nvflare.apis.fl_api.strategies.base.agg_strategy import AggStrategy


# 2. FedOpt (e.g., FedAdam, FedYogi)

@register_strategy("fedopt")
class FedOpt(AggStrategy):
    def __init__(self, optimizer, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optimizer

    def aggregate(self, updates, round_number):
        gradients = self.aggregator.aggregate(updates)
        return self.optimizer.apply(gradients)