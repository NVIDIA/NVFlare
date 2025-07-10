from abc import ABC
from dataclasses import Field
from typing import Optional, Callable, List, Tuple, Dict

from nvflare.apis.fl_api import Strategy
from nvflare.apis.fl_api.aggregator.aggregators import DefaultAggregator, FuncAggregatorWrapper
from nvflare.apis.fl_api.communication.wf_comm_client_layers import MessageType
from nvflare.apis.fl_api.interfaces.strategy import StrategyConfig
from nvflare.apis.fl_api.strategies.base_strategy import BaseStrategy
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.utils.math_utils import parse_compare_criteria


class BaseAvgConfig(StrategyConfig):
    aggregator: Optional[Aggregator] = Field(None, description="Aggregator instance to use.")
    aggregation_fn: Optional[Callable[[List[MessageType]], MessageType]] = Field(
        None, description="Aggregator function")


class BaseAvgStrategy(BaseStrategy, ABC):
    def __init__(self, strategy_config: Optional[BaseAvgConfig] = None, **kwargs):
        super().__init__(strategy_config, **kwargs)

        self.stop_condition: Optional[Tuple[str, float, Callable]] = parse_compare_criteria(
            self.strategy_config.stop_cond
        )

        if self.strategy_config.aggregator:
            self.aggregator = self.strategy_config.aggregator
        elif self.strategy_config.aggregation_fn:
            self.aggregator = FuncAggregatorWrapper(self.strategy_config.aggregation_fn)
        else:
            self.aggregator = DefaultAggregator()

    def aggregate(self, model: FLModel, updates: Dict[str, FLModel]) -> FLModel:
        """Aggregate client updates using the configured aggregator."""
        result = self.aggregator.aggregate(list(updates.values()))
        return FLModelUtils.update_model(model, result)
