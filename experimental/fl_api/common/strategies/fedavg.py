from typing import List, Optional, Callable

from experimental.fl_api.common.interfaces.message_type import MessageType
from experimental.fl_api.common.interfaces.strategy import StrategyConfig
from experimental.fl_api.common.strategies.base_strategy import BaseStrategy
from nvflare.app_common.abstract.aggregator import Aggregator


class FedAvgConfig(StrategyConfig):
    aggregator: Aggregator = None
    aggregation_fn: Optional[Callable[[List[MessageType]], MessageType]] = None


class FedAvg(BaseStrategy):
    def __init__(
            self,
            strategy_config: Optional[FedAvgConfig] = None,
            **kwargs
    ):
        super().__init__(strategy_config, **kwargs)
        self.best_model = None

    def coordinate(
            self,
            available_clients: List[str],
            **kwargs,
    ) :

        print("Start FedAvg.")
        model = self.load_model()

        for r in range(self.strategy_config.start_round, self.strategy_config.num_rounds):
            print(f"Round {r} started.")
            model.current_round = r

            clients = self.sample_clients(available_clients)
            results = self.send_model_and_wait(targets=clients, fl_model=model)
            model = self.aggregate(model, results)

            print(f"Round {r} global metrics: {model.metrics}")
            self.best_model = self.select_best_model(model, self.best_model, self.stop_condition)
            self.save_model(self.best_model)

            if self.should_stop(model.metrics, self.stop_condition):
                break
        print("Finished FedAvg.")

