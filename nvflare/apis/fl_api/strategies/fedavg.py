from typing import List, Optional, Any

from nvflare.apis.fl_api.strategies.base_avg_strategy import BaseAvgStrategy, BaseAvgConfig

from nvflare.apis.fl_api.communication.wf_comm_client_layers import MessageType


class FedAvgConfig(BaseAvgConfig):


class FedAvg(BaseAvgStrategy):
    def __init__(
            self,
            strategy_config: Optional[FedAvgConfig] = None,
            **kwargs
    ):
        super().__init__(strategy_config, **kwargs)
        self.best_model = None
        self.current_round = None

    def coordinate(
            self,
            available_clients: List[str],
            **kwargs,
    ) :

        print("Start FedAvg.")
        model = self.load_model()

        for r in range(self.strategy_config.start_round, self.strategy_config.num_rounds):
            print(f"Round {self.current_round} started.")
            self.current_round = r
            model.current_round = r

            clients = self.sample_clients(available_clients)

            results = self.send_model_and_wait(targets=clients, fl_model=model)
            model = self.aggregate(model, results)

            print(f"Round {r} global metrics: {model.metrics}")
            self.best_model = self.select_best_model(model, self.best_model, self.stop_condition)
            self.save_model(self.best_model)

            if self.should_stop(model.metrics, self.stop_condition):
                print(
                    f"Stopping at round={self.current_round} out of total_rounds={self.strategy_config.num_rounds}. Early stop condition satisfied.")
                break
        print("Finished FedAvg.")

