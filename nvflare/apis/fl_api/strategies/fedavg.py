from typing import List, Optional, Callable, Any, Dict, Tuple
from pydantic import Field
from nvflare.apis.fl_api.interfaces.strategy import Strategy, StrategyConfig, FrameworkType
from nvflare.apis.fl_api.aggregator.aggregators import Aggregator, FuncAggregatorWrapper, DefaultAggregator
from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer
from nvflare.apis.fl_api.message.message_type import MessageType
from nvflare.apis.fl_api.strategies.strategy_core import StrategyCore
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.utils.math_utils import parse_compare_criteria


class FedAvgConfig(StrategyConfig):
    num_clients: int = Field(3, description="Number of clients to select per round.")
    num_rounds: int = Field(5, description="Total number of federated learning rounds.")
    start_round: int = Field(0, description="Starting round index.")
    framework: FrameworkType = Field(FrameworkType.PYTORCH, description="ML framework to use.")
    stop_cond: Optional[str] = Field(None, description="Stopping condition as a string expression.")
    aggregator: Optional[Aggregator] = Field(None, description="Aggregator instance to use.")
    aggregation_fn: Optional[Callable[[List[MessageType]], MessageType]] = Field(None,
                                                                                 description="Custom aggregation function.")
    # add default pytorch load/save model functions
    load_model_fn: Optional[Callable[[], Any]] = Field(None, description="Function to load initial model.")
    save_model_fn: Optional[Callable[[Any], None]] = Field(None, description="Function to save model.")
    sample_clients_fn: Optional[Callable[[int], List[str]]] = Field(None, description="Function to sample clients.")

class FedAvg(Strategy):
    def __init__(
            self,
            strategy_config: Optional[FedAvgConfig] = None,
            communicator: Optional[CommunicationLayer] = None,
            **kwargs
    ):
        """
        Federated Averaging strategy.
        Args:
            strategy_config: Config object for FedAvg.
            communicator: CommunicationLayer instance for client communication.
            **kwargs: Additional config fields.
        """

        config = strategy_config or FedAvgConfig(**kwargs)
        super().__init__(strategy_config=config, communicator=communicator)

        if config.aggregator:
            self.aggregator = config.aggregator
        elif config.aggregation_fn:
            self.aggregator = FuncAggregatorWrapper(config.aggregation_fn)
        else:
            self.aggregator = DefaultAggregator()

        self.best_model = None
        self.current_round = None
        self.stop_condition: Optional[Tuple[str, float, Callable]] = \
            parse_compare_criteria(self.strategy_config.stop_cond)

    def load_model(self) -> Any:
        """Load initial model. Should be implemented by user or subclass."""
        if self.strategy_config.load_model_fn:
            return self.strategy_config.load_model_fn()


    def save_model(self, model: FLModel) -> None:
        if self.strategy_config.save_model_fn:
            self.strategy_config.save_model_fn(model)

    def sample_clients(self, clients: List[str]) -> List[str]:
        if self.strategy_config.sample_clients_fn:
            return self.strategy_config.sample_clients_fn(clients)
        else:
            return clients

    def average(self, updates: Dict[str, FLModel]) -> FLModel:
        """Aggregate client updates using the configured aggregator."""
        updates = list(updates.values())
        model = self.aggregator.aggregate(updates)
        return model

    def coordinate(
            self,
            selected_clients: List[str],
            **kwargs,
    ) :
        """
        Main coordination loop for FedAvg.
        """
        print("Start FedAvg.")
        model: FLModel = self.load_model()
        for r in range(self.strategy_config.start_round, self.strategy_config.num_rounds):
            self.current_round = r
            print(f"Round {self.current_round} started.")
            clients = self.sample_clients(selected_clients)
            results = self.send_model_and_wait(targets=clients, fl_model=model)
            aggregate_results = self.average(results)
            model = FLModelUtils.update_model(model, aggregate_results)

            print(f"Round {r} global metrics: {model.metrics}")
            self.select_best_model(model)
            self.save_model(self.best_model)

            if self.should_stop(getattr(model, 'metrics', None), self.strategy_config.stop_cond):
                print(
                    f"Stopping at round={self.current_round} out of total_rounds={self.strategy_config.num_rounds}. Early stop condition satisfied.")
                break
        print("Finished FedAvg.")

    def send_model_and_wait(self, targets: List[str], fl_model: FLModel) -> Dict[str, FLModel]:
        """Send model to clients and wait for responses."""
        fl_model.context = self.strategy_config.dict()
        fl_model.meta["round"] = self.current_round

        message: Dict[str, MessageType] = self.communicator.broadcast_and_wait(sites=targets, message=input_message)
        results = {}
        for site, response in message.items():
            if not isinstance(response, FLModel):
                raise RuntimeError(f"Expected FLModel, got {type(response)}")
            results[site] = response
        return results

    def select_best_model(self, curr_model: FLModel):
        if self.best_model is None:
            self.best_model = curr_model
            return

        if self.stop_condition:
            metric, _, op_fn = self.stop_condition
            if FLModelUtils.is_curr_model_better(self.best_model, curr_model, metric, op_fn):
                print("Current model is new best model.")
                self.best_model = curr_model
        else:
            self.best_model = curr_model

    def should_stop(self, metrics: Optional[Dict] = None, stop_condition: Optional[Tuple[str, float, Callable]] = None):
        if stop_condition is None or metrics is None:
            return False

        key, target, op_fn = stop_condition
        value = metrics.get(key, None)

        if value is None:
            raise RuntimeError(f"stop criteria key '{key}' doesn't exists in metrics")

        return op_fn(value, target)
