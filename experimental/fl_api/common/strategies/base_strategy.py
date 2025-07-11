from abc import ABC
from typing import Optional, List, Dict, Callable, Tuple

from experimental.fl_api import Strategy
from experimental.fl_api.common.interfaces.message_type import FedModel
from experimental.fl_api.nvflare.communication.wf_comm_client_layers import MessageType
from experimental.fl_api.common.interfaces.strategy import StrategyConfig


class BaseStrategyConfig(StrategyConfig):
    stop_cond: Optional[Tuple[str, float, Callable]] = None

    # The number of checks with no improvement after which
    # the FL will be stopped. If set to `None`, this parameter is disabled.
    # If stop_condition is None, patience does not apply
    patience: Optional[int] = None


class BaseStrategy(Strategy, ABC):
    def __init__(self, strategy_config: Optional[StrategyConfig] = None, **kwargs):
        super().__init__(strategy_config, **kwargs)
        self.no_improvement_cks = 0

    def load_model(self) -> FedModel:
        """Load initial model. Should be implemented by user or subclass."""
        if self.strategy_config.load_model_fn:
            return self.strategy_config.load_model_fn()

    def save_model(self, model: FedModel) -> None:
        if self.strategy_config.save_model_fn:
            self.strategy_config.save_model_fn(model)

    def sample_clients(self, clients: List[str]) -> List[str]:
        if self.strategy_config.sample_clients_fn:
            return self.strategy_config.sample_clients_fn(clients)
        else:
            return clients

    def send_model_and_wait(self, targets: List[str], fl_model: FedModel) -> Dict[str, FedModel]:
        """Send model to clients and wait for responses."""
        fl_model.context = self.strategy_config.dict()

        message: Dict[str, MessageType] = self.communicator.broadcast_and_wait(sites=targets, message=fl_model)
        results = {}
        for site, response in message.items():
            if not isinstance(response, FedModel):
                raise RuntimeError(f"Expected FedModel, got {type(response)}")
            results[site] = response
        return results

    def select_best_model(
        self, curr_model: FedModel, best_model: FedModel, stop_condition: Optional[Tuple[str, float, Callable]]
    ) -> FedModel:
        if best_model is None:
            best_model = curr_model
            return best_model

        if stop_condition:
            metric, _, op_fn = stop_condition
            patience = self.strategy_config.patience
            if self.is_curr_model_better(best_model, curr_model, metric, op_fn):
                if patience:
                    self.no_improvement_cks = 0
                print("Current model is new best model.")
                best_model = curr_model
            else:
                if patience:
                    self.no_improvement_cks += 1
        else:
            best_model = curr_model

        return best_model

    def should_stop(self, metrics: Optional[Dict] = None, stop_condition: Optional[Tuple[str, float, Callable]] = None):
        if stop_condition is None or metrics is None:
            return False

        patience = self.strategy_config.patience
        no_improvement_cks = self.no_improvement_cks

        if patience and (patience < no_improvement_cks):
            print(f"Exceeded the number of checks ({patience}) without improvements")
            return True

        key, target, op_fn = stop_condition
        value = metrics.get(key, None)
        if value is None:
            raise RuntimeError(f"stop criteria key '{key}' doesn't exists in metrics")

        if op_fn(value, target):
            print(f"Early stop condition satisfied: {stop_condition}")
            return True

        return False

    def is_curr_model_better(self, best_model: FedModel, curr_model: FedModel, target_metric: str, op_fn: Callable) -> bool:
        curr_metrics = curr_model.metrics
        if curr_metrics is None:
            return False
        if target_metric not in curr_metrics:
            return False

        best_metrics = best_model.metrics
        return op_fn(curr_metrics.get(target_metric), best_metrics.get(target_metric))
