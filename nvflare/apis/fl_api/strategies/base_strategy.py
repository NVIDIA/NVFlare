from abc import ABC
from dataclasses import Field
from typing import Optional, List, Dict, Callable, Tuple

from nvflare.apis.fl_api import Strategy
from nvflare.apis.fl_api.communication.wf_comm_client_layers import MessageType
from nvflare.apis.fl_api.interfaces.strategy import StrategyConfig
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


class BaseStrategyConfig(StrategyConfig):
    stop_cond: Optional[Tuple[str, float, Callable]] = Field(None, description="Stop condition")

    # The number of checks with no improvement after which
    # the FL will be stopped. If set to `None`, this parameter is disabled.
    # If stop_condition is None, patience does not apply
    patience: Optional[int] = Field(None, description="number of checks with no improvement")


class BaseStrategy(Strategy, ABC):
    def __init__(self, strategy_config: Optional[StrategyConfig] = None, **kwargs):
        super().__init__(strategy_config, **kwargs)
        self.no_improvement_cks = 0

    def load_model(self) -> FLModel:
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

    def send_model_and_wait(self, targets: List[str], fl_model: FLModel) -> Dict[str, FLModel]:
        """Send model to clients and wait for responses."""
        fl_model.context = self.strategy_config.dict()

        message: Dict[str, MessageType] = self.communicator.broadcast_and_wait(sites=targets, message=fl_model)
        results = {}
        for site, response in message.items():
            if not isinstance(response, FLModel):
                raise RuntimeError(f"Expected FLModel, got {type(response)}")
            results[site] = response
        return results

    def select_best_model(
        self, curr_model: FLModel, best_model: FLModel, stop_condition: Optional[Tuple[str, float, Callable]]
    ) -> FLModel:
        if best_model is None:
            best_model = curr_model
            return best_model

        if stop_condition:
            metric, _, op_fn = stop_condition
            patience = self.strategy_config.patience
            if FLModelUtils.is_curr_model_better(best_model, curr_model, metric, op_fn):
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
