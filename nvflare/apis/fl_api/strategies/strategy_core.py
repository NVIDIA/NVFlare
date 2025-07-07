from abc import ABC
from nvflare.apis.fl_api import Strategy
from nvflare.app_common.abstract.fl_model import FLModel


class StrategyCore(Strategy):

    def load_model(self) -> FLModel:
        """Load initial model. Should be implemented by user or subclass."""
        pass

    def save_model(self, model: FLModel) -> None:
        pass
