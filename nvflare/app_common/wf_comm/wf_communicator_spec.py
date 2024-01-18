from abc import ABC
from typing import List, Dict, Optional

from nvflare.fuel.utils.class_utils import instantiate_class
from nvflare.fuel.utils.component_builder import ComponentBuilder
from nvflare.fuel.utils.fobs import fobs


class WFCommunicatorSpec(ABC):

    def __init__(self):
        self.strategy = None
        self.strategy_config: Optional[Dict] = None

    def set_strategy_config(self, strategy_config):
        if strategy_config is None:
            raise ValueError("strategy_config is None")

        if not isinstance(strategy_config, dict):
            raise ValueError(f"strategy_config should be Dict, found '{type(strategy_config)}'")

        self.strategy_config = strategy_config

    def get_strategy(self):
        # if self.strategy is None and isinstance(self.strategy_config, dict):
        print(f"{self.strategy_config=}")
        if isinstance(self.strategy_config, dict):
            strategy = ComponentBuilder().build_component(self.strategy_config)
            if strategy is None:
                raise ValueError("strategy should provided, but get None")
            self.strategy = strategy

        return self.strategy

    def set_serializers(self, serializer_class_paths: List[str] = None):
        if serializer_class_paths:
            for class_path in serializer_class_paths:
                fobs.register(instantiate_class(class_path, {}))
