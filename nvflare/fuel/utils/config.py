from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict


class ConfigFormat(Enum):
    # use file format extension as value indicator
    JSON = ".json"
    PYHOCON = ".conf"
    OMEGACONF = ".yml"

    JSON_DEFAULT = ".json.default"
    PYHOCON_DEFAULT = ".conf.default"
    OMEGACONF_DEFAULT = ".yml.default"


class Config(ABC):
    @abstractmethod
    def to_dict(self) -> Dict:
        pass

    @abstractmethod
    def to_conf(self, element: Dict) -> str:
        pass

