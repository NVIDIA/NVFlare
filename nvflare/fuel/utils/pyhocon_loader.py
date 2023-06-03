from typing import Dict

from pyhocon import ConfigFactory
from pyhocon import ConfigTree
from pyhocon.converter import HOCONConverter

from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel.utils.config_loader import ConfigLoader, Config


class PyhoconConfig(Config):
    def __init__(self, conf: ConfigTree):
        self.conf = conf
        self.format = ConfigFormat.PYHOCON

    def to_dict(self) -> Dict:
        return self._convert_conf_item(self.conf)

    def to_conf(self, element: Dict) -> str:
        config = ConfigFactory.from_dict(element)
        return HOCONConverter.to_hocon(config)

    def _convert_conf_item(self, conf_item):
        result = {}
        if isinstance(conf_item, ConfigTree):
            if len(conf_item) > 0:
                for key, item in conf_item.items():
                    new_key = key.strip('"')  # for dotted keys enclosed with "" to not be interpreted as nested key
                    new_value = self._convert_conf_item(item)
                    result[new_key] = new_value
        elif isinstance(conf_item, list):
            if len(conf_item) > 0:
                result = [self._convert_conf_item(item) for item in conf_item]
            else:
                result = []
        elif conf_item is True:
            return True
        elif conf_item is False:
            return False
        else:
            return conf_item

        return result


class PyhoconLoader(ConfigLoader):
    def __init__(self):
        self.format = ConfigFormat.PYHOCON

    def load_config(self, file_path: str) -> Config:
        config = ConfigFactory.parse_file(file_path)
        conf: ConfigTree = config.get_config("config")
        return PyhoconConfig(conf)
