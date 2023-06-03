import json
from typing import Dict

from nvflare.fuel.utils.config import ConfigFormat, Config
from nvflare.fuel.utils.config_loader import ConfigLoader
from nvflare.security.logging import secure_format_exception


class JsonConfig(Config):
    def __init__(self, conf: Dict):
        self.conf = conf
        self.format = ConfigFormat.JSON

    def to_dict(self) -> Dict:
        return self.conf

    def to_conf(self, element: Dict) -> str:
        return json.dumps(element)


class JsonConfigLoader(ConfigLoader):
    def __init__(self):
        self.format = ConfigFormat.JSON

    def load_config(self, file_path: str) -> Config:
        with open(file_path, "r") as file:
            try:
                conf = json.load(file)
                return JsonConfig(conf)
            except Exception as e:
                print("Error loading config file {}: {}".format(file_path, secure_format_exception(e)))
                raise e
