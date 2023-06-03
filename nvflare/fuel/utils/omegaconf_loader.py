from typing import Dict

import yaml

from nvflare.fuel.utils.config import ConfigFormat, Config
from nvflare.fuel.utils.config_loader import ConfigLoader
from nvflare.fuel.utils.json_config_loader import JsonConfig
from nvflare.security.logging import secure_format_exception


class OmegaConfLoader(ConfigLoader):
    def __init__(self):
        self.format = ConfigFormat.OMEGACONF

    def load_config(self, file_path: str) -> Config:
        with open(file_path, "r") as file:
            try:
                conf = yaml.safe_load(file)
                return JsonConfig(conf)
            except Exception as e:
                print("Error loading config file {}: {}".format(file_path, secure_format_exception(e)))
        raise e

