# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from typing import Dict, List, Optional

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils.config import Config, ConfigFormat, ConfigLoader
from nvflare.security.logging import secure_format_exception


class JsonConfig(Config):
    def __init__(self, conf: Dict, file_path: Optional[str] = None):
        self.conf = conf
        self.format = ConfigFormat.JSON
        self.file_path = file_path

    def get_conf(self):
        return self.conf

    def get_format(self):
        return self.format

    def get_location(self) -> Optional[str]:
        return self.file_path

    def to_dict(self) -> Dict:
        return self.conf

    def to_conf(self, element: Dict) -> str:
        return json.dumps(element)

    def get_int(self, key: str, default=None) -> Optional[int]:
        value = self.conf.get(key, default)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            raise ConfigError("{key} has type '{type}' rather than 'int'".format(key=key, type=type(value).__name__))

    def get_float(self, key: str, conf=None, default=None) -> Optional[float]:
        value = self.conf.get(key, default)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            raise ConfigError("{key} has type '{type}' rather than 'float'".format(key=key, type=type(value).__name__))

    def get_bool(self, key: str, default=None) -> Optional[bool]:
        v = self.conf.get(key, default)

        if v is None:
            return v
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return v != 0
        if isinstance(v, str):
            v = v.lower()
            return v in ["true", "t", "yes", "y", "1"]
        raise ConfigError(f"{key} has type '{type(v).__name__}' value '{v}' cannot be converted to bool ")

    def get_str(self, key: str, default=None) -> Optional[str]:
        value = self.conf.get(key, default)
        try:
            return str(value) if value is not None else None
        except (TypeError, ValueError):
            raise ConfigError(
                "{key} has type '{type}' cannot covert to 'str'".format(key=key, type=type(value).__name__)
            )

    def get_list(self, key: str, default=None) -> Optional[List]:
        value = self.conf.get(key, default)
        if value:
            if isinstance(value, List):
                return value
            else:
                raise ConfigError(
                    "{key} has type '{type}' rather than 'List'".format(key=key, type=type(value).__name__)
                )
        else:
            return None

    def get_config(self, key: str, default=None):
        value = self.conf.get(key, default)
        if value:
            if isinstance(value, Dict):
                return JsonConfig(value, self.file_path)
            else:
                raise ConfigError(
                    "{key} has type '{type}' rather than 'Dict' or 'JsonConfig' ".format(
                        key=key, type=type(value).__name__
                    )
                )
        else:
            return None


class JsonConfigLoader(ConfigLoader):
    def __init__(self):
        self.format = ConfigFormat.JSON
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_config(
        self, file_path: str, default_file_path: Optional[str] = None, overwrite_config: Optional[Dict] = None
    ) -> Config:

        conf_dict = self._from_file(file_path)
        if default_file_path:
            default_conf_dict = self._from_file(default_file_path)
            conf_dict = {**default_conf_dict, **conf_dict}
        if overwrite_config:
            conf_dict = {**conf_dict, **overwrite_config}

        return JsonConfig(conf_dict, file_path)

    def load_config_from_str(self, config_str: str) -> Config:
        try:
            conf = json.loads(config_str)
            return JsonConfig(conf)
        except Exception as e:
            self.logger.error("Error loading config {}: {}".format(config_str, secure_format_exception(e)))
            raise e

    def load_config_from_dict(self, config_dict: dict) -> Config:
        return JsonConfig(config_dict)

    def _from_file(self, path) -> Dict:
        with open(path, "r") as file:
            try:
                return json.load(file)
            except Exception as e:
                self.logger.error("Error loading config file {}: {}".format(path, secure_format_exception(e)))
                raise e
