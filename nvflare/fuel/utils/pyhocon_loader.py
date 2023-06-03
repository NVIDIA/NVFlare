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

from typing import Dict, Optional, List

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree
from pyhocon.converter import HOCONConverter

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils.config import ConfigFormat, Config, ConfigLoader


class PyhoconConfig(Config):

    def __init__(self, conf: ConfigTree, file_path: Optional[str] = None):
        self.conf = conf
        self.format = ConfigFormat.PYHOCON
        self.file_path = file_path

    def get_conf(self):
        return self.conf

    def get_format(self):
        return self.format

    def get_location(self) -> Optional[str]:
        return self.file_path

    def to_dict(self) -> Dict:
        return self._convert_conf_item(self.conf)

    def to_conf(self, element: Dict) -> str:
        config = CF.from_dict(element)
        return HOCONConverter.to_hocon(config)

    def get_int(self, key: str, default=None) -> Optional[int]:
        return self.conf.get_int(key, default)

    def get_float(self, key: str, conf=None, default=None) -> Optional[float]:
        return self.conf.get_float(key, default)

    def get_bool(self, key: str, default=None) -> Optional[bool]:
        try:
            return self.conf.get_bool(key, default)
        except (TypeError, ValueError):
            v = self.conf.get(key, default)
            if v is None:
                return v
            if isinstance(v, int):
                return v != 0
            if isinstance(v, str):
                v = v.lower()
                return v in ["true", "t", "yes", "y", "1"]
            raise ConfigError(f"{key} has type '{type(v).__name__}' value '{v}' cannot be converted to bool ")

    def get_str(self, key: str, default=None) -> Optional[str]:
        return self.conf.get_string(key, default)

    def get_list(self, key: str, default=None) -> Optional[List]:
        return self.conf.get_list(key, default)

    def get_config(self, key: str, default=None):
        conf: ConfigTree = self.conf.get_config(key, default)
        return PyhoconConfig(conf, self.file_path)

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

    def load_config(self,
                    file_path: str,
                    default_file_path: Optional[str] = None,
                    overwrite_config: Optional[Dict] = None) -> Config:

        config: ConfigTree = self._from_file(file_path)
        if default_file_path:
            config = config.with_fallback(self._from_file(default_file_path))
        if overwrite_config:
            config = CF.from_dict(overwrite_config).with_fallback(config)

        conf: ConfigTree = config.get_config("config")
        return PyhoconConfig(conf, file_path)

    def load_config_from_str(self, config_str: str) -> Config:
        config = CF.parse_string(config_str)
        conf: ConfigTree = config.get_config("config")
        return PyhoconConfig(conf)

    def load_config_from_dict(self, config_dict: dict) -> Config:
        config = CF.from_dict(config_dict)
        conf: ConfigTree = config.get_config("config")
        return PyhoconConfig(conf)

    def _from_file(self, file_path):
        return CF.parse_file(file_path)
