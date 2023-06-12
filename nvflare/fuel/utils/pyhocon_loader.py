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

from typing import Dict, Optional

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree
from pyhocon.converter import HOCONConverter

from nvflare.fuel.utils.config import Config, ConfigFormat, ConfigLoader


class PyhoconConfig(Config):
    def __init__(self, conf: ConfigTree, file_path: Optional[str] = None):
        super(PyhoconConfig, self).__init__(ConfigFormat.PYHOCON)
        self.conf = conf
        self.file_path = file_path

    def get_native_conf(self):
        return self.conf

    def get_location(self) -> Optional[str]:
        return self.file_path

    def to_dict(self) -> Dict:
        return self._convert_conf_item(self.conf)

    def to_conf_str(self, element: Dict) -> str:
        config = CF.from_dict(element)
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
        super(PyhoconLoader, self).__init__(ConfigFormat.PYHOCON)

    def load_config(self, file_path: str) -> Config:
        config: ConfigTree = self._from_file(file_path)
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
