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
from typing import Dict, Optional

from nvflare.fuel.utils.config import Config, ConfigFormat, ConfigLoader
from nvflare.security.logging import secure_format_exception


class JsonConfig(Config):
    def __init__(self, conf: Dict, file_path: Optional[str] = None):
        super(JsonConfig, self).__init__(conf, ConfigFormat.JSON, file_path)

    def to_dict(self, resolve: Optional[bool] = True) -> Dict:
        return self.conf

    def to_str(self, element: Optional[Dict] = None) -> str:
        if element is None:
            return json.dumps(self.conf)
        else:
            return json.dumps(element)


class JsonConfigLoader(ConfigLoader):
    def __init__(self):
        super(JsonConfigLoader, self).__init__(fmt=ConfigFormat.JSON)
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_config(self, file_path: str) -> Config:
        conf_dict = self._from_file(file_path)
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
