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

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, List


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
    def get_conf(self):
        pass

    @abstractmethod
    def get_format(self):
        pass

    @abstractmethod
    def get_location(self) -> Optional[str]:
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        pass

    @abstractmethod
    def to_conf(self, element: Dict) -> str:
        pass

    @abstractmethod
    def get_int(self, key: str, default=None) -> Optional[int]:
        pass

    @abstractmethod
    def get_float(self, key: str, conf=None, default=None) -> Optional[float]:
        pass

    @abstractmethod
    def get_bool(self, key: str, default=None) -> Optional[bool]:
        pass

    @abstractmethod
    def get_str(self, key: str, default=None) -> Optional[str]:
        pass

    @abstractmethod
    def get_list(self, key: str, default=None) -> Optional[List]:
        pass

    @abstractmethod
    def get_config(self, key: str, default=None):
        pass


class ConfigLoader(ABC):

    @abstractmethod
    def load_config(self,
                    file_path: str,
                    default_file_path: Optional[str] = None,
                    overwrite_config: Optional[Dict] = None) -> Config:
        pass

    def load_config_from_str(self, config_str: str) -> Config:
        raise NotImplementedError

    def load_config_from_dict(self, config_dict: dict) -> Config:
        raise NotImplementedError
