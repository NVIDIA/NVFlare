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
from typing import Dict, List, Optional


class ConfigFormat(Enum):
    JSON = "JSON"
    PYHOCON = "PYHOCON"
    OMEGACONF = "OMEGACONF"

    @classmethod
    def _config_format_extensions(cls):
        return {
            "JSON": [".json", ".json.default"],
            "PYHOCON": [".conf", ".conf.default"],
            "OMEGACONF": [".yml", ".yml.default"],
        }

    @classmethod
    def search_order(cls):
        return [cls.JSON, cls.PYHOCON, cls.OMEGACONF]

    @classmethod
    def ordered_search_extensions(cls) -> List:
        search_sequence: List[ConfigFormat] = cls.search_order()
        extensions = []
        for fmt in search_sequence:
            exts = cls.get_extensions(fmt.name)
            if exts:
                for ext in exts:
                    extensions.append((fmt, ext))

        return extensions

    @classmethod
    def get_extensions(cls, fmt: str) -> Optional[List[str]]:
        return cls._config_format_extensions().get(fmt, None)

    @classmethod
    def config_exts(cls) -> str:
        search_sequence: List[ConfigFormat] = cls.search_order()
        extensions = []
        for fmt in search_sequence:
            exts = cls.get_extensions(fmt.name)
            if exts:
                extensions.extend(exts)

        return "|".join(iter(extensions))


class Config(ABC):
    @abstractmethod
    def get_conf(self):
        # Return the original underline config object representation if you prefer to use it directly
        # Pyhocon → ConfigTree
        # JSON → Dict
        # OMEGACONF → Conf
        pass

    @abstractmethod
    def get_format(self):
        pass

    @abstractmethod
    def get_location(self) -> Optional[str]:
        # return the real file path of the reel configuration.
        # For example, if initial config_fed_json.json was initially input file path. but the real file path is
        # config_fed_json.conf. the get location will be return config_fed_json.conf
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
    def load_config(
        self, file_path: str, default_file_path: Optional[str] = None, overwrite_config: Optional[Dict] = None
    ) -> Config:
        """
        configuration from default_file_path will be the default config if specified
        configuration from file_path will be the merge with default config overwrite the same key
        configuration from overwrite_config if provided will be the merge with config overwrite the same key
        Args:
            file_path: file path for configuration to be loaded
            default_file_path: file path for default configuration
            overwrite_config: dict config that will be overwrite the final config if provided

        Returns:

        """

        pass

    def load_config_from_str(self, config_str: str) -> Config:
        raise NotImplementedError

    def load_config_from_dict(self, config_dict: dict) -> Config:
        raise NotImplementedError
