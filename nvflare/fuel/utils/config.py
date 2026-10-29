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
from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, List, Optional


class ConfigFormat(Enum):
    JSON = "JSON"
    # Legacy identifiers are retained for explicit unsupported-format errors.
    PYHOCON = "PYHOCON"
    OMEGACONF = "OMEGACONF"

    @classmethod
    def config_ext_formats(cls):
        return OrderedDict(
            {
                ".json": ConfigFormat.JSON,
                ".conf": ConfigFormat.JSON,
                # Discover YAML overrides so loading reports an error instead of using defaults.
                ".yml": ConfigFormat.OMEGACONF,
                ".yaml": ConfigFormat.OMEGACONF,
                ".json.default": ConfigFormat.JSON,
                ".conf.default": ConfigFormat.JSON,
                ".yml.default": ConfigFormat.OMEGACONF,
                ".yaml.default": ConfigFormat.OMEGACONF,
            }
        )

    @classmethod
    def extensions(cls, target_fmt=None) -> List[str]:
        return [ext for ext, fmt in cls.config_ext_formats().items() if target_fmt is None or fmt == target_fmt]


class Config(ABC):
    def __init__(self, conf: Any, fmt: ConfigFormat, file_path: Optional[str] = None):
        self.format = fmt
        self.conf = conf
        self.file_path = file_path

    def get_format(self) -> ConfigFormat:
        """Returns the current config objects ConfigFormat.

        Returns:
            return ConfigFormat
        """
        return self.format

    def get_exts(self) -> List[str]:
        return ConfigFormat.extensions(self.format)

    def get_native_conf(self):
        """Returns the underlying JSON-compatible Python value.

        Returns:
            A native config object
        """

        return self.conf

    def get_location(self) -> Optional[str]:
        """Returns the file path where this configuration is loaded from.

        Returns:
            None if the config is not from file; else return file path

        """
        return self.file_path

    @abstractmethod
    def to_dict(self, resolve: Optional[bool] = True) -> Dict:
        """Returns configuration data as a dictionary.

        Args:
            resolve: retained for compatibility; JSON values remain literal.

        Returns:
            A converted configuration as dict

        """

    @abstractmethod
    def to_str(self, element: Optional[Dict] = None) -> str:
        """Serialize an element or the full configuration as JSON.

        Args:
            element: Optional[Dict]. default to None. dictionary representation of config

        Returns:
            string representation of the configuration in given format for the element or config

        """


class ConfigLoader(ABC):
    def __init__(self, fmt: ConfigFormat):
        self.format = fmt

    def get_format(self) -> ConfigFormat:
        """Returns the current ConfigLoader's ConfigFormat.

        Returns:
            A ConfigFormat

        """
        return self.format

    @abstractmethod
    def load_config(self, file_path: str) -> Config:
        """Load configuration from config file.

        Args:
            file_path (str): file path for configuration to be loaded

        Returns:
            A Config

        """

    def load_config_from_str(self, config_str: str) -> Config:
        """Load Configuration based on the string representation of the underline configuration.

           For example, Json String for Jsonformat. python conf string or yaml string presentation

        Args:
            config_str (str): string for configuration to be loaded

        Returns:
            A Config

        """
        raise NotImplementedError

    def load_config_from_dict(self, config_dict: dict) -> Config:
        """Load Configuration based on a given config dict.

        Args:
            config_dict (dict): dict for configuration to be loaded

        Returns:
            A Config

        """
        raise NotImplementedError
