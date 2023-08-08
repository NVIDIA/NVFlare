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
    PYHOCON = "PYHOCON"
    OMEGACONF = "OMEGACONF"

    @classmethod
    def config_ext_formats(cls):
        return OrderedDict(
            {
                ".json": ConfigFormat.JSON,
                ".conf": ConfigFormat.PYHOCON,
                ".yml": ConfigFormat.OMEGACONF,
                ".json.default": ConfigFormat.JSON,
                ".conf.default": ConfigFormat.PYHOCON,
                ".yml.default": ConfigFormat.OMEGACONF,
            }
        )

    @classmethod
    def extensions(cls, target_fmt=None) -> List[str]:
        if target_fmt is None:
            return [ext for ext, fmt in cls.config_ext_formats().items()]
        else:
            return [ext for ext, fmt in cls.config_ext_formats().items() if fmt == target_fmt]


class Config(ABC):
    def __init__(self, conf: Any, fmt: ConfigFormat, file_path: Optional[str] = None):
        self.format = fmt
        self.conf = conf
        self.file_path = file_path

    def get_format(self) -> ConfigFormat:
        """returns the current config objects ConfigFormat
        Returns:
            return ConfigFormat
        """
        return self.format

    def get_exts(self) -> List[str]:
        return ConfigFormat.extensions(self.format)

    def get_native_conf(self):
        """Return the original underline config object representation if you prefer to use it directly
           Pyhocon → ConfigTree
           JSON → Dict
           OMEGACONF → ConfigDict

        Returns: Any,
            return native config objects
        """

        return self.conf

    def get_location(self) -> Optional[str]:
        """return the file path where this configuration is loaded from

        Returns:
            return None if the config is not from file else return file path

        """
        return self.file_path

    @abstractmethod
    def to_dict(self, resolve: Optional[bool] = True) -> Dict:
        """convert underline config object to dictionary
        Args:
            resolve: optional argument to indicate if the variable need to be resolved when convert to dictionary
                     not all underline configuration format support this.
                     If not supported, it is treated default valueTrue.

        Returns:
            Returns: converted configuration as dict

        """

    @abstractmethod
    def to_str(self, element: Optional[Dict] = None) -> str:
        """convert dict element to the str representation of the underline configuration, if element is not None
           For example, for JsonFormat, the method return json string
           for PyhoconFormat, the method return pyhocon string
           for OmegaconfFormat, the method returns YAML string representation

           If the element is None, return the underline config to string presentation
        Args:
            element: Optional[Dict]. default to None. dictionary representation of config
        Returns:
            string representation of the configuration in given format for the element or config

        """


class ConfigLoader(ABC):
    def __init__(self, fmt: ConfigFormat):
        self.format = fmt

    def get_format(self) -> ConfigFormat:
        """returns the current configLoader's ConfigFormat
        Returns:
            return ConfigFormat
        """
        return self.format

    @abstractmethod
    def load_config(self, file_path: str) -> Config:
        """load configuration from config file: file_path
        Args:
            file_path: file path for configuration to be loaded
        Returns:
            return Config
        """

    def load_config_from_str(self, config_str: str) -> Config:
        """Load Configuration based on the string representation of the underline configuration
            for example, Json String for Jsonformat. python conf string or yaml string presentation

        Args:
            config_str:
        Returns:
            return Config

        """
        raise NotImplementedError

    def load_config_from_dict(self, config_dict: dict) -> Config:
        """Load Configuration based for given config dict.

        Args:
            config_dict:
        Returns:
            Config

        """
        raise NotImplementedError
