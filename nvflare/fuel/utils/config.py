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
from typing import Dict, Optional


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


class Config(ABC):
    @abstractmethod
    def get_native_conf(self):
        """
        Return the original underline config object representation if you prefer to use it directly
        Pyhocon → ConfigTree
        JSON → Dict
        OMEGACONF → ConfigDict

        Returns: Any, native config objects
        """

        pass

    @abstractmethod
    def get_format(self):
        """
        Returns: ConfigFormat, returns the current config objects ConfigFormat
        """

        pass

    @abstractmethod
    def get_location(self) -> Optional[str]:
        """
        Returns: None if the config is not from file else return the file path where this configuration is loaded from

        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        """
        convert underline config object to dictionary
        Returns:Dict

        """
        pass

    @abstractmethod
    def to_conf_str(self, element: Dict) -> str:
        """
            convert dict element to the str representation of the underline configuration.
            For example, for JsonFormat, the method return json string
            for PyhoconFormat, the method return pyhocon string
            for OmegaconfFormat, the method returns YAML string representation
        Args:
            element: dict

        Returns: str string representation of the configuration in given format

        """
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
            overwrite_config: dict config that will  overwrite the final config if provided

        Returns:nvflare.fuel.utils.Config
        """

        pass

    def load_config_from_str(self, config_str: str) -> Config:
        """
        Load Configuration based on the string representation of the underline configuration
        for example, Json String for Jsonformat. python conf string or yaml string presentation
        Args:
            config_str:
        Returns:nvflare.fuel.utils.Config

        """
        raise NotImplementedError

    def load_config_from_dict(self, config_dict: dict) -> Config:
        """
        Load Configuration based for given config dict.

        Args:
            config_dict:
        Returns:nvflare.fuel.utils.Config

        """
        raise NotImplementedError
