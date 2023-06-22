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

import logging
import os
import pathlib
from typing import List, Optional

from nvflare.fuel.utils.config import Config, ConfigFormat, ConfigLoader
from nvflare.fuel.utils.json_config_loader import JsonConfigLoader
from nvflare.fuel_opt.utils.omegaconf_loader import OmegaConfLoader
from nvflare.fuel_opt.utils.pyhocon_loader import PyhoconLoader


class ConfigFactory:
    logger = logging.getLogger(__qualname__)
    _fmt2Loader = {
        ConfigFormat.JSON: JsonConfigLoader,
        ConfigFormat.PYHOCON: PyhoconLoader,
        ConfigFormat.OMEGACONF: OmegaConfLoader,
    }

    @staticmethod
    def search_config_format(
        init_file_path, search_dirs: Optional[List[str]] = None
    ) -> (Optional[ConfigFormat], Optional[str]):

        """find the configuration format and the location (file_path) for given initial init_file_path and search directories.
            for example, the initial config file path given is config_client.json
            the search function will ignore the .json extension and search "config_client.xxx" in the given directory in
            specified extension search order. The first found file_path will be used as configuration.
            the ".xxx" is one of the extensions defined in the configuration format.

        Args:
            init_file_path: initial file_path for the configuration
            search_dirs: search directory. If none, the parent directory of init_file_path will be used as search dir

        Returns:
            Tuple of None,None or ConfigFormat and real configuration path

        """
        logger = ConfigFactory.logger
        if not search_dirs:
            parent_dir = pathlib.Path(init_file_path).parent
            search_dirs = [str(parent_dir)]
        # we ignore the original extension
        file_basename = os.path.splitext(pathlib.Path(init_file_path).name)[0]
        ext2fmt_map = ConfigFormat.config_ext_formats()
        for search_dir in search_dirs:
            logger.debug(f"search file:{file_basename} basename, search dirs = {search_dirs}")
            for ext in ext2fmt_map:
                fmt = ext2fmt_map[ext]
                file = os.path.join(search_dir, file_basename, ext)
                if os.path.isfile:
                    return fmt, file

        return None, None

    @staticmethod
    def load_config(file_path: str, search_dirs: Optional[List[str]] = None) -> Optional[Config]:

        """Find the configuration for given initial init_file_path and search directories.
            for example, the initial config file path given is config_client.json
            the search function will ignore the .json extension and search "config_client.xxx" in the given directory in
            specified extension search order. The first found file_path will be used as configuration.
            the ".xxx" is one of the extensions defined in the configuration format.
        Args:
            file_path: initial file path
            search_dirs: search directory. If none, the parent directory of init_file_path will be used as search dir

        Returns:
            None if not found, or Config

        """
        config_format, real_config_file_path = ConfigFactory.search_config_format(file_path, search_dirs)
        config_loader = ConfigFactory.get_config_loader(config_format)
        if config_loader:
            conf = config_loader.load_config(real_config_file_path)
            return conf
        else:
            return None

    @staticmethod
    def get_config_loader(config_format: ConfigFormat) -> Optional[ConfigLoader]:

        """return ConfigLoader for given config_format

        Args:
            config_format: ConfigFormat

        Returns:
            the matching ConfigLoader for the given format

        """
        if config_format is None:
            return None
        return ConfigFactory._fmt2Loader.get(config_format)
