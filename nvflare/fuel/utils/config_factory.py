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
from nvflare.fuel.utils.omegaconf_loader import OmegaConfLoader
from nvflare.fuel.utils.pyhocon_loader import PyhoconLoader


class ConfigFactory:
    logger = logging.getLogger(__qualname__)

    @staticmethod
    def search_config_format(
        init_file_path, search_dirs: Optional[List[str]] = None
    ) -> (Optional[ConfigFormat], Optional[str]):
        # we ignore the original extension
        logger = ConfigFactory.logger
        if not search_dirs:
            parent_dir = pathlib.Path(init_file_path).parent
            search_dirs = [str(parent_dir)]
        file_path = os.path.splitext(pathlib.Path(init_file_path).name)[0]
        for search_dir in search_dirs:
            logger.debug(f"search file:{file_path} basename, search dirs = {search_dirs}")
            for root, dirs, files in os.walk(search_dir):
                for fmt, ext in ConfigFormat.ordered_search_extensions():
                    found_one = ConfigFactory._search_config_file(file_path, fmt, ext, root, files)
                    if found_one:
                        return found_one

        return None, None

    @staticmethod
    def _search_config_file(base_filename, fmt, ext, root, files):
        file = f"{base_filename}{ext}"
        if file in files:
            config_file_path = os.path.join(root, file)
            return fmt, config_file_path

        return None, None

    @staticmethod
    def has_config(init_file_path: str, search_dirs: Optional[List[str]] = None) -> bool:
        _, real_file_path = ConfigFactory.search_config_format(init_file_path, search_dirs)
        return real_file_path is not None

    @staticmethod
    def match_config(parent, init_file_path, match_fn) -> bool:
        # we ignore the original extension
        basename = os.path.splitext(init_file_path)[0]
        for fmt, ext in ConfigFormat.ordered_search_extensions():
            ConfigFactory.logger.debug(f"search format {fmt.name} with ext {ext} in {basename}{ext}")
            if match_fn(parent, f"{basename}{ext}"):
                return True
        return False

    @staticmethod
    def load_config(file_path: str, search_dirs: Optional[List[str]] = None) -> Optional[Config]:
        config_format, real_config_file_path = ConfigFactory.search_config_format(file_path, search_dirs)
        config_loader = ConfigFactory.get_config_loader(config_format)
        if config_loader:
            conf = config_loader.load_config(real_config_file_path)
            return conf
        else:
            return None

    @staticmethod
    def get_config_loader(config_format: ConfigFormat) -> Optional[ConfigLoader]:
        if config_format is None:
            return None
        if config_format == ConfigFormat.JSON:
            return JsonConfigLoader()
        elif config_format == ConfigFormat.OMEGACONF:
            return OmegaConfLoader()
        elif config_format == ConfigFormat.PYHOCON:
            return PyhoconLoader()
        else:
            raise NotImplementedError(
                f"configuration format {config_format.name} with file ext {config_format.value} is not implemented"
            )
