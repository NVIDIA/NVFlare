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
import os
import pathlib
from typing import List, Optional, Tuple

from nvflare.fuel.utils.config import Config, ConfigFormat, ConfigLoader
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel.utils.json_config_loader import JsonConfigLoader
from nvflare.fuel.utils.log_utils import get_module_logger


def uses_config_features(text: str, config_format: ConfigFormat) -> bool:
    """Whether a HOCON or OmegaConf config uses features beyond plain data, or YAML aliases."""
    if config_format not in (ConfigFormat.PYHOCON, ConfigFormat.OMEGACONF):
        return False
    try:
        if config_format == ConfigFormat.PYHOCON:
            json.loads(text)
            return False
        import yaml

        # Aliases let a few bytes expand into an exponentially large config. Scalars are checked
        # after YAML unescaping, so "\u0024{" cannot hide an interpolation.
        return any(
            isinstance(e, yaml.AliasEvent) or (isinstance(e, yaml.ScalarEvent) and "${" in e.value)
            for e in yaml.parse(text, Loader=yaml.SafeLoader)
        )
    except Exception:
        return True


class ConfigFactory:
    logger = get_module_logger(__module__, __qualname__)

    # HOCON includes and substitutions, and OmegaConf interpolations, read files, URLs, and environment
    # variables. Only configs under these site folders may use them; others, such as job configs, must be plain data.
    _trusted_dirs: List[str] = []

    OmegaConfLoader, omega_import_ok = optional_import(
        module="nvflare.fuel_opt.utils.omegaconf_loader", name="OmegaConfLoader"
    )
    PyhoconLoader, pyhocon_import_ok = optional_import(
        module="nvflare.fuel_opt.utils.pyhocon_loader", name="PyhoconLoader"
    )

    _fmt2Loader = {
        ConfigFormat.JSON: JsonConfigLoader(),
    }

    if omega_import_ok:
        _fmt2Loader.update({ConfigFormat.OMEGACONF: OmegaConfLoader()})

    if pyhocon_import_ok:
        _fmt2Loader.update({ConfigFormat.PYHOCON: PyhoconLoader()})

    @staticmethod
    def trust_config_dirs(*dirs: str):
        ConfigFactory._trusted_dirs.extend(os.path.realpath(d) for d in dirs)

    @staticmethod
    def is_trusted_config(file_path: str) -> bool:
        real_path = os.path.realpath(file_path)
        return any(real_path.startswith(d + os.sep) for d in ConfigFactory._trusted_dirs)

    @staticmethod
    def check_plain_data(text: str, config_format: ConfigFormat, source: str):
        if uses_config_features(text, config_format):
            raise ValueError(
                f"{source} uses HOCON or OmegaConf features, which only site configs may use; "
                "submit the job with FLARE tools to render it"
            )

    @staticmethod
    def search_config_format(
        init_file_path: str, search_dirs: Optional[List[str]] = None, target_fmt: Optional[ConfigFormat] = None
    ) -> Tuple[Optional[ConfigFormat], Optional[str]]:
        """Finds the configuration format and the location (file_path) for given initial init_file_path and search directories.

        For example, the initial config file path given is `config_client.json`
        the search function will ignore the .json extension and search "config_client.xxx" in the given directory in
        specified extension search order. The first found file_path will be used as configuration.
        the ".xxx" is one of the extensions defined in the configuration format.

        Args:
            init_file_path: initial file_path for the configuration
            search_dirs: search directory. If none, the parent directory of init_file_path will be used as search dir
            target_fmt: (ConfigFormat) if specified, only this format searched, ignore all other formats.

        Returns:
            Tuple of None,None or ConfigFormat and real configuration file path

        """
        logger = ConfigFactory.logger
        if not search_dirs:  # empty or None
            parent_dir = pathlib.Path(init_file_path).parent
            search_dirs = [str(parent_dir)]

        target_exts = None
        if target_fmt:
            target_exts = ConfigFormat.extensions(target_fmt)

        # we ignore the original extension
        file_basename = ConfigFactory.get_file_basename(init_file_path)
        ext2fmt_map = ConfigFormat.config_ext_formats()
        extensions = target_exts if target_fmt else ext2fmt_map.keys()
        for search_dir in search_dirs:
            logger.debug(f"search file basename:'{file_basename}', search dir = {search_dir}")
            for ext in extensions:
                fmt = ext2fmt_map[ext]
                filename = f"{file_basename}{ext}"
                for root, dirs, files in os.walk(search_dir):
                    if filename in files:
                        config_file = os.path.join(root, filename)
                        return fmt, config_file

        return None, None

    @staticmethod
    def get_file_basename(init_file_path):
        base_path = os.path.basename(init_file_path)
        # split keeps the first-dot rule that "config.json.default" relies on and
        # returns the name unchanged when there is no dot, where find() gave -1
        # and the slice dropped the last character.
        file_basename = base_path.split(".", 1)[0]
        return file_basename

    @staticmethod
    def load_config(
        file_path: str, search_dirs: Optional[List[str]] = None, target_fmt: Optional[ConfigFormat] = None
    ) -> Optional[Config]:
        """Finds the configuration for given initial init_file_path and search directories.

        For example, the initial config file path given is `config_client.json`
        the search function will ignore the .json extension and search "config_client.xxx" in the given directory in
        specified extension search order. The first found file_path will be used as configuration.
        the ".xxx" is one of the extensions defined in the configuration format.

        Args:
            file_path: initial file path
            search_dirs: search directory. If none, the parent directory of init_file_path will be used as search dir
            target_fmt: (ConfigFormat) if specified, only this format searched, ignore all other formats.

        Returns:
            None if not found, or Config

        """
        config_format, real_config_file_path = ConfigFactory.search_config_format(file_path, search_dirs, target_fmt)
        if config_format is not None and real_config_file_path is not None:
            config_loader = ConfigFactory.get_config_loader(config_format)
            if config_loader:
                if config_format != ConfigFormat.JSON and not ConfigFactory.is_trusted_config(real_config_file_path):
                    with open(real_config_file_path, encoding="utf-8", errors="replace") as f:
                        ConfigFactory.check_plain_data(f.read(), config_format, real_config_file_path)
                conf = config_loader.load_config(file_path=real_config_file_path)
                return conf
            else:
                return None
        return None

    @staticmethod
    def get_config_loader(config_format: ConfigFormat) -> Optional[ConfigLoader]:
        """Returns ConfigLoader for given config_format

        Args:
            config_format: ConfigFormat

        Returns:
            the matching ConfigLoader for the given format

        """
        if config_format is None:
            return None
        return ConfigFactory._fmt2Loader.get(config_format)

    @staticmethod
    def match_config(parent, init_file_path, match_fn) -> bool:
        # we ignore the original extension
        basename = os.path.splitext(pathlib.Path(init_file_path).name)[0]
        ext2fmt_map = ConfigFormat.config_ext_formats()
        for ext in ext2fmt_map:
            if match_fn(parent, f"{basename}{ext}"):
                return True
        return False

    @staticmethod
    def has_config(init_file_path: str, search_dirs: Optional[List[str]] = None) -> bool:
        fmt, real_file_path = ConfigFactory.search_config_format(init_file_path, search_dirs)
        return real_file_path is not None
