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
import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Union

from nvflare.fuel.utils.config import Config, ConfigFormat
from nvflare.fuel.utils.config_factory import ConfigFactory

ENV_VAR_PREFIX = "NVFLARE_"


def find_file_in_dir(file_basename, path) -> Union[None, str]:
    """
    Find a file from a directory and return the full path of the file, if found

    Args:
        file_basename: base name of the file to be found
        path: the directory from where the file is to be found

    Returns: the full path of the file, if found; None if not found
    """
    for root, dirs, files in os.walk(path):
        if file_basename in files:
            return os.path.join(root, file_basename)
    return None


def search_file(file_basename: str, dirs: List[str]) -> Union[None, str]:
    """
    Find a file by searching a list of dirs and return the one in the last dir.

    Args:
        file_basename: base name of the file to be found
        dirs: list of directories to search

    Returns: the full path of the file, if found; None if not found

    """
    if isinstance(dirs, str):
        dirs = [dirs]
    for d in dirs:
        f = find_file_in_dir(file_basename, d)
        if f:
            return f
    return None


class ConfigService:
    """
    The ConfigService provides a global configuration service that can be used by any component at any layer.
    The ConfigService manages config information and makes it available to any component, in two ways:
    1. Config info is preloaded into predefined sections. Callers can get the config data by a section name.
    2. Manages config path (a list of directories) and loads file from the path.

    Only JSON file loading is supported.
    """

    logger = logging.getLogger(__name__)
    _sections = {}
    _config_path = []
    _cmd_args = None
    _var_dict = None
    _var_values = {}

    @classmethod
    def initialize(cls, section_files: Dict[str, str], config_path: List[str], parsed_args=None, var_dict=None):
        """
        Initialize the ConfigService.
        Configuration is divided into sections, and each section must have a JSON config file.
        Only specify the base name of the config file.
        Config path is provided to locate config files. Files are searched in the order of provided
        config_dirs. If multiple directories contain the same file name, then the first one is used.

        Args:
            section_files: dict: section name => config file
            config_path: list of config directories
            parsed_args: command args for starting the program
            var_dict: dict for additional vars

        Returns:

        """
        if not isinstance(section_files, dict):
            raise TypeError(f"section_files must be dict but got {type(section_files)}")

        if not isinstance(config_path, list):
            raise TypeError(f"config_dirs must be list but got {type(config_path)}")

        if not config_path:
            raise ValueError("config_dirs is empty")

        if var_dict and not isinstance(var_dict, dict):
            raise ValueError(f"var_dict must dict but got {type(var_dict)}")

        for d in config_path:
            if not isinstance(d, str):
                raise ValueError(f"config_dirs must contain str but got {type(d)}")

            if not os.path.exists(d):
                raise ValueError(f"'directory {d}' does not exist")

            if not os.path.isdir(d):
                raise ValueError(f"'{d}' is not a valid directory")

        cls._config_path = config_path

        for section, file_basename in section_files.items():
            cls._sections[section] = cls.load_config_dict(file_basename, cls._config_path)

        cls._var_dict = var_dict
        if parsed_args:
            if not isinstance(parsed_args, argparse.Namespace):
                raise ValueError(f"parsed_args must be argparse.Namespace but got {type(parsed_args)}")
            cls._cmd_args = dict(parsed_args.__dict__)

    @classmethod
    def reset(cls):
        """Reset the ConfigServer to its initial state. All registered sections and cached var values
        are cleared.  This method is mainly used for test purpose.

        Returns:

        """
        cls._sections = {}
        cls._config_path = []
        cls._cmd_args = None
        cls._var_dict = None
        cls._var_values = {}

    @classmethod
    def get_section(cls, name: str):
        """Get the specified section.

        Args:
            name: name of the section

        Returns: the section of the specified name, or None if the section is not found.

        """
        return cls._sections.get(name)

    @classmethod
    def add_section(cls, section_name: str, data: dict, overwrite_existing: bool = True):
        """
        Add a section to the config data.

        Args:
            section_name: name of the section to be added
            data: data of the section
            overwrite_existing: if section already exists, whether to overwrite

        Returns: None

        """
        if not isinstance(section_name, str):
            raise TypeError(f"section name must be str but got {type(section_name)}")
        if not isinstance(data, dict):
            raise TypeError(f"config data must be dict but got {type(data)}")

        if overwrite_existing or section_name not in cls._sections:
            cls._sections[section_name] = data

    @classmethod
    def load_configuration(cls, file_basename: str) -> Optional[Config]:
        """Load config data from the specified file basename.
        The full name of the config file will be determined by ConfigFactory.

        Args:
            file_basename: the basename of the config file.

        Returns: config data loaded, or None if the config file is not found.

        """
        return ConfigFactory.load_config(file_basename, cls._config_path)

    @classmethod
    def load_config_dict(
        cls, file_basename: str, search_dirs: Optional[List] = None, raise_exception: bool = True
    ) -> Optional[Dict]:
        """
        Load a specified config file ( ignore extension)

        Args:
            raise_exception: if True raise exception when error occurs
            file_basename: base name of the config file to be loaded.
            for example: file_basename = config_fed_server.json
            what the function does is to search for config file that matches
            config_fed_server.[json|json.default|conf|conf.default|yml|yml.default]
            in given search directories: cls._config_path
            if json or json.default is not found;
            then switch to Pyhoncon [.conf] or corresponding default file; if still not found; then we switch
            to YAML files. We use OmegaConf to load YAML
            search_dirs: which directories to search.

        Returns: Dictionary from the configuration
                if not found, exception will be raised.
        """
        conf = ConfigFactory.load_config(file_basename, search_dirs)
        if conf:
            return conf.to_dict()
        else:
            if raise_exception:
                raise FileNotFoundError(cls.config_not_found_msg(file_basename, search_dirs))
            return None

    @classmethod
    def config_not_found_msg(cls, file_basename, search_dirs):
        basename = os.path.splitext(file_basename)[0]
        conf_exts = "|".join(ConfigFormat.config_ext_formats().keys())
        msg = f"cannot find file '{basename}[{conf_exts}]'"
        msg = f"{msg} from search paths: '{search_dirs}'" if search_dirs else msg
        return msg

    @classmethod
    def find_file(cls, file_basename: str) -> Union[None, str]:
        """
        Find specified file from the config path.
        Caller is responsible for loading/processing the file. This is useful for non-JSON files.

        Args:
            file_basename: base name of the file to be found

        Returns: full name of the file if found; None if not.

        """
        if not isinstance(file_basename, str):
            raise TypeError(f"file_basename must be str but got {type(file_basename)}")
        return search_file(file_basename, cls._config_path)

    @classmethod
    def _get_from_config(cls, func, name: str, conf, default):
        v, src = cls._get_var_from_source(name, conf)
        cls.logger.debug(f"got var {name} from {src}")
        if v is None:
            return default

        # convert to right data type
        return func(name, v)

    @classmethod
    def _any_var(cls, func, name, conf, default):
        if name in cls._var_values:
            return cls._var_values.get(name)
        v = cls._get_from_config(func, name, conf, default)
        if v is not None:
            cls._var_values[name] = v
        return v

    @staticmethod
    def _get_var_from_os_env(name: str):
        if not name.startswith(ENV_VAR_PREFIX):
            env_var_name = ENV_VAR_PREFIX + name
        else:
            env_var_name = name

        env_var_name = env_var_name.upper()
        if env_var_name in os.environ:
            return os.environ.get(env_var_name)
        else:
            return None

    @classmethod
    def _get_var_from_config_sources(cls, name: str, conf):
        if conf is None:
            return None

        # conf could be:
        #   a single config source (a section name or a dict)
        #   a list of config sources
        if not isinstance(conf, list):
            conf = [conf]

        # check each conf source until the var is found
        for src in conf:
            if isinstance(src, str):
                # this is a section name
                src = cls.get_section(src)

            if isinstance(src, dict):
                v = src.get(name)
                if v is not None:
                    return v

        # No source has this var
        return None

    @classmethod
    def _get_var_from_source(cls, name: str, conf):
        if not isinstance(name, str):
            raise ValueError(f"var name must be str but got {type(name)}")

        # see whether command args have it
        if cls._cmd_args and name in cls._cmd_args:
            return cls._cmd_args.get(name), "cmd_args"

        if cls._var_dict and name in cls._var_dict:
            return cls._var_dict.get(name), "var_dict"

        value = cls._get_var_from_config_sources(name, conf)
        if value is not None:
            return value, "config"

        # finally check os env
        return cls._get_var_from_os_env(name), "env"

    @classmethod
    def _to_int(cls, name: str, v):
        try:
            return int(v)
        except Exception as e:
            raise ValueError(f"var {name}'s value '{v}' cannot be converted to int: {e}")

    @classmethod
    def get_int_var(cls, name: str, conf=None, default=None):
        """Get configured int value of the specified var

        Args:
            name: name of the var
            conf: source config
            default: value to return if the var is not found

        Returns: configured value of the var, or the default value if var is not configured

        """
        return cls._any_var(cls._to_int, name, conf, default)

    @classmethod
    def _to_float(cls, name: str, v):
        try:
            return float(v)
        except Exception as e:
            raise ValueError(f"var {name}'s value '{v}' cannot be converted to float: {e}")

    @classmethod
    def get_float_var(cls, name: str, conf=None, default=None):
        """Get configured float value of the specified var

        Args:
            name: name of the var
            conf: source config
            default: value to return if the var is not found

        Returns: configured value of the var, or the default value if var is not configured

        """
        return cls._any_var(cls._to_float, name, conf, default)

    @classmethod
    def _to_bool(cls, name: str, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return v != 0
        if isinstance(v, str):
            v = v.lower()
            return v in ["true", "t", "yes", "y", "1"]
        raise ValueError(f"var {name}'s value '{v}' cannot be converted to bool")

    @classmethod
    def get_bool_var(cls, name: str, conf=None, default=None):
        """Get configured bool value of the specified var

        Args:
            name: name of the var
            conf: source config
            default: value to return if the var is not found

        Returns: configured value of the var, or the default value if var is not configured

        """
        return cls._any_var(cls._to_bool, name, conf, default)

    @classmethod
    def _to_str(cls, name: str, v):
        try:
            return str(v)
        except Exception as e:
            raise ValueError(f"var {name}'s value '{v}' cannot be converted to str: {e}")

    @classmethod
    def get_str_var(cls, name: str, conf=None, default=None):
        """Get configured str value of the specified var

        Args:
            name: name of the var
            conf: source config
            default: value to return if the var is not found

        Returns: configured value of the var, or the default value if var is not configured

        """
        return cls._any_var(cls._to_str, name, conf, default)

    @classmethod
    def _to_dict(cls, name: str, v):
        if isinstance(v, dict):
            return v

        if isinstance(v, str):
            # assume it's a json str
            try:
                v2 = json.loads(v)
            except Exception as e:
                raise ValueError(f"var {name}'s value '{v}' cannot be converted to dict: {e}")

            if not isinstance(v2, dict):
                raise ValueError(f"var {name}'s value '{v}' does not represent a dict")
            return v2
        else:
            raise ValueError(f"var {name}'s value '{v}' does not represent a dict")

    @classmethod
    def get_dict_var(cls, name: str, conf=None, default=None):
        """Get configured dict value of the specified var

        Args:
            name: name of the var
            conf: source config
            default: value to return if the var is not found

        Returns: configured value of the var, or the default value if var is not configured

        """
        return cls._any_var(cls._to_dict, name, conf, default)

    @classmethod
    def get_var_values(cls):
        """Get cached var values.

        Returns:

        """
        return cls._var_values
