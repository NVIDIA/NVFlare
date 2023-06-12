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
import os
from typing import Dict, List, Union

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
            process_start_cmd_args: command args for starting the program
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
            cls._sections[section] = cls.load_json(file_basename)

        cls._var_dict = var_dict
        if parsed_args:
            if not isinstance(parsed_args, argparse.Namespace):
                raise ValueError(f"parsed_args must be argparse.Namespace but got {type(parsed_args)}")
            cls._cmd_args = dict(parsed_args.__dict__)

    @classmethod
    def get_section(cls, name: str):
        return cls._sections.get(name)

    @classmethod
    def add_section(cls, section_name: str, data: dict, overwrite_existing: bool = True):
        """
        Add a section to the config data.

        Args:
            section_name: name of the section to be added
            data: data of the section
            overwrite_existing: if section already exists, whether to overwrite

        Returns:

        """
        if not isinstance(section_name, str):
            raise TypeError(f"section name must be str but got {type(section_name)}")
        if not isinstance(data, dict):
            raise TypeError(f"config data must be dict but got {type(data)}")

        if overwrite_existing or section_name not in cls._sections:
            cls._sections[section_name] = data

    @classmethod
    def load_json(cls, file_basename: str) -> dict:
        """
        Load a specified JSON config file

        Args:
            file_basename: base name of the config file to be loaded

        Returns:

        """
        file_path = cls.find_file(file_basename)
        if not file_path:
            raise FileNotFoundError(f"cannot find file '{file_basename}' from search path '{cls._config_path}'")
        return json.load(open(file_path, "rt"))

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
    def _get_var(cls, name: str, conf):
        if not isinstance(name, str):
            raise ValueError(f"var name must be str but got {type(name)}")

        # see whether command args have it
        if cls._cmd_args and name in cls._cmd_args:
            return cls._cmd_args.get(name)
        if cls._var_dict and name in cls._var_dict:
            return cls._var_dict.get(name)

        # check OS env vars
        if not name.startswith(ENV_VAR_PREFIX):
            env_var_name = ENV_VAR_PREFIX + name
        else:
            env_var_name = name

        env_var_name = env_var_name.upper()
        if env_var_name in os.environ:
            return os.environ.get(env_var_name)

        if isinstance(conf, dict):
            return conf.get(name)

    @classmethod
    def _int_var(cls, name: str, conf=None, default=None):
        v = cls._get_var(name, conf)
        if v is None:
            return default
        try:
            return int(v)
        except:
            raise ValueError(f"var {name}'s value '{v}' cannot be converted to int")

    @classmethod
    def _any_var(cls, func, name, conf, default):
        if name in cls._var_values:
            return cls._var_values.get(name)
        v = func(name, conf, default)
        if v is not None:
            cls._var_values[name] = v
        return v

    @classmethod
    def get_int_var(cls, name: str, conf=None, default=None):
        return cls._any_var(cls._int_var, name, conf, default)

    @classmethod
    def _float_var(cls, name: str, conf=None, default=None):
        v = cls._get_var(name, conf)
        if v is None:
            return default
        try:
            return float(v)
        except:
            raise ValueError(f"var {name}'s value '{v}' cannot be converted to float")

    @classmethod
    def get_float_var(cls, name: str, conf=None, default=None):
        return cls._any_var(cls._float_var, name, conf, default)

    @classmethod
    def _bool_var(cls, name: str, conf=None, default=None):
        v = cls._get_var(name, conf)
        if v is None:
            return default
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
        return cls._any_var(cls._bool_var, name, conf, default)

    @classmethod
    def _str_var(cls, name: str, conf=None, default=None):
        v = cls._get_var(name, conf)
        if v is None:
            return default
        try:
            return str(v)
        except:
            raise ValueError(f"var {name}'s value '{v}' cannot be converted to str")

    @classmethod
    def get_str_var(cls, name: str, conf=None, default=None):
        return cls._any_var(cls._str_var, name, conf, default)

    @classmethod
    def get_var_values(cls):
        return cls._var_values
