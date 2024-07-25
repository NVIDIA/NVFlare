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
import os
import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree, HOCONConverter

from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel_opt.utils.pyhocon_loader import PyhoconConfig
from nvflare.tool.job.job_client_const import CONFIG_CONF, JOB_TEMPLATES


def get_home_dir() -> Path:
    return Path.home()


def get_package_root() -> Path:
    """
    Get the nvflare package root folder, e.g.
        /usr/local/python/3.10/lib/python3.10/site-packages/nvflare
    """
    return pathlib.Path(__file__).parent.parent.absolute().resolve()


def get_hidden_nvflare_config_path(hidden_nvflare_dir: str) -> str:
    """
    Get the path for the hidden nvflare configuration file.
    Args:
        hidden_nvflare_dir: ~/.nvflare directory
    Returns:
        str: The path to the hidden nvflare configuration file.
    """
    hidden_nvflare_config_file = os.path.join(hidden_nvflare_dir, CONFIG_CONF)
    return str(hidden_nvflare_config_file)


def get_or_create_hidden_nvflare_dir():
    hidden_nvflare_dir = get_hidden_nvflare_dir()
    if not hidden_nvflare_dir.exists():
        try:
            hidden_nvflare_dir.mkdir(exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Error creating the hidden nvflare directory: {e}")

    return hidden_nvflare_dir


def get_hidden_nvflare_dir() -> pathlib.Path:
    home_dir = get_home_dir()
    hidden_nvflare_dir = pathlib.Path(home_dir) / ".nvflare"
    return hidden_nvflare_dir


def load_config(config_file_path) -> Optional[ConfigTree]:
    if os.path.isfile(config_file_path):
        return CF.parse_file(config_file_path)
    else:
        return None


def find_startup_kit_location() -> str:
    nvflare_config = load_hidden_config()
    return nvflare_config.get_string("startup_kit.path", None) if nvflare_config else None


def load_hidden_config() -> ConfigTree:
    hidden_dir = get_or_create_hidden_nvflare_dir()
    hidden_nvflare_config_file = get_hidden_nvflare_config_path(str(hidden_dir))
    nvflare_config = load_config(hidden_nvflare_config_file)
    return nvflare_config


def create_startup_kit_config(nvflare_config: ConfigTree, startup_kit_dir: Optional[str] = None) -> ConfigTree:
    """
    Args:
        startup_kit_dir: specified startup kit location
        nvflare_config (ConfigTree): The existing nvflare configuration.

    Returns:
        ConfigTree: The merged configuration tree.
    """
    old_startup_kit_dir = nvflare_config.get_string("startup_kit", None)
    if old_startup_kit_dir is None and (startup_kit_dir is not None and not os.path.isdir(startup_kit_dir)):
        raise ValueError(f"invalid startup kit location '{startup_kit_dir}'")
    if startup_kit_dir:
        startup_kit_dir = get_startup_kit_dir(startup_kit_dir)
        conf_str = f"""
            startup_kit {{
                path = "{startup_kit_dir}"
            }}
        """
        conf: ConfigTree = CF.parse_string(conf_str)

        return conf.with_fallback(nvflare_config)
    else:
        return nvflare_config


def create_poc_workspace_config(nvflare_config: ConfigTree, poc_workspace_dir: Optional[str] = None) -> ConfigTree:
    """
    Args:
        poc_workspace_dir: specified poc_workspace_dir
        nvflare_config (ConfigTree): The existing nvflare configuration.

    Returns:
        ConfigTree: The merged configuration tree.
    """
    if poc_workspace_dir is None:
        return nvflare_config

    poc_workspace_dir = os.path.abspath(poc_workspace_dir)

    conf_str = f"""
        poc_workspace {{
            path = {poc_workspace_dir}
        }}
    """
    conf: ConfigTree = CF.parse_string(conf_str)

    return conf.with_fallback(nvflare_config)


def create_job_template_config(nvflare_config: ConfigTree, job_templates_dir: Optional[str] = None) -> ConfigTree:
    """
    Args:
        job_templates_dir: specified job template directory
        nvflare_config (ConfigTree): The existing nvflare configuration.

    Returns:
        ConfigTree: The merged configuration tree.
    """
    if job_templates_dir is None:
        return nvflare_config

    job_templates_dir = os.path.abspath(job_templates_dir)
    check_dir(job_templates_dir)
    conf_str = f"""
        job_template {{
            path = {job_templates_dir}
        }}
    """
    conf: ConfigTree = CF.parse_string(conf_str)

    return conf.with_fallback(nvflare_config)


def check_dir(dir_path: str):
    if not dir_path or not os.path.isdir(dir_path):
        raise ValueError(f"directory {dir_path} doesn't exists")


def get_startup_kit_dir(startup_kit_dir: Optional[str] = None) -> str:
    if not startup_kit_dir:
        # load from config file:
        startup_kit_dir = find_startup_kit_location()
        if startup_kit_dir is None:
            startup_kit_dir = os.getenv("NVFLARE_STARTUP_KIT_DIR")

        if startup_kit_dir is None or len(startup_kit_dir.strip()) == 0:
            raise ValueError("startup kit directory is not specified")

    check_startup_dir(startup_kit_dir)
    startup_kit_dir = os.path.abspath(startup_kit_dir)
    return startup_kit_dir


def check_startup_dir(startup_kit_dir):
    if not startup_kit_dir or not os.path.isdir(startup_kit_dir):
        raise ValueError(
            f"startup_kit_dir '{startup_kit_dir}' must be a valid and non-empty path. "
            f"use 'nvflare poc' command to 'prepare' if you are using POC mode. Or use"
            f" 'nvflare config' to setup startup_kit_dir location if you are in production"
        )


def find_job_templates_location(job_templates_dir: Optional[str] = None):
    def check_job_templates_dir(job_temp_dir: str):
        if job_temp_dir:
            if not os.path.isdir(job_temp_dir):
                raise ValueError(f"Invalid job template directory {job_temp_dir}")

    if job_templates_dir is None:
        nvflare_home = os.environ.get("NVFLARE_HOME", None)
        if nvflare_home:
            job_templates_dir = os.path.join(nvflare_home, JOB_TEMPLATES)

    if job_templates_dir is None:
        nvflare_config = load_hidden_config()
        job_templates_dir = nvflare_config.get_string("job_template.path", None) if nvflare_config else None

    if job_templates_dir:
        check_job_templates_dir(job_templates_dir)

    if not job_templates_dir:
        # get default template from nvflare wheel if package installed
        from nvflare.tool import job as job_module

        template_path = os.path.join(os.path.dirname(job_module.__file__), "templates")
        if os.path.isdir(template_path):
            job_templates_dir = template_path
        else:
            job_templates_dir = None

    if not job_templates_dir:
        raise ValueError(
            "Required job_template directory is not specified. "
            "Please check ~/.nvflare/config.conf or set env variable NVFLARE_HOME "
        )

    return job_templates_dir


def get_curr_dir():
    return os.path.curdir


def is_dir_empty(path: str):
    targe_dir = os.listdir(path)
    return len(targe_dir) == 0


def hocon_to_string(target_fmt: ConfigFormat, dst_config: ConfigTree):
    if target_fmt == ConfigFormat.JSON:
        return HOCONConverter.to_json(dst_config)
    elif target_fmt == ConfigFormat.PYHOCON:
        return HOCONConverter.to_hocon(config=dst_config, level=1)
    elif target_fmt == ConfigFormat.OMEGACONF:
        from nvflare.fuel_opt.utils.omegaconf_loader import OmegaConfLoader

        loader = OmegaConfLoader()
        dst_dict_config = PyhoconConfig(dst_config).to_dict()
        omega_conf = loader.load_config_from_dict(dst_dict_config)
        return omega_conf.to_str()


def save_configs(app_configs: Dict[str, Tuple], keep_origin_format: bool = True):
    for app_name, (dst_config, dst_path) in app_configs.items():
        save_config(dst_config, dst_path, keep_origin_format)


def save_config(dst_config: ConfigTree, dst_path, keep_origin_format: bool = True):
    if dst_path is None or dst_path.rindex(".") == -1:
        raise ValueError(f"configuration file path '{dst_path}' can't be None or has no extension")

    require_clean_up = False
    if keep_origin_format:
        original_ext = os.path.basename(dst_path).split(".")[1]
        fmt = ConfigFormat.config_ext_formats().get(f".{original_ext}", None)
        if fmt is None:
            raise ValueError(f"invalid file extension {dst_path}, no corresponding configuration format")
        dst_config_path = dst_path

    else:
        fmt = ConfigFormat.PYHOCON
        ext = ConfigFormat.extensions(fmt)[0]
        if dst_path.endswith(ext):
            dst_config_path = dst_path
        else:
            filename = f"{os.path.basename(dst_path).split('.')[0]}{ext}"
            dst_config_path = os.path.join(os.path.dirname(dst_path), filename)
            require_clean_up = True

    config_str = hocon_to_string(fmt, dst_config)
    with open(dst_config_path, "w") as outfile:
        outfile.write(f"{config_str}\n")

    if require_clean_up:
        if os.path.exists(dst_path):
            os.remove(dst_path)


def get_hidden_config() -> (str, ConfigTree):
    hidden_nvflare_config_file = get_hidden_nvflare_config_path(str(get_or_create_hidden_nvflare_dir()))
    conf = load_hidden_config()
    nvflare_config = CF.parse_string("{}") if not conf else conf
    return hidden_nvflare_config_file, nvflare_config


def print_hidden_config(dst_path: str, dst_config: ConfigTree):
    original_ext = os.path.basename(dst_path).split(".")[1]
    fmt = ConfigFormat.config_ext_formats().get(f".{original_ext}", None)
    config_str = hocon_to_string(fmt, dst_config)
    print(config_str)


def find_in_list(arr: List, item) -> bool:
    if arr is None:
        return False

    found = False
    for a in arr:
        if a.__eq__(item):
            return True

    return found


def append_if_not_in_list(arr: List, item) -> List:
    if item is None:
        return arr

    if arr is None:
        arr = []

    if not find_in_list(arr, item):
        arr.append(item)

    return arr
