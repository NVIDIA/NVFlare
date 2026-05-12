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
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree, HOCONConverter

from nvflare.fuel.utils.config import ConfigFormat
from nvflare.tool.job.job_client_const import CONFIG_CONF, JOB_TEMPLATES

CONFIG_VERSION = "version"
CURRENT_CONFIG_VERSION = 2


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


def load_config(config_file_path) -> ConfigTree | None:
    if os.path.isfile(config_file_path):
        return CF.parse_file(config_file_path)
    else:
        return None


def find_startup_kit_config_keys(nvflare_config: ConfigTree) -> list[str]:
    """Return old startup-kit config keys that should no longer be persisted."""
    if not nvflare_config:
        return []

    keys = []
    for key in ("startup_kit.path", "startup_kit", "poc.startup_kit", "prod.startup_kit"):
        try:
            if nvflare_config.get(key, None) is not None:
                keys.append(key)
        except Exception:
            pass
    return keys


def remove_startup_kit_config_keys(nvflare_config: ConfigTree) -> ConfigTree:
    """Remove obsolete startup kit config keys without touching the startup_kits registry."""
    if not nvflare_config:
        return nvflare_config

    for key in ("startup_kit.path", "startup_kit", "poc.startup_kit", "prod.startup_kit"):
        try:
            nvflare_config.pop(key, None)
        except Exception:
            pass
    return nvflare_config


def load_hidden_config_state() -> tuple[str, ConfigTree | None, bool]:
    hidden_dir = get_or_create_hidden_nvflare_dir()
    hidden_nvflare_config_file = get_hidden_nvflare_config_path(str(hidden_dir))
    nvflare_config = load_config(hidden_nvflare_config_file)
    return hidden_nvflare_config_file, nvflare_config, False


def backup_hidden_config_file(hidden_nvflare_config_file: str) -> str | None:
    if not os.path.exists(hidden_nvflare_config_file):
        return None

    backup_base = f"{hidden_nvflare_config_file}.bak"
    backup_path = backup_base
    suffix = 1
    while os.path.exists(backup_path):
        backup_path = f"{backup_base}{suffix}"
        suffix += 1

    shutil.copy2(hidden_nvflare_config_file, backup_path)
    return backup_path


def load_hidden_config() -> ConfigTree:
    _, nvflare_config, _ = load_hidden_config_state()
    return nvflare_config


def create_poc_workspace_config(nvflare_config: ConfigTree, poc_workspace_dir: str | None = None) -> ConfigTree:
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
        {CONFIG_VERSION} = {CURRENT_CONFIG_VERSION}
        poc {{
            workspace = "{poc_workspace_dir}"
        }}
    """
    conf: ConfigTree = CF.parse_string(conf_str)

    return conf.with_fallback(nvflare_config)


def create_job_template_config(nvflare_config: ConfigTree, job_templates_dir: str | None = None) -> ConfigTree:
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


def find_job_templates_location(job_templates_dir: str | None = None):
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


def save_configs(app_configs: dict[str, tuple], keep_origin_format: bool = True):
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
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(dst_config_path), delete=False) as outfile:
            temp_path = outfile.name
            outfile.write(f"{config_str}\n")
            outfile.flush()
            os.fsync(outfile.fileno())
        os.replace(temp_path, dst_config_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    if require_clean_up:
        if os.path.exists(dst_path):
            os.remove(dst_path)


def get_hidden_config() -> (str, ConfigTree):
    hidden_nvflare_config_file, conf, _ = load_hidden_config_state()
    nvflare_config = CF.parse_string("{}") if not conf else conf
    return hidden_nvflare_config_file, nvflare_config


def print_hidden_config(dst_path: str, dst_config: ConfigTree):
    original_ext = os.path.basename(dst_path).split(".")[1]
    fmt = ConfigFormat.config_ext_formats().get(f".{original_ext}", None)
    config_str = hocon_to_string(fmt, dst_config)
    print(config_str)


def find_in_list(arr: list, item) -> bool:
    if arr is None:
        return False

    return any(a == item for a in arr)


def append_if_not_in_list(arr: list, item) -> list:
    if item is None:
        return arr

    if arr is None:
        arr = []

    if not find_in_list(arr, item):
        arr.append(item)

    return arr
