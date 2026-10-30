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
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

from nvflare.fuel.utils.config import ConfigFormat
from nvflare.tool.job.job_client_const import CONFIG_JSON, JOB_TEMPLATES

CONFIG_VERSION = "version"
CURRENT_CONFIG_VERSION = 2
LEGACY_CONFIG_CONF = "config.conf"

_warned_legacy_config_paths = set()


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
    hidden_nvflare_config_file = os.path.join(hidden_nvflare_dir, CONFIG_JSON)
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


def get_config_value(config: dict, dotted_key: str, default: Any = None) -> Any:
    node = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_config_value(config: dict, dotted_key: str, value: Any) -> dict:
    *parents, leaf = dotted_key.split(".")
    node = config
    for part in parents:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[leaf] = value
    return config


def pop_config_value(config: dict, dotted_key: str, default: Any = None) -> Any:
    parent_key, _, leaf = dotted_key.rpartition(".")
    parent = get_config_value(config, parent_key) if parent_key else config
    return parent.pop(leaf, default) if isinstance(parent, dict) else default


def warn_if_legacy_hidden_config(config_file_path: str) -> None:
    legacy_path = os.path.join(os.path.dirname(config_file_path), LEGACY_CONFIG_CONF)
    if os.path.isfile(config_file_path) or not os.path.isfile(legacy_path):
        return
    if legacy_path in _warned_legacy_config_paths:
        return
    _warned_legacy_config_paths.add(legacy_path)
    print(
        f"Warning: {legacy_path} is no longer read; settings are now stored in {config_file_path}. "
        "Re-register startup kits with 'nvflare config add <id> <startup-kit-dir>' and 'nvflare config use <id>', "
        "and set the POC workspace with 'nvflare poc config --pw <dir>'.",
        file=sys.stderr,
    )


def load_config(config_file_path) -> Optional[dict]:
    if not os.path.isfile(config_file_path):
        warn_if_legacy_hidden_config(config_file_path)
        return None
    with open(config_file_path) as f:
        return json.load(f)


def load_hidden_config_state() -> Tuple[str, Optional[dict], bool]:
    hidden_dir = get_or_create_hidden_nvflare_dir()
    hidden_nvflare_config_file = get_hidden_nvflare_config_path(str(hidden_dir))
    nvflare_config = load_config(hidden_nvflare_config_file)
    return hidden_nvflare_config_file, nvflare_config, False


def load_hidden_config() -> Optional[dict]:
    _, nvflare_config, _ = load_hidden_config_state()
    return nvflare_config


def create_poc_workspace_config(nvflare_config: dict, poc_workspace_dir: Optional[str] = None) -> dict:
    if poc_workspace_dir is None:
        return nvflare_config

    nvflare_config[CONFIG_VERSION] = CURRENT_CONFIG_VERSION
    return set_config_value(nvflare_config, "poc.workspace", os.path.abspath(poc_workspace_dir))


def create_job_template_config(nvflare_config: dict, job_templates_dir: Optional[str] = None) -> dict:
    if job_templates_dir is None:
        return nvflare_config

    job_templates_dir = os.path.abspath(job_templates_dir)
    check_dir(job_templates_dir)
    return set_config_value(nvflare_config, "job_template.path", job_templates_dir)


def check_dir(dir_path: str):
    if not dir_path or not os.path.isdir(dir_path):
        raise ValueError(f"directory {dir_path} doesn't exists")


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
        job_templates_dir = get_config_value(nvflare_config, "job_template.path") if nvflare_config else None

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
            "Please check ~/.nvflare/config.json or set env variable NVFLARE_HOME "
        )

    return job_templates_dir


def get_curr_dir():
    return os.path.curdir


def is_dir_empty(path: str):
    targe_dir = os.listdir(path)
    return len(targe_dir) == 0


def save_config(dst_config: dict, dst_path: str):
    if not dst_path or not dst_path.endswith(tuple(ConfigFormat.extensions())):
        raise ValueError(f"invalid configuration file path '{dst_path}', expecting one of {ConfigFormat.extensions()}")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(dst_path), delete=False) as outfile:
            temp_path = outfile.name
            json.dump(dst_config, outfile, indent=2)
            outfile.write("\n")
            outfile.flush()
            os.fsync(outfile.fileno())
        os.replace(temp_path, dst_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def find_in_list(arr: List, item) -> bool:
    if arr is None:
        return False

    return any(a == item for a in arr)


def append_if_not_in_list(arr: List, item) -> List:
    if item is None:
        return arr

    if arr is None:
        arr = []

    if not find_in_list(arr, item):
        arr.append(item)

    return arr
