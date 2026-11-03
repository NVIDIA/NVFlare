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

"""Locating and loading FLARE JSON config files.

A config is named by its basename (``config_fed_client.json`` -> ``config_fed_client``) and may exist as
``<basename>.json`` or the lower-priority fallback ``<basename>.json.default``.
"""

import json
import os
import pathlib
from typing import List, Optional

from nvflare.fuel.utils.log_utils import get_module_logger
from nvflare.security.logging import secure_format_exception

CONFIG_EXTENSIONS = (".json", ".json.default")

logger = get_module_logger(__name__)


def find_config_file(file_path: str, search_dirs: Optional[List[str]] = None) -> Optional[str]:
    """Finds the config file for file_path's basename, ignoring file_path's extension.

    Each search dir is walked recursively, trying each extension of CONFIG_EXTENSIONS in priority order;
    the first match wins. With no search dirs, file_path's parent dir is searched.

    Args:
        file_path: config file name or path, e.g. "config_fed_client.json"
        search_dirs: dirs to search; defaults to the parent dir of file_path

    Returns:
        path of the config file found, or None
    """
    if not search_dirs:
        search_dirs = [str(pathlib.Path(file_path).parent)]

    # cut at the first dot so "x.json.default" and "x.json" both map to "x"
    basename = os.path.basename(file_path).split(".", 1)[0]
    for search_dir in search_dirs:
        for ext in CONFIG_EXTENSIONS:
            filename = f"{basename}{ext}"
            for root, _dirs, files in os.walk(search_dir):
                if filename in files:
                    return os.path.join(root, filename)
    return None


def load_config(file_path: str, search_dirs: Optional[List[str]] = None) -> Optional[dict]:
    """Finds the config file (see find_config_file) and loads it; returns None if not found."""
    path = find_config_file(file_path, search_dirs)
    if path is None:
        return None
    with open(path, "r") as f:
        try:
            return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file {path}: {secure_format_exception(e)}")
            raise
