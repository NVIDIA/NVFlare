# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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


SITE_CONFIG_FORMAT_VERSION = 1
SITE_CONFIG_MAX_SIZE = 64 * 1024
SITE_CONFIG_ALLOWED_KEYS = {"format_version", "resources", "labels", "capabilities"}


def validate_site_config(site_config, max_size=SITE_CONFIG_MAX_SIZE):
    """Validate and normalize client-supplied site config.

    Returns:
        A tuple of (sanitized_config, error). If the config is invalid, sanitized_config is None and error is set.
    """
    if site_config is None:
        return None, None

    if not isinstance(site_config, dict):
        return None, f"site config must be a dict but got {type(site_config)}"

    site_config = dict(site_config)
    unknown_keys = set(site_config.keys()) - SITE_CONFIG_ALLOWED_KEYS
    if unknown_keys:
        return None, f"site config contains unsupported top-level keys: {sorted(str(k) for k in unknown_keys)}"

    format_version = site_config.get("format_version", SITE_CONFIG_FORMAT_VERSION)
    if not isinstance(format_version, int) or isinstance(format_version, bool):
        return None, f"site config format_version must be int but got {type(format_version)}"
    if format_version != SITE_CONFIG_FORMAT_VERSION:
        return None, f"unsupported site config format_version: {format_version}"
    site_config["format_version"] = format_version

    resources = site_config.get("resources")
    if resources is not None and not isinstance(resources, dict):
        return None, f"site config resources must be a dict but got {type(resources)}"

    labels = site_config.get("labels")
    if labels is not None and not isinstance(labels, dict):
        return None, f"site config labels must be a dict but got {type(labels)}"

    capabilities = site_config.get("capabilities")
    if capabilities is not None:
        if not isinstance(capabilities, list):
            return None, f"site config capabilities must be a list but got {type(capabilities)}"
        if not all(isinstance(c, str) for c in capabilities):
            return None, "site config capabilities must contain only strings"

    try:
        normalized = json.dumps(site_config, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as ex:
        return None, f"site config must be JSON-serializable: {ex}"

    encoded_size = len(normalized.encode("utf-8"))
    if encoded_size > max_size:
        return None, f"site config size {encoded_size} exceeds max size {max_size}"

    return json.loads(normalized), None


def load_site_config_from_file(file_path, max_size=SITE_CONFIG_MAX_SIZE):
    """Load and validate a site_config.json file.

    Returns:
        A tuple of (sanitized_config, error). Missing files are not errors.
    """
    if not file_path or not os.path.exists(file_path):
        return None, None

    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        return None, f"site config file size {file_size} exceeds max size {max_size}"

    try:
        with open(file_path, "rt", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as ex:
        return None, f"could not load site config file {file_path}: {ex}"

    return validate_site_config(config, max_size=max_size)


def load_site_config_from_workspace(workspace, max_size=SITE_CONFIG_MAX_SIZE):
    if not workspace:
        return None, None

    try:
        file_path = workspace.get_site_config_file_path()
    except Exception as ex:
        return None, f"could not determine site config file path: {ex}"

    return load_site_config_from_file(file_path, max_size=max_size)
