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

"""Utilities for reading configuration files."""

import json
import os
from typing import Any

from nvflare.apis.fl_constant import JobConstants
from nvflare.apis.fl_context import FLContext


def get_job_config_value(fl_ctx: FLContext, config_file: str, key: str, default: Any = None) -> Any:
    """Generic function to read from any job config file.

    Args:
        fl_ctx: FLContext
        config_file: Name of the config file (e.g., JobConstants.CLIENT_JOB_CONFIG)
        key: The configuration key to read
        default: Default value if key is not found or reading fails. Defaults to None.

    Returns:
        The configuration value if found, otherwise the default value.
    """
    try:
        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        config_dir = workspace.get_app_config_dir(fl_ctx.get_job_id())
        config_file_path = os.path.join(config_dir, config_file)

        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as f:
                config_data = json.load(f)
                return config_data.get(key, default)
    except Exception:
        # Silently return default on any error
        pass

    return default


def get_client_config_value(fl_ctx: FLContext, key: str, default: Any = None) -> Any:
    """Read a value from config_fed_client.json.

    This utility function reads top-level configuration values from the client config JSON file.
    Jobs can set these values using recipe.add_client_config({"key": value}).

    Args:
        fl_ctx: FLContext
        key: The configuration key to read
        default: Default value if key is not found or reading fails. Defaults to None.

    Returns:
        The configuration value if found, otherwise the default value.

    Example:
        ```python
        from nvflare.utils.configs import get_client_config_value
        from nvflare.client.constants import EXTERNAL_PRE_INIT_TIMEOUT

        # In your executor's initialize method:
        timeout = get_client_config_value(fl_ctx, EXTERNAL_PRE_INIT_TIMEOUT, default=300.0)
        ```
    """
    return get_job_config_value(fl_ctx, JobConstants.CLIENT_JOB_CONFIG, key, default)


def get_server_config_value(fl_ctx: FLContext, key: str, default: Any = None) -> Any:
    """Read a value from config_fed_server.json.

    This utility function reads top-level configuration values from the server config JSON file.
    Jobs can set these values using recipe.add_server_config({"key": value}).

    Args:
        fl_ctx: FLContext
        key: The configuration key to read
        default: Default value if key is not found or reading fails. Defaults to None.

    Returns:
        The configuration value if found, otherwise the default value.

    Example:
        ```python
        from nvflare.utils.configs import get_server_config_value

        # In your controller's initialize method:
        custom_param = get_server_config_value(fl_ctx, "custom_param", default="default_value")
        ```
    """
    return get_job_config_value(fl_ctx, JobConstants.SERVER_JOB_CONFIG, key, default)
