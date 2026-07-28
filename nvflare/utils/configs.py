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
from nvflare.fuel.utils.secret_utils import resolve_secret_refs


def get_job_config_value(
    fl_ctx: FLContext,
    config_file: str,
    key: str,
    default: Any = None,
    *,
    resolve_refs: bool = True,
) -> Any:
    """Generic function to read from any job config file.

    By default, secret references in the selected value are resolved recursively at this runtime
    boundary. This helper reads the exported JSON directly, so ordinary NVFlare placeholders
    such as ``{SITE_NAME}`` are not expanded here and remain literal in the returned value, even
    when they appear alongside secret references.

    Args:
        fl_ctx: FLContext
        config_file: Name of the config file (e.g., JobConstants.CLIENT_JOB_CONFIG)
        key: The configuration key to read
        default: Default value if key is not found or reading fails. Defaults to None.
        resolve_refs: Whether to resolve secret references in the selected value. Defaults to
            True. Internal consumers that may serialize the value again must set this to False
            and reject references rather than persisting resolved secret material.

    Returns:
        The configuration value if found, otherwise the default value.

    Raises:
        ValueError: If ``resolve_refs`` is True and the selected value contains a malformed
            reference, a referenced environment variable is unset, or a referenced file cannot
            be read.
    """
    value_found = False
    value = default
    try:
        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        config_dir = workspace.get_app_config_dir(fl_ctx.get_job_id())
        config_file_path = os.path.join(config_dir, config_file)

        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as f:
                config_data = json.load(f)
                value = config_data.get(key, default)
                value_found = key in config_data
    except Exception:
        # Silently return default on any error
        return default

    # When requested, resolve outside the broad file-reading exception handler so an unavailable
    # secret reference fails explicitly instead of silently becoming the caller's default.
    if not value_found:
        return default
    return resolve_secret_refs(value) if resolve_refs else value


def get_client_config_value(fl_ctx: FLContext, key: str, default: Any = None, *, resolve_refs: bool = True) -> Any:
    """Read a value from config_fed_client.json.

    This utility function reads top-level configuration values from the client config JSON file.
    Jobs can set these values using recipe.add_client_config({"key": value}). By default, secret
    references in the selected value are resolved recursively when it is read; dictionary keys
    are unchanged.

    Args:
        fl_ctx: FLContext
        key: The configuration key to read
        default: Default value if key is not found or reading fails. Defaults to None.
        resolve_refs: Whether to resolve secret references in the selected value. Defaults to
            True. Consumers that may serialize the returned value must disable resolution and
            reject references.

    Returns:
        The configuration value if found, otherwise the default value.

    Raises:
        ValueError: If ``resolve_refs`` is True and the selected value contains a malformed
            reference, a referenced environment variable is unset, or a referenced file cannot
            be read.

    Example:
        ```python
        from nvflare.utils.configs import get_client_config_value
        from nvflare.client.constants import EXTERNAL_PRE_INIT_TIMEOUT

        # In your executor's initialize method:
        timeout = get_client_config_value(fl_ctx, EXTERNAL_PRE_INIT_TIMEOUT, default=300.0)
        ```
    """
    return get_job_config_value(
        fl_ctx,
        JobConstants.CLIENT_JOB_CONFIG,
        key,
        default,
        resolve_refs=resolve_refs,
    )


def get_server_config_value(fl_ctx: FLContext, key: str, default: Any = None, *, resolve_refs: bool = True) -> Any:
    """Read a value from config_fed_server.json.

    This utility function reads top-level configuration values from the server config JSON file.
    Jobs can set these values using recipe.add_server_config({"key": value}). By default, secret
    references in the selected value are resolved recursively when it is read; dictionary keys
    are unchanged.

    Args:
        fl_ctx: FLContext
        key: The configuration key to read
        default: Default value if key is not found or reading fails. Defaults to None.
        resolve_refs: Whether to resolve secret references in the selected value. Defaults to
            True. Consumers that may serialize the returned value must disable resolution and
            reject references.

    Returns:
        The configuration value if found, otherwise the default value.

    Raises:
        ValueError: If ``resolve_refs`` is True and the selected value contains a malformed
            reference, a referenced environment variable is unset, or a referenced file cannot
            be read.

    Example:
        ```python
        from nvflare.utils.configs import get_server_config_value

        # In your controller's initialize method:
        custom_param = get_server_config_value(fl_ctx, "custom_param", default="default_value")
        ```
    """
    return get_job_config_value(
        fl_ctx,
        JobConstants.SERVER_JOB_CONFIG,
        key,
        default,
        resolve_refs=resolve_refs,
    )
