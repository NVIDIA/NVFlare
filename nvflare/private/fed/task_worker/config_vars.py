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

"""Late environment substitution for task worker constructor arguments."""

import json
import os
import re

from nvflare.fuel.common.excepts import ConfigError

ENV_WORKER_CONFIG_VARS = "NVFLARE_TASK_CONFIG_ENV_NAMES"
# Keep code selection, routing, and shared codecs resolved for CJ authorization.
CJ_EXECUTOR_ARGS = frozenset(
    {
        "command",
        "task_script_path",
        "execution_mode",
        "result_wait_timeout",
        "generator_id",
        "params_exchange_format",
        "server_expected_format",
    }
)
_ENV_REF = re.compile(r"\{\$([^{}]+)\}")


def worker_config_env_names():
    """The launcher sends variable names to both pods, with values only on the worker."""
    raw_names = os.environ.get(ENV_WORKER_CONFIG_VARS)
    if not raw_names:
        return set()
    try:
        names = json.loads(raw_names)
    except ValueError:
        raise ConfigError(f"{ENV_WORKER_CONFIG_VARS} must be a JSON array of environment names") from None
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ConfigError(f"{ENV_WORKER_CONFIG_VARS} must be a JSON array of environment names")
    return set(names)


def resolve_worker_config_env(args, *, exclude=()):
    """Substitute worker values once, after the component has been authorized."""
    names = worker_config_env_names()
    if not names:
        return args

    def substitute(match):
        name = match[1]
        if name not in names:
            return match[0]
        value = os.environ.get(name)
        if value is None:
            raise ConfigError(f"task worker environment variable {name} is not set")
        return value

    def expand(value):
        if isinstance(value, str):
            return _ENV_REF.sub(substitute, value)
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return {key: value if key in exclude else expand(value) for key, value in args.items()}
