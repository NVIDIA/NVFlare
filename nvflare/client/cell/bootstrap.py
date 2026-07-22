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
"""Owner-only bootstrap contract between ExternalProcessBackend and its trainer.

The launch-scoped token and trainer FQCN reject stale processes. Key strings and schema
versions are persisted protocol and must remain stable.
"""

import json
import os
import tempfile

BOOTSTRAP_FILE_ENV_VAR = "NVFLARE_CLIENT_API_BOOTSTRAP"

CELL_API_TYPE = "CELL_API"

# Bootstrap schema version is independent of the post-connection Cell protocol version.
BOOTSTRAP_SCHEMA_VERSION = 1

# Typed schema currently accepts launched external-process mode only.
EXTERNAL_PROCESS_EXECUTION_MODE = "external_process"

# Limit exposure of the launch token to the file owner.
BOOTSTRAP_FILE_PERMISSION = 0o600


class BootstrapKey:
    """Stable keys shared by the backend and trainer."""

    # Self-identifying fields distinguish this from legacy configs without environment hints.
    SCHEMA_VERSION = "schema_version"
    EXECUTION_MODE = "execution_mode"

    # Launch-scoped FQCNs prevent stale trainers from colliding with a later launch.
    CONNECT_URL = "connect_url"
    CJ_FQCN = "cj_fqcn"
    TRAINER_FQCN = "trainer_fqcn"

    LAUNCH_TOKEN = "launch_token"

    JOB_ID = "job_id"
    SITE_NAME = "site_name"

    # Legacy TASK_EXCHANGE shape needed before the first task arrives.
    TASK_EXCHANGE = "task_exchange"

    MEMORY_GC_ROUNDS = "memory_gc_rounds"
    CUDA_EMPTY_CACHE = "cuda_empty_cache"


_REQUIRED_STRING_FIELDS = (
    BootstrapKey.CJ_FQCN,
    BootstrapKey.TRAINER_FQCN,
    BootstrapKey.JOB_ID,
    BootstrapKey.SITE_NAME,
    BootstrapKey.CONNECT_URL,
    BootstrapKey.LAUNCH_TOKEN,
)


def get_bootstrap_client_api_type(config: dict, path: str = "<bootstrap config>") -> str | None:
    """Return ``CELL_API_TYPE`` for a typed config or ``None`` for a legacy config.

    If either envelope marker exists, require both and reject unsupported values rather
    than silently selecting the legacy engine.
    """
    has_schema = BootstrapKey.SCHEMA_VERSION in config
    has_execution_mode = BootstrapKey.EXECUTION_MODE in config
    if not has_schema and not has_execution_mode:
        return None
    if not has_schema or not has_execution_mode:
        missing = BootstrapKey.SCHEMA_VERSION if not has_schema else BootstrapKey.EXECUTION_MODE
        raise ValueError(f"invalid Client API bootstrap config {path}: missing required field {missing!r}")

    schema_version = config[BootstrapKey.SCHEMA_VERSION]
    if type(schema_version) is not int or schema_version != BOOTSTRAP_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Client API bootstrap schema_version {schema_version!r} in {path}; "
            f"supported version is {BOOTSTRAP_SCHEMA_VERSION}"
        )

    execution_mode = config[BootstrapKey.EXECUTION_MODE]
    if execution_mode != EXTERNAL_PROCESS_EXECUTION_MODE:
        raise ValueError(
            f"unsupported Client API bootstrap execution_mode {execution_mode!r} in {path}; "
            f"supported mode is {EXTERNAL_PROCESS_EXECUTION_MODE!r}"
        )

    for field in _REQUIRED_STRING_FIELDS:
        if field not in config:
            raise ValueError(f"invalid Client API bootstrap config {path}: missing required field {field!r}")
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"invalid Client API bootstrap config {path}: field {field!r} must be a non-empty string")
    return CELL_API_TYPE


def write_bootstrap_config(path: str, config: dict) -> None:
    """Atomically write an owner-only bootstrap file.

    A sibling temporary preserves an existing file on failure and avoids following a
    planted destination symlink.
    """
    target_path = os.path.abspath(path)
    config_dir = os.path.dirname(target_path)
    fd, tmp_path = tempfile.mkstemp(dir=config_dir, prefix=".client_api_bootstrap-", suffix=".tmp")
    fd_owned = True
    try:
        if hasattr(os, "fchmod"):
            # mkstemp is already 0600 on POSIX; enforce the contract explicitly.
            os.fchmod(fd, BOOTSTRAP_FILE_PERMISSION)
        with os.fdopen(fd, "w") as f:
            fd_owned = False
            json.dump(config, f, indent=2)
        os.replace(tmp_path, target_path)
    except BaseException:
        if fd_owned:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise


def read_bootstrap_config(path: str) -> dict:
    """Read a JSON-object bootstrap config."""
    with open(path, "r") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"invalid bootstrap config {path}: expect a JSON dict but got {type(config)}")
    return config
