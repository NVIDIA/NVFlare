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
"""Typed connection profiles for Cell-based Client API execution.

External-process profiles are owner-written launch bootstraps. Attach profiles are
pre-provisioned independently of a job. Key strings and schema versions are persisted
protocol and must remain stable.
"""

import json
import os
import tempfile

BOOTSTRAP_FILE_ENV_VAR = "NVFLARE_CLIENT_API_BOOTSTRAP"

CELL_API_TYPE = "CELL_API"

# Bootstrap schema version is independent of the post-connection Cell protocol version.
BOOTSTRAP_SCHEMA_VERSION = 1

# Supported typed Cell Client API profile modes.
EXTERNAL_PROCESS_EXECUTION_MODE = "external_process"
ATTACH_EXECUTION_MODE = "attach"

# Limit exposure of the launch token to the file owner.
BOOTSTRAP_FILE_PERMISSION = 0o600


def bootstrap_file_name(seq: int) -> str:
    """Return a launch-scoped bootstrap name so stale processes retain stale credentials."""
    return f"client_api_bootstrap_{seq}.json"


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
    SECURE_MODE = "secure_mode"
    CONNECTION_SECURITY = "connection_security"
    CA_CERT = "ca_cert"

    ATTACH_ID = "attach_id"
    RENDEZVOUS_DIR = "rendezvous_dir"
    JOB_WAIT_TIMEOUT = "job_wait_timeout"

    # Legacy TASK_EXCHANGE shape needed before the first task arrives.
    TASK_EXCHANGE = "task_exchange"

    MEMORY_GC_ROUNDS = "memory_gc_rounds"
    CUDA_EMPTY_CACHE = "cuda_empty_cache"


_EXTERNAL_REQUIRED_STRING_FIELDS = (
    BootstrapKey.CJ_FQCN,
    BootstrapKey.TRAINER_FQCN,
    BootstrapKey.JOB_ID,
    BootstrapKey.SITE_NAME,
    BootstrapKey.CONNECT_URL,
    BootstrapKey.LAUNCH_TOKEN,
)

_ATTACH_REQUIRED_STRING_FIELDS = (
    BootstrapKey.ATTACH_ID,
    BootstrapKey.SITE_NAME,
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
    if execution_mode not in (EXTERNAL_PROCESS_EXECUTION_MODE, ATTACH_EXECUTION_MODE):
        raise ValueError(
            f"unsupported Client API bootstrap execution_mode {execution_mode!r} in {path}; "
            f"supported modes are {EXTERNAL_PROCESS_EXECUTION_MODE!r} and {ATTACH_EXECUTION_MODE!r}"
        )

    required_fields = (
        _EXTERNAL_REQUIRED_STRING_FIELDS
        if execution_mode == EXTERNAL_PROCESS_EXECUTION_MODE
        else _ATTACH_REQUIRED_STRING_FIELDS
    )
    for field in required_fields:
        if field not in config:
            raise ValueError(f"invalid Client API bootstrap config {path}: missing required field {field!r}")
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"invalid Client API bootstrap config {path}: field {field!r} must be a non-empty string")
    if BootstrapKey.SECURE_MODE in config and type(config[BootstrapKey.SECURE_MODE]) is not bool:
        raise ValueError(
            f"invalid Client API bootstrap config {path}: field {BootstrapKey.SECURE_MODE!r} must be a bool"
        )
    if execution_mode == ATTACH_EXECUTION_MODE:
        from nvflare.apis.fl_constant import ConnectionSecurity
        from nvflare.client.cell.attach import validate_attach_id, validate_attach_profile

        validate_attach_id(config[BootstrapKey.ATTACH_ID])
        connect_url = config.get(BootstrapKey.CONNECT_URL)
        rendezvous_dir = config.get(BootstrapKey.RENDEZVOUS_DIR)
        if bool(connect_url) == bool(rendezvous_dir):
            raise ValueError(
                f"invalid Client API bootstrap config {path}: attach requires exactly one of "
                f"{BootstrapKey.CONNECT_URL!r} or {BootstrapKey.RENDEZVOUS_DIR!r}"
            )
        if rendezvous_dir:
            if not isinstance(rendezvous_dir, str) or not os.path.isabs(rendezvous_dir):
                raise ValueError(
                    f"invalid Client API bootstrap config {path}: field "
                    f"{BootstrapKey.RENDEZVOUS_DIR!r} must be an absolute path"
                )
            if config.get(BootstrapKey.CONNECTION_SECURITY) not in (None, ConnectionSecurity.CLEAR):
                raise ValueError(
                    f"invalid Client API bootstrap config {path}: shared-file rendezvous supports only "
                    f"{BootstrapKey.CONNECTION_SECURITY!r}={ConnectionSecurity.CLEAR!r}"
                )
            if BootstrapKey.CJ_FQCN in config:
                raise ValueError(
                    f"invalid Client API bootstrap config {path}: shared-file rendezvous discovers "
                    f"{BootstrapKey.CJ_FQCN!r}; do not configure it"
                )
            connection_security = ConnectionSecurity.CLEAR
        else:
            if not isinstance(connect_url, str) or not connect_url.strip():
                raise ValueError(
                    f"invalid Client API bootstrap config {path}: field "
                    f"{BootstrapKey.CONNECT_URL!r} must be a non-empty string"
                )
            cj_fqcn = config.get(BootstrapKey.CJ_FQCN)
            if not isinstance(cj_fqcn, str) or not cj_fqcn:
                raise ValueError(
                    f"invalid Client API bootstrap config {path}: a direct attach profile requires "
                    f"field {BootstrapKey.CJ_FQCN!r}"
                )
            from nvflare.fuel.f3.cellnet.fqcn import FQCN

            cj_path = FQCN.split(cj_fqcn)
            if len(cj_path) != 2 or cj_path[0] != config[BootstrapKey.SITE_NAME] or FQCN.validate(cj_fqcn):
                raise ValueError(
                    f"invalid Client API bootstrap config {path}: field "
                    f"{BootstrapKey.CJ_FQCN!r} must be '<site_name>.<job_id>'"
                )
            connection_security = validate_attach_profile(
                connect_url,
                config.get(BootstrapKey.CONNECTION_SECURITY),
            )
        ca_cert = config.get(BootstrapKey.CA_CERT)
        if ca_cert is not None and (not isinstance(ca_cert, str) or not ca_cert.strip()):
            raise ValueError(
                f"invalid Client API bootstrap config {path}: field "
                f"{BootstrapKey.CA_CERT!r} must be a non-empty string"
            )
        if rendezvous_dir and ca_cert:
            raise ValueError(
                f"invalid Client API bootstrap config {path}: shared-file rendezvous does not use "
                f"{BootstrapKey.CA_CERT!r}"
            )
        if connection_security != ConnectionSecurity.CLEAR and not ca_cert:
            raise ValueError(
                f"invalid Client API bootstrap config {path}: "
                f"{connection_security!r} attach requires field {BootstrapKey.CA_CERT!r}"
            )
        if BootstrapKey.SECURE_MODE in config:
            expected_secure = connection_security != ConnectionSecurity.CLEAR
            if config[BootstrapKey.SECURE_MODE] is not expected_secure:
                raise ValueError(
                    f"invalid Client API bootstrap config {path}: "
                    f"{BootstrapKey.SECURE_MODE!r} disagrees with "
                    f"{BootstrapKey.CONNECTION_SECURITY!r}"
                )
    if BootstrapKey.JOB_WAIT_TIMEOUT in config:
        value = config[BootstrapKey.JOB_WAIT_TIMEOUT]
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0):
            raise ValueError(
                f"invalid Client API bootstrap config {path}: field "
                f"{BootstrapKey.JOB_WAIT_TIMEOUT!r} must be a number >= 0 or None"
            )
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
