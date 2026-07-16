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
"""Bootstrap config contract for external_process trainers.

Design: docs/design/client_api_execution_modes.md ("external_process", "Session setup").
The external_process backend writes this file (mode 0600) before launching the trainer
command and points the trainer at it through ``BOOTSTRAP_FILE_ENV_VAR``. The trainer-side
Cell engine reads it in ``flare.init()`` to build its child cell and perform the HELLO
handshake. Like nvflare/client/cell/defs.py, the key values here are cross-track wire/file
contract shared with the trainer-side engine: renaming a value is a protocol break.

The launch token is launch-scoped, not job-scoped: the backend regenerates it on every
launch and invalidates it when it stops the process, so a stale trainer that survived
termination cannot authenticate against a later launch. On the V1 trusted single-tenant
host the token is a rendezvous/binding id, not a secret defended against co-tenants --
which is why the file mode (0600) and a plain token match (no challenge-response) are
sufficient; see the design doc's "Session setup" and Appendix B for the attach-mode
contrast.
"""

import json
import os
import tempfile

# Environment variable the backend sets on the launched trainer process; its value is the
# absolute path of the bootstrap config file.
BOOTSTRAP_FILE_ENV_VAR = "NVFLARE_CLIENT_API_BOOTSTRAP"

# Client API type selected by this self-identifying bootstrap. Mirrored by
# ClientAPIType.CELL_API in nvflare/client/api_context.py.
CELL_API_TYPE = "CELL_API"

# Version of the typed bootstrap-file envelope. This controls how flare.init() identifies
# and parses the persisted rendezvous file; the trainer's compiled protocol constant controls
# messages exchanged after the Cell session is established.
BOOTSTRAP_SCHEMA_VERSION = 1

# Supported execution mode for this first typed bootstrap contract. Attach will use the
# same Cell Client API engine later, but has a different authentication/session contract
# and must not be accepted accidentally before that mode is implemented.
EXTERNAL_PROCESS_EXECUTION_MODE = "external_process"

# Bootstrap files are readable/writable by the owner only: the launch token must not be
# readable by other local users (same rationale as nvflare/client/config.py).
BOOTSTRAP_FILE_PERMISSION = 0o600


class BootstrapKey:
    """Keys of the bootstrap config file. Frozen file contract shared with the trainer engine."""

    # Self-identifying envelope. These fields let ``flare.init(config_file=...)`` distinguish
    # this file from the legacy ExProcess Client API config without relying on inherited env.
    SCHEMA_VERSION = "schema_version"
    EXECUTION_MODE = "execution_mode"

    # Cell construction: the URL the trainer's child cell connects to (the CJ cell's
    # internal listener) and the exact FQCN the child cell must bind. The backend
    # prescribes a fresh FQCN leaf per launch so a stale process's cell can never
    # collide with the current launch's cell name.
    CONNECT_URL = "connect_url"
    CJ_FQCN = "cj_fqcn"
    TRAINER_FQCN = "trainer_fqcn"

    # HELLO handshake material (see nvflare/client/cell/defs.py Topic.HELLO).
    LAUNCH_TOKEN = "launch_token"

    # Job identity, echoed in HELLO and used by the trainer-side Client API meta.
    JOB_ID = "job_id"
    SITE_NAME = "site_name"

    # The task-exchange config (train/eval/submit-model task names, train_with_evaluation,
    # exchange format / transfer type) the trainer-side Client API needs before the first
    # task arrives; same shape as ConfigKey.TASK_EXCHANGE in the in_process task meta.
    TASK_EXCHANGE = "task_exchange"

    # Memory-management knobs of the frozen executor surface, applied by the trainer-side
    # Client API (the ex-process analog of InProcessClientAPI.configure_memory_management).
    MEMORY_GC_ROUNDS = "memory_gc_rounds"
    CUDA_EMPTY_CACHE = "cuda_empty_cache"


def get_bootstrap_client_api_type(config: dict, path: str = "<bootstrap config>") -> str | None:
    """Returns the Client API type selected by a typed bootstrap config.

    An untyped dict is deliberately returned as ``None`` so legacy ExProcess configs retain
    their existing environment-based selection. If either typed-envelope marker is present,
    however, both markers must be valid: silently treating a malformed or future bootstrap as
    a legacy config could start the wrong Client API engine with misleading downstream errors.

    Args:
        config: parsed JSON config dict.
        path: source path used in validation errors.

    Returns:
        ``CELL_API_TYPE`` for the supported Cell bootstrap, or ``None`` for an untyped config.

    Raises:
        ValueError: if the typed envelope is incomplete or unsupported.
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
    return CELL_API_TYPE


def write_bootstrap_config(path: str, config: dict) -> None:
    """Writes the bootstrap config file with owner-only permission (0600).

    The content is first written to an owner-only sibling temporary file and then
    atomically replaces the destination. This keeps a valid existing bootstrap intact
    when serialization fails and replaces, rather than follows, a planted destination
    symlink.

    Args:
        path: absolute path of the bootstrap config file.
        config: the bootstrap config dict (BootstrapKey keys).
    """
    target_path = os.path.abspath(path)
    config_dir = os.path.dirname(target_path)
    fd, tmp_path = tempfile.mkstemp(dir=config_dir, prefix=".client_api_bootstrap-", suffix=".tmp")
    fd_owned = True
    try:
        if hasattr(os, "fchmod"):
            # mkstemp already creates 0600 on POSIX; set it explicitly as part of the
            # persisted credential contract.
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
    """Reads the bootstrap config file written by the external_process backend.

    Args:
        path: absolute path of the bootstrap config file.

    Returns:
        The bootstrap config dict.

    Raises:
        ValueError: if the file content is not a JSON dict.
    """
    with open(path, "r") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"invalid bootstrap config {path}: expect a JSON dict but got {type(config)}")
    return config
