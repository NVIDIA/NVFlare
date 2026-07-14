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

# Environment variable the backend sets on the launched trainer process; its value is the
# absolute path of the bootstrap config file.
BOOTSTRAP_FILE_ENV_VAR = "NVFLARE_CLIENT_API_BOOTSTRAP"

# CLIENT_API_TYPE value the backend sets on the launched trainer so its flare.init() resolves
# to the Cell engine (CellClientAPI). Canonical here (part of the launch contract) and mirrored
# by ClientAPIType.CELL_API in nvflare/client/api_context.py.
CELL_API_TYPE = "CELL_API"

# Bootstrap files are readable/writable by the owner only: the launch token must not be
# readable by other local users (same rationale as nvflare/client/config.py).
BOOTSTRAP_FILE_PERMISSION = 0o600


class BootstrapKey:
    """Keys of the bootstrap config file. Frozen file contract shared with the trainer engine."""

    # Cell construction: the URL the trainer's child cell connects to (the CJ cell's
    # internal listener) and the exact FQCN the child cell must bind. The backend
    # prescribes a fresh FQCN leaf per launch so a stale process's cell can never
    # collide with the current launch's cell name.
    CONNECT_URL = "connect_url"
    CJ_FQCN = "cj_fqcn"
    TRAINER_FQCN = "trainer_fqcn"

    # HELLO handshake material (see nvflare/client/cell/defs.py Topic.HELLO).
    LAUNCH_TOKEN = "launch_token"
    PROTOCOL_VERSION = "protocol_version"

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


def write_bootstrap_config(path: str, config: dict) -> None:
    """Writes the bootstrap config file with owner-only permission (0600).

    The file is created (or truncated) with mode 0600 atomically at open time, so the
    launch token is never observable at a wider mode. An os.fchmod follows for the
    pre-existing-file case, where the open mode argument does not apply.

    Args:
        path: absolute path of the bootstrap config file.
        config: the bootstrap config dict (BootstrapKey keys).
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, BOOTSTRAP_FILE_PERMISSION)
    with os.fdopen(fd, "w") as f:
        if hasattr(os, "fchmod"):
            # Unix-only before Python 3.13; on Windows the POSIX permission model does not
            # apply (the O_CREAT mode above already bounds fresh-file permissions on POSIX)
            os.fchmod(fd, BOOTSTRAP_FILE_PERMISSION)
        json.dump(config, f, indent=2)


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
