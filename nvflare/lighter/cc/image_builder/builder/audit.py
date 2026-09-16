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

"""Allowlisted public appraisal metadata; no errors, tokens or application data."""

import datetime
import os
import stat
from pathlib import Path

from .common import canonical, read_json


def record(config, identity, decision):
    return {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cvm_build_id": config["build_id"],
        "vault_id": identity.get("luks_uuid"),
        "measurement": identity.get("measurements"),
        "policy_id": config["attestation_policy_id"],
        "decision": decision,
    }


def append(path, line):
    # Clear, workload-writable audit storage is untrusted. Never follow a
    # substituted symlink or write a special file. Its loss or a full
    # filesystem must not bypass shutdown.
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if stat.S_ISREG(os.fstat(fd).st_mode):
            os.write(fd, line)
    finally:
        os.close(fd)


def emit(decision):
    try:
        config = read_json("/etc/cvm/runtime.json")
        if Path("/etc/cvm/dev_mode").exists():
            return
        identity = Path("/run/cvm/binding.json")
        value = record(config, read_json(identity) if identity.exists() else {}, decision)
        line = canonical(value) + b"\n"
        print("CVM_ATTESTATION=" + line.decode().strip(), flush=True)
        append("/applog/attestation.log", line)
    except Exception:
        # Audit storage is not an authority and can be unavailable during a
        # failed bootstrap. Never log exception objects or delay failure logic.
        pass
