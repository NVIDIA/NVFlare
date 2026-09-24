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

"""Emit guest authorization decisions without disclosing secrets."""

import datetime
import os
import queue
import stat
import threading
from pathlib import Path

from ..common.io import canonical, read_json

# One daemon writer and a bounded queue per process. Regular-file/block-device
# I/O can sleep despite O_NONBLOCK; it must never run on the supervisor thread.
# Advisory records may be dropped on congestion or process exit.
_records = queue.Queue(maxsize=8)
_writer = None


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
            remaining = memoryview(line)
            while remaining:
                # EAGAIN can leave an incomplete advisory record. Do not wait or
                # retry a full/untrusted log filesystem and delay shutdown.
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("Audit write made no progress")
                remaining = remaining[written:]
    finally:
        os.close(fd)


def _write(decision):
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


def _drain():
    while True:
        _write(_records.get())


def emit(decision):
    """Queue an advisory record without waiting for any host-controlled I/O."""
    global _writer
    try:
        if _writer is None:
            _writer = threading.Thread(target=_drain, name="cvm-audit", daemon=True)
            _writer.start()
        _records.put_nowait(decision)
    except (RuntimeError, queue.Full):
        pass
