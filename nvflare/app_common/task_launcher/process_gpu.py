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

"""Exclusive whole-GPU accounting for a trusted local Process task partition.

All participating CJs on this host must use the same persistent local directory.
The partition must be excluded from job-lifetime and external GPU allocators.
Records deliberately survive process death. There is no timeout, automatic
reclamation, or restart/adoption: an administrator must prove orphan settlement
before repairing records. This is accounting for trusted code, not a sandbox.
"""

import json
import logging
import os
import sqlite3
import stat
import subprocess
import time
import uuid
from contextlib import contextmanager

from nvflare.apis.task_launcher_spec import TaskLauncherError


def _gpu_uuid(value):
    if not isinstance(value, str) or not value.startswith("GPU-"):
        raise ValueError("devices must be full physical NVIDIA GPU UUIDs")
    try:
        canonical = "GPU-" + str(uuid.UUID(value[4:]))
    except ValueError as error:
        raise ValueError("devices must be full physical NVIDIA GPU UUIDs") from error
    if canonical.lower() != value.lower():
        raise ValueError("devices must be full hyphenated GPU UUIDs")
    return canonical


class ProcessGpuPool:
    """Durable task-attempt reservations in a site-owned local SQLite ledger.

    SQLite serializes whole-attempt admission across processes. FULL synchronous
    transactions on a local filesystem are required; NFS/shared network storage,
    directory replacement, manual ledger deletion, and disk corruption are outside
    this profile. Allocated records never expire when their owner disappears.
    """

    def __init__(self, config):
        if not isinstance(config, dict) or set(config) != {"id", "devices", "exclusive", "reservation_root"}:
            raise ValueError("gpu_partition requires id, devices, exclusive and reservation_root")
        if config["exclusive"] is not True:
            raise ValueError("gpu_partition must be site-approved exclusive to Process task workers")
        if not isinstance(config["id"], str) or not config["id"].strip() or "\x00" in config["id"]:
            raise ValueError("gpu_partition id must be nonempty text without NUL")
        if not isinstance(config["devices"], list) or not config["devices"]:
            raise ValueError("gpu_partition devices must be a nonempty list")
        self.devices = tuple(sorted(_gpu_uuid(value) for value in config["devices"]))
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("gpu_partition devices must be distinct")
        root = config["reservation_root"]
        if not isinstance(root, str) or not os.path.isabs(root) or "\x00" in root:
            raise ValueError("reservation_root must be an absolute local directory")
        self.root = os.path.realpath(root)
        self.partition_id = config["id"]
        self.path = os.path.join(self.root, "process-gpu.sqlite3")
        self._failed = False
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        info = os.stat(self.root)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("reservation_root must be owned by the site UID with mode 0700")
        with self._transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS attempts (identity TEXT PRIMARY KEY, token TEXT NOT NULL, "
                "partition_id TEXT NOT NULL, devices TEXT NOT NULL, released INTEGER NOT NULL DEFAULT 0, "
                "acquired_at REAL NOT NULL, released_at REAL, execution_id TEXT, owner_pid INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS allocations (device TEXT PRIMARY KEY, identity TEXT NOT NULL, "
                "token TEXT NOT NULL)"
            )

    @contextmanager
    def _transaction(self):
        if self._failed:
            raise TaskLauncherError("Process GPU ledger is fenced after an uncertain transaction")
        if os.path.lexists(self.path):
            info = os.lstat(self.path)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise TaskLauncherError("Process GPU ledger is not a site-owned regular file")
        connection = sqlite3.connect(self.path, timeout=5.0)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except sqlite3.Error as error:
            # Never interpret a failed storage operation as free capacity.
            self._failed = True
            connection.rollback()
            raise TaskLauncherError("Process GPU ledger transaction failed; retain ownership for inspection") from error
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def validate_host(self):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,mig.mode.current", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        available = {}
        for line in result.stdout.splitlines():
            device, mig_mode = (value.strip() for value in line.split(",", 1))
            available[_gpu_uuid(device)] = mig_mode
        for device in self.devices:
            if device not in available or available[device] not in {"Disabled", "[N/A]", "N/A"}:
                raise TaskLauncherError("Process task partition requires locally present whole GPUs with MIG disabled")

    def acquire(self, identity, count):
        identity = json.dumps(tuple(identity), separators=(",", ":"))
        token = str(uuid.uuid4())
        with self._transaction() as connection:
            if any(name.startswith(".release-") for name in os.listdir(self.root)):
                raise TaskLauncherError("Process GPU release is unconfirmed; retain the pool for inspection")
            if connection.execute("SELECT 1 FROM attempts WHERE identity=?", (identity,)).fetchone():
                raise TaskLauncherError("task attempt was already admitted; resubmit its saved result")
            used = {row[0] for row in connection.execute("SELECT device FROM allocations")}
            devices = tuple(device for device in self.devices if device not in used)[:count]
            if not count or len(devices) != count:
                raise TaskLauncherError("approved Process GPU partition has insufficient free capacity")
            connection.execute(
                "INSERT INTO attempts(identity,token,partition_id,devices,acquired_at,owner_pid) VALUES(?,?,?,?,?,?)",
                (identity, token, self.partition_id, json.dumps(devices), time.time(), os.getpid()),
            )
            connection.executemany(
                "INSERT INTO allocations(device,identity,token) VALUES(?,?,?)",
                [(device, identity, token) for device in devices],
            )
        return _ProcessGpuReservation(self, identity, token, devices)


class _ProcessGpuReservation:
    def __init__(self, pool, identity, token, devices):
        self.pool, self.identity, self.token, self.devices = pool, identity, token, devices
        self.released = False

    def record_execution(self, execution_id):
        with self.pool._transaction() as connection:
            changed = connection.execute(
                "UPDATE attempts SET execution_id=? WHERE identity=? AND token=? AND released=0",
                (execution_id, self.identity, self.token),
            ).rowcount
            if changed != 1:
                raise TaskLauncherError("Process GPU reservation ownership changed during launch")

    def release(self):
        if self.released:
            return
        # A separate, durable admission fence precedes the release transaction.
        # If commit confirmation is lost (even after commit took effect), other
        # CJs must not infer available capacity from absent allocation rows.
        fence = os.path.join(self.pool.root, ".release-" + self.token)
        fd = os.open(fence, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"identity": self.identity, "token": self.token, "devices": self.devices}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            directory_fd = os.open(self.pool.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            # An incomplete fence still blocks all new admission. Allocation
            # rows are untouched until the durability barrier succeeds.
            raise
        with self.pool._transaction() as connection:
            row = connection.execute(
                "SELECT token,devices,released FROM attempts WHERE identity=?", (self.identity,)
            ).fetchone()
            owned = tuple(
                sorted(
                    row[0]
                    for row in connection.execute(
                        "SELECT device FROM allocations WHERE identity=? AND token=?", (self.identity, self.token)
                    )
                )
            )
            if (
                row is None
                or row[0] != self.token
                or row[2] != 0
                or tuple(json.loads(row[1])) != self.devices
                or owned != self.devices
            ):
                raise TaskLauncherError("Process GPU reservation ownership changed; retain evidence")
            connection.execute("DELETE FROM allocations WHERE identity=? AND token=?", (self.identity, self.token))
            connection.execute(
                "UPDATE attempts SET released=1,released_at=? WHERE identity=?", (time.time(), self.identity)
            )
        self.released = True
        # The release commit is confirmed before opening admission. Failure to
        # remove this extra fence can reduce capacity, but cannot make resource
        # release uncertain or permit overcommit. Retain it for an administrator.
        try:
            os.unlink(fence)
        except OSError as error:
            logging.getLogger(__name__).warning("GPU release confirmed; admission fence cleanup failed: %s", error)
