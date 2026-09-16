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

"""Small, shared contracts for the builder, launcher and measured guest."""

import base64
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import re
import resource
import subprocess
import tempfile
import time
from pathlib import Path

HEADER_BYTES = 16777216
STORAGE_PROFILE = "luks2-xts-random-hmac-sha256-v1"
PLATFORMS = ("amd_sev_snp", "intel_tdx")
DISK_ROLES = ("root", "applog", "user-config", "user-data", "vault")
ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")


class BuildError(Exception):
    """An actionable, non-secret diagnostic safe to print."""


def require(condition, message):
    if not condition:
        raise BuildError(message)


def disk_device(role, *, wait=False):
    """Select a disk by its launch-assigned serial, never Linux probe order."""
    require(role in DISK_ROLES, "Unknown disk role")
    path = Path("/dev/disk/by-id") / ("scsi-0QEMU_QEMU_HARDDISK_cvm-" + role)
    if wait:
        deadline = time.monotonic() + 30
        while not path.is_block_device():
            require(time.monotonic() < deadline, "Required CVM disk is missing")
            time.sleep(0.1)
    return str(path)


def identifier(value):
    require(isinstance(value, str) and ID.fullmatch(value), "Invalid immutable identifier")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def write_json(path, value, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".cvm-", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(argv, *, input=None, pass_fds=(), timeout=3600, cwd=None, env=None):
    """No shell, no echoed arguments/output: child errors can contain secrets."""
    try:
        result = subprocess.run(
            [str(a) for a in argv],
            input=input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=pass_fds,
            timeout=timeout,
            cwd=cwd,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuildError(f"{Path(argv[0]).name} could not complete ({type(exc).__name__})") from None
    require(result.returncode == 0, f"{Path(argv[0]).name} failed (exit {result.returncode}); no output logged")
    return result.stdout


@contextlib.contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise BuildError("Resource is already in use") from None
        yield stream


def binding(header):
    require(len(header) == HEADER_BYTES, "Truncated or unsupported vault header")
    return hashlib.sha256(b"nvflare-vault-v2\x00" + header).digest()


def binding_id(platform, value):
    require(platform in PLATFORMS and len(value) == 32, "Invalid platform/binding")
    if platform == "amd_sev_snp":
        return base64.urlsafe_b64encode(value).decode().rstrip("=")
    return (value + bytes(16)).hex()


def qemu_binding(platform, value):
    require(platform in PLATFORMS and len(value) == 32, "Invalid platform/binding")
    return base64.b64encode(value + (bytes(16) if platform == "intel_tdx" else b"")).decode()


def resource_path(build_id, platform, value):
    return f"keys/{identifier(build_id)}/{binding_id(platform, value)}"


def validate_resource(path):
    parts = path.split("/")
    require(len(parts) == 3 and parts[0] == "keys", "Invalid resource namespace")
    identifier(parts[1])
    tag = parts[2]
    snp = re.fullmatch(r"[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]", tag)
    tdx = re.fullmatch(r"[0-9a-f]{64}0{32}", tag)
    require(snp or tdx, "Noncanonical binding identifier")
    return parts


def validate_core_policy(path=Path("/proc/sys/kernel/core_pattern")):
    # Linux ignores RLIMIT_CORE for pipe handlers, and exec resets dumpability.
    # A child holding a key must therefore never inherit a piped core collector.
    require(not path.read_text().strip().startswith("|"), "Disable piped core collection before handling secrets")


def protect_process():
    """Lock all Python copies as well as FDs; never silently allow swapping."""
    require(os.name == "posix" and Path("/proc/swaps").exists(), "Secret handling requires Linux")
    require(len(Path("/proc/swaps").read_text().splitlines()) == 1, "Disable build/guest swap before handling secrets")
    validate_core_policy()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(4, 0, 0, 0, 0) == 0, "Cannot disable process dumps")
    require(libc.mlockall(1 | 2) == 0, "Cannot lock memory; increase LimitMEMLOCK / CAP_IPC_LOCK")


@contextlib.contextmanager
def memory_file(data, *, sealed=False):
    require(hasattr(os, "memfd_create"), "Linux memfd is required")
    fd = os.memfd_create("cvm-private", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        with os.fdopen(os.dup(fd), "wb") as stream:
            stream.write(data)
            stream.flush()
        os.lseek(fd, 0, os.SEEK_SET)
        if sealed:
            fcntl.fcntl(
                fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
            )
        yield fd
    finally:
        if not sealed:
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, bytes(len(data)))
            os.ftruncate(fd, 0)
        os.close(fd)
