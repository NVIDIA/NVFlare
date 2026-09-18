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

"""Scoped process execution, locks and protected secret handling."""

import contextlib
import ctypes
import fcntl
import os
import resource
import subprocess
from pathlib import Path

from .errors import BuildError, require


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
