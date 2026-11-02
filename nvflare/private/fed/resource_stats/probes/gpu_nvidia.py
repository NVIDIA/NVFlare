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


"""Count CUDA devices visible to this process using only the system Driver API.

Keep this standalone helper limited to the standard library. It inherits the
execution's environment and device access, without importing NVFlare or a
framework, inspecting UUIDs, or initializing CUDA in the application process.
"""

import ctypes
import json
import os
import platform
import select
import signal
import sys
import threading
import time

CUDA_SUCCESS = 0
CUDA_ERROR_NOT_INITIALIZED = 3
CUDA_ERROR_NO_DEVICE = 100
MAX_CUDA_DEVICES = 4096


def _load_driver():
    name = {"Linux": "libcuda.so.1", "Darwin": "libcuda.dylib"}.get(platform.system())
    if name is not None:
        try:
            return ctypes.CDLL(name)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return None


def _bind(driver, name, argtypes):
    try:
        function = getattr(driver, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        return function
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _result(function, *args):
    if function is not None:
        try:
            return function(*args)
        except Exception:
            pass
    return None


def _count(driver, *, initialized_only):
    if not initialized_only:
        status = _result(_bind(driver, "cuInit", [ctypes.c_uint]), 0)
        if status == CUDA_ERROR_NO_DEVICE:
            return {"count": 0}
        if status != CUDA_SUCCESS:
            return None
    count = ctypes.c_int(-1)
    status = _result(_bind(driver, "cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)]), ctypes.byref(count))
    if status == CUDA_ERROR_NO_DEVICE:
        return {"count": 0}
    if initialized_only and status == CUDA_ERROR_NOT_INITIALIZED and os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return {"count": 0}
    return {"count": count.value} if status == CUDA_SUCCESS and 0 <= count.value <= MAX_CUDA_DEVICES else None


def probe_gpu():
    """Initialize only the helper's Driver and return its visible device count."""
    driver = _load_driver()
    if driver is None:
        return None
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return {"count": 0}
    return _count(driver, initialized_only=False)


def probe_initialized_gpu():
    """Reuse application-owned CUDA state without initialization or metadata calls."""
    driver = _load_driver()
    return _count(driver, initialized_only=True) if driver is not None else None


def _watch_owner(parent_pid):
    # getppid observes the owning process, unlike Linux PDEATHSIG, which follows
    # the particular thread that created the helper. EOF also releases a hold;
    # checking the process covers descendants that inherited the pipe writer.
    try:
        while os.getppid() == parent_pid:
            if select.select([sys.stdin.fileno()], [], [], 0.1)[0]:
                break
    except (OSError, ValueError):
        pass
    os._exit(0)


def _protect_owner(parent_pid, deadline):
    remaining = deadline - time.monotonic()
    if os.getppid() != parent_pid or remaining <= 0:
        raise SystemExit(1)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    monitor = threading.Thread(target=_watch_owner, args=(parent_pid,), daemon=True)
    monitor.start()
    return monitor


def main(*, parent_pid=None, deadline=None):
    """Emit one count; hold positive initialization until the execution releases it."""
    monitor = _protect_owner(parent_pid, deadline) if parent_pid is not None else None
    value = probe_gpu()
    if monitor is not None:
        signal.setitimer(signal.ITIMER_REAL, 0)
    sys.stdout.buffer.write(json.dumps(value, allow_nan=False).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()
    if monitor is not None and value is not None and value["count"] > 0:
        monitor.join()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--hold":
        main(parent_pid=int(sys.argv[2]), deadline=float(sys.argv[3]))
    else:
        main()
