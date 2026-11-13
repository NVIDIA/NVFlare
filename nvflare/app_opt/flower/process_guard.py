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

"""Independent Flower cleanup owner; executed by path to keep startup lightweight.

The job alone holds the write end of stdin. EOF means normal shutdown or job
death, including SIGKILL. The guard lives in a separate session and retains
process identities, so cleanup also reaches detached, reparented workers.
"""

import json
import os
import selectors
import sys
import time

import psutil

OWNER_ENV = "NVFLARE_FLOWER_PROCESS_OWNER"


def capture_owned(roots, owner=None):
    """Walk one process-table snapshot, preserving identity across PID reuse."""
    processes = {p.pid: p for p in psutil.process_iter(["ppid", "create_time"])}
    children = {}
    for process in processes.values():
        children.setdefault(process.info["ppid"], []).append(process)
    found = {p for p in roots if p.is_running() and processes.get(p.pid) == p}
    pending = list(found)
    while pending:
        for child in children.get(pending.pop().pid, ()):
            if child not in found and child.is_running():
                found.add(child)
                pending.append(child)
    if owner:
        for process in processes.values():
            try:
                if process.environ().get(OWNER_ENV) == owner:
                    found.add(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    return found


def is_alive(process):
    try:
        return process.is_running() and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


def cleanup(owned, owner, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        owned = capture_owned(owned, owner)
        alive = [p for p in owned if is_alive(p)]
        if not alive:
            return
        errors = []
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
            except psutil.Error as error:
                errors.append(str(error))
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Flower guard could not clean workers {[p.pid for p in alive]}: {errors}")
        time.sleep(0.05)


def main(owner, launch_fence):
    owned = set()
    pending = b""
    try:
        os.set_blocking(sys.stdin.fileno(), False)
        with selectors.DefaultSelector() as selector:
            selector.register(sys.stdin, selectors.EVENT_READ)
            print("ready", flush=True)
            while True:
                if selector.select(0.5):
                    data = os.read(sys.stdin.fileno(), 65536)
                    if not data:
                        break
                    lines = (pending + data).split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        pid, created = json.loads(line)
                        try:
                            root = psutil.Process(pid)
                            if root.create_time() == created:
                                owned.add(root)
                        except psutil.NoSuchProcess:
                            pass
                owned = capture_owned(owned)
    finally:
        # The job can die before Popen returns a PID. A child between fork and
        # exec has no owner marker yet. Its launch wrapper closes this inherited
        # descriptor only after exec has installed the marked environment.
        try:
            while os.read(launch_fence, 1):
                pass
        finally:
            os.close(launch_fence)
            cleanup(owned, owner)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
