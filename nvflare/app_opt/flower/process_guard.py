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

The job holds the write end of stdin. EOF means normal shutdown or job death,
including SIGKILL, unless a fork-without-exec helper inherited that descriptor.
The guard retains observed process identities and discovers orphans by their
inherited owner marker. Unobserved workers without a readable marker cannot be
recovered after their ancestors exit.
"""

import json
import logging
import os
import selectors
import sys
import time

import psutil

OWNER_ENV = "NVFLARE_FLOWER_PROCESS_OWNER"


def capture_owned(roots, owner=None, errors=None):
    """Retain ownership on inspection failures; optionally collect incomplete scans."""
    errors = errors if errors is not None else []

    def failed(pid, error):
        detail = RuntimeError(f"Flower process inspection failed for PID {pid}: {error!r}")
        errors.append(detail)
        logging.getLogger(__name__).warning("%s", detail)

    processes = {}
    try:
        for process in psutil.process_iter(["ppid", "create_time"]):
            processes[process.pid] = process
    except (psutil.Error, OSError) as error:
        failed("process table", error)
    children = {}
    for process in processes.values():
        children.setdefault(process.info["ppid"], []).append(process)
    found = set()
    for process in roots:
        try:
            if process.is_running():
                found.add(process)
        except psutil.NoSuchProcess:
            pass
        except (psutil.Error, OSError) as error:
            found.add(process)
            failed(process.pid, error)
    pending = list(found)
    while pending:
        for child in children.get(pending.pop().pid, ()):
            try:
                if child not in found and child.is_running():
                    found.add(child)
                    pending.append(child)
            except psutil.NoSuchProcess:
                pass
            except (psutil.Error, OSError) as error:
                failed(child.pid, error)
    if owner:
        for process in processes.values():
            try:
                if process.environ().get(OWNER_ENV) == owner:
                    found.add(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except (psutil.Error, OSError) as error:
                failed(process.pid, error)
    return found


def is_alive(process):
    try:
        return process.is_running() and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


def cleanup(owned, owner, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        errors = []
        owned = capture_owned(owned, owner, errors)
        alive = []
        for process in owned:
            try:
                if is_alive(process):
                    alive.append(process)
            except psutil.NoSuchProcess:
                pass
            except (psutil.Error, OSError) as error:
                alive.append(process)
                errors.append(f"PID {process.pid}: {error!r}")
        if not alive and not errors:
            return
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
            except (psutil.Error, OSError) as error:
                errors.append(f"PID {process.pid}: {error!r}")
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
                try:
                    owned = capture_owned(owned)
                except (psutil.Error, OSError):
                    logging.getLogger(__name__).exception("Flower guard scan failed; retaining ownership and retrying")
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
