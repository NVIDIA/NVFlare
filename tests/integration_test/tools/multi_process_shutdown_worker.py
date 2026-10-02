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

"""Plain child-process fixtures for the multiprocess shutdown integration tests."""

import subprocess
import sys
import time
from pathlib import Path


def _rank(directory, rank):
    (directory / f"ready-{rank}").touch()
    deadline = time.monotonic() + 30.0
    while not (directory / "close").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("CLOSE was not delivered")
        time.sleep(0.01)
    (directory / f"finished-{rank}").touch()


def _launcher(directory):
    children = []
    try:
        for rank in range(2):
            children.append(subprocess.Popen([sys.executable, __file__, "rank", str(directory), str(rank)]))
        for child in children:
            if child.wait(timeout=35.0) != 0:
                raise RuntimeError("rank failed")
        (directory / "launcher-finished").touch()
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5.0)


def _exited_launcher(directory):
    # Keep a rank in this launcher's process group after the launcher exits.
    child = subprocess.Popen([sys.executable, __file__, "rank", str(directory), "0"])
    try:
        deadline = time.monotonic() + 10.0
        while not (directory / "ready-0").exists():
            if child.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("rank did not start")
            time.sleep(0.01)
    except BaseException:
        child.kill()
        child.wait(timeout=5.0)
        raise


def main():
    mode, directory = sys.argv[1], Path(sys.argv[2])
    if mode == "launcher":
        _launcher(directory)
    elif mode == "rank":
        _rank(directory, sys.argv[3])
    elif mode == "exited-launcher":
        _exited_launcher(directory)
    elif mode == "hung":
        (directory / "ready").touch()
        time.sleep(30.0)
    else:
        raise ValueError(f"unknown worker mode: {mode}")


if __name__ == "__main__":
    main()
