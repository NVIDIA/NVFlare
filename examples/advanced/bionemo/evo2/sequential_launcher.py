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
"""Serialize external Evo2 trainers that share one simulation GPU."""

from __future__ import annotations

import argparse
import fcntl
import os
import shlex
from pathlib import Path
from typing import NoReturn


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _inner_command(values: list[str]) -> list[str]:
    command = values[1:] if values and values[0] == "--" else values
    if not command:
        raise ValueError("An inner trainer command is required after '--'.")
    return command


def exec_with_lock(lock_file: str | os.PathLike[str], command: list[str]) -> NoReturn:
    """Acquire the workspace lock and replace this process with ``command``."""

    lock_path = Path(lock_file).expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        print(f"Waiting for Evo2 training lock: {lock_path}", flush=True)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # Python creates descriptors as non-inheritable. Clearing close-on-exec
        # keeps this lock held by the Python/torchrun process for its full life.
        os.set_inheritable(lock_fd, True)
        print(f"Acquired Evo2 training lock; starting: {shlex.join(command)}", flush=True)
        os.execvp(command[0], command)
    finally:
        os.close(lock_fd)


def main(argv: list[str] | None = None) -> NoReturn:
    args = define_parser().parse_args(argv)
    try:
        command = _inner_command(args.command)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    exec_with_lock(args.lock_file, command)


if __name__ == "__main__":
    main()
