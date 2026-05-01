#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Run a command with optional timeout and write combined output to a log file."""

import argparse
import os
import signal
import subprocess
import sys

TERMINATE_GRACE_SECONDS = 30
KILL_GRACE_SECONDS = 5


def terminate_process_group(process, log):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        log.write(
            f"ERROR: process group did not exit after SIGTERM within {TERMINATE_GRACE_SECONDS} seconds; "
            "sending SIGKILL\n"
        )
        log.flush()

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        log.write(f"ERROR: process did not exit within {KILL_GRACE_SECONDS} seconds after SIGKILL\n")
        log.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("command is required after --")

    with open(args.log, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            log.write(f"\nERROR: run exceeded timeout of {args.timeout:.0f} seconds\n")
            log.flush()
            terminate_process_group(process, log)
            return 124


if __name__ == "__main__":
    sys.exit(main())
