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

"""Bound offline launcher workflows independently of product API deadlines."""

import os
import signal
import subprocess
import tempfile

import pytest


def run_launch(arguments, *, env, calls, timeout=60):
    # This is a completion watchdog, not a launcher latency assertion. Allow for
    # interpreter startup and CI scheduling while retaining a hard failure bound.
    # Files avoid blocking output collection if a descendant retains a pipe.
    with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(mode="w+") as stderr:
        process = subprocess.Popen(arguments, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Own the complete shell/Python/kubectl group, including descendants
            # left behind when the shell exits or the test fails.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        stdout.seek(0)
        stderr.seek(0)
        output, errors = stdout.read(), stderr.read()
    if timed_out:
        progress = "\n".join(calls.read_text().splitlines()[-10:]) if calls.exists() else "(no kubectl calls)"
        pytest.fail(
            f"Launcher did not finish within {timeout}s: {arguments!r}\n"
            f"stdout:\n{output}\nstderr:\n{errors}\nRecent kubectl calls:\n{progress}",
            pytrace=False,
        )
    return subprocess.CompletedProcess(arguments, process.returncode, output, errors)
