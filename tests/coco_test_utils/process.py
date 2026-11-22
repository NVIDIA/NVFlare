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


"""Own the process tree and watchdog for offline CoCo CLI contracts."""

import os
import signal
import subprocess
from pathlib import Path

# These tests assert command behavior, not throughput. Keep scheduling/startup
# outside any product policy deadline; a stuck command must still fail and reap.
COMMAND_TIMEOUT = 90


def run_coco_command(command, *, timeout=COMMAND_TIMEOUT, check=False, input=None, diagnostic_paths=(), **kwargs):
    """Run a command in its own session, killing owned descendants on failure."""
    if kwargs.pop("capture_output", False):
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if input is not None:
        kwargs["stdin"] = subprocess.PIPE
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate(timeout=10)
            if isinstance(error, subprocess.TimeoutExpired):
                details = []
                for path in diagnostic_paths:
                    path = Path(path)
                    if path.is_file():
                        details.append(f"{path.name}:\n{path.read_text(errors='replace')[-8192:]}")
                raise AssertionError(
                    f"CoCo command exceeded its {timeout}s watchdog: {command!r}\n"
                    f"stdout: {stdout!r}\nstderr: {stderr!r}\n" + "\n".join(details)
                ) from error
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result
