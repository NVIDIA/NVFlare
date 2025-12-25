# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import os
import signal
import subprocess
from typing import List, Optional

log = logging.getLogger(__name__)

_POSIX_SPAWN_SUPPORTED = hasattr(os, "posix_spawn") and os.name == "posix"


class ProcessAdapter:
    def __init__(self, process: Optional[subprocess.Popen] = None, pid: Optional[int] = None):
        """Adapter to manage a process, whether created via subprocess.Popen or os.posix_spawn.

        Args:
            process: The subprocess.Popen object (if created via subprocess)
            pid: The process ID (if created via posix_spawn, or fallback for process.pid)
        """
        self.process = process
        self.pid = pid if pid is not None else (process.pid if process else None)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._return_code: Optional[int] = None

        if self.pid is None:
            raise ValueError("ProcessAdapter requires either a process object or a pid.")

    def terminate(self) -> None:
        """Terminate the process group.

        Sends SIGKILL to the entire process group. No need to call process.terminate()
        separately since SIGKILL already terminates all processes in the group.
        """
        self._kill_process_group()

    def poll(self) -> Optional[int]:
        """Check if the process has terminated.

        Returns:
            None if process is still running, otherwise the exit code.
        """
        if self.process:
            return self.process.poll()

        return self._poll_pid()

    def wait(self) -> None:
        """Wait for the process to terminate."""
        if self.process:
            self.process.wait()
            return

        if self.pid is None:
            return

        if self._return_code is None:
            try:
                _, status = os.waitpid(self.pid, 0)
                self._return_code = self._decode_status(status)
            except ChildProcessError:
                pass

    def _poll_pid(self) -> Optional[int]:
        if self.pid is None:
            return None

        if self._return_code is not None:
            return self._return_code

        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            # Process already reaped or doesn't exist, treat as terminated
            if self._return_code is None:
                self._return_code = -1
            return self._return_code

        if pid == 0:
            return None

        self._return_code = self._decode_status(status)
        return self._return_code

    def _decode_status(self, status: int) -> int:
        if hasattr(os, "waitstatus_to_exitcode"):
            return os.waitstatus_to_exitcode(status)

        if os.WIFEXITED(status):
            return os.WEXITSTATUS(status)
        if os.WIFSIGNALED(status):
            return -os.WTERMSIG(status)
        # Fallback/Error case
        return -1

    def _kill_process_group(self):
        if self.pid is None:
            return

        if not hasattr(os, "killpg") or not hasattr(os, "getpgid"):
            return

        try:
            pgid = os.getpgid(self.pid)
        except ProcessLookupError:
            # Process already gone; nothing left to terminate.
            return
        except PermissionError as exc:
            self.logger.warning("Unable to read pgid for %s (%s)", self.pid, exc)
            pgid = self.pid

        try:
            os.killpg(pgid, signal.SIGKILL)
            self.logger.debug("kill signal sent")
        except ProcessLookupError:
            # Group already terminated, treat as success.
            return
        except Exception as exc:
            self.logger.warning("Failed to kill process group %s (%s)", pgid, exc)


def spawn_process(cmd_args: List[str], env: dict) -> ProcessAdapter:
    """Launch a process using posix_spawn if available, falling back to subprocess.Popen.

    This method attempts to use os.posix_spawn with setsid=True to avoid fork() related issues
    (such as gRPC deadlocks). If posix_spawn is unavailable or fails, it falls back to
    subprocess.Popen with preexec_fn=os.setsid.

    Args:
        cmd_args: The command arguments as a list of strings.
        env: The environment variables dictionary.

    Returns:
        ProcessAdapter: An adapter wrapping the launched process.
    """
    if _POSIX_SPAWN_SUPPORTED and cmd_args:
        try:
            # Note: 'setsid' is a potential extension or patch in some python environments.
            # We wrap it in try-except to gracefully fallback if not supported.
            path = cmd_args[0]
            pid = os.posix_spawn(path, cmd_args, env, setsid=True)
            log.info("Launch the job in process ID: %s (posix_spawn)", pid)
            return ProcessAdapter(pid=pid)
        except TypeError as exc:
            # Happens when this interpreter lacks posix_spawn(..., setsid=...) support and silently falls back to fork.
            log.warning("posix_spawn missing setsid support (%s); falling back to subprocess.", exc)
        except Exception as exc:
            # Covers launch failures unrelated to setsid (e.g. binary missing, permission issues).
            log.warning("posix_spawn failed (%s); falling back to subprocess.", exc)

    preexec_fn = os.setsid if hasattr(os, "setsid") else None
    process = subprocess.Popen(cmd_args, preexec_fn=preexec_fn, env=env)
    log.info("Launch the job in process ID: %s (subprocess)", process.pid)

    return ProcessAdapter(process=process)
