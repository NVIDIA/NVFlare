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
import shlex
import subprocess
import sys
import threading


class ProcessManager:
    def __init__(self):
        """Constructor of ProcessManager.
        ProcessManager provides methods for managing the lifecycle of a subprocess (start, stop, poll), as well
        as the handling of log file to be used by the subprocess.

        NOTE: the methods of ProcessManager are not thread safe.

        """
        self.process = None
        self.log_file = None
        self.log_prefix = None

    def start(
        self,
        command: str,
        log_prefix: str = None,
        cwd=None,
        env=None,
        log_file=None,
        log_stdout=True,
    ):
        """Start the subprocess.

        Args:
            command: the command to be executed
            log_prefix: prefix to be prepended to log message when writing to stdout
            cwd: current work dir for the process to be started
            env: system env for the process
            log_file: log file for the process. It can be: a file object, a path to the file, or None.
                If None, no log file will be used.
            log_stdout: whether to output log messages to stdout.

        Returns: None

        """
        self.log_prefix = log_prefix

        lf = log_file
        if log_file and isinstance(log_file, str):
            lf = open(log_file, "a")
            self.log_file = lf

        if lf and log_stdout:
            stdout = subprocess.PIPE
        elif lf and not log_stdout:
            stdout = lf
        else:
            stdout = None

        command_seq = shlex.split(command)
        self.process = subprocess.Popen(
            command_seq,
            universal_newlines=True,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            stdout=stdout,
        )

        if stdout == subprocess.PIPE:
            log_writer = threading.Thread(target=self._write_log, daemon=True)
            log_writer.start()

    def _write_log(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                break

            self.log_file.write(line)
            self.log_file.flush()
            if self.log_prefix:
                line = f"[{self.log_prefix}] {line}"
            sys.stdout.write(line)
            sys.stdout.flush()

    def poll(self):
        """Perform a poll request on the process.

        Returns: None if the process is still running; an exit code (int) if process is not running.

        """
        if not self.process:
            return 0
        return self.process.poll()

    def stop(self):
        """Stop the process.
        If the process is still running, kill the process. If a log file is open, close the log file.

        Returns: the exit code of the process. If killed, returns -9.

        """
        rc = self.poll()
        if rc is None:
            # process is still alive
            try:
                self.process.kill()
                rc = -9
            except:
                # ignore kill error
                pass

        # close the log file if any
        if self.log_file:
            print("closed subprocess log file!")
            self.log_file.close()
            self.log_file = None
        return rc


def start_process(command: str, log_prefix=None, cwd=None, env=None, log_file=None, log_stdout=True) -> ProcessManager:
    """Convenience function for starting a subprocess.

    Args:
        command: the command to be executed
        log_prefix: log message prefix
        cwd: current work dir of the process
        env: system env for the process
        log_file: log file to be used for the process
        log_stdout: whether to write log messages to stdout

    Returns: a ProcessManager object.

    """
    mgr = ProcessManager()
    mgr.start(command, log_prefix, cwd, env, log_file, log_stdout)
    return mgr
