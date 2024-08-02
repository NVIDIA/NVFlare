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
import os
import shlex
import subprocess
import sys
import threading

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils.obj_utils import get_logger
from nvflare.fuel.utils.validation_utils import check_object_type, check_str


class CommandDescriptor:
    def __init__(
        self,
        cmd: str,
        cwd=None,
        env=None,
        log_file_name: str = "",
        log_stdout: bool = True,
        stdout_msg_prefix: str = None,
    ):
        """Constructor of CommandDescriptor.
        A CommandDescriptor describes the requirements of the new process to be started.

        Args:
            cmd: the command to be executed to start the new process
            cwd: current work dir for the new process
            env: system env for the new process
            log_file_name: base name of the log file.
            log_stdout: whether to output log messages to stdout.
            stdout_msg_prefix: prefix to be prepended to log message when writing to stdout.
                Since multiple processes could be running within the same terminal window, the prefix can help
                differentiate log messages from these processes.
        """
        check_str("cmd", cmd)

        if cwd:
            check_str("cwd", cwd)

        if env:
            check_object_type("env", env, dict)

        if log_file_name:
            check_str("log_file_name", log_file_name)

        if stdout_msg_prefix:
            check_str("stdout_msg_prefix", stdout_msg_prefix)

        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self.log_file_name = log_file_name
        self.log_stdout = log_stdout
        self.stdout_msg_prefix = stdout_msg_prefix


class ProcessManager:
    def __init__(self, cmd_desc: CommandDescriptor):
        """Constructor of ProcessManager.
        ProcessManager provides methods for managing the lifecycle of a subprocess (start, stop, poll), as well
        as the handling of log file to be used by the subprocess.

        Args:
            cmd_desc: the CommandDescriptor that describes the command of the new process to be started

        NOTE: the methods of ProcessManager are not thread safe.

        """
        check_object_type("cmd_desc", cmd_desc, CommandDescriptor)
        self.process = None
        self.cmd_desc = cmd_desc
        self.log_file = None
        self.msg_prefix = None
        self.file_lock = threading.Lock()
        self.logger = get_logger(self)

    def start(
        self,
        fl_ctx: FLContext,
    ):
        """Start the new process.

        Args:
            fl_ctx: FLContext object.

        Returns: None

        """
        job_id = fl_ctx.get_job_id()

        if self.cmd_desc.stdout_msg_prefix:
            site_name = fl_ctx.get_identity_name()
            self.msg_prefix = f"[{self.cmd_desc.stdout_msg_prefix}@{site_name}]"

        if self.cmd_desc.log_file_name:
            ws = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            if not isinstance(ws, Workspace):
                self.logger.error(
                    f"FL context prop {FLContextKey.WORKSPACE_OBJECT} should be Workspace but got {type(ws)}"
                )
                raise RuntimeError("bad FLContext object")

            run_dir = ws.get_run_dir(job_id)
            log_file_path = os.path.join(run_dir, self.cmd_desc.log_file_name)
            self.log_file = open(log_file_path, "a")

        env = os.environ.copy()
        if self.cmd_desc.env:
            env.update(self.cmd_desc.env)

        command_seq = shlex.split(self.cmd_desc.cmd)
        self.process = subprocess.Popen(
            command_seq,
            stderr=subprocess.STDOUT,
            cwd=self.cmd_desc.cwd,
            env=env,
            stdout=subprocess.PIPE,
        )

        log_writer = threading.Thread(target=self._write_log, daemon=True)
        log_writer.start()

    def _write_log(self):
        # write messages from the process's stdout pipe to log file and sys.stdout.
        # note that depending on how the process flushes out its output, the messages may be buffered/delayed.
        while True:
            line = self.process.stdout.readline()
            if not line:
                break

            assert isinstance(line, bytes)
            line = line.decode("utf-8")
            # use file_lock to ensure file integrity since the log file could be closed by the self.stop() method!
            with self.file_lock:
                if self.log_file:
                    self.log_file.write(line)
                    self.log_file.flush()

            if self.cmd_desc.log_stdout:
                assert isinstance(line, str)
                if self.msg_prefix and not line.startswith("\r"):
                    line = f"{self.msg_prefix} {line}"
                sys.stdout.write(line)
                sys.stdout.flush()

    def poll(self):
        """Perform a poll request on the process.

        Returns: None if the process is still running; an exit code (int) if process is not running.

        """
        if not self.process:
            raise RuntimeError("there is no process to poll")
        return self.process.poll()

    def stop(self) -> int:
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
        with self.file_lock:
            if self.log_file:
                self.logger.debug("closed subprocess log file!")
                self.log_file.close()
                self.log_file = None
        return rc


def start_process(cmd_desc: CommandDescriptor, fl_ctx: FLContext) -> ProcessManager:
    """Convenience function for starting a subprocess.

    Args:
        cmd_desc: the CommandDescriptor the describes the command to be executed
        fl_ctx: FLContext object

    Returns: a ProcessManager object.

    """
    mgr = ProcessManager(cmd_desc)
    mgr.start(fl_ctx)
    return mgr
