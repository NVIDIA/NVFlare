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
        """

        Args:
            cmd: the command to be executed
            cwd: current work dir for the process to be started
            env: system env for the process
            log_file_name: base name of the log file.
            log_stdout: whether to output log messages to stdout.
            stdout_msg_prefix: prefix to be prepended to log message when writing to stdout
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

        NOTE: the methods of ProcessManager are not thread safe.

        """
        check_object_type("cmd_desc", cmd_desc, CommandDescriptor)
        self.process = None
        self.cmd_desc = cmd_desc
        self.log_file = None
        self.msg_prefix = None
        self.logger = get_logger(self)

    def start(
        self,
        fl_ctx: FLContext,
    ):
        """Start the subprocess.

        Args:
            fl_ctx: FLContext object.

        Returns: None

        """
        job_id = fl_ctx.get_job_id()

        if self.cmd_desc.stdout_msg_prefix:
            site_name = fl_ctx.get_identity_name()
            self.msg_prefix = f"[{self.cmd_desc.stdout_msg_prefix}@{site_name}]"

        lf = None
        if self.cmd_desc.log_file_name:
            ws = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            if not isinstance(ws, Workspace):
                self.logger.error(
                    f"FL context prop {FLContextKey.WORKSPACE_OBJECT} should be Workspace but got {type(ws)}"
                )
                raise RuntimeError("bad FLContext object")

            run_dir = ws.get_run_dir(job_id)
            log_file_path = os.path.join(run_dir, self.cmd_desc.log_file_name)

            lf = open(log_file_path, "a")
            self.log_file = lf

        if lf and self.cmd_desc.log_stdout:
            stdout = subprocess.PIPE
        elif lf and not self.cmd_desc.log_stdout:
            stdout = lf
        else:
            stdout = None

        env = os.environ.copy()
        if self.cmd_desc.env:
            env.update(self.cmd_desc.env)

        command_seq = shlex.split(self.cmd_desc.cmd)
        self.process = subprocess.Popen(
            command_seq,
            universal_newlines=True,
            stderr=subprocess.STDOUT,
            cwd=self.cmd_desc.cwd,
            env=self.cmd_desc.env,
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
            if self.msg_prefix:
                line = f"{self.msg_prefix} {line}"
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
            self.logger.debug("closed subprocess log file!")
            self.log_file.close()
            self.log_file = None
        return rc


def start_process(cmd_desc: CommandDescriptor, fl_ctx: FLContext) -> ProcessManager:
    """Convenience function for starting a subprocess.

    Args:
        cmd_desc: the command to be executed
        fl_ctx: FLContext object

    Returns: a ProcessManager object.

    """
    mgr = ProcessManager(cmd_desc)
    mgr.start(fl_ctx)
    return mgr
