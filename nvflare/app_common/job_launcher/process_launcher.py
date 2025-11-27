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
import logging
import os
import shlex
import signal
import subprocess
from abc import abstractmethod
from typing import Optional, Sequence

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobReturnCode, add_launcher
from nvflare.apis.workspace import Workspace
from nvflare.utils.job_launcher_utils import add_custom_dir_to_path, extract_job_image

JOB_RETURN_CODE_MAPPING = {0: JobReturnCode.SUCCESS, 1: JobReturnCode.EXECUTION_ERROR, 9: JobReturnCode.ABORTED}
_POSIX_SPAWN_SUPPORTED = hasattr(os, "posix_spawn") and os.name == "posix"


class ProcessHandle(JobHandleSpec):
    def __init__(self, process: Optional[subprocess.Popen] = None, pid: Optional[int] = None):
        super().__init__()

        self.process = process
        self.pid = pid if pid is not None else (process.pid if process else None)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._return_code: Optional[int] = None

        if self.pid is None:
            raise ValueError("ProcessHandle requires either a process object or a pid.")

    def terminate(self):
        self._kill_process_group()

        if self.process:
            self.process.terminate()

    def poll(self):
        if self.process:
            code = self.process.poll()
            if code is None:
                return JobReturnCode.UNKNOWN
            return JOB_RETURN_CODE_MAPPING.get(code, JobReturnCode.EXECUTION_ERROR)

        return self._poll_pid()

    def wait(self):
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

    def _poll_pid(self) -> JobReturnCode:
        if self.pid is None:
            return JobReturnCode.UNKNOWN

        if self._return_code is not None:
            return JOB_RETURN_CODE_MAPPING.get(self._return_code, JobReturnCode.EXECUTION_ERROR)

        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            return JobReturnCode.UNKNOWN

        if pid == 0:
            return JobReturnCode.UNKNOWN

        self._return_code = self._decode_status(status)
        return JOB_RETURN_CODE_MAPPING.get(self._return_code, JobReturnCode.EXECUTION_ERROR)

    def _decode_status(self, status: int) -> int:
        if hasattr(os, "waitstatus_to_exitcode"):
            return os.waitstatus_to_exitcode(status)

        if os.WIFEXITED(status):
            return os.WEXITSTATUS(status)
        if os.WIFSIGNALED(status):
            return -os.WTERMSIG(status)
        return JobReturnCode.EXECUTION_ERROR

    def _kill_process_group(self):
        if self.pid is None:
            return

        if not hasattr(os, "killpg") or not hasattr(os, "getpgid"):
            return

        try:
            pgid = os.getpgid(self.pid)
        except Exception:
            pgid = self.pid

        try:
            os.killpg(pgid, signal.SIGKILL)
            self.logger.debug("kill signal sent")
        except Exception:
            pass


class ProcessJobLauncher(JobLauncherSpec):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self._use_posix_spawn = _POSIX_SPAWN_SUPPORTED

    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:

        new_env = os.environ.copy()
        workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        job_id = job_meta.get(JobConstants.JOB_ID)
        app_custom_folder = workspace_obj.get_app_custom_dir(job_id)
        if app_custom_folder != "":
            add_custom_dir_to_path(app_custom_folder, new_env)

        command = self.get_command(job_meta, fl_ctx)
        argv = shlex.split(command, True)

        if self._use_posix_spawn and argv:
            try:
                pid = self._spawn_with_posix(argv, new_env)
            except Exception as exc:
                self.logger.warning("posix_spawn failed (%s); falling back to subprocess.", exc)
            else:
                self.logger.info("Launch the job in process ID: {}".format(pid))
                return ProcessHandle(pid=pid)

        preexec_fn = os.setsid if hasattr(os, "setsid") else None
        process = subprocess.Popen(argv, preexec_fn=preexec_fn, env=new_env)

        self.logger.info("Launch the job in process ID: {}".format(process.pid))

        return ProcessHandle(process=process)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.BEFORE_JOB_LAUNCH:
            job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
            job_image = extract_job_image(job_meta, fl_ctx.get_identity_name())
            if not job_image:
                add_launcher(self, fl_ctx)

    def _spawn_with_posix(self, argv: Sequence[str], env: dict) -> int:
        if not _POSIX_SPAWN_SUPPORTED:
            raise RuntimeError("posix_spawn is not supported on this platform.")

        path = argv[0]
        return os.posix_spawn(path, argv, env, setsid=True)

    @abstractmethod
    def get_command(self, job_meta, fl_ctx) -> str:
        """To generate the command to launcher the job in sub-process

        Args:
            fl_ctx: FLContext
            job_meta: job meta data

        Returns:
            launch command

        """
        pass
