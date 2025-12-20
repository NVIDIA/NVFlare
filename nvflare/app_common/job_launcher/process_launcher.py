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
import subprocess
from abc import abstractmethod
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobReturnCode, add_launcher
from nvflare.apis.workspace import Workspace
from nvflare.utils.job_launcher_utils import add_custom_dir_to_path, extract_job_image
from nvflare.utils.process_utils import ProcessAdapter, spawn_process

JOB_RETURN_CODE_MAPPING = {0: JobReturnCode.SUCCESS, 1: JobReturnCode.EXECUTION_ERROR, 9: JobReturnCode.ABORTED}


class ProcessHandle(JobHandleSpec):
    def __init__(
        self,
        process: Optional[subprocess.Popen] = None,
        pid: Optional[int] = None,
        process_adapter: Optional[ProcessAdapter] = None,
    ):
        super().__init__()

        if process_adapter:
            self.adapter = process_adapter
        elif process or pid is not None:
            self.adapter = ProcessAdapter(process=process, pid=pid)
        else:
            raise ValueError("ProcessHandle requires a process object, a pid, or a ProcessAdapter.")

    def terminate(self):
        self.adapter.terminate()

    def poll(self):
        code = self.adapter.poll()
        if code is None:
            return JobReturnCode.UNKNOWN
        return JOB_RETURN_CODE_MAPPING.get(code, JobReturnCode.EXECUTION_ERROR)

    def wait(self):
        self.adapter.wait()


class ProcessJobLauncher(JobLauncherSpec):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:

        new_env = os.environ.copy()
        workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        job_id = job_meta.get(JobConstants.JOB_ID)
        app_custom_folder = workspace_obj.get_app_custom_dir(job_id)
        if app_custom_folder != "":
            add_custom_dir_to_path(app_custom_folder, new_env)

        command = self.get_command(job_meta, fl_ctx)
        argv = shlex.split(command, True)

        # Use the spawn_process utility which handles the choice between posix_spawn and subprocess
        process_adapter = spawn_process(argv, new_env)

        return ProcessHandle(process_adapter=process_adapter)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.BEFORE_JOB_LAUNCH:
            job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
            job_image = extract_job_image(job_meta, fl_ctx.get_identity_name())
            if not job_image:
                add_launcher(self, fl_ctx)

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
