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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobReturnCode, add_launcher
from nvflare.apis.workspace import Workspace
from nvflare.utils.job_launcher_utils import add_custom_dir_to_path, extract_job_image

JOB_RETURN_CODE_MAPPING = {0: JobReturnCode.SUCCESS, 1: JobReturnCode.EXECUTION_ERROR, 9: JobReturnCode.ABORTED}


class ProcessHandle(JobHandleSpec):
    def __init__(self, process):
        super().__init__()

        self.process = process
        self.logger = logging.getLogger(self.__class__.__name__)

    def terminate(self):
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), 9)
                self.logger.debug("kill signal sent")
            except:
                pass

            self.process.terminate()

    def poll(self):
        if self.process:
            return JOB_RETURN_CODE_MAPPING.get(self.process.poll(), JobReturnCode.EXECUTION_ERROR)
        else:
            return JobReturnCode.UNKNOWN

    def wait(self):
        if self.process:
            self.process.wait()


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
        # use os.setsid to create new process group ID
        process = subprocess.Popen(shlex.split(command, True), preexec_fn=os.setsid, env=new_env)

        self.logger.info("Launch the job in process ID: {}".format(process.pid))

        return ProcessHandle(process)

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
