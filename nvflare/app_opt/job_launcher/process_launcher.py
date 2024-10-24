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
import sys
from abc import abstractmethod

from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.job_launcher_spec import JobHandleSpec, JobLauncherSpec
from nvflare.private.fed.utils.fed_utils import add_custom_dir_to_path


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
            return self.process.poll()
        else:
            return None

    def wait(self):
        if self.process:
            self.process.wait()


class ProcessJobLauncher(JobLauncherSpec):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

    def launch_job(self, launch_data: dict, fl_ctx: FLContext) -> JobHandleSpec:

        command, new_env = self.get_command(launch_data, fl_ctx)
        # use os.setsid to create new process group ID
        process = subprocess.Popen(shlex.split(command, True), preexec_fn=os.setsid, env=new_env)

        self.logger.info("Launch the job in process ID: {}".format(process.pid))

        return ProcessHandle(process)

    def can_launch(self, launch_data: dict) -> bool:
        job_image = launch_data.get(JobConstants.JOB_IMAGE)
        if job_image:
            return False
        else:
            return True

    @abstractmethod
    def get_command(self, launch_data, fl_ctx) -> (str, dict):
        """To generate the command to launcher the job in sub-process

        Args:
            fl_ctx: FLContext
            launch_data: job launcher data

        Returns:
            launch command, environment dict

        """
        pass

