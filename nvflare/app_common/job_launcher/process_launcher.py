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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec
from nvflare.private.fed.utils.fed_utils import add_custom_dir_to_path, extract_job_image


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

    def launch_job(self, meta_data: dict, fl_ctx: FLContext) -> JobHandleSpec:

        new_env = os.environ.copy()
        workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        args = fl_ctx.get_prop(FLContextKey.ARGS)
        client = fl_ctx.get_prop(FLContextKey.SITE_OBJ)
        job_id = meta_data.get(JobMetaKey.JOB_ID)
        server_config = fl_ctx.get_prop(FLContextKey.SERVER_CONFIG)
        if not server_config:
            raise RuntimeError(f"missing {FLContextKey.SERVER_CONFIG} in FL context")
        service = server_config[0].get("service", {})
        if not isinstance(service, dict):
            raise RuntimeError(f"expect server config data to be dict but got {type(service)}")

        app_custom_folder = workspace_obj.get_app_custom_dir(job_id)
        if app_custom_folder != "":
            add_custom_dir_to_path(app_custom_folder, new_env)

        command_options = ""
        for t in args.set:
            command_options += " " + t
        command = (
            f"{sys.executable} -m nvflare.private.fed.app.client.worker_process -m "
            + args.workspace
            + " -w "
            + (workspace_obj.get_startup_kit_dir())
            + " -t "
            + client.token
            + " -d "
            + client.ssid
            + " -n "
            + job_id
            + " -c "
            + client.client_name
            + " -p "
            + str(client.cell.get_internal_listener_url())
            + " -g "
            + service.get("target")
            + " -scheme "
            + service.get("scheme", "grpc")
            + " -s fed_client.json "
            " --set" + command_options + " print_conf=True"
        )
        # use os.setsid to create new process group ID
        process = subprocess.Popen(shlex.split(command, True), preexec_fn=os.setsid, env=new_env)

        self.logger.info("Worker child process ID: {}".format(process.pid))

        return ProcessHandle(process)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.GET_JOB_LAUNCHER:
            job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
            job_image = extract_job_image(job_meta, fl_ctx.get_identity_name())
            if not job_image:
                job_launcher: list = fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER, [])
                job_launcher.append(self)
                fl_ctx.set_prop(FLContextKey.JOB_LAUNCHER, job_launcher, private=True, sticky=False)

