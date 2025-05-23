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
import time
from abc import abstractmethod

import docker
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobReturnCode, add_launcher
from nvflare.apis.workspace import Workspace
from nvflare.utils.job_launcher_utils import extract_job_image, generate_client_command, generate_server_command


class DOCKER_STATE:
    CREATED = "created"
    RESTARTING = "restarting"
    RUNNING = "running"
    PAUSED = "paused"
    EXITED = "exited"
    DEAD = "dead"


JOB_RETURN_CODE_MAPPING = {
    DOCKER_STATE.CREATED: JobReturnCode.UNKNOWN,
    DOCKER_STATE.RESTARTING: JobReturnCode.UNKNOWN,
    DOCKER_STATE.RUNNING: JobReturnCode.UNKNOWN,
    DOCKER_STATE.PAUSED: JobReturnCode.UNKNOWN,
    DOCKER_STATE.EXITED: JobReturnCode.SUCCESS,
    DOCKER_STATE.DEAD: JobReturnCode.ABORTED,
}


class DockerJobHandle(JobHandleSpec):
    def __init__(self, container, timeout=None):
        super().__init__()

        self.container = container
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)

    def terminate(self):
        if self.container:
            self.container.stop()

    def poll(self):
        container = self._get_container()
        if container:
            if container.status in [DOCKER_STATE.EXITED, DOCKER_STATE.DEAD]:
                container.remove(force=True)
                self.logger.debug(f"docker completes state: {container.status}")
            return JOB_RETURN_CODE_MAPPING.get(container.status, JobReturnCode.UNKNOWN)

    def wait(self):
        if self.container:
            self.enter_states([DOCKER_STATE.EXITED, DOCKER_STATE.DEAD], self.timeout)

    def _get_container(self):
        try:
            docker_client = docker.from_env()
            # Get the container object
            container = docker_client.containers.get(self.container.id)
            # Get the container state
            # state = container.attrs['State']
            return container
        except:
            return None

    def enter_states(self, job_states_to_enter: list, timeout=None):
        starting_time = time.time()
        if not isinstance(job_states_to_enter, (list, tuple)):
            job_states_to_enter = [job_states_to_enter]
        while True:
            container = self._get_container()
            if container:
                self.logger.debug(f"container state: {container.status}, job states to enter: {job_states_to_enter}")
                if container.status in job_states_to_enter:
                    return True
                elif timeout is not None and time.time() - starting_time > timeout:
                    return False
                time.sleep(1)
            else:
                return False


class DockerJobLauncher(JobLauncherSpec):
    def __init__(self, mount_path: str = "/workspace", network: str = "nvflare-network", timeout=None):
        super().__init__()

        self.mount_path = mount_path
        self.network = network
        self.timeout = timeout

    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:
        self.logger.debug("DockerJobLauncher start to launch job")
        job_image = extract_job_image(job_meta, fl_ctx.get_identity_name())

        workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        job_id = job_meta.get(JobConstants.JOB_ID)
        app_custom_folder = workspace_obj.get_app_custom_dir(job_id)

        python_path = f"{app_custom_folder}:$PYTHONPATH" if app_custom_folder != "" else "$PYTHONPATH"
        job_name, cmd = self.get_command(job_meta, fl_ctx)
        command = f' /bin/bash -c "export PYTHONPATH={python_path};{cmd}"'
        self.logger.info(f"Launch image:{job_image}, run command: {command}")

        docker_workspace = os.environ.get("NVFL_DOCKER_WORKSPACE")
        self.logger.info(f"launch_job {job_id} in docker_workspace: {docker_workspace}")
        docker_client = docker.from_env()
        try:
            container = docker_client.containers.run(
                job_image,
                command=command,
                name=job_name,
                network=self.network,
                detach=True,
                # remove=True,
                volumes={
                    docker_workspace: {
                        "bind": self.mount_path,
                        "mode": "rw",
                    },
                },
                # ports=ports,  # Map container ports to host ports (optional)
            )
            self.logger.info(f"Launch the job in DockerJobLauncher using image: {job_image}")

            handle = DockerJobHandle(container)
            try:
                if handle.enter_states([DOCKER_STATE.RUNNING], timeout=self.timeout):
                    return handle
                else:
                    handle.terminate()
                    return None
            except:
                handle.terminate()
                return None

        except docker.errors.ImageNotFound:
            self.logger.error(f"Failed to launcher job: {job_id} in DockerJobLauncher. Image '{job_image}' not found.")
            return None
        except docker.errors.APIError as e:
            self.logger.error(f"Error starting container: {e}")
            return None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.BEFORE_JOB_LAUNCH:
            job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
            job_image = extract_job_image(job_meta, fl_ctx.get_identity_name())
            if job_image:
                add_launcher(self, fl_ctx)

    @abstractmethod
    def get_command(self, job_meta, fl_ctx) -> (str, str):
        """To generate the command to launcher the job in sub-process

        Args:
            fl_ctx: FLContext
            job_meta: job launcher data

        Returns:
            (container name, launch command)

        """
        pass


class ClientDockerJobLauncher(DockerJobLauncher):
    def get_command(self, job_meta, fl_ctx) -> (str, str):
        job_id = job_meta.get(JobConstants.JOB_ID)
        client_name = fl_ctx.get_identity_name()
        command = generate_client_command(fl_ctx)

        return f"{client_name}-{job_id}", command


class ServerDockerJobLauncher(DockerJobLauncher):
    def get_command(self, job_meta, fl_ctx) -> (str, str):
        job_id = job_meta.get(JobConstants.JOB_ID)
        command = generate_server_command(fl_ctx)

        return f"server-{job_id}", command
