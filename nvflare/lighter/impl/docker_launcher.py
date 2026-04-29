# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os

from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher, ServerDockerJobLauncher
from nvflare.lighter import utils
from nvflare.lighter.constants import CommConfigArg, CtxKey, PropKey, ProvFileName, TemplateSectionKey
from nvflare.lighter.entity import Participant
from nvflare.lighter.spec import Builder, Project, ProvisionContext

_LAUNCHER_IDS = {"process_launcher", "docker_launcher", "k8s_launcher"}
_PASSTHROUGH_RESOURCE_MANAGER_PATH = (
    "nvflare.app_common.resource_managers.passthrough_resource_manager.PassthroughResourceManager"
)


class DockerLauncherBuilder(Builder):
    """Generates start_docker.sh per site and injects DockerJobLauncher into resources.json.

    The site admin is responsible for building the Docker image before deployment.
    This builder only needs the image name to embed in start_docker.sh.

    Usage in project.yml:
        - path: nvflare.lighter.impl.docker_launcher.DockerLauncherBuilder
          args:
            docker_image: my-nvflare-image:latest

    Each participant that has ``run_in_docker: true`` in its props will get a
    ``startup/start_docker.sh`` generated alongside the standard ``startup/start.sh``.
    Running start_docker.sh starts the SP/CP container in Docker mode; job containers
    (SJ/CJ) are then launched automatically by DockerJobLauncher when jobs are submitted.
    """

    def __init__(
        self,
        docker_image: str = "nvflare:latest",
        network: str = "nvflare-network",
        python_path: str = "/usr/local/bin/python",
        default_job_container_kwargs: dict = None,
        default_job_env: dict = None,
        set_passthrough_resource_manager: bool = True,
    ):
        """
        Args:
            docker_image: Docker image name for SP/CP containers. The site admin must
                          build and tag this image before running start_docker.sh.
                          Job images (SJ/CJ) are specified per job in meta.json.
            network: Docker network name shared by the site container and per-job containers.
            python_path: Python executable path inside the job containers.
            default_job_container_kwargs: Default container kwargs passed to DockerJobLauncher.
            default_job_env: Default environment variables passed to DockerJobLauncher.
            set_passthrough_resource_manager: whether to replace process-mode resource
                          scheduling components with PassthroughResourceManager.
        """
        self.docker_image = docker_image
        self.network = network
        self.python_path = python_path
        self.default_job_container_kwargs = default_job_container_kwargs or {}
        self.default_job_env = default_job_env or {}
        self.set_passthrough_resource_manager = set_passthrough_resource_manager

    def _inject_launcher(self, dest_dir: str, path: str, args: dict):
        """Replace any existing job launcher component with DockerJobLauncher."""
        resources_file = os.path.join(dest_dir, ProvFileName.RESOURCES_JSON_DEFAULT)
        with open(resources_file, "rt") as f:
            resources = json.load(f)

        components = resources.get("components", [])
        resources["components"] = [c for c in components if c.get("id") not in _LAUNCHER_IDS]
        resources["components"].append({"id": "docker_launcher", "path": path, "args": args})
        utils.write(resources_file, json.dumps(resources, indent=4), "t")

    def _set_resource_manager(self, dest_dir: str):
        """Replace process-mode resource scheduling components for Docker runtime."""
        resources_file = os.path.join(dest_dir, ProvFileName.RESOURCES_JSON_DEFAULT)
        with open(resources_file, "rt") as f:
            resources = json.load(f)

        components = resources.get("components", [])
        new_components = []
        found_resource_manager = False
        for component in components:
            component_id = component.get("id")
            if component_id == "resource_consumer":
                continue
            if component_id == "resource_manager":
                component = {
                    "id": "resource_manager",
                    "path": _PASSTHROUGH_RESOURCE_MANAGER_PATH,
                    "args": {},
                }
                found_resource_manager = True
            new_components.append(component)

        if not found_resource_manager:
            new_components.insert(
                0,
                {
                    "id": "resource_manager",
                    "path": _PASSTHROUGH_RESOURCE_MANAGER_PATH,
                    "args": {},
                },
            )

        resources["components"] = new_components
        utils.write(resources_file, json.dumps(resources, indent=4), "t")

    def _set_internal_listener_host(self, participant: Participant):
        """Override internal listener host to 0.0.0.0 so SJ/CJ containers on the Docker network can connect."""
        comm_config_args = participant.get_prop(PropKey.COMM_CONFIG_ARGS)
        if comm_config_args is not None:
            comm_config_args[CommConfigArg.HOST] = "0.0.0.0"

    def _build_server(self, server: Participant, ctx: ProvisionContext):
        fed_learn_port = ctx.get(CtxKey.FED_LEARN_PORT)

        # Inject launcher config — workspace resolved at runtime from NVFL_DOCKER_WORKSPACE
        dest_dir = ctx.get_local_dir(server)
        if self.set_passthrough_resource_manager:
            self._set_resource_manager(dest_dir)
        self._inject_launcher(
            dest_dir,
            path=ServerDockerJobLauncher.__module__ + ".ServerDockerJobLauncher",
            args={
                "network": self.network,
                "python_path": self.python_path,
                "default_job_container_kwargs": self.default_job_container_kwargs,
                "default_job_env": self.default_job_env,
            },
        )

        # Auto-inject 0.0.0.0 binding so SJ containers can reach SP via Docker DNS
        self._set_internal_listener_host(server)

        dest_dir = ctx.get_kit_dir(server)
        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.DOCKER_LAUNCHER_SERVER_SH,
            ProvFileName.DOCKER_LAUNCHER_SH,
            replacement={
                "fed_learn_port": fed_learn_port,
                "server_name": server.name,
                "docker_image": self.docker_image,
                "network": self.network,
            },
            exe=True,
        )

    def _build_client(self, client: Participant, ctx: ProvisionContext):
        fed_learn_port = ctx.get(CtxKey.FED_LEARN_PORT)

        # Inject launcher config — workspace resolved at runtime from NVFL_DOCKER_WORKSPACE
        dest_dir = ctx.get_local_dir(client)
        if self.set_passthrough_resource_manager:
            self._set_resource_manager(dest_dir)
        self._inject_launcher(
            dest_dir,
            path=ClientDockerJobLauncher.__module__ + ".ClientDockerJobLauncher",
            args={
                "network": self.network,
                "python_path": self.python_path,
                "default_job_container_kwargs": self.default_job_container_kwargs,
                "default_job_env": self.default_job_env,
            },
        )

        # Auto-inject 0.0.0.0 binding so CJ containers can reach CP via Docker DNS
        self._set_internal_listener_host(client)

        dest_dir = ctx.get_kit_dir(client)
        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.DOCKER_LAUNCHER_CLIENT_SH,
            ProvFileName.DOCKER_LAUNCHER_SH,
            replacement={
                "fed_learn_port": fed_learn_port,
                "docker_image": self.docker_image,
                "client_name": client.name,
                "network": self.network,
            },
            exe=True,
        )

    def initialize(self, project: Project, ctx: ProvisionContext):
        ctx.load_templates("docker_launcher_template.yml")

    def build(self, project: Project, ctx: ProvisionContext):
        fed_learn_port = ctx.get(CtxKey.FED_LEARN_PORT)
        admin_port = ctx.get(CtxKey.ADMIN_PORT)
        if admin_port != fed_learn_port:
            raise ValueError(
                f"Docker mode requires fed_learn_port == admin_port, "
                f"but got fed_learn_port={fed_learn_port}, admin_port={admin_port}. "
                f"Remove the explicit admin_port from project.yml or set it to match fed_learn_port."
            )

        server = project.get_server()
        if server:
            self._build_server(server, ctx)

        for client in project.get_clients():
            self._build_client(client, ctx)
