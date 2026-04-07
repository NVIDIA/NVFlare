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
from nvflare.lighter.constants import CtxKey, PropKey, ProvFileName, TemplateSectionKey
from nvflare.lighter.entity import Participant
from nvflare.lighter.spec import Builder, Project, ProvisionContext


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

    def __init__(self, docker_image: str = "nvflare:latest"):
        """
        Args:
            docker_image: Docker image name for SP/CP containers. The site admin must
                          build and tag this image before running start_docker.sh.
                          Job images (SJ/CJ) are specified per job in meta.json.
        """
        self.docker_image = docker_image

    def _inject_launcher(self, dest_dir: str, path: str, args: dict):
        """Inject DockerJobLauncher component into resources.json."""
        resources_file = os.path.join(dest_dir, ProvFileName.RESOURCES_JSON_DEFAULT)
        with open(resources_file, "rt") as f:
            resources = json.load(f)
        resources["components"].append({"id": "docker_launcher", "path": path, "args": args})
        utils.write(resources_file, json.dumps(resources, indent=4), "t")

    def _build_server(self, server: Participant, ctx: ProvisionContext):
        fed_learn_port = ctx.get(CtxKey.FED_LEARN_PORT)

        lh = server.get_listening_host()

        # Inject launcher config — workspace resolved at runtime from NVFL_DOCKER_WORKSPACE
        dest_dir = ctx.get_local_dir(server)
        self._inject_launcher(
            dest_dir,
            path=ServerDockerJobLauncher.__module__ + ".ServerDockerJobLauncher",
            args={
                "network": "nvflare-network",
                "mount_path": "/var/nvflare/workspace",
                "python_path": "/usr/local/bin/python",
                "pending_timeout": 30,
            },
        )

        run_in_docker = server.get_prop(PropKey.RUN_IN_DOCKER)
        if run_in_docker:
            if not lh:
                raise RuntimeError(f"run_in_docker requires listening_host but it's missing from {server.name}")
            if not lh.port:
                raise RuntimeError(f"run_in_docker requires listening_host.port but it's missing from {server.name}")

            dest_dir = ctx.get_kit_dir(server)
            ctx.build_from_template(
                dest_dir,
                TemplateSectionKey.DOCKER_LAUNCHER_SERVER_SH,
                ProvFileName.DOCKER_LAUNCHER_SH,
                replacement={
                    "fed_learn_port": fed_learn_port,
                    "communication_port": lh.port,
                    "server_name": server.name,
                    "docker_image": self.docker_image,
                },
                exe=True,
            )

    def _build_client(self, client: Participant, ctx: ProvisionContext):
        fed_learn_port = ctx.get(CtxKey.FED_LEARN_PORT)

        lh = client.get_listening_host()

        # Inject launcher config — workspace resolved at runtime from NVFL_DOCKER_WORKSPACE
        dest_dir = ctx.get_local_dir(client)
        self._inject_launcher(
            dest_dir,
            path=ClientDockerJobLauncher.__module__ + ".ClientDockerJobLauncher",
            args={
                "network": "nvflare-network",
                "mount_path": "/var/nvflare/workspace",
                "python_path": "/usr/local/bin/python",
                "pending_timeout": 30,
            },
        )

        run_in_docker = client.get_prop(PropKey.RUN_IN_DOCKER)
        if run_in_docker:
            if not lh:
                raise RuntimeError(f"run_in_docker requires listening_host but it's missing from {client.name}")
            if not lh.port:
                raise RuntimeError(f"run_in_docker requires listening_host.port but it's missing from {client.name}")

            dest_dir = ctx.get_kit_dir(client)
            ctx.build_from_template(
                dest_dir,
                TemplateSectionKey.DOCKER_LAUNCHER_CLIENT_SH,
                ProvFileName.DOCKER_LAUNCHER_SH,
                replacement={
                    "fed_learn_port": fed_learn_port,
                    "communication_port": lh.port,
                    "docker_image": self.docker_image,
                    "client_name": client.name,
                },
                exe=True,
            )

    def initialize(self, project: Project, ctx: ProvisionContext):
        ctx.load_templates("docker_launcher_template.yml")

    def build(self, project: Project, ctx: ProvisionContext):
        server = project.get_server()
        if server:
            self._build_server(server, ctx)

        for client in project.get_clients():
            self._build_client(client, ctx)
