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

import copy
import json
import os
import shutil

import yaml

from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher, ServerDockerJobLauncher
from nvflare.lighter import utils
from nvflare.lighter.constants import CtxKey, PropKey, ProvFileName, TemplateSectionKey
from nvflare.lighter.spec import Builder, Project, ProvisionContext


class DockerBuilder(Builder):
    def __init__(
        self, docker_image="nvflare-docker:0.0.1", base_image="python:3.8", requirements_file="requirements.txt"
    ):
        """Build docker compose file."""
        self.docker_image = docker_image
        self.base_image = base_image
        self.requirements_file = requirements_file
        self.services = {}
        self.compose_file_path = None

    def _build_overseer(self, overseer):
        protocol = overseer.props.get("protocol", "http")
        default_port = "443" if protocol == "https" else "80"
        port = overseer.props.get("port", default_port)
        info_dict = copy.deepcopy(self.services["__overseer__"])
        info_dict["volumes"] = [f"./{overseer.name}:" + "${WORKSPACE}"]
        info_dict["ports"] = [f"{port}:{port}"]
        info_dict["build"] = "nvflare_compose"
        info_dict["container_name"] = overseer.name
        self.services[overseer.name] = info_dict

    def _build_server(self, server, ctx: ProvisionContext):
        fed_learn_port = ctx.get(CtxKey.FED_LEARN_PORT)
        admin_port = ctx.get(CtxKey.ADMIN_PORT)

        info_dict = copy.deepcopy(self.services["__flserver__"])
        info_dict["volumes"][0] = f"./{server.name}:" + "${WORKSPACE}"
        info_dict["ports"] = [f"{fed_learn_port}:{fed_learn_port}", f"{admin_port}:{admin_port}"]
        info_dict["build"] = "nvflare_compose"
        for i in range(len(info_dict["command"])):
            if info_dict["command"][i] == "flserver":
                info_dict["command"][i] = server.name
            if info_dict["command"][i] == "org=__org_name__":
                info_dict["command"][i] = f"org={server.org}"
        info_dict["container_name"] = server.name
        self.services[server.name] = info_dict

        # local folder creation
        dest_dir = ctx.get_local_dir(server)
        with open(os.path.join(dest_dir, ProvFileName.RESOURCES_JSON_DEFAULT), "rt") as f:
            resources = json.load(f)
            resources["components"].append(
                {
                    "id": "docker_launcher",
                    "path": ServerDockerJobLauncher().__module__ + "." + "ServerDockerJobLauncher",
                    "args": {},
                }
            )
        utils.write(os.path.join(dest_dir, ProvFileName.RESOURCES_JSON_DEFAULT), json.dumps(resources, indent=4), "t")

        communication_port = server.get_prop(CtxKey.DOCKER_COMM_PORT)
        if communication_port:
            replacement_dict = {"comm_host_name": "server-parent", "communication_port": communication_port}
            ctx.build_from_template(
                dest_dir,
                TemplateSectionKey.COMM_CONFIG,
                ProvFileName.COMM_CONFIG,
                replacement=replacement_dict,
                exe=True,
            )

    def _build_client(self, client, ctx: ProvisionContext):
        info_dict = copy.deepcopy(self.services["__flclient__"])
        info_dict["volumes"] = [f"./{client.name}:" + "${WORKSPACE}"]
        info_dict["build"] = "nvflare_compose"
        for i in range(len(info_dict["command"])):
            if info_dict["command"][i] == "flclient":
                info_dict["command"][i] = client.name
            if info_dict["command"][i] == "uid=__flclient__":
                info_dict["command"][i] = f"uid={client.name}"
            if info_dict["command"][i] == "org=__org_name__":
                info_dict["command"][i] = f"org={client.org}"
        info_dict["container_name"] = client.name
        self.services[client.name] = info_dict

        # local folder creation
        dest_dir = ctx.get_local_dir(client)
        with open(os.path.join(dest_dir, ProvFileName.RESOURCES_JSON_DEFAULT), "rt") as f:
            resources = json.load(f)
            resources["components"].append(
                {
                    "id": "docker_launcher",
                    "path": ClientDockerJobLauncher().__module__ + "." + "ClientDockerJobLauncher",
                    "args": {},
                }
            )
        utils.write(os.path.join(dest_dir, ProvFileName.RESOURCES_JSON_DEFAULT), json.dumps(resources, indent=4), "t")

        communication_port = client.get_prop(PropKey.DOCKER_COMM_PORT)
        if communication_port:
            replacement_dict = {"comm_host_name": client.name + "-parent", "communication_port": communication_port}
            ctx.build_from_template(
                dest_dir,
                TemplateSectionKey.COMM_CONFIG,
                ProvFileName.COMM_CONFIG,
                replacement=replacement_dict,
                exe=True,
            )

    def build(self, project: Project, ctx: ProvisionContext):
        compose = ctx.yaml_load_template_section(TemplateSectionKey.COMPOSE_YAML)
        self.services = compose.get("services")
        self.compose_file_path = os.path.join(ctx.get_wip_dir(), ProvFileName.COMPOSE_YAML)
        overseer = project.get_overseer()
        if overseer:
            self._build_overseer(overseer)
        server = project.get_server()
        if server:
            self._build_server(server, ctx)

        for client in project.get_clients():
            self._build_client(client, ctx)

        self.services.pop("__overseer__", None)
        self.services.pop("__flserver__", None)
        self.services.pop("__flclient__", None)
        compose["services"] = self.services
        with open(self.compose_file_path, "wt") as f:
            yaml.dump(compose, f)
        env_file_path = os.path.join(ctx.get_wip_dir(), ProvFileName.ENV)
        with open(env_file_path, "wt") as f:
            f.write("WORKSPACE=/workspace\n")
            f.write("PYTHON_EXECUTABLE=/usr/local/bin/python3\n")
            f.write("IMAGE_NAME=nvflare-service\n")
        compose_build_dir = os.path.join(ctx.get_wip_dir(), ProvFileName.COMPOSE_BUILD_DIR)
        os.mkdir(compose_build_dir)
        with open(os.path.join(compose_build_dir, ProvFileName.DOCKERFILE), "wt") as f:
            f.write(f"FROM {self.base_image}\n")
            f.write(ctx.get_template_section(TemplateSectionKey.DOCKERFILE))
        replacement_dict = {"image": self.docker_image}
        ctx.build_from_template(
            compose_build_dir,
            TemplateSectionKey.DOCKER_BUILD_SH,
            ProvFileName.DOCKER_BUILD_SH,
            replacement=replacement_dict,
            exe=True,
        )
        try:
            shutil.copyfile(self.requirements_file, os.path.join(compose_build_dir, ProvFileName.REQUIREMENTS_TXT))
        except Exception:
            f = open(os.path.join(compose_build_dir, ProvFileName.REQUIREMENTS_TXT), "wt")
            f.close()
