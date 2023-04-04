# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import yaml

from nvflare.lighter.spec import Builder
from nvflare.lighter.utils import sh_replace


class StaticFileBuilder(Builder):
    def __init__(
        self,
        enable_byoc=False,
        config_folder="",
        app_validator="",
        download_job_url="",
        docker_image="",
        snapshot_persistor="",
        overseer_agent="",
        components="",
    ):
        """Build all static files from template.

        Uses the information from project.yml through project to go through the participants and write the contents of
        each file with the template, and replacing with the appropriate values from project.yml.

        Usually, two main categories of files are created in all FL participants, static and dynamic. Static files
        have similar contents among different participants, with small differences.  For example, the differences in
        sub_start.sh are client name and python module.  Those are basically static files.  This builder uses template
        file and string replacement to generate those static files for each participant.

        Args:
            enable_byoc: for each participant, true to enable loading of code in the custom folder of applications
            config_folder: usually "config"
            app_validator: optional path to an app validator to verify that uploaded app has the expected structure
            docker_image: when docker_image is set to a docker image name, docker.sh will be generated on server/client/admin
        """
        self.enable_byoc = enable_byoc
        self.config_folder = config_folder
        self.docker_image = docker_image
        self.download_job_url = download_job_url
        self.app_validator = app_validator
        self.overseer_agent = overseer_agent
        self.snapshot_persistor = snapshot_persistor
        self.components = components

    def _write(self, file_full_path, content, mode, exe=False):
        mode = mode + "w"
        with open(file_full_path, mode) as f:
            f.write(content)
        if exe:
            os.chmod(file_full_path, 0o755)

    def _build_overseer(self, overseer, ctx):
        dest_dir = self.get_kit_dir(overseer, ctx)
        self._write(
            os.path.join(dest_dir, "start.sh"),
            self.template["start_svr_sh"],
            "t",
            exe=True,
        )
        protocol = overseer.props.get("protocol", "http")
        api_root = overseer.props.get("api_root", "/api/v1/")
        default_port = "443" if protocol == "https" else "80"
        port = overseer.props.get("port", default_port)
        replacement_dict = {"port": port, "hostname": overseer.name}
        admins = self.project.get_participants_by_type("admin", first_only=False)
        privilege_dict = dict()
        for admin in admins:
            role = admin.props.get("role")
            if role in privilege_dict:
                privilege_dict[role].append(admin.subject)
            else:
                privilege_dict[role] = [admin.subject]
        self._write(
            os.path.join(dest_dir, "privilege.yml"),
            yaml.dump(privilege_dict, Dumper=yaml.Dumper),
            "t",
            exe=False,
        )

        if self.docker_image:
            self._write(
                os.path.join(dest_dir, "docker.sh"),
                sh_replace(self.template["docker_svr_sh"], replacement_dict),
                "t",
                exe=True,
            )
        self._write(
            os.path.join(dest_dir, "gunicorn.conf.py"),
            sh_replace(self.template["gunicorn_conf_py"], replacement_dict),
            "t",
            exe=False,
        )
        self._write(
            os.path.join(dest_dir, "start.sh"),
            self.template["start_ovsr_sh"],
            "t",
            exe=True,
        )
        if port:
            ctx["overseer_end_point"] = f"{protocol}://{overseer.name}:{port}{api_root}"
        else:
            ctx["overseer_end_point"] = f"{protocol}://{overseer.name}{api_root}"

    def _build_server(self, server, ctx):
        config = json.loads(self.template["fed_server"])
        dest_dir = self.get_kit_dir(server, ctx)
        server_0 = config["servers"][0]
        server_0["name"] = self.project_name
        admin_port = server.props.get("admin_port", 8003)
        ctx["admin_port"] = admin_port
        fed_learn_port = server.props.get("fed_learn_port", 8002)
        ctx["fed_learn_port"] = fed_learn_port
        ctx["server_name"] = server.name
        server_0["service"]["target"] = f"{server.name}:{fed_learn_port}"
        server_0["admin_host"] = server.name
        server_0["admin_port"] = admin_port
        # if self.download_job_url:
        #     server_0["download_job_url"] = self.download_job_url
        # config["enable_byoc"] = server.enable_byoc
        # if self.app_validator:
        #     config["app_validator"] = {"path": self.app_validator}
        if self.overseer_agent:
            overseer_agent = copy.deepcopy(self.overseer_agent)
            if overseer_agent.get("overseer_exists", True):
                overseer_agent["args"] = {
                    "role": "server",
                    "overseer_end_point": ctx.get("overseer_end_point", ""),
                    "project": self.project_name,
                    "name": server.name,
                    "fl_port": str(fed_learn_port),
                    "admin_port": str(admin_port),
                }
            overseer_agent.pop("overseer_exists", None)
            config["overseer_agent"] = overseer_agent
        # if self.snapshot_persistor:
        #     config["snapshot_persistor"] = self.snapshot_persistor
        # components = server.props.get("components", [])
        # config["components"] = list()
        # for comp in components:
        #     temp_dict = {"id": comp}
        #     temp_dict.update(components[comp])
        #     config["components"].append(temp_dict)
        # provisioned_client_list = list()
        # for client in self.project.get_participants_by_type("client", first_only=False):
        #     provisioned_client_list.append(client.name)
        # config["provisioned_client_list"] = provisioned_client_list
        self._write(os.path.join(dest_dir, "fed_server.json"), json.dumps(config, indent=2), "t")
        replacement_dict = {
            "admin_port": admin_port,
            "fed_learn_port": fed_learn_port,
            "config_folder": self.config_folder,
            "docker_image": self.docker_image,
            "org_name": server.org,
        }
        if self.docker_image:
            self._write(
                os.path.join(dest_dir, "docker.sh"),
                sh_replace(self.template["docker_svr_sh"], replacement_dict),
                "t",
                exe=True,
            )
        self._write(
            os.path.join(dest_dir, "start.sh"),
            self.template["start_svr_sh"],
            "t",
            exe=True,
        )
        self._write(
            os.path.join(dest_dir, "sub_start.sh"),
            sh_replace(self.template["sub_start_svr_sh"], replacement_dict),
            "t",
            exe=True,
        )
        self._write(
            os.path.join(dest_dir, "stop_fl.sh"),
            self.template["stop_fl_sh"],
            "t",
            exe=True,
        )
        # local folder creation
        dest_dir = self.get_local_dir(server, ctx)
        self._write(
            os.path.join(dest_dir, "log.config.default"),
            self.template["log_config"],
            "t",
        )
        self._write(
            os.path.join(dest_dir, "resources.json.default"),
            self.template["local_server_resources"],
            "t",
        )
        self._write(
            os.path.join(dest_dir, "privacy.json.sample"),
            self.template["sample_privacy"],
            "t",
        )
        self._write(
            os.path.join(dest_dir, "authorization.json.default"),
            self.template["default_authz"],
            "t",
        )

        # workspace folder file
        self._write(
            os.path.join(self.get_ws_dir(server, ctx), "readme.txt"),
            self.template["readme_fs"],
            "t",
        )

    def _build_client(self, client, ctx):
        config = json.loads(self.template["fed_client"])
        dest_dir = self.get_kit_dir(client, ctx)
        fed_learn_port = ctx.get("fed_learn_port")
        server_name = ctx.get("server_name")
        # config["servers"][0]["service"]["target"] = f"{server_name}:{fed_learn_port}"
        config["servers"][0]["name"] = self.project_name
        # config["enable_byoc"] = client.enable_byoc
        replacement_dict = {
            "client_name": f"{client.subject}",
            "config_folder": self.config_folder,
            "docker_image": self.docker_image,
            "org_name": client.org,
        }
        if self.overseer_agent:
            overseer_agent = copy.deepcopy(self.overseer_agent)
            if overseer_agent.get("overseer_exists", True):
                overseer_agent["args"] = {
                    "role": "client",
                    "overseer_end_point": ctx.get("overseer_end_point", ""),
                    "project": self.project_name,
                    "name": client.subject,
                }
            overseer_agent.pop("overseer_exists", None)
            config["overseer_agent"] = overseer_agent
        # components = client.props.get("components", [])
        # config["components"] = list()
        # for comp in components:
        #     temp_dict = {"id": comp}
        #     temp_dict.update(components[comp])
        #     config["components"].append(temp_dict)

        self._write(os.path.join(dest_dir, "fed_client.json"), json.dumps(config, indent=2), "t")
        if self.docker_image:
            self._write(
                os.path.join(dest_dir, "docker.sh"),
                sh_replace(self.template["docker_cln_sh"], replacement_dict),
                "t",
                exe=True,
            )
        self._write(
            os.path.join(dest_dir, "start.sh"),
            self.template["start_cln_sh"],
            "t",
            exe=True,
        )
        self._write(
            os.path.join(dest_dir, "sub_start.sh"),
            sh_replace(self.template["sub_start_cln_sh"], replacement_dict),
            "t",
            exe=True,
        )
        self._write(
            os.path.join(dest_dir, "stop_fl.sh"),
            self.template["stop_fl_sh"],
            "t",
            exe=True,
        )
        # local folder creation
        dest_dir = self.get_local_dir(client, ctx)
        self._write(
            os.path.join(dest_dir, "log.config.default"),
            self.template["log_config"],
            "t",
        )
        self._write(
            os.path.join(dest_dir, "resources.json.default"),
            self.template["local_client_resources"],
            "t",
        )
        self._write(
            os.path.join(dest_dir, "privacy.json.sample"),
            self.template["sample_privacy"],
            "t",
        )
        self._write(
            os.path.join(dest_dir, "authorization.json.default"),
            self.template["default_authz"],
            "t",
        )

        # workspace folder file
        self._write(
            os.path.join(self.get_ws_dir(client, ctx), "readme.txt"),
            self.template["readme_fc"],
            "t",
        )

    def _build_admin(self, admin, ctx):
        config = json.loads(self.template["fed_admin"])
        dest_dir = self.get_kit_dir(admin, ctx)
        admin_port = ctx.get("admin_port")
        server_name = ctx.get("server_name")

        replacement_dict = {
            "cn": f"{server_name}",
            "admin_port": f"{admin_port}",
            "docker_image": self.docker_image,
        }
        agent_config = dict()
        if self.overseer_agent:
            overseer_agent = copy.deepcopy(self.overseer_agent)
            if overseer_agent.get("overseer_exists", True):
                overseer_agent["args"] = {
                    "role": "admin",
                    "overseer_end_point": ctx.get("overseer_end_point", ""),
                    "project": self.project_name,
                    "name": admin.subject,
                }
            overseer_agent.pop("overseer_exists", None)
            agent_config["overseer_agent"] = overseer_agent
        config["admin"].update(agent_config)
        self._write(os.path.join(dest_dir, "fed_admin.json"), json.dumps(config, indent=2), "t")
        if self.docker_image:
            self._write(
                os.path.join(dest_dir, "docker.sh"),
                sh_replace(self.template["docker_adm_sh"], replacement_dict),
                "t",
                exe=True,
            )
        self._write(
            os.path.join(dest_dir, "fl_admin.sh"),
            sh_replace(self.template["fl_admin_sh"], replacement_dict),
            "t",
            exe=True,
        )
        self._write(
            os.path.join(dest_dir, "readme.txt"),
            self.template["readme_am"],
            "t",
        )

    def build(self, project, ctx):
        self.template = ctx.get("template")
        self.project_name = project.name
        self.project = project
        overseer = project.get_participants_by_type("overseer")
        if overseer:
            self._build_overseer(overseer, ctx)
        servers = project.get_participants_by_type("server", first_only=False)
        for server in servers:
            self._build_server(server, ctx)

        for client in project.get_participants_by_type("client", first_only=False):
            self._build_client(client, ctx)

        for admin in project.get_participants_by_type("admin", first_only=False):
            self._build_admin(admin, ctx)
