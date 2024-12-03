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

from nvflare.lighter import utils
from nvflare.lighter.spec import Builder, Participant


class StaticFileBuilder(Builder):
    def __init__(
        self,
        enable_byoc=False,
        config_folder="",
        scheme="grpc",
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
        self.scheme = scheme
        self.docker_image = docker_image
        self.download_job_url = download_job_url
        self.app_validator = app_validator
        self.overseer_agent = overseer_agent
        self.snapshot_persistor = snapshot_persistor
        self.components = components

    def get_server_name(self, server):
        return server.name

    def get_overseer_name(self, overseer):
        return overseer.name

    def _build_overseer(self, overseer, ctx):
        dest_dir = self.get_kit_dir(overseer, ctx)
        utils._write(
            os.path.join(dest_dir, "start.sh"),
            self.template["start_svr_sh"],
            "t",
            exe=True,
        )
        protocol = overseer.props.get("protocol", "http")
        api_root = overseer.props.get("api_root", "/api/v1/")
        default_port = "443" if protocol == "https" else "80"
        port = overseer.props.get("port", default_port)
        replacement_dict = {"port": port, "hostname": self.get_overseer_name(overseer)}
        admins = self.project.get_participants_by_type("admin", first_only=False)
        privilege_dict = dict()
        for admin in admins:
            role = admin.props.get("role")
            if role in privilege_dict:
                privilege_dict[role].append(admin.subject)
            else:
                privilege_dict[role] = [admin.subject]
        utils._write(
            os.path.join(dest_dir, "privilege.yml"),
            yaml.dump(privilege_dict, Dumper=yaml.Dumper),
            "t",
            exe=False,
        )

        if self.docker_image:
            utils._write(
                os.path.join(dest_dir, "docker.sh"),
                utils.sh_replace(self.template["docker_svr_sh"], replacement_dict),
                "t",
                exe=True,
            )
        utils._write(
            os.path.join(dest_dir, "gunicorn.conf.py"),
            utils.sh_replace(self.template["gunicorn_conf_py"], replacement_dict),
            "t",
            exe=False,
        )
        utils._write(
            os.path.join(dest_dir, "start.sh"),
            self.template["start_ovsr_sh"],
            "t",
            exe=True,
        )
        if port:
            ctx["overseer_end_point"] = f"{protocol}://{self.get_overseer_name(overseer)}:{port}{api_root}"
        else:
            ctx["overseer_end_point"] = f"{protocol}://{self.get_overseer_name(overseer)}{api_root}"

    def _build_server(self, server, ctx):
        config = json.loads(self.template["fed_server"])
        dest_dir = self.get_kit_dir(server, ctx)
        server_0 = config["servers"][0]
        server_0["name"] = self.project_name
        admin_port = server.get_prop("admin_port", 8003)
        ctx["admin_port"] = admin_port
        fed_learn_port = server.get_prop("fed_learn_port", 8002)
        ctx["fed_learn_port"] = fed_learn_port
        ctx["server_name"] = self.get_server_name(server)
        server_0["service"]["target"] = f"{self.get_server_name(server)}:{fed_learn_port}"
        server_0["service"]["scheme"] = self.scheme
        server_0["admin_host"] = self.get_server_name(server)
        server_0["admin_port"] = admin_port

        self._prepare_overseer_agent(server, config, "server", ctx)

        utils._write(os.path.join(dest_dir, "fed_server.json"), json.dumps(config, indent=2), "t")
        replacement_dict = {
            "admin_port": admin_port,
            "fed_learn_port": fed_learn_port,
            "config_folder": self.config_folder,
            "docker_image": self.docker_image,
            "org_name": server.org,
            "type": "server",
            "cln_uid": "",
        }
        if self.docker_image:
            utils._write(
                os.path.join(dest_dir, "docker.sh"),
                utils.sh_replace(self.template["docker_svr_sh"], replacement_dict),
                "t",
                exe=True,
            )
        utils._write(
            os.path.join(dest_dir, "start.sh"),
            self.template["start_svr_sh"],
            "t",
            exe=True,
        )
        utils._write(
            os.path.join(dest_dir, "sub_start.sh"),
            utils.sh_replace(self.template["sub_start_sh"], replacement_dict),
            "t",
            exe=True,
        )
        utils._write(
            os.path.join(dest_dir, "stop_fl.sh"),
            self.template["stop_fl_sh"],
            "t",
            exe=True,
        )
        # local folder creation
        dest_dir = self.get_local_dir(server, ctx)
        utils._write(
            os.path.join(dest_dir, "log.config.default"),
            self.template["log_config"],
            "t",
        )
        utils._write(
            os.path.join(dest_dir, "resources.json.default"),
            self.template["local_server_resources"],
            "t",
        )
        utils._write(
            os.path.join(dest_dir, "privacy.json.sample"),
            self.template["sample_privacy"],
            "t",
        )
        utils._write(
            os.path.join(dest_dir, "authorization.json.default"),
            self.template["default_authz"],
            "t",
        )

        # workspace folder file
        utils._write(
            os.path.join(self.get_ws_dir(server, ctx), "readme.txt"),
            self.template["readme_fs"],
            "t",
        )

    def _build_client(self, client, ctx):
        project = ctx["project"]
        server = project.get_server()
        if not server:
            raise ValueError("missing server definition in project")
        config = json.loads(self.template["fed_client"])
        dest_dir = self.get_kit_dir(client, ctx)
        config["servers"][0]["service"]["scheme"] = self.scheme
        config["servers"][0]["name"] = self.project_name
        config["servers"][0]["identity"] = server.name  # the official identity of the server
        replacement_dict = {
            "client_name": f"{client.subject}",
            "config_folder": self.config_folder,
            "docker_image": self.docker_image,
            "org_name": client.org,
            "type": "client",
            "cln_uid": f"uid={client.subject}",
        }

        self._prepare_overseer_agent(client, config, "client", ctx)

        utils._write(os.path.join(dest_dir, "fed_client.json"), json.dumps(config, indent=2), "t")
        if self.docker_image:
            utils._write(
                os.path.join(dest_dir, "docker.sh"),
                utils.sh_replace(self.template["docker_cln_sh"], replacement_dict),
                "t",
                exe=True,
            )
        utils._write(
            os.path.join(dest_dir, "start.sh"),
            self.template["start_cln_sh"],
            "t",
            exe=True,
        )
        utils._write(
            os.path.join(dest_dir, "sub_start.sh"),
            utils.sh_replace(self.template["sub_start_sh"], replacement_dict),
            "t",
            exe=True,
        )
        utils._write(
            os.path.join(dest_dir, "stop_fl.sh"),
            self.template["stop_fl_sh"],
            "t",
            exe=True,
        )
        # local folder creation
        dest_dir = self.get_local_dir(client, ctx)
        utils._write(
            os.path.join(dest_dir, "log.config.default"),
            self.template["log_config"],
            "t",
        )
        utils._write(
            os.path.join(dest_dir, "resources.json.default"),
            self.template["local_client_resources"],
            "t",
        )
        utils._write(
            os.path.join(dest_dir, "privacy.json.sample"),
            self.template["sample_privacy"],
            "t",
        )
        utils._write(
            os.path.join(dest_dir, "authorization.json.default"),
            self.template["default_authz"],
            "t",
        )

        # workspace folder file
        utils._write(
            os.path.join(self.get_ws_dir(client, ctx), "readme.txt"),
            self.template["readme_fc"],
            "t",
        )

    def _check_host_name(self, host_name: str, server: Participant) -> str:
        if host_name == server.get_default_host():
            # Use the default host - OK
            return ""

        available_host_names = server.get_host_names()
        if available_host_names and host_name in available_host_names:
            # use alternative host name - OK
            return ""

        return f"unknown host name '{host_name}'"

    def _prepare_overseer_agent(self, participant, config, role, ctx):
        project = ctx["project"]
        server = project.get_server()
        if not server:
            raise ValueError(f"Missing server definition in project {project.name}")

        fl_port = server.get_prop("fed_learn_port", 8002)
        admin_port = server.get_prop("admin_port", 8003)

        if self.overseer_agent:
            overseer_agent = copy.deepcopy(self.overseer_agent)
            if overseer_agent.get("overseer_exists", True):
                if role == "server":
                    overseer_agent["args"] = {
                        "role": role,
                        "overseer_end_point": ctx.get("overseer_end_point", ""),
                        "project": self.project_name,
                        "name": server.name,
                        "fl_port": str(fl_port),
                        "admin_port": str(admin_port),
                    }
                else:
                    overseer_agent["args"] = {
                        "role": role,
                        "overseer_end_point": ctx.get("overseer_end_point", ""),
                        "project": self.project_name,
                        "name": participant.subject,
                    }
            else:
                # do not use overseer system
                # Dummy overseer agent is used here
                if role == "server":
                    # the server expects the "connect_to" to be the same as its name
                    # otherwise the host name generated by the dummy agent won't be accepted!
                    connect_to = server.name
                else:
                    connect_to = participant.get_connect_to()
                    if connect_to:
                        err = self._check_host_name(connect_to, server)
                        if err:
                            raise ValueError(f"bad connect_to in {participant.subject}: {err}")
                    else:
                        # connect_to is not explicitly specified: use the server's name by default
                        # Note: by doing this dynamically, we guarantee the sp_end_point to be correct, even if the
                        # project.yaml does not specify the default server host correctly!
                        connect_to = server.get_default_host()

                # change the sp_end_point to use connect_to
                agent_args = overseer_agent.get("args")
                if agent_args:
                    sp_end_point = agent_args.get("sp_end_point")
                    if sp_end_point:
                        # format of the sp_end_point:  server_host_name:fl_port:admin_port
                        agent_args["sp_end_point"] = f"{connect_to}:{fl_port}:{admin_port}"

            overseer_agent.pop("overseer_exists", None)
            config["overseer_agent"] = overseer_agent

    def _build_admin(self, admin, ctx):
        dest_dir = self.get_kit_dir(admin, ctx)
        admin_port = ctx.get("admin_port")
        server_name = ctx.get("server_name")

        replacement_dict = {
            "cn": f"{server_name}",
            "admin_port": f"{admin_port}",
            "docker_image": self.docker_image,
        }

        config = self.prepare_admin_config(admin, ctx)

        utils._write(os.path.join(dest_dir, "fed_admin.json"), json.dumps(config, indent=2), "t")
        if self.docker_image:
            utils._write(
                os.path.join(dest_dir, "docker.sh"),
                utils.sh_replace(self.template["docker_adm_sh"], replacement_dict),
                "t",
                exe=True,
            )
        utils._write(
            os.path.join(dest_dir, "fl_admin.sh"),
            utils.sh_replace(self.template["fl_admin_sh"], replacement_dict),
            "t",
            exe=True,
        )
        utils._write(
            os.path.join(dest_dir, "readme.txt"),
            self.template["readme_am"],
            "t",
        )

    def prepare_admin_config(self, admin, ctx):
        config = json.loads(self.template["fed_admin"])
        agent_config = dict()
        self._prepare_overseer_agent(admin, agent_config, "admin", ctx)
        config["admin"].update(agent_config)

        provision_mode = ctx.get("provision_mode")
        if provision_mode == "poc":
            # in poc mode, we change to use "local_cert" as the cred_type so that the user won't be
            # prompted for username when starting the admin console
            config["admin"]["username"] = admin.name
            config["admin"]["cred_type"] = "local_cert"
        return config

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
