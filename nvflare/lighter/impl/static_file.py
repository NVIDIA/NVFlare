# Copyright (c) 2021, NVIDIA CORPORATION.
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

from nvflare.lighter.spec import Builder
from nvflare.lighter.utils import sh_replace


class StaticFileBuilder(Builder):
    def __init__(self, enable_byoc=False, config_folder="", docker_image=""):
        self.enable_byoc = enable_byoc
        self.config_folder = config_folder
        self.docker_image = docker_image

    def _write(self, file_full_path, content, mode, exe=False):
        mode = mode + "w"
        with open(file_full_path, mode) as f:
            f.write(content)
        if exe:
            os.chmod(file_full_path, 0o755)

    def _build_server(self, server, ctx):
        config = json.loads(self.template["fed_server"])
        dest_dir = self.get_kit_dir(server, ctx)
        server_0 = config["servers"][0]
        server_0["name"] = self.study_name
        admin_port = server.props.get("admin_port", 8003)
        ctx["admin_port"] = admin_port
        fed_learn_port = server.props.get("fed_learn_port", 8002)
        ctx["fed_learn_port"] = fed_learn_port
        ctx["server_name"] = server.name
        server_0["service"]["target"] = f"{server.name}:{fed_learn_port}"
        server_0["admin_host"] = server.name
        server_0["admin_port"] = admin_port
        config["enable_byoc"] = server.enable_byoc
        self._write(os.path.join(dest_dir, "fed_server.json"), json.dumps(config), "t")
        replacement_dict = {
            "admin_port": admin_port,
            "fed_learn_port": fed_learn_port,
            "config_folder": self.config_folder,
            "docker_image": self.docker_image,
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
            os.path.join(dest_dir, "log.config"),
            self.template["log_config"],
            "t",
        )
        self._write(
            os.path.join(dest_dir, "readme.txt"),
            self.template["readme_fs"],
            "t",
        )
        self._write(
            os.path.join(dest_dir, "stop_fl.sh"),
            self.template["stop_fl_sh"],
            "t",
            exe=True,
        )

    def _build_client(self, client, ctx):
        config = json.loads(self.template["fed_client"])
        dest_dir = self.get_kit_dir(client, ctx)
        fed_learn_port = ctx.get("fed_learn_port")
        server_name = ctx.get("server_name")
        config["servers"][0]["service"]["target"] = f"{server_name}:{fed_learn_port}"
        config["servers"][0]["name"] = self.study_name
        config["enable_byoc"] = client.enable_byoc
        replacement_dict = {
            "client_name": f"{client.subject}",
            "config_folder": self.config_folder,
            "docker_image": self.docker_image,
        }

        self._write(os.path.join(dest_dir, "fed_client.json"), json.dumps(config), "t")
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
            os.path.join(dest_dir, "log.config"),
            self.template["log_config"],
            "t",
        )
        self._write(
            os.path.join(dest_dir, "readme.txt"),
            self.template["readme_fc"],
            "t",
        )
        self._write(
            os.path.join(dest_dir, "stop_fl.sh"),
            self.template["stop_fl_sh"],
            "t",
            exe=True,
        )

    def _build_admin(self, admin, ctx):
        dest_dir = self.get_kit_dir(admin, ctx)
        admin_port = ctx.get("admin_port")
        server_name = ctx.get("server_name")

        replacement_dict = {
            "cn": f"{server_name}",
            "admin_port": f"{admin_port}",
            "docker_image": self.docker_image,
        }
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

    def build(self, study, ctx):
        self.template = ctx.get("template")
        server = study.get_participants_by_type("server")
        self.study_name = study.name
        self._build_server(server, ctx)

        for client in study.get_participants_by_type("client", first_only=False):
            self._build_client(client, ctx)

        for admin in study.get_participants_by_type("admin", first_only=False):
            self._build_admin(admin, ctx)
