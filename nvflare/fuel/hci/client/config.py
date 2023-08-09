# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os
import re
import sys

from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils.json_scanner import Node
from nvflare.fuel.utils.wfconf import ConfigContext
from nvflare.private.json_configer import JsonConfigurator

from .event import EventHandler

FL_PACKAGES = ["nvflare"]
FL_MODULES = ["ha"]


class FLAdminClientStarterConfigurator(JsonConfigurator):
    """FL Admin Client startup configurator."""

    def __init__(self, workspace: Workspace):
        """Uses the json configuration to start the FL admin client.

        Args:
            workspace: the workspace object
        """
        base_pkgs = FL_PACKAGES
        module_names = FL_MODULES

        custom_dir = workspace.get_client_custom_dir()
        if os.path.isdir(custom_dir):
            sys.path.append(custom_dir)

        admin_config_file_path = workspace.get_admin_startup_file_path()
        config_files = [admin_config_file_path]
        resources_file_path = workspace.get_resources_file_path()
        if resources_file_path:
            config_files.append(resources_file_path)

        JsonConfigurator.__init__(
            self,
            config_file_name=config_files,
            base_pkgs=base_pkgs,
            module_names=module_names,
            exclude_libs=True,
        )

        self.workspace = workspace
        self.admin_config_file_path = config_files
        self.overseer_agent = None
        self.handlers = []

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        """Process config element.

        Args:
            config_ctx: config context
            node: element node
        """
        element = node.element
        path = node.path()

        if path == "admin.overseer_agent":
            self.overseer_agent = self.build_component(element)
            return

        if re.search(r"^handlers\.#[0-9]+$", path):
            c = self.build_component(element)
            if not isinstance(c, EventHandler):
                raise ConfigError(f"component must be EventHandler but got {type(c)}")
            self.handlers.append(c)
            return

    def start_config(self, config_ctx: ConfigContext):
        """Start the config process.

        Args:
            config_ctx: config context
        """
        super().start_config(config_ctx)

        try:
            admin = self.config_data["admin"]
            if admin.get("client_key"):
                admin["client_key"] = self.workspace.get_file_path_in_startup(admin["client_key"])
            if admin.get("client_cert"):
                admin["client_cert"] = self.workspace.get_file_path_in_startup(admin["client_cert"])
            if admin.get("ca_cert"):
                admin["ca_cert"] = self.workspace.get_file_path_in_startup(admin["ca_cert"])

            if admin.get("upload_dir"):
                admin["upload_dir"] = self.workspace.get_file_path_in_root(admin["upload_dir"])
            if admin.get("download_dir"):
                admin["download_dir"] = self.workspace.get_file_path_in_root(admin["download_dir"])
        except Exception:
            raise ValueError(f"Client config error: '{self.admin_config_file_path}'")
