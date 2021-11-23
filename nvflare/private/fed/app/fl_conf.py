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

import logging
import logging.config
import os

from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.fuel.utils.json_scanner import Node
from nvflare.fuel.utils.wfconf import ConfigContext
from nvflare.private.fed.client.base_client_deployer import BaseClientDeployer
from nvflare.private.json_configer import JsonConfigurator
from .trainers.server_deployer import ServerDeployer

FL_PACKAGES = ["nvflare"]
FL_MODULES = ["server", "client", "app"]


class FLServerStarterConfiger(JsonConfigurator):
    def __init__(
        self,
        app_root: str,
        server_config_file_name=None,
        log_config_file_name=None,
        kv_list=None,
        logging_config=True,
    ):

        base_pkgs = FL_PACKAGES
        module_names = FL_MODULES

        if kv_list:
            assert isinstance(kv_list, list), "cmd_vars must be list, but got {}".format(type(kv_list))
            self.cmd_vars = parse_vars(kv_list)
        else:
            self.cmd_vars = {}

        if logging_config:
            log_config_file_path = os.path.join(app_root, log_config_file_name)
            assert os.path.isfile(log_config_file_path), "missing log config file {}".format(log_config_file_path)
            logging.config.fileConfig(fname=log_config_file_path, disable_existing_loggers=False)

        server_config_file_name = os.path.join(app_root, server_config_file_name)

        JsonConfigurator.__init__(
            self,
            config_file_name=server_config_file_name,
            base_pkgs=base_pkgs,
            module_names=module_names,
            exclude_libs=True,
        )

        self.app_root = app_root
        self.server_config_file_name = server_config_file_name

        self.config_validator = None
        self.enable_byoc = False

    def start_config(self, config_ctx: ConfigContext):
        super().start_config(config_ctx)

        # loading server specifications
        try:
            for server in self.config_data["servers"]:
                if server.get("ssl_private_key"):
                    server["ssl_private_key"] = os.path.join(self.app_root, server["ssl_private_key"])
                if server.get("ssl_cert"):
                    server["ssl_cert"] = os.path.join(self.app_root, server["ssl_cert"])
                if server.get("ssl_root_cert"):
                    server["ssl_root_cert"] = os.path.join(self.app_root, server["ssl_root_cert"])
        except Exception:
            raise ValueError("Server config error: '{}'".format(self.server_config_file_name))

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        # JsonConfigurator.process_config_element(self, config_ctx, node)

        element = node.element
        path = node.path()

        if path == "enable_byoc":
            self.enable_byoc = element
            return

    def finalize_config(self, config_ctx: ConfigContext):
        secure_train = False
        if self.cmd_vars.get("secure_train"):
            secure_train = self.cmd_vars["secure_train"]
        if not secure_train:
            self.enable_byoc = True

        build_ctx = {
            "secure_train": secure_train,
            "app_validator": self.config_validator,
            "server_config": self.config_data["servers"],
            "server_host": self.cmd_vars.get("host", None),
            "enable_byoc": self.enable_byoc
        }

        deployer = ServerDeployer()
        deployer.build(build_ctx)
        self.deployer = deployer


class FLClientStarterConfiger(JsonConfigurator):
    def __init__(
        self,
        app_root: str,
        client_config_file_name=None,
        log_config_file_name=None,
        kv_list=None,
        logging_config=True,
    ):

        base_pkgs = FL_PACKAGES
        module_names = FL_MODULES

        if kv_list:
            assert isinstance(kv_list, list), "cmd_vars must be list, but got {}".format(type(kv_list))
            self.cmd_vars = parse_vars(kv_list)
        else:
            self.cmd_vars = {}

        if logging_config:
            log_config_file_path = os.path.join(app_root, log_config_file_name)
            assert os.path.isfile(log_config_file_path), "missing log config file {}".format(log_config_file_path)
            logging.config.fileConfig(fname=log_config_file_path, disable_existing_loggers=False)

        client_config_file_name = os.path.join(app_root, client_config_file_name)

        JsonConfigurator.__init__(
            self,
            config_file_name=client_config_file_name,
            base_pkgs=base_pkgs,
            module_names=module_names,
            exclude_libs=True,
        )

        self.app_root = app_root
        self.client_config_file_name = client_config_file_name
        self.enable_byoc = False

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        # JsonConfigurator.process_config_element(self, config_ctx, node)

        element = node.element
        path = node.path()

        if path == "enable_byoc":
            self.enable_byoc = element
            return

    def start_config(self, config_ctx: ConfigContext):
        super().start_config(config_ctx)

        try:
            client = self.config_data["client"]
            if client.get("ssl_private_key"):
                client["ssl_private_key"] = os.path.join(self.app_root, client["ssl_private_key"])
            if client.get("ssl_cert"):
                client["ssl_cert"] = os.path.join(self.app_root, client["ssl_cert"])
            if client.get("ssl_root_cert"):
                client["ssl_root_cert"] = os.path.join(self.app_root, client["ssl_root_cert"])
        except Exception:
            raise ValueError("Client config error: '{}'".format(self.client_config_file_name))

    def finalize_config(self, config_ctx: ConfigContext):
        secure_train = False
        if self.cmd_vars.get("secure_train"):
            secure_train = self.cmd_vars["secure_train"]
        if not secure_train:
            self.enable_byoc = True

        build_ctx = {
            "client_name": self.cmd_vars.get("uid", ""),
            "server_config": self.config_data.get("servers", []),
            "client_config": self.config_data["client"],
            "secure_train": secure_train,
            "server_host": self.cmd_vars.get("host", None),
            "enable_byoc": self.enable_byoc,
        }

        self.base_deployer = BaseClientDeployer()
        self.base_deployer.build(build_ctx)
        # self.trainer = trainer
        # self.client_trainer = trainer

    # def build_nested(self, config_dict, app_root):
    #     if get_config_classname(config_dict) == "ClaraEvaluator":
    #         config_dict["args"]["app_root"] = app_root
    #     for k in config_dict["args"]:
    #         element = config_dict["args"][k]
    #         if isinstance(element, dict) and element.get("args") is not None:
    #             config_dict["args"][k] = self.build_nested(config_dict["args"][k], app_root)
    #     return self.build_component(config_dict)
