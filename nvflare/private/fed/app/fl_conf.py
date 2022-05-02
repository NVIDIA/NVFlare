# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

"""FL Server / Client startup configer."""

import logging
import logging.config
import os
import re

from nvflare.apis.fl_component import FLComponent
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.fuel.utils.json_scanner import Node
from nvflare.fuel.utils.wfconf import ConfigContext, ConfigError
from nvflare.private.defs import SSLConstants
from nvflare.private.json_configer import JsonConfigurator

from .deployer.base_client_deployer import BaseClientDeployer
from .deployer.server_deployer import ServerDeployer
from .fl_app_validator import FLAppValidator

FL_PACKAGES = ["nvflare"]
FL_MODULES = ["server", "client", "app"]


class FLServerStarterConfiger(JsonConfigurator):
    """FL Server startup configer."""

    def __init__(
        self,
        app_root: str,
        server_config_file_name=None,
        log_config_file_name=None,
        kv_list=None,
        logging_config=True,
    ):
        """Init the FLServerStarterConfiger.

        Args:
            app_root: application root
            server_config_file_name: server config filename
            log_config_file_name: log config filename
            kv_list: key value pair list
            logging_config: True/False
        """
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

        self.components = {}  # id => component
        self.handlers = []

        self.app_root = app_root
        self.server_config_file_name = server_config_file_name

        self.deployer = None
        self.app_validator = None
        self.enable_byoc = False
        self.snapshot_persistor = None
        self.overseer_agent = None

    def start_config(self, config_ctx: ConfigContext):
        """Start the config process.

        Args:
            config_ctx: config context

        """
        super().start_config(config_ctx)

        # loading server specifications
        try:
            for server in self.config_data["servers"]:
                if server.get(SSLConstants.PRIVATE_KEY):
                    server[SSLConstants.PRIVATE_KEY] = os.path.join(self.app_root, server[SSLConstants.PRIVATE_KEY])
                if server.get(SSLConstants.CERT):
                    server[SSLConstants.CERT] = os.path.join(self.app_root, server[SSLConstants.CERT])
                if server.get(SSLConstants.ROOT_CERT):
                    server[SSLConstants.ROOT_CERT] = os.path.join(self.app_root, server[SSLConstants.ROOT_CERT])
        except Exception:
            raise ValueError("Server config error: '{}'".format(self.server_config_file_name))

    def build_component(self, config_dict):
        t = super().build_component(config_dict)
        if isinstance(t, FLComponent):
            if type(t).__name__ not in [type(h).__name__ for h in self.handlers]:
                self.handlers.append(t)
        return t

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        """Process the config element.

        Args:
            config_ctx: config context
            node: element node

        """
        # JsonConfigurator.process_config_element(self, config_ctx, node)

        element = node.element
        path = node.path()

        if path == "enable_byoc":
            self.enable_byoc = element
            return

        if path == "app_validator" and isinstance(element, dict):
            self.app_validator = self.build_component(element)
            return

        if path == "snapshot_persistor":
            self.snapshot_persistor = self.build_component(element)
            return

        if path == "overseer_agent":
            self.overseer_agent = self.build_component(element)
            return

        if re.search(r"^components\.#[0-9]+$", path):
            c = self.build_component(element)
            cid = element.get("id", None)
            if not cid:
                raise ConfigError("missing component id")

            if not isinstance(cid, str):
                raise ConfigError('"id" must be str but got {}'.format(type(cid)))

            if cid in self.components:
                raise ConfigError('duplicate component id "{}"'.format(cid))

            self.components[cid] = c
            return

    def finalize_config(self, config_ctx: ConfigContext):
        """Finalize the config process.

        Args:
            config_ctx: config context

        """
        secure_train = False
        if self.cmd_vars.get("secure_train"):
            secure_train = self.cmd_vars["secure_train"]
        if not secure_train:
            self.enable_byoc = True

        custom_validators = [self.app_validator] if self.app_validator else []
        self.app_validator = FLAppValidator(custom_validators=custom_validators)

        build_ctx = {
            "secure_train": secure_train,
            "app_validator": self.app_validator,
            "server_config": self.config_data["servers"],
            "server_host": self.cmd_vars.get("host", None),
            "enable_byoc": self.enable_byoc,
            "snapshot_persistor": self.snapshot_persistor,
            "overseer_agent": self.overseer_agent,
            "server_components": self.components,
            "server_handlers": self.handlers,
        }

        deployer = ServerDeployer()
        deployer.build(build_ctx)
        self.deployer = deployer


class FLClientStarterConfiger(JsonConfigurator):
    """FL Client startup configer."""

    def __init__(
        self,
        app_root: str,
        client_config_file_name=None,
        log_config_file_name=None,
        kv_list=None,
        logging_config=True,
    ):
        """Init the FLClientStarterConfiger.

        Args:
            app_root: application root
            client_config_file_name: client config filename
            log_config_file_name: log config filename
            kv_list: key value pair list
            logging_config: True/False
        """
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

        self.components = {}  # id => component
        self.handlers = []

        self.app_root = app_root
        self.client_config_file_name = client_config_file_name
        self.enable_byoc = False
        self.base_deployer = None
        self.overseer_agent = None

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        """Process config element.

        Args:
            config_ctx: config context
            node: element node
        """
        element = node.element
        path = node.path()

        if path == "enable_byoc":
            self.enable_byoc = element
            return

        if path == "overseer_agent":
            self.overseer_agent = self.build_component(element)
            return

        if re.search(r"^components\.#[0-9]+$", path):
            c = self.build_component(element)
            cid = element.get("id", None)
            if not cid:
                raise ConfigError("missing component id")

            if not isinstance(cid, str):
                raise ConfigError('"id" must be str but got {}'.format(type(cid)))

            if cid in self.components:
                raise ConfigError('duplicate component id "{}"'.format(cid))

            self.components[cid] = c
            return

    def build_component(self, config_dict):
        t = super().build_component(config_dict)
        if isinstance(t, FLComponent):
            if type(t).__name__ not in [type(h).__name__ for h in self.handlers]:
                self.handlers.append(t)
        return t

    def start_config(self, config_ctx: ConfigContext):
        """Start the config process.

        Args:
            config_ctx: config context
        """
        super().start_config(config_ctx)

        try:
            client = self.config_data["client"]
            if client.get(SSLConstants.PRIVATE_KEY):
                client[SSLConstants.PRIVATE_KEY] = os.path.join(self.app_root, client[SSLConstants.PRIVATE_KEY])
            if client.get(SSLConstants.CERT):
                client[SSLConstants.CERT] = os.path.join(self.app_root, client[SSLConstants.CERT])
            if client.get(SSLConstants.ROOT_CERT):
                client[SSLConstants.ROOT_CERT] = os.path.join(self.app_root, client[SSLConstants.ROOT_CERT])
        except Exception:
            raise ValueError("Client config error: '{}'".format(self.client_config_file_name))

    def finalize_config(self, config_ctx: ConfigContext):
        """Finalize the config process.

        Args:
            config_ctx: config context
        """
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
            "overseer_agent": self.overseer_agent,
            "client_components": self.components,
            "client_handlers": self.handlers,
        }

        self.base_deployer = BaseClientDeployer()
        self.base_deployer.build(build_ctx)


class FLAdminClientStarterConfigurator(JsonConfigurator):
    """FL Admin Client startup configurator."""

    def __init__(self, app_root: str, admin_config_file_name=None):
        """Uses the json configuration to start the FL admin client.

        Args:
            app_root: application root
            admin_config_file_name: admin config filename
        """
        base_pkgs = FL_PACKAGES
        module_names = FL_MODULES

        admin_config_file_name = os.path.join(app_root, admin_config_file_name)

        JsonConfigurator.__init__(
            self,
            config_file_name=admin_config_file_name,
            base_pkgs=base_pkgs,
            module_names=module_names,
            exclude_libs=True,
        )

        self.app_root = app_root
        self.admin_config_file_name = admin_config_file_name
        self.base_deployer = None
        self.overseer_agent = None

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

    def start_config(self, config_ctx: ConfigContext):
        """Start the config process.

        Args:
            config_ctx: config context
        """
        super().start_config(config_ctx)

        try:
            admin = self.config_data["admin"]
            if admin.get("client_key"):
                admin["client_key"] = os.path.join(self.app_root, admin["client_key"])
            if admin.get("client_cert"):
                admin["client_cert"] = os.path.join(self.app_root, admin["client_cert"])
            if admin.get("ca_cert"):
                admin["ca_cert"] = os.path.join(self.app_root, admin["ca_cert"])
            if admin.get("upload_dir"):
                admin["upload_dir"] = os.path.join(os.path.dirname(self.app_root), admin["upload_dir"])
            if admin.get("download_dir"):
                admin["download_dir"] = os.path.join(os.path.dirname(self.app_root), admin["download_dir"])
        except Exception:
            raise ValueError("Client config error: '{}'".format(self.admin_config_file_name))
