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

"""FL Server / Client startup configer."""

import os
import re
import sys

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import SiteType, SystemConfigs
from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.json_scanner import Node
from nvflare.fuel.utils.wfconf import ConfigContext, ConfigError
from nvflare.private.defs import SSLConstants
from nvflare.private.fed.utils.fed_utils import configure_logging
from nvflare.private.json_configer import JsonConfigurator
from nvflare.private.privacy_manager import PrivacyManager, Scope

from .deployer.base_client_deployer import BaseClientDeployer
from .deployer.server_deployer import ServerDeployer
from .fl_app_validator import FLAppValidator

FL_PACKAGES = ["nvflare"]
FL_MODULES = ["server", "client", "app_common", "private", "app_opt"]


class FLServerStarterConfiger(JsonConfigurator):
    """FL Server startup configer."""

    def __init__(self, workspace: Workspace, args, kv_list=None):
        """Init the FLServerStarterConfiger.

        Args:
            workspace: the workspace object
            kv_list: key value pair list
        """
        site_custom_folder = workspace.get_site_custom_dir()
        if os.path.isdir(site_custom_folder):
            sys.path.append(site_custom_folder)

        self.args = args

        base_pkgs = FL_PACKAGES
        module_names = FL_MODULES

        if kv_list:
            assert isinstance(kv_list, list), "cmd_vars must be list, but got {}".format(type(kv_list))
            self.cmd_vars = parse_vars(kv_list)
        else:
            self.cmd_vars = {}

        configure_logging(workspace)

        server_startup_file_path = workspace.get_server_startup_file_path()
        resource_config_path = workspace.get_resources_file_path()
        config_files = [server_startup_file_path, resource_config_path]
        if args.job_id:
            # this is for job process
            job_resources_file_path = workspace.get_job_resources_file_path()
            if os.path.exists(job_resources_file_path):
                config_files.append(job_resources_file_path)

        JsonConfigurator.__init__(
            self,
            config_file_name=config_files,
            base_pkgs=base_pkgs,
            module_names=module_names,
            exclude_libs=True,
        )

        self.components = {}  # id => component
        self.handlers = []

        self.workspace = workspace
        self.server_config_file_names = config_files

        self.deployer = None
        self.app_validator = None
        self.snapshot_persistor = None
        self.overseer_agent = None
        self.site_org = ""

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
                    server[SSLConstants.PRIVATE_KEY] = self.workspace.get_file_path_in_startup(
                        server[SSLConstants.PRIVATE_KEY]
                    )
                if server.get(SSLConstants.CERT):
                    server[SSLConstants.CERT] = self.workspace.get_file_path_in_startup(server[SSLConstants.CERT])
                if server.get(SSLConstants.ROOT_CERT):
                    server[SSLConstants.ROOT_CERT] = self.workspace.get_file_path_in_startup(
                        server[SSLConstants.ROOT_CERT]
                    )
        except Exception:
            raise ValueError(f"Server config error: '{self.server_config_file_names}'")

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

        custom_validators = [self.app_validator] if self.app_validator else []
        self.app_validator = FLAppValidator(site_type=SiteType.SERVER, custom_validators=custom_validators)

        build_ctx = {
            "secure_train": secure_train,
            "app_validator": self.app_validator,
            "server_config": self.config_data["servers"],
            "server_host": self.cmd_vars.get("host", None),
            "site_org": self.cmd_vars.get("org", ""),
            "snapshot_persistor": self.snapshot_persistor,
            "overseer_agent": self.overseer_agent,
            "server_components": self.components,
            "server_handlers": self.handlers,
        }

        deployer = ServerDeployer()
        deployer.build(build_ctx)
        self.deployer = deployer
        self.site_org = build_ctx["site_org"]

        ConfigService.initialize(
            section_files={
                SystemConfigs.STARTUP_CONF: os.path.basename(self.server_config_file_names[0]),
                SystemConfigs.RESOURCES_CONF: os.path.basename(self.server_config_file_names[1]),
            },
            config_path=[self.args.workspace],
            parsed_args=self.args,
            var_dict=self.cmd_vars,
        )


class FLClientStarterConfiger(JsonConfigurator):
    """FL Client startup configer."""

    def __init__(self, workspace: Workspace, args, kv_list=None):
        """Init the FLClientStarterConfiger.

        Args:
            workspace: the workspace object
            kv_list: key value pair list
        """
        site_custom_folder = workspace.get_site_custom_dir()
        if os.path.isdir(site_custom_folder):
            sys.path.append(site_custom_folder)

        self.args = args

        base_pkgs = FL_PACKAGES
        module_names = FL_MODULES

        if kv_list:
            assert isinstance(kv_list, list), "cmd_vars must be list, but got {}".format(type(kv_list))
            self.cmd_vars = parse_vars(kv_list)
        else:
            self.cmd_vars = {}

        configure_logging(workspace)

        client_startup_file_path = workspace.get_client_startup_file_path()
        resources_file_path = workspace.get_resources_file_path()
        config_files = [client_startup_file_path, resources_file_path]

        if args.job_id:
            # this is for job process
            job_resources_file_path = workspace.get_job_resources_file_path()
            if os.path.exists(job_resources_file_path):
                config_files.append(job_resources_file_path)

        JsonConfigurator.__init__(
            self,
            config_file_name=config_files,
            base_pkgs=base_pkgs,
            module_names=module_names,
            exclude_libs=True,
        )

        self.components = {}  # id => component
        self.handlers = []

        self.workspace = workspace
        self.client_config_file_names = config_files
        self.base_deployer = None
        self.overseer_agent = None
        self.site_org = ""
        self.app_validator = None

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        """Process config element.

        Args:
            config_ctx: config context
            node: element node
        """
        element = node.element
        path = node.path()

        if path == "app_validator" and isinstance(element, dict):
            self.app_validator = self.build_component(element)
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
                client[SSLConstants.PRIVATE_KEY] = self.workspace.get_file_path_in_startup(
                    client[SSLConstants.PRIVATE_KEY]
                )
            if client.get(SSLConstants.CERT):
                client[SSLConstants.CERT] = self.workspace.get_file_path_in_startup(client[SSLConstants.CERT])
            if client.get(SSLConstants.ROOT_CERT):
                client[SSLConstants.ROOT_CERT] = self.workspace.get_file_path_in_startup(client[SSLConstants.ROOT_CERT])
        except Exception:
            raise ValueError(f"Client config error: '{self.client_config_file_names}'")

    def finalize_config(self, config_ctx: ConfigContext):
        """Finalize the config process.

        Args:
            config_ctx: config context
        """
        secure_train = False
        if self.cmd_vars.get("secure_train"):
            secure_train = self.cmd_vars["secure_train"]

        build_ctx = {
            "client_name": self.cmd_vars.get("uid", ""),
            "site_org": self.cmd_vars.get("org", ""),
            "server_config": self.config_data.get("servers", []),
            "client_config": self.config_data["client"],
            "secure_train": secure_train,
            "server_host": self.cmd_vars.get("host", None),
            "overseer_agent": self.overseer_agent,
            "client_components": self.components,
            "client_handlers": self.handlers,
        }

        custom_validators = [self.app_validator] if self.app_validator else []
        self.app_validator = FLAppValidator(site_type=SiteType.CLIENT, custom_validators=custom_validators)
        self.site_org = build_ctx["site_org"]
        self.base_deployer = BaseClientDeployer()
        self.base_deployer.build(build_ctx)

        ConfigService.initialize(
            section_files={
                SystemConfigs.STARTUP_CONF: os.path.basename(self.client_config_file_names[0]),
                SystemConfigs.RESOURCES_CONF: os.path.basename(self.client_config_file_names[1]),
            },
            config_path=[self.args.workspace],
            parsed_args=self.args,
            var_dict=self.cmd_vars,
        )


class FLAdminClientStarterConfigurator(JsonConfigurator):
    """FL Admin Client startup configurator."""

    def __init__(self, workspace: Workspace):
        """Uses the json configuration to start the FL admin client.

        Args:
            workspace: the workspace object
        """
        base_pkgs = FL_PACKAGES
        module_names = FL_MODULES

        admin_config_file_path = workspace.get_admin_startup_file_path()

        JsonConfigurator.__init__(
            self,
            config_file_name=admin_config_file_path,
            base_pkgs=base_pkgs,
            module_names=module_names,
            exclude_libs=True,
        )

        self.workspace = workspace
        self.admin_config_file_path = admin_config_file_path
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


class PrivacyConfiger(JsonConfigurator):
    def __init__(self, workspace: Workspace, names_only: bool):
        """Uses the json configuration to start the FL admin client.

        Args:
            workspace: the workspace object
        """
        self.privacy_manager = None
        self.scopes = []
        self.default_scope_name = None
        self.components = {}
        self.current_scope = None
        self.names_only = names_only

        privacy_file_path = workspace.get_site_privacy_file_path()
        JsonConfigurator.__init__(
            self,
            config_file_name=privacy_file_path,
            base_pkgs=FL_PACKAGES,
            module_names=FL_MODULES,
            exclude_libs=True,
        )

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        """Process config element.

        Args:
            config_ctx: config context
            node: element node
        """
        element = node.element
        path = node.path()

        if re.search(r"^scopes\.#[0-9]+$", path):
            scope = Scope()
            self.current_scope = scope
            self.scopes.append(scope)
            return

        if re.search(r"^scopes\.#[0-9]+\.name$", path):
            self.current_scope.set_name(element)
            return

        if path == "default_scope":
            self.default_scope_name = element
            return

        if not self.names_only:
            if re.search(r"^scopes\.#[0-9]+\.properties$", path):
                self.current_scope.set_props(element)
                return

            if re.search(r"^scopes.#[0-9]+\.task_data_filters\.#[0-9]+$", path):
                f = self.build_component(element)
                if f:
                    self.current_scope.add_task_data_filter(f)
                return

            if re.search(r"^scopes.#[0-9]+\.task_result_filters\.#[0-9]+$", path):
                f = self.build_component(element)
                if f:
                    self.current_scope.add_task_result_filter(f)
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
        self.privacy_manager = PrivacyManager(
            scopes=self.scopes, default_scope_name=self.default_scope_name, components=self.components
        )


def create_privacy_manager(workspace: Workspace, names_only: bool):
    privacy_file_path = workspace.get_site_privacy_file_path()
    if not os.path.isfile(privacy_file_path):
        # privacy policy not defined
        mgr = PrivacyManager(scopes=None, default_scope_name=None, components=None)
    else:
        configer = PrivacyConfiger(workspace, names_only)
        configer.configure()
        mgr = configer.privacy_manager
    return mgr
