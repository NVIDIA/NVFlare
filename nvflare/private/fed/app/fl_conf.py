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

"""FL Server / Client startup configure."""

import os
import re
import sys

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ConnectionSecurity, ConnPropKey, FilterKey, SiteType, SystemConfigs
from nvflare.apis.workspace import Workspace
from nvflare.fuel.data_event.utils import set_scope_property
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.json_scanner import Node
from nvflare.fuel.utils.url_utils import make_url
from nvflare.fuel.utils.wfconf import ConfigContext, ConfigError
from nvflare.private.defs import SSLConstants
from nvflare.private.json_configer import JsonConfigurator
from nvflare.private.privacy_manager import PrivacyManager, Scope

from .deployer.base_client_deployer import BaseClientDeployer
from .deployer.server_deployer import ServerDeployer
from .fl_app_validator import FLAppValidator

FL_PACKAGES = ["nvflare"]
FL_MODULES = ["server", "client", "app_common", "private"]


class FLServerStarterConfiger(JsonConfigurator):
    """FL Server startup configure."""

    def __init__(self, workspace: Workspace, args, kv_list=None):
        """Init the FLServerStarterConfiger.

        Args:
            workspace: the workspace object
            kv_list: key value pair list
        """
        site_custom_folder = workspace.get_site_custom_dir()
        if os.path.isdir(site_custom_folder) and site_custom_folder not in sys.path:
            sys.path.append(site_custom_folder)

        self.args = args

        base_pkgs = FL_PACKAGES
        module_names = FL_MODULES

        if kv_list:
            assert isinstance(kv_list, list), "cmd_vars must be list, but got {}".format(type(kv_list))
            self.cmd_vars = parse_vars(kv_list)
        else:
            self.cmd_vars = {}

        config_files = workspace.get_config_files_for_startup(is_server=True, for_job=True if args.job_id else False)

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
    """FL Client startup configure."""

    def __init__(self, workspace: Workspace, args, kv_list=None):
        """Init the FLClientStarterConfiger.

        Args:
            workspace: the workspace object
            kv_list: key value pair list
        """
        site_custom_folder = workspace.get_site_custom_dir()
        if os.path.isdir(site_custom_folder) and site_custom_folder not in sys.path:
            sys.path.append(site_custom_folder)

        self.args = args

        base_pkgs = FL_PACKAGES
        module_names = FL_MODULES

        if kv_list:
            assert isinstance(kv_list, list), "cmd_vars must be list, but got {}".format(type(kv_list))
            self.cmd_vars = parse_vars(kv_list)
        else:
            self.cmd_vars = {}

        config_files = workspace.get_config_files_for_startup(is_server=False, for_job=True if args.job_id else False)

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

    def _determine_conn_props(self, client_name, config_data: dict):
        relay_fqcn = None
        relay_url = None
        relay_conn_security = None

        # relay info is set in the client's relay__resources.json.
        # If relay is used, then connect via the specified relay; if not, try to connect the Server directly
        relay_config = config_data.get(ConnPropKey.RELAY_CONFIG)
        self.logger.debug(f"got relay config: {relay_config}")
        if relay_config:
            if relay_config:
                relay_fqcn = relay_config.get(ConnPropKey.FQCN)
                scheme = relay_config.get(ConnPropKey.SCHEME)
                addr = relay_config.get(ConnPropKey.ADDRESS)
                relay_conn_security = relay_config.get(ConnPropKey.CONNECTION_SECURITY)
                secure = True
                if relay_conn_security == ConnectionSecurity.CLEAR:
                    secure = False
                relay_url = make_url(scheme, addr, secure)
            else:
                self.logger.debug("no relay defined: connect to server directly")
        else:
            self.logger.debug("no relay_config: connect to server directly")

        if relay_fqcn:
            cp_fqcn = FQCN.join([relay_fqcn, client_name])
        else:
            cp_fqcn = client_name

        if relay_fqcn:
            relay_conn_props = {
                ConnPropKey.FQCN: relay_fqcn,
                ConnPropKey.URL: relay_url,
                ConnPropKey.CONNECTION_SECURITY: relay_conn_security,
            }
            set_scope_property(client_name, ConnPropKey.RELAY_CONN_PROPS, relay_conn_props)

        client = self.config_data["client"]

        if hasattr(self.args, "job_id") and self.args.job_id:
            # this is CJ
            sp_scheme = self.args.sp_scheme
            sp_target = self.args.sp_target
            root_url = f"{sp_scheme}://{sp_target}"
            root_conn_props = {
                ConnPropKey.FQCN: FQCN.ROOT_SERVER,
                ConnPropKey.URL: root_url,
                ConnPropKey.CONNECTION_SECURITY: client.get(ConnPropKey.CONNECTION_SECURITY),
            }
            set_scope_property(client_name, ConnPropKey.ROOT_CONN_PROPS, root_conn_props)

            cp_conn_props = {
                ConnPropKey.FQCN: cp_fqcn,
                ConnPropKey.URL: self.args.parent_url,
                ConnPropKey.CONNECTION_SECURITY: self.args.parent_conn_sec,
            }
        else:
            # this is CP
            cp_conn_props = {
                ConnPropKey.FQCN: cp_fqcn,
            }
        set_scope_property(client_name, ConnPropKey.CP_CONN_PROPS, cp_conn_props)

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

            client_name = self.cmd_vars.get("uid", None)
            if not client_name:
                raise ConfigError("missing 'uid' from command args")

            conn_sec = client.get(ConnPropKey.CONNECTION_SECURITY)
            if conn_sec:
                set_scope_property(client_name, ConnPropKey.CONNECTION_SECURITY, conn_sec)

            self._determine_conn_props(client_name, self.config_data)

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


class PrivacyConfiger(JsonConfigurator):
    def __init__(self, workspace: Workspace, names_only: bool, is_server=False):
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
        self.is_server = is_server

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
                direction = element.get("direction")
                if direction:
                    direction = direction.lower()
                else:
                    direction = FilterKey.OUT if self.is_server else FilterKey.IN
                if f:
                    self.current_scope.add_task_data_filter(f, direction)
                return

            if re.search(r"^scopes.#[0-9]+\.task_result_filters\.#[0-9]+$", path):
                f = self.build_component(element)
                direction = element.get("direction")
                if direction:
                    direction = direction.lower()
                else:
                    direction = FilterKey.IN if self.is_server else FilterKey.OUT
                if f:
                    self.current_scope.add_task_result_filter(f, direction)
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


def create_privacy_manager(workspace: Workspace, names_only: bool, is_server=False):
    privacy_file_path = workspace.get_site_privacy_file_path()
    if not os.path.isfile(privacy_file_path):
        # privacy policy not defined
        mgr = PrivacyManager(scopes=None, default_scope_name=None, components=None)
    else:
        configer = PrivacyConfiger(workspace, names_only, is_server=is_server)
        configer.configure()
        mgr = configer.privacy_manager
    return mgr
