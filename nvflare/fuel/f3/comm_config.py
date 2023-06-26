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
import logging

from nvflare.fuel.f3.drivers.net_utils import MAX_PAYLOAD_SIZE
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.security.logging import secure_format_exception

_comm_config_files = ["comm_config.json", "comm_config.json.default"]


DEFAULT_MAX_MSG_SIZE = MAX_PAYLOAD_SIZE


class VarName:

    MAX_MESSAGE_SIZE = "max_message_size"
    ALLOW_ADHOC_CONNS = "allow_adhoc_conns"
    ADHOC_CONN_SCHEME = "adhoc_conn_scheme"
    INTERNAL_CONN_SCHEME = "internal_conn_scheme"
    BACKBONE_CONN_GEN = "backbone_conn_gen"
    SUBNET_HEARTBEAT_INTERVAL = "subnet_heartbeat_interval"
    SUBNET_TROUBLE_THRESHOLD = "subnet_trouble_threshold"
    COMM_DRIVER_PATH = "comm_driver_path"


class CommConfigurator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        config = None
        for file_name in _comm_config_files:
            try:
                config = ConfigService.load_json(file_name)
                if config:
                    break
            except FileNotFoundError:
                self.logger.debug(f"config file {file_name} not found from config path")
                config = None
            except Exception as ex:
                self.logger.error(f"failed to load config file {file_name}: {secure_format_exception(ex)}")
                config = None
        self.config = config

    def get_config(self):
        return self.config

    def get_max_message_size(self):
        return ConfigService.get_int_var(VarName.MAX_MESSAGE_SIZE, self.config, default=DEFAULT_MAX_MSG_SIZE)

    def allow_adhoc_connections(self, default):
        return ConfigService.get_bool_var(VarName.ALLOW_ADHOC_CONNS, self.config, default=default)

    def get_adhoc_connection_scheme(self, default):
        return ConfigService.get_str_var(VarName.ADHOC_CONN_SCHEME, self.config, default=default)

    def get_internal_connection_scheme(self, default):
        return ConfigService.get_str_var(VarName.INTERNAL_CONN_SCHEME, self.config, default=default)

    def get_backbone_connection_generation(self, default):
        return ConfigService.get_int_var(VarName.BACKBONE_CONN_GEN, self.config, default=default)

    def get_subnet_heartbeat_interval(self, default):
        return ConfigService.get_int_var(VarName.SUBNET_HEARTBEAT_INTERVAL, self.config, default)

    def get_subnet_trouble_threshold(self, default):
        return ConfigService.get_int_var(VarName.SUBNET_TROUBLE_THRESHOLD, self.config, default)

    def get_comm_driver_path(self, default):
        return ConfigService.get_str_var(VarName.COMM_DRIVER_PATH, self.config, default=default)
