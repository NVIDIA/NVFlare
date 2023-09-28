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
from nvflare.fuel.utils.config import Config
from nvflare.fuel.utils.config_service import ConfigService

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
    HEARTBEAT_INTERVAL = "heartbeat_interval"
    STREAMING_CHUNK_SIZE = "streaming_chunk_size"
    STREAMING_ACK_WAIT = "streaming_ack_wait"
    STREAMING_WINDOW_SIZE = "streaming_window_size"
    STREAMING_ACK_INTERVAL = "streaming_ack_interval"
    STREAMING_MAX_OUT_SEQ_CHUNKS = "streaming_max_out_seq_chunks"
    STREAMING_READ_TIMEOUT = "streaming_read_timeout"


class CommConfigurator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        config: Config = ConfigService.load_configuration(file_basename=_comm_config_files[0])
        self.config = None if config is None else config.to_dict()

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
        return ConfigService.get_int_var(VarName.SUBNET_HEARTBEAT_INTERVAL, self.config, default=default)

    def get_subnet_trouble_threshold(self, default):
        return ConfigService.get_int_var(VarName.SUBNET_TROUBLE_THRESHOLD, self.config, default=default)

    def get_comm_driver_path(self, default):
        return ConfigService.get_str_var(VarName.COMM_DRIVER_PATH, self.config, default=default)

    def get_heartbeat_interval(self, default):
        return ConfigService.get_int_var(VarName.HEARTBEAT_INTERVAL, self.config, default=default)

    def get_streaming_chunk_size(self, default):
        return ConfigService.get_int_var(VarName.STREAMING_CHUNK_SIZE, self.config, default=default)

    def get_streaming_ack_wait(self, default):
        return ConfigService.get_int_var(VarName.STREAMING_ACK_WAIT, self.config, default=default)

    def get_streaming_window_size(self, default):
        return ConfigService.get_int_var(VarName.STREAMING_WINDOW_SIZE, self.config, default=default)

    def get_streaming_ack_interval(self, default):
        return ConfigService.get_int_var(VarName.STREAMING_ACK_INTERVAL, self.config, default=default)

    def get_streaming_max_out_seq_chunks(self, default):
        return ConfigService.get_int_var(VarName.STREAMING_MAX_OUT_SEQ_CHUNKS, self.config, default=default)

    def get_streaming_read_timeout(self, default):
        return ConfigService.get_int_var(VarName.STREAMING_READ_TIMEOUT, self.config, default)
