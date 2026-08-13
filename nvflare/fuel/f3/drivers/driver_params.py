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
from enum import Enum


class DriverParams(str, Enum):

    # URL components. Those parameters are part of the URL, no need to be included in query string
    # URL = SCHEME://HOST:PORT/PATH;PARAMS?QUERY#FRAG
    URL = "url"
    SCHEME = "scheme"
    HOST = "host"
    PORT = "port"
    PATH = "path"
    PARAMS = "params"
    FRAG = "frag"
    QUERY = "query"

    # Other parameters
    CA_CERT = "ca_cert"
    SERVER_CERT = "server_cert"
    SERVER_KEY = "server_key"
    CLIENT_CERT = "client_cert"
    CLIENT_KEY = "client_key"
    CONNECTION_SECURITY = "connection_security"
    CUSTOM_CA_CERT = "custom_ca_cert"
    SECURE = "secure"
    PORTS = "ports"
    SOCKET = "socket"
    LOCAL_ADDR = "local_addr"
    PEER_ADDR = "peer_addr"
    PEER_CN = "peer_cn"
    IMPLEMENTED_CONN_SEC = "implemented_conn_sec"


class DriverCap(str, Enum):

    SEND_HEARTBEAT = "send_heartbeat"
    SUPPORT_SSL = "support_ssl"
