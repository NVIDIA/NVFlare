# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import errno
import json
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import grpc

from nvflare.apis.fl_constant import ConnectionSecurity, SecureTrainConst
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.grpc.utils import get_grpc_client_credentials
from nvflare.fuel.f3.drivers.net_utils import enhance_credential_info, get_ssl_context
from nvflare.utils.process_utils import popen_in_new_session

SUPPORTED_SCHEMES = ("grpc", "agrpc", "http", "https", "tcp", "stcp", "atcp", "satcp")


class NVFlareConfig:
    SERVER = "fed_server.json"
    CLIENT = "fed_client.json"
    ADMIN = "fed_admin.json"


class NVFlareRole:
    SERVER = "server"
    CLIENT = "client"
    ADMIN = "admin"


def try_write_dir(path: str):
    try:
        created = False
        if not os.path.exists(path):
            created = True
            os.makedirs(path, exist_ok=False)
        fd, name = tempfile.mkstemp(dir=path)
        with os.fdopen(fd, "w") as fp:
            fp.write("dummy")
        os.remove(name)
        if created:
            shutil.rmtree(path)
    except OSError as e:
        return e


def try_bind_address(host: str, port: int):
    """Tries to bind to address."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            return e
        try:
            # BSD permits a reuse-enabled loopback bind over a live wildcard listener.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(1)
                if probe.connect_ex((host, port)) != errno.ECONNREFUSED:
                    return e
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        except OSError as retry_error:
            return retry_error
    finally:
        sock.close()
    return None


def split_by_len(item, max_len):
    return [item[ind : ind + max_len] for ind in range(0, len(item), max_len)]


def _client_probe_params(startup: str):
    startup = Path(startup)
    client_config = startup / NVFlareConfig.CLIENT
    if client_config.is_file():
        config = json.loads(client_config.read_text())["client"]
        ca_field, cert_field, key_field = (
            SecureTrainConst.SSL_ROOT_CERT,
            SecureTrainConst.SSL_CERT,
            SecureTrainConst.PRIVATE_KEY,
        )
    else:
        admin_config = startup / NVFlareConfig.ADMIN
        config = json.loads(admin_config.read_text())["admin"] if admin_config.is_file() else {}
        ca_field, cert_field, key_field = DriverParams.CA_CERT, DriverParams.CLIENT_CERT, DriverParams.CLIENT_KEY
    security = config.get("connection_security", ConnectionSecurity.MTLS)
    params = {
        DriverParams.CONNECTION_SECURITY: security,
        DriverParams.SECURE: security != ConnectionSecurity.CLEAR,
        DriverParams.CA_CERT: str(startup / config.get(ca_field, "rootCA.pem")),
        DriverParams.CLIENT_CERT: str(startup / config.get(cert_field, "client.crt")),
        DriverParams.CLIENT_KEY: str(startup / config.get(key_field, "client.key")),
    }
    enhance_credential_info(params)
    return params


def check_grpc_server_running(startup: str, host: str, port: int, token=None) -> bool:
    params = _client_probe_params(startup)
    if params[DriverParams.SECURE]:
        call_credentials = grpc.metadata_call_credentials(
            lambda context, callback: callback((("x-custom-token", token),), None)
        )
        credentials = get_grpc_client_credentials(params)
        composite_credentials = grpc.composite_channel_credentials(credentials, call_credentials)
        channel = grpc.secure_channel(target=f"{host}:{port}", credentials=composite_credentials)
    else:
        channel = grpc.insecure_channel(target=f"{host}:{port}")

    try:
        grpc.channel_ready_future(channel).result(timeout=10)
    except grpc.FutureTimeoutError:
        return False
    finally:
        channel.close()
    return True


def check_socket_server_running(startup: str, host: str, port: int, scheme: str = "https", *, tcp_only=False) -> bool:
    """Check if socket-based server is running and accessible.

    This function performs a socket connection test with optional SSL/TLS.
    It's used for HTTP/WebSocket and TCP-based FL servers.

    Args:
        startup: Path to startup directory containing certificates
        host: Server hostname or IP address
        port: Server port number
        scheme: configured transport scheme
        tcp_only: probe reachability without loading credentials

    Returns:
        True if server is accessible, False otherwise
    """
    context = None
    try:
        if not tcp_only:
            params = _client_probe_params(startup)
            params[DriverParams.SCHEME] = scheme
            context = get_ssl_context(params, ssl_server=False)
        with socket.create_connection((host, port), timeout=10) as sock:
            if context:
                with context.wrap_socket(sock, server_hostname=host):
                    pass
        return True
    except OSError:
        return False


def run_command_in_subprocess(command):
    new_env = os.environ.copy()
    process = popen_in_new_session(
        shlex.split(command),
        shell=False,
        env=new_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return process


def get_communication_scheme(package_path: str, config_name: str, default_scheme: str = "http") -> str:
    """Read the communication scheme from package configuration files.

    This function checks multiple sources to determine the communication scheme:
    1. For servers: fed_server.json (service.scheme)
    2. For all packages: comm_config.json in local/ or startup/ directories

    Args:
        package_path: Path to the package directory
        config_name: Name of the configuration file (fed_server.json, fed_client.json, fed_admin.json)
        default_scheme: Default scheme to return if no scheme is found

    Returns:
        The communication scheme (e.g., "grpc", "http")
    """
    # First try to read from fed_xxx.json
    startup = os.path.join(package_path, "startup")
    fed_config_file = os.path.join(startup, config_name)
    if os.path.exists(fed_config_file):
        try:
            with open(fed_config_file, "r") as f:
                fed_config = json.load(f)
                server_conf = fed_config.get("servers", [{}])[0]
                service_config = server_conf.get("service", {})
                scheme = service_config.get("scheme")
                if scheme:
                    return scheme.lower()
        except Exception:
            pass

    return default_scheme
