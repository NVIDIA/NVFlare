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
import os
import random
import socket
import ssl
from ssl import SSLContext
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlparse

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.utils.argument_utils import str2bool
from nvflare.security.logging import secure_format_exception

log = logging.getLogger(__name__)

LO_PORT = 1025
HI_PORT = 65535
MAX_ITER_SIZE = 10
RANDOM_TRIES = 20
BIND_TIME_OUT = 5
SECURE_SCHEMES = {"https", "wss", "grpcs", "agrpcs", "ngrpcs", "stcp", "satcp"}

# GRPC can't handle frame size over 2G. So the limit is set to (2G-2M)
MAX_FRAME_SIZE = 2 * 1024 * 1024 * 1024 - (2 * 1024 * 1024)
MAX_HEADER_SIZE = 1024 * 1024
MAX_PAYLOAD_SIZE = MAX_FRAME_SIZE - 16 - MAX_HEADER_SIZE

SSL_SERVER_PRIVATE_KEY = "server.key"
SSL_SERVER_CERT = "server.crt"
SSL_CLIENT_PRIVATE_KEY = "client.key"
SSL_CLIENT_CERT = "client.crt"
SSL_ROOT_CERT = "rootCA.pem"
CUSTOM_ROOT_CERT = "customRootCA.pem"


def ssl_required(params: dict) -> bool:
    """Check if SSL is required"""
    scheme = params.get(DriverParams.SCHEME.value, None)
    return scheme in SECURE_SCHEMES or str2bool(params.get(DriverParams.SECURE.value))


def get_ssl_context(params: dict, ssl_server: bool) -> Optional[SSLContext]:
    if not ssl_required(params):
        params[DriverParams.IMPLEMENTED_CONN_SEC.value] = "clear"
        return None

    conn_security = params.get(DriverParams.CONNECTION_SECURITY.value, ConnectionSecurity.MTLS)
    if ssl_server:
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ca_path = params.get(DriverParams.CA_CERT.value)
        cert_path = params.get(DriverParams.SERVER_CERT.value)
        key_path = params.get(DriverParams.SERVER_KEY.value)

        if not cert_path or not key_path:
            raise RuntimeError(f"not cert or key for SSL server: {params=}")

        if conn_security == ConnectionSecurity.TLS:
            # do not require client auth
            ctx.verify_mode = ssl.CERT_NONE
            params[DriverParams.IMPLEMENTED_CONN_SEC] = "Server TLS: client auth not required"
        else:
            ctx.verify_mode = ssl.CERT_REQUIRED
            params[DriverParams.IMPLEMENTED_CONN_SEC] = "Server mTLS: client auth required"
    else:
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ctx.verify_mode = ssl.CERT_REQUIRED
        if conn_security == ConnectionSecurity.TLS:
            # one-way SSL: use custom CA cert if provided
            params[DriverParams.IMPLEMENTED_CONN_SEC] = "Client TLS: Custom CA Cert used"
            ca_path = params.get(DriverParams.CUSTOM_CA_CERT)
            if not ca_path:
                # no custom CA cert: use provisioned CA cert
                ca_path = params.get(DriverParams.CA_CERT.value)
                params[DriverParams.IMPLEMENTED_CONN_SEC] = "Client TLS: Flare CA Cert used"
            cert_path = None
            key_path = None
        else:
            # two-way SSL: use provisioned cert
            ca_path = params.get(DriverParams.CA_CERT.value)
            cert_path = params.get(DriverParams.CLIENT_CERT.value)
            key_path = params.get(DriverParams.CLIENT_KEY.value)
            params[DriverParams.IMPLEMENTED_CONN_SEC] = "Client mTLS: Flare credentials used"

    if not ca_path:
        scheme = params.get(DriverParams.SCHEME.value, "Unknown")
        role = "Server" if ssl_server else "Client"
        raise CommError(CommError.BAD_CONFIG, f"{role} certificate parameters are missing for scheme {scheme}")

    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.load_verify_locations(ca_path)
    if cert_path:
        ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)

    return ctx


def get_address(params: dict) -> str:
    host = params.get(DriverParams.HOST.value, "0.0.0.0")
    port = params.get(DriverParams.PORT.value, 0)
    if not host:
        host = "0.0.0.0"
    return f"{host}:{port}"


def parse_port_range(entry: Any):

    if isinstance(entry, int):
        return range(entry, entry + 1)

    parts = entry.split("-")
    if len(parts) == 1:
        num = int(parts[0])
        return range(num, num + 1)
    lo = int(parts[0]) if parts[0] else LO_PORT
    hi = int(parts[1]) if parts[1] else HI_PORT
    return range(lo, hi + 1)


def parse_port_list(ranges: Any) -> list:
    all_ranges = []
    if isinstance(ranges, list):
        for r in ranges:
            all_ranges.append(parse_port_range(r))
    else:
        all_ranges.append(parse_port_range(ranges))

    return all_ranges


def check_tcp_port(port) -> bool:
    result = False
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(BIND_TIME_OUT)
    try:
        s.bind(("", port))
        result = True
    except Exception as e:
        log.debug(f"Port {port} binding error: {secure_format_exception(e)}")
    finally:
        s.close()

    return result


def get_open_tcp_port(resources: dict) -> Optional[int]:

    port = resources.get(DriverParams.PORT)
    if port:
        return port

    ports = resources.get(DriverParams.PORTS)
    if ports:
        all_ports = parse_port_list(ports)
    else:
        all_ports = [range(LO_PORT, HI_PORT + 1)]

    for port_range in all_ports:
        if len(port_range) <= MAX_ITER_SIZE:
            for port in port_range:
                if check_tcp_port(port):
                    return port
        else:
            for i in range(RANDOM_TRIES):
                port = random.randint(port_range.start, port_range.stop - 1)
                if check_tcp_port(port):
                    return port

    return None


def parse_url(url: str) -> dict:
    """Parse URL into a dictionary, saving original URL also"""
    if not url:
        return {}

    params = {DriverParams.URL.value: url}
    parsed_url = urlparse(url)
    params[DriverParams.SCHEME.value] = parsed_url.scheme
    parts = parsed_url.netloc.split(":")
    if len(parts) >= 1:
        host = parts[0]
        # Host is required in URL. 0 is used as the placeholder for empty host
        if host == "0":
            host = ""
        params[DriverParams.HOST.value] = host
    if len(parts) >= 2:
        params[DriverParams.PORT.value] = parts[1]

    params[DriverParams.PATH.value] = parsed_url.path
    params[DriverParams.PARAMS.value] = parsed_url.params
    params[DriverParams.QUERY.value] = parsed_url.query
    params[DriverParams.FRAG.value] = parsed_url.fragment

    if parsed_url.query:
        for k, v in parse_qsl(parsed_url.query):
            # Only last one is saved if duplicate keys
            params[k] = v

    return params


def encode_url(params: dict) -> str:

    temp = params.copy()

    # Original URL is not needed
    temp.pop(DriverParams.URL.value, None)

    scheme = temp.pop(DriverParams.SCHEME.value, None)
    host = temp.pop(DriverParams.HOST.value, None)
    if not host:
        host = "0"
    port = temp.pop(DriverParams.PORT.value, None)
    path = temp.pop(DriverParams.PATH.value, None)
    parameters = temp.pop(DriverParams.PARAMS.value, None)
    # Encoded query is not needed
    temp.pop(DriverParams.QUERY.value, None)
    frag = temp.pop(DriverParams.FRAG.value, None)

    url = f"{scheme}://{host}"
    if port:
        url += ":" + str(port)

    if path:
        url += path

    if parameters:
        url += ";" + parameters

    if temp:
        url += "?" + urlencode(temp)

    if frag:
        url += "#" + frag

    return url


def short_url(params: dict) -> str:
    """Get a short url to be used in logs"""

    url = params.get(DriverParams.URL.value)
    if url:
        return url

    subset = {
        k: params[k]
        for k in {DriverParams.SCHEME.value, DriverParams.HOST.value, DriverParams.PORT.value, DriverParams.PATH.value}
    }

    return encode_url(subset)


def get_tcp_urls(scheme: str, resources: dict) -> (str, str):
    """Generate URL pairs for connecting and listening for TCP-based protocols

    Args:
        scheme: The transport scheme
        resources: The resource restrictions like port ranges

    Returns:
        a tuple with connecting and listening URL
    Raises:
        CommError: If any error happens while sending the request
    """

    host = resources.get("host") if resources else None
    if not host:
        host = "localhost"

    port = get_open_tcp_port(resources)
    if not port:
        raise CommError(CommError.BAD_CONFIG, "Can't find an open port in the specified range")

    # Always listen on all interfaces
    listening_url = f"{scheme}://0:{port}"
    connect_url = f"{scheme}://{host}:{port}"

    return connect_url, listening_url


def enhance_credential_info(params: dict):
    """Enhance the params by loading additional cert and key from the folder that contains the CA cert.

    This is necessary because the params initially only contains basic credentials:
    - for server, only CA cert, and the server's cert and key;
    - for client, only CA cert, the client's cert and key.

    However, a client could also behave like a server for other processes, and could have a server cert as well.
    This function loads all certs and keys, regardless the role of the process.

    Args:
        params: the dict that contains initial credentials

    Returns: None
    """

    # Must have CA since all other certs/keys are assumed to be in the same folder as the CA cert.
    ca_path = params.get(DriverParams.CA_CERT.value)
    if not ca_path:
        return

    # assume all SSL credential files are in the same folder with CA cert
    cred_folder = os.path.dirname(ca_path)

    client_cert_path = params.get(DriverParams.CLIENT_CERT.value)
    if not client_cert_path:
        # see whether the client cert file exists
        client_cert_path = os.path.join(cred_folder, SSL_CLIENT_CERT)
        if os.path.exists(client_cert_path):
            params[DriverParams.CLIENT_CERT.value] = client_cert_path

    client_key_path = params.get(DriverParams.CLIENT_KEY.value)
    if not client_key_path:
        # see whether the client key file exists
        client_key_path = os.path.join(cred_folder, SSL_CLIENT_PRIVATE_KEY)
        if os.path.exists(client_key_path):
            params[DriverParams.CLIENT_KEY.value] = client_key_path

    server_cert_path = params.get(DriverParams.SERVER_CERT.value)
    if not server_cert_path:
        # see whether the server cert file exists
        server_cert_path = os.path.join(cred_folder, SSL_SERVER_CERT)
        if os.path.exists(server_cert_path):
            params[DriverParams.SERVER_CERT.value] = server_cert_path

    server_key_path = params.get(DriverParams.SERVER_KEY.value)
    if not server_key_path:
        # see whether the server key file exists
        server_key_path = os.path.join(cred_folder, SSL_SERVER_PRIVATE_KEY)
        if os.path.exists(server_key_path):
            params[DriverParams.SERVER_KEY.value] = server_key_path

    custom_ca_cert_path = params.get(DriverParams.CUSTOM_CA_CERT.value)
    if not custom_ca_cert_path:
        # see whether the custom CA cert file exists
        custom_ca_cert_path = os.path.join(cred_folder, CUSTOM_ROOT_CERT)
        if os.path.exists(custom_ca_cert_path):
            params[DriverParams.CUSTOM_CA_CERT.value] = custom_ca_cert_path
