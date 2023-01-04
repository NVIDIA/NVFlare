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
import logging
import random
import socket
import ssl
from ssl import SSLContext
from typing import Any, Optional

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.driver import DriverParams

log = logging.getLogger(__name__)

LO_PORT = 1025
HI_PORT = 65535
MAX_ITER_SIZE = 10
RANDOM_TRIES = 20


def get_ssl_context(params: dict, server: bool) -> Optional[SSLContext]:
    scheme = params.get(DriverParams.SCHEME.value)
    if scheme not in ("https", "wss", "stcp"):
        return None

    ca_path = params.get(DriverParams.CA_CERT.value)
    if server:
        cert_path = params.get(DriverParams.SERVER_CERT.value)
        key_path = params.get(DriverParams.SERVER_KEY.value)
    else:
        cert_path = params.get(DriverParams.CLIENT_CERT.value)
        key_path = params.get(DriverParams.CLIENT_KEY.value)

    if not all([ca_path, cert_path, key_path]):
        raise CommError(CommError.BAD_CONFIG, f"Certificate parameters are required for scheme {scheme}")

    if server:
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    else:
        ctx = ssl.create_default_context()

    # This feature is only supported on 3.7+
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = False
    ctx.load_verify_locations(ca_path)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)

    return ctx


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
    try:
        s.bind(("", port))
        result = True
    except Exception as e:
        log.debug(f"Port {port} binding error: {e}")
    s.close()

    return result


def get_open_tcp_port(resources: dict) -> Optional[int]:

    port = resources.get(DriverParams.PORT)
    if port:
        return port

    ports = resources.get(DriverParams.PORTS)
    if not ports:
        port = random.randint(LO_PORT, HI_PORT)
        return port

    all_ports = parse_port_list(ports)

    for port_range in all_ports:
        if len(port_range) <= MAX_ITER_SIZE:
            for port in port_range:
                if check_tcp_port(port):
                    return port
        else:
            for i in range(RANDOM_TRIES):
                port = random.randint(port_range.start, port_range.stop)
                if check_tcp_port(port):
                    return port

    return None


def get_client_ip():
    """Return localhost IP.

    More robust than ``socket.gethostbyname(socket.gethostname())``. See
    https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib/28950776#28950776
    for more details.

    Returns:
        The host IP

    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # doesn't even have to be reachable
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip
