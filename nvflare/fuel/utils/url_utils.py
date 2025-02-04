# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

_SECURE_SCHEME_MAPPING = {"tcp": "stcp", "grpc": "grpcs", "http": "https"}
_CLEAR_SCHEME_MAPPING = {"stcp": "tcp", "grpcs": "grpc", "https": "http"}


def make_url(scheme: str, address, secure: bool) -> str:
    """Make a full URL based on specified info

    Args:
        scheme: scheme of the url
        address: host address. Multiple formats are supported:
            str: this is a string that contains host name and optionally port number (e.g. localhost:1234)
            dict: contains item "host" and optionally "port"
            tuple or list: contains 1 or 2 items for host and port
        secure: whether secure connection is required

    Returns:

    """
    if secure:
        if scheme in _SECURE_SCHEME_MAPPING.values():
            # already secure scheme
            secure_scheme = scheme
        else:
            secure_scheme = _SECURE_SCHEME_MAPPING.get(scheme)

        if not secure_scheme:
            raise ValueError(f"unsupported scheme '{scheme}'")

        scheme = secure_scheme
    else:
        if scheme in _CLEAR_SCHEME_MAPPING.values():
            # already clear scheme
            clear_scheme = scheme
        else:
            clear_scheme = _CLEAR_SCHEME_MAPPING.get(scheme)

        if not clear_scheme:
            raise ValueError(f"unsupported scheme '{scheme}'")

        scheme = clear_scheme

    if isinstance(address, str):
        if not address:
            raise ValueError("address must not be empty")
        return f"{scheme}://{address}"
    else:
        port = None
        if isinstance(address, (tuple, list)):
            if len(address) < 1:
                raise ValueError("address must not be empty")
            if len(address) > 2:
                raise ValueError(f"invalid address {address}")
            host = address[0]
            if len(address) > 1:
                port = address[1]
        elif isinstance(address, dict):
            if len(address) < 1:
                raise ValueError("address must not be empty")
            if len(address) > 2:
                raise ValueError(f"invalid address {address}")

            host = address.get("host")
            if not host:
                raise ValueError(f"invalid address {address}: missing 'host'")

            port = address.get("port")
            if not port and len(address) > 1:
                raise ValueError(f"invalid address {address}: missing 'port'")
        else:
            raise ValueError(f"invalid address: {address}")

        if not isinstance(host, str):
            raise ValueError(f"invalid host '{host}': must be str but got {type(host)}")

        if port:
            if not isinstance(port, (str, int)):
                raise ValueError(f"invalid port '{port}': must be str or int but got {type(port)}")
            port_str = f":{port}"
        else:
            port_str = ""
        return f"{scheme}://{host}{port_str}"
