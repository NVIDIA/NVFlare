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


def make_url(scheme: str, address, secure: bool) -> str:
    if secure:
        scheme = _SECURE_SCHEME_MAPPING.get(scheme)
        if not scheme:
            raise ValueError(f"unsupported scheme '{scheme}'")

    if isinstance(address, str):
        return f"{scheme}://{address}"
    else:
        port = None
        if isinstance(address, (tuple, list)):
            host = address[0]
            if len(address) > 1:
                port = address[1]
        elif isinstance(address, dict):
            host = address["host"]
            port = address.get("port")
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
