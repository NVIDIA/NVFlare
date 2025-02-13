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

import socket


def get_open_ports(number) -> list:
    """Gets the number of open ports from the system.
    Note: the returned ports are only guaranteed to be accessible from the same host.

    Args:
        number: number of ports
    Returns:
        A list of open_ports
    """
    ports = []
    sockets = []
    for i in range(number):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.listen(1)

        # to prevent the same port number used multiple times, we only close it after all ports are obtained
        sockets.append(s)
        port = s.getsockname()[1]
        if port > 0:
            ports.append(port)

    # close obtained ports
    for s in sockets:
        s.close()

    if len(ports) != number:
        raise RuntimeError(
            "Could not get enough open ports from the system. Needed {} but got {}.".format(number, len(ports))
        )
    return ports


def get_local_addresses(number, host_name=None) -> list:
    """Return a list of local addresses

    Args:
        number: the number of addresses wanted
        host_name: name of the local host

    Returns:

    """
    if not host_name:
        host_name = "127.0.0.1"
    ports = get_open_ports(number)
    return [f"{host_name}:{p}" for p in ports]
