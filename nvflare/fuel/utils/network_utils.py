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

    Args:
        number: number of ports
    Returns:
        A list of open_ports
    """
    ports = []
    for i in range(number):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()
        if port > 0:
            ports.append(port)
    if len(ports) != number:
        raise RuntimeError(
            "Could not get enough open ports from the system. Needed {} but got {}.".format(number, len(ports))
        )
    return ports
