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
from dataclasses import dataclass
from enum import IntEnum

from nvflare.fuel.f3.drivers.net_utils import short_url


class Mode(IntEnum):
    ACTIVE = 0
    PASSIVE = 1


@dataclass
class ConnectorInfo:
    """Connector information"""

    handle: str
    # noinspection PyUnresolvedReferences
    driver: "Driver"
    params: dict
    mode: Mode
    total_conns: int
    curr_conns: int
    started: bool
    stopping: bool

    def __str__(self):
        url = short_url(self.params)
        return f"[{self.handle} {self.mode.name} {url}]"
