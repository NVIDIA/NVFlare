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
from typing import Any, Dict, List

from nvflare.fuel.f3.drivers.base_driver import BaseDriver
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo
from nvflare.fuel.f3.drivers.driver_params import DriverCap


class WarpDriver(BaseDriver):
    """A dummy driver to test custom driver loading"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def supported_transports() -> List[str]:
        return ["warp"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {DriverCap.SEND_HEARTBEAT.value: True, DriverCap.SUPPORT_SSL.value: False}

    def listen(self, connector: ConnectorInfo):
        self.connector = connector

    def connect(self, connector: ConnectorInfo):
        self.connector = connector

    def shutdown(self):
        self.close_all()

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        return "warp:enterprise"
