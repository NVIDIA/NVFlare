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
from nvflare.fuel.f3.driver import DriverSpec
from nvflare.fuel.f3.endpoint import Endpoint


class ConnManager:

    def __init__(self, local_endpoint: Endpoint):
        self.local_endpoint = local_endpoint
        connectors = []
        listeners = []

    def add_connector(self, driver: DriverSpec):
        self.connectors.append(driver)

    def add_listeners(self, driver: DriverSpec):
        self.listeners.append(driver)
