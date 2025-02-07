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


import os
from enum import Enum
from typing import Optional

from nvflare.client.constants import CLIENT_API_CONFIG
from nvflare.fuel.data_event.data_bus import DataBus

from .api_spec import CLIENT_API_KEY, CLIENT_API_TYPE_KEY, APISpec
from .ex_process.api import ExProcessClientAPI
from .in_process.api import InProcessClientAPI

DEFAULT_CONFIG = f"config/{CLIENT_API_CONFIG}"
data_bus = DataBus()


class ClientAPIType(Enum):
    IN_PROCESS_API = "IN_PROCESS_API"
    EX_PROCESS_API = "EX_PROCESS_API"


class APIContext:
    def __init__(self, rank: Optional[str] = None, config_file: str = None):
        self.rank = rank
        self.config_file = config_file if config_file else DEFAULT_CONFIG

        api_type_name = os.environ.get(CLIENT_API_TYPE_KEY, ClientAPIType.IN_PROCESS_API.value)
        api_type = ClientAPIType(api_type_name)
        self.api = self._create_client_api(api_type)
        self.api.init(rank=self.rank)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup the client API when the context ends."""
        if self.api:
            self.api.shutdown()
            self.api = None

    def _create_client_api(self, api_type: ClientAPIType) -> APISpec:
        """Creates a new client_api based on the provided API type."""
        if api_type == ClientAPIType.IN_PROCESS_API:
            api = data_bus.get_data(CLIENT_API_KEY)
            if not isinstance(api, InProcessClientAPI):
                raise RuntimeError(f"api {api} is not a valid InProcessClientAPI")
            return api
        else:
            return ExProcessClientAPI(config_file=self.config_file)
