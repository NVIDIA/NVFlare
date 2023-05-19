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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from requests import Response

from .fl_context import FLContext


@dataclass
class SP:
    name: str = ""
    fl_port: str = ""
    admin_port: str = ""
    service_session_id: str = ""
    primary: bool = False
    props: dict = field(default_factory=dict)


class OverseerAgent(ABC):
    def __init__(self):
        self.overseer_info = {}

    def initialize(self, fl_ctx: FLContext):
        pass

    @abstractmethod
    def set_secure_context(self, ca_path: str, cert_path: str = "", prv_key_path: str = ""):
        pass

    @abstractmethod
    def start(self, update_callback=None, conditional_cb=False):
        pass

    @abstractmethod
    def pause(self):
        pass

    @abstractmethod
    def resume(self):
        pass

    @abstractmethod
    def end(self):
        pass

    @abstractmethod
    def is_shutdown(self) -> bool:
        """Return whether the agent receives a shutdown request."""
        pass

    @abstractmethod
    def get_primary_sp(self) -> SP:
        """Return current primary service provider.

        If primary sp not available, such as not reported by SD, connection to SD not established yet
        the name and ports will be empty strings.
        """
        pass

    @abstractmethod
    def promote_sp(self, sp_end_point, headers=None) -> Response:
        pass

    @abstractmethod
    def set_state(self, state) -> Response:
        pass
