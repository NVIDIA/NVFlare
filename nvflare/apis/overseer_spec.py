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

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict

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
    def initialize(self, fl_ctx: FLContext):
        pass

    def set_secure_context(self, ca_path: str, cert_path: str = "", prv_key_path: str = ""):
        pass

    def start(self, update_callback=None, conditional_cb=False):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def end(self):
        pass

    def get_primary_sp(self) -> SP:
        """Return current primary service provider.

        If primary sp not available, such as not reported by SD, connection to SD not established yet
        the name and ports will be empty strings.
        """
        pass

    def promote_sp(self, sp_end_point, headers=None):
        pass

    def add_payload(self, payload: Dict[str, Any]):
        pass

    def get_overseer_status(self) -> Dict[str, Any]:
        """

        Returns:
            Dict[str, Any]: [description]
        """
        pass
