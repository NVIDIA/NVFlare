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

from nvflare.apis.overseer_spec import OverseerAgent
from nvflare.fuel.hci.reg import CommandModule
from nvflare.ha.ha_admin_cmds import HACommandModule

from .api_spec import ServiceFinder


class ServiceFinderByOverseer(ServiceFinder):
    def __init__(self, overseer_agent: OverseerAgent):
        assert isinstance(overseer_agent, OverseerAgent), "overseer_agent must be OverseerAgent but got {}".format(
            type(overseer_agent)
        )

        self.overseer_agent = overseer_agent
        self.sp_address_changed_cb = None
        self.host = ""
        self.port = 0
        self.ssid = ""

    def set_secure_context(self, ca_cert_path: str, cert_path: str, private_key_path: str):
        self.overseer_agent.set_secure_context(ca_path=ca_cert_path, cert_path=cert_path, prv_key_path=private_key_path)

    def get_command_module(self) -> CommandModule:
        return HACommandModule(self.overseer_agent)

    def start(self, sp_address_changed_cb):
        if not callable(sp_address_changed_cb):
            raise TypeError("sp_address_changed_cb must be callable but got {}".format(type(sp_address_changed_cb)))

        self.sp_address_changed_cb = sp_address_changed_cb
        self.overseer_agent.start(self._overseer_callback)

    def _overseer_callback(self, overseer_agent):
        sp = overseer_agent.get_primary_sp()
        if not sp or not sp.primary:
            return

        port_num = int(sp.admin_port)
        if self.host != sp.name or self.port != port_num or self.ssid != sp.service_session_id:
            # SP changed!
            self.host = sp.name
            self.port = port_num
            self.ssid = sp.service_session_id
            if self.sp_address_changed_cb is not None:
                self.sp_address_changed_cb(self.host, self.port, self.ssid)

    def stop(self):
        self.overseer_agent.end()
