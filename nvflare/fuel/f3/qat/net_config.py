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

from nvflare.fuel.utils.config_service import ConfigService


class NetConfig:
    def __init__(self, config_file_name="net_config.json"):
        self.config = ConfigService.load_config_dict(config_file_name)
        if not self.config:
            raise RuntimeError(f"cannot load {config_file_name}")

    def get_root_url(self):
        return self.config.get("root_url")

    def get_children(self, me: str):
        my_config = self.config.get(me)
        if my_config:
            return my_config.get("children", [])
        else:
            return []

    def get_clients(self):
        server_config = self.config.get("server")
        if server_config:
            return server_config.get("clients", [])
        else:
            return []

    def get_admin(self) -> (str, str):
        admin_config = self.config.get("admin")
        if admin_config:
            return admin_config.get("host"), admin_config.get("port")
        return "", ""
