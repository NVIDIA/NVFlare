# Copyright (c) 2021, NVIDIA CORPORATION.
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

from nvflare.private.fed.server.server_cmd_modules import ServerCommandModules
from nvflare.private.fed.server.fed_server import FederatedServer


class ServerDeployer:
    def __init__(self):
        # self.app_validator = None
        self.services = None
        self.cmd_modules = ServerCommandModules.cmd_modules

    def build(self, build_ctx):
        self.server_config = build_ctx["server_config"]
        self.secure_train = build_ctx["secure_train"]
        self.config_validator = build_ctx["app_validator"]
        self.host = build_ctx["server_host"]
        self.enable_byoc = build_ctx["enable_byoc"]

    def train(self):
        self.services = self.deploy()
        self.start_training(self.services)

    def start_training(self, services):
        services.start()

    # def create_builder(self, services):
    #     services.set_builder(builder=self.model)

    def create_fl_server(self, args, secure_train=False):
        # We only deploy the first server right now .....
        first_server = sorted(self.server_config)[0]
        wait_after_min_clients = first_server.get("wait_after_min_clients", 10)
        heart_beat_timeout = 600
        if first_server["heart_beat_timeout"]:
            heart_beat_timeout = first_server["heart_beat_timeout"]

        if self.host:
            target = first_server["service"].get("target", None)
            first_server["service"]["target"] = self.host + ":" + target.split(":")[1]

        services = FederatedServer(
            project_name=first_server.get("name", ""),
            min_num_clients=first_server.get("min_num_clients", 1),
            max_num_clients=first_server.get("max_num_clients", 100),
            wait_after_min_clients=wait_after_min_clients,
            cmd_modules=self.cmd_modules,
            heart_beat_timeout=heart_beat_timeout,
            args=args,
            secure_train=secure_train,
            enable_byoc=self.enable_byoc
        )
        return first_server, services

    def deploy(self, args):
        first_server, services = self.create_fl_server(args, secure_train=self.secure_train)
        services.deploy(grpc_args=first_server, secure_train=self.secure_train)
        # self.create_builder(services)
        print("deployed FL server trainer.")
        return services

    def close(self):
        if self.services:
            self.services.close()
