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
import shutil
import tempfile

from nvflare.apis.event_type import EventType
from nvflare.apis.utils.common_utils import get_open_ports
from nvflare.private.fed.client.admin import FedAdminAgent
from nvflare.private.fed.client.admin_msg_sender import AdminMessageSender
from nvflare.private.fed.client.client_req_processors import ClientRequestProcessors
from nvflare.private.fed.client.fed_client import FederatedClient
from nvflare.private.fed.simulator.simulator_client_engine import SimulatorClientEngine
from nvflare.private.fed.simulator.simulator_server import SimulatorServer

from ..server.server_train import create_admin_server
from .base_client_deployer import BaseClientDeployer
from .server_deployer import ServerDeployer


class SimulatorDeployer(ServerDeployer):
    def __init__(self):
        super().__init__()
        self.open_ports = get_open_ports(2)
        self.admin_storage = tempfile.mkdtemp()

    def create_fl_server(self, args, secure_train=False):
        simulator_server = self._create_simulator_server_config(self.admin_storage)

        self.services = SimulatorServer(
            project_name=simulator_server.get("name", ""),
            max_num_clients=simulator_server.get("max_num_clients", 100),
            cmd_modules=self.cmd_modules,
            args=args,
            secure_train=secure_train,
            enable_byoc=self.enable_byoc,
            snapshot_persistor=self.snapshot_persistor,
            overseer_agent=self.overseer_agent,
        )

        admin_server = create_admin_server(
            self.services,
            server_conf=simulator_server,
            args=args,
            secure_train=False,
        )
        admin_server.start()
        self.services.set_admin_server(admin_server)

        return simulator_server, self.services

    def create_fl_client(self, client_name, args):
        client_config, build_ctx = self._create_simulator_client_config(client_name)

        deployer = BaseClientDeployer()
        deployer.build(build_ctx)
        federated_client = deployer.create_fed_client(args)

        federated_client.register()
        federated_client.start_heartbeat()
        federated_client.run_manager = None

        return federated_client, client_config, args

    def create_admin_agent(self, server_args, federated_client: FederatedClient, args, rank=0):
        sender = AdminMessageSender(
            client_name=federated_client.token,
            server_args=server_args,
            secure=False,
        )
        client_engine = SimulatorClientEngine(federated_client, federated_client.token, sender, args, rank)
        admin_agent = FedAdminAgent(
            client_name="admin_agent",
            sender=sender,
            app_ctx=client_engine,
        )
        admin_agent.app_ctx.set_agent(admin_agent)
        federated_client.set_client_engine(client_engine)
        for processor in ClientRequestProcessors.request_processors:
            admin_agent.register_processor(processor)

        client_engine.fire_event(EventType.SYSTEM_START, client_engine.new_context())

        return admin_agent

    def _create_simulator_server_config(self, admin_storage):
        simulator_server = {
            "name": "simulator_server",
            "service": {
                "target": "localhost:" + str(self.open_ports[0]),
                "options": [
                    ["grpc.max_send_message_length", 2147483647],
                    ["grpc.max_receive_message_length", 2147483647],
                ],
            },
            "admin_host": "localhost",
            "admin_port": self.open_ports[1],
            "max_num_clients": 100,
            "heart_beat_timeout": 600,
            "num_server_workers": 4,
            "compression": "Gzip",
            "admin_storage": admin_storage,
            "download_job_url": "http://download.server.com/",
        }
        return simulator_server

    def _create_simulator_client_config(self, client_name):
        client_config = {
            "servers": [
                {
                    "name": "simulator_server",
                    "service": {
                        "target": "localhost:" + str(self.open_ports[0]),
                        "options": [
                            ["grpc.max_send_message_length", 2147483647],
                            ["grpc.max_receive_message_length", 2147483647],
                        ],
                    },
                }
            ],
            "client": {"retry_timeout": 30, "compression": "Gzip"},
        }

        build_ctx = {
            "client_name": client_name,
            "server_config": client_config.get("servers", []),
            "client_config": client_config["client"],
            "server_host": None,
            "secure_train": False,
            "enable_byoc": True,
            "overseer_agent": None,
            "client_components": {},
            "client_handlers": None,
        }

        return client_config, build_ctx

    def close(self):
        shutil.rmtree(self.admin_storage)
        super().close()
