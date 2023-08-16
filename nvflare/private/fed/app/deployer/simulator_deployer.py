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

import json
import os
import shutil
import tempfile

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.utils.dict_utils import augment
from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.private.fed.app.utils import create_admin_server
from nvflare.private.fed.simulator.simulator_client_engine import SimulatorParentClientEngine
from nvflare.private.fed.simulator.simulator_server import SimulatorServer
from nvflare.security.logging import secure_format_exception

from .base_client_deployer import BaseClientDeployer
from .server_deployer import ServerDeployer


class SimulatorDeployer(ServerDeployer):
    def __init__(self):
        super().__init__()
        self.open_ports = get_open_ports(2)
        self.admin_storage = tempfile.mkdtemp()

    def create_fl_server(self, args, secure_train=False):
        simulator_server = self._create_simulator_server_config(self.admin_storage, args.max_clients)

        heart_beat_timeout = simulator_server.get("heart_beat_timeout", 600)

        services = SimulatorServer(
            project_name=simulator_server.get("name", ""),
            max_num_clients=simulator_server.get("max_num_clients", 100),
            cmd_modules=self.cmd_modules,
            args=args,
            secure_train=secure_train,
            snapshot_persistor=self.snapshot_persistor,
            overseer_agent=self.overseer_agent,
            heart_beat_timeout=heart_beat_timeout,
        )
        services.deploy(args, grpc_args=simulator_server)

        admin_server = create_admin_server(
            services,
            server_conf=simulator_server,
            args=args,
            secure_train=False,
        )
        admin_server.start()
        services.set_admin_server(admin_server)

        # mpm.add_cleanup_cb(admin_server.stop)

        return simulator_server, services

    def create_fl_client(self, client_name, args):
        client_config, build_ctx = self._create_simulator_client_config(client_name, args)

        deployer = BaseClientDeployer()
        deployer.build(build_ctx)
        federated_client = deployer.create_fed_client(args)

        self._create_client_cell(client_config, client_name, federated_client)

        federated_client.register()

        client_engine = SimulatorParentClientEngine(federated_client, federated_client.token, args)
        federated_client.set_client_engine(client_engine)
        # federated_client.start_heartbeat()
        federated_client.run_manager = None

        return federated_client, client_config, args, build_ctx

    def _create_client_cell(self, client_config, client_name, federated_client):
        target = client_config["servers"][0].get("service").get("target")
        scheme = client_config["servers"][0].get("service").get("scheme", "grpc")
        credentials = {}
        parent_url = None
        cell = Cell(
            fqcn=client_name,
            root_url=scheme + "://" + target,
            secure=self.secure_train,
            credentials=credentials,
            create_internal_listener=False,
            parent_url=parent_url,
        )
        cell.start()
        federated_client.cell = cell
        federated_client.communicator.cell = cell
        # if self.engine:
        #     self.engine.admin_agent.register_cell_cb()

        mpm.add_cleanup_cb(cell.stop)

    def _create_simulator_server_config(self, admin_storage, max_clients):
        simulator_server = {
            "name": "simulator_server",
            "service": {
                "target": "localhost:" + str(self.open_ports[0]),
                "scheme": "tcp",
            },
            "admin_host": "localhost",
            "admin_port": self.open_ports[1],
            "max_num_clients": max_clients,
            "heart_beat_timeout": 600,
            "num_server_workers": 4,
            "compression": "Gzip",
            "admin_storage": admin_storage,
            "download_job_url": "http://download.server.com/",
        }
        return simulator_server

    def _create_simulator_client_config(self, client_name, args):
        client_config = {
            "servers": [
                {
                    "name": "simulator_server",
                    "service": {
                        "target": "localhost:" + str(self.open_ports[0]),
                        "scheme": "tcp",
                    },
                }
            ],
            "client": {"retry_timeout": 30, "compression": "Gzip"},
        }

        resources = os.path.join(args.workspace, "local/resources.json")
        if os.path.exists(resources):
            with open(resources) as file:
                try:
                    data = json.load(file)
                    augment(to_dict=client_config, from_dict=data, from_override_to=False)
                except Exception as e:
                    raise RuntimeError(f"Error processing config file {resources}: {secure_format_exception(e)}")

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
