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
from typing import List, Tuple, Union

from nvflare.fox.api.app import ClientApp, ServerApp

from .simulator import Simulator as Sim1


class Simulator:

    def __init__(
        self,
        root_dir: str,
        experiment_name: str,
        server,
        client,
        server_objects: dict[str, object] = None,
        client_objects: dict[str, object] = None,
        max_workers: int = 100,
        num_clients: Union[int, Tuple[int, int]] = 2,
    ):
        server_app: ServerApp = ServerApp(server)
        client_app: ClientApp = ClientApp(client)

        if server_objects:
            for name, obj in server_objects.items():
                server_app.add_collab_object(name, obj)

        if client_objects:
            for name, obj in client_objects.items():
                client_app.add_collab_object(name, obj)

        self.simulator = Sim1(
            root_dir=root_dir,
            experiment_name=experiment_name,
            server_app=server_app,
            client_app=client_app,
            max_workers=max_workers,
            num_clients=num_clients,
        )

        self.server_app = server_app
        self.client_app = client_app

    def add_server_outgoing_call_filters(self, pattern: str, filters: List[object]):
        self.server_app.add_outgoing_call_filters(pattern, filters)

    def add_server_incoming_call_filters(self, pattern: str, filters: List[object]):
        self.server_app.add_incoming_call_filters(pattern, filters)

    def add_server_outgoing_result_filters(self, pattern: str, filters: List[object]):
        self.server_app.add_outgoing_result_filters(pattern, filters)

    def add_server_incoming_result_filters(self, pattern: str, filters: List[object]):
        self.server_app.add_incoming_result_filters(pattern, filters)

    def add_client_outgoing_call_filters(self, pattern: str, filters: List[object]):
        self.client_app.add_outgoing_call_filters(pattern, filters)

    def add_client_incoming_call_filters(self, pattern: str, filters: List[object]):
        self.client_app.add_incoming_call_filters(pattern, filters)

    def add_client_outgoing_result_filters(self, pattern: str, filters: List[object]):
        self.client_app.add_outgoing_result_filters(pattern, filters)

    def add_client_incoming_result_filters(self, pattern: str, filters: List[object]):
        self.client_app.add_incoming_result_filters(pattern, filters)

    def set_server_prop(self, name: str, value):
        self.server_app.set_prop(name, value)

    def set_client_prop(self, name: str, value):
        self.client_app.set_prop(name, value)

    def run(self):
        return self.simulator.run()
