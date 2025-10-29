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
from typing import Tuple, Union

from nvflare.fox.api.app import ClientApp, ServerApp
from nvflare.fox.api.strategy import Strategy

from .simulator import Simulator


class Simulator2:

    def __init__(
        self,
        root_dir: str,
        experiment_name: str,
        strategy: Strategy,
        server_objects: dict[str, object],
        client_objects: dict[str, object],
        max_workers: int = 100,
        num_clients: Union[int, Tuple[int, int]] = 2,
    ):
        server_app: ServerApp = ServerApp(strategy=strategy)
        client_app: ClientApp = ClientApp()

        for name, obj in server_objects.items():
            server_app.add_collab_object(name, obj)

        for name, obj in client_objects.items():
            client_app.add_collab_object(name, obj)

        self.simulator = Simulator(
            root_dir=root_dir,
            experiment_name=experiment_name,
            server_app=server_app,
            client_app=client_app,
            max_workers=max_workers,
            num_clients=num_clients,
        )

    def run(self):
        return self.simulator.run()
