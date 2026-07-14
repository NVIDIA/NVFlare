# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Dict, List, Optional, Tuple, Union

from nvflare.collab.api.app import ClientApp, ServerApp
from nvflare.collab.runtime.local.app_runner import AppRunner


class InProcessRunner:
    """High-level Collab simulation runner.

    Provides a simple API to run Collab simulations with server and client objects.
    Supports both in-process and subprocess execution modes.
    """

    def __init__(
        self,
        root_dir: str,
        experiment_name: str,
        server,
        client,
        server_objects: Dict[str, object] = None,
        client_objects: Dict[str, object] = None,
        max_workers: int = 100,
        num_clients: Union[int, Tuple[int, int]] = 2,
        per_site_props: Optional[Dict[str, dict]] = None,
        # Subprocess execution options
        inprocess: bool = True,
        run_cmd: Optional[str] = None,
        training_module: Optional[str] = None,
        subprocess_timeout: float = 300.0,
    ):
        """Initialize InProcessRunner.

        Args:
            root_dir: Root directory for simulation output.
            experiment_name: Name of the experiment.
            server: Server object with @collab.main methods.
            client: Client object with @collab.publish methods.
            server_objects: Additional server collab objects.
            client_objects: Additional client collab objects.
            max_workers: Maximum worker threads.
            num_clients: Number of clients or (height, width) tuple.
            inprocess: If True, execute in-process. If False, use subprocess.
            run_cmd: Command prefix for subprocess (e.g., "torchrun --nproc_per_node=4").
            training_module: Python module containing @collab.publish methods
                            (required when inprocess=False).
            subprocess_timeout: Timeout for subprocess operations.
        """
        server_app: ServerApp = ServerApp(server)
        client_app: ClientApp = ClientApp(client)

        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.max_workers = max_workers
        self.num_clients = num_clients
        self.per_site_props = per_site_props

        # Subprocess options
        self.inprocess = inprocess
        self.run_cmd = run_cmd
        self.training_module = training_module
        self.subprocess_timeout = subprocess_timeout

        if server_objects:
            for name, obj in server_objects.items():
                server_app.add_collab_object(name, obj)

        if client_objects:
            for name, obj in client_objects.items():
                client_app.add_collab_object(name, obj)

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

    def set_server_resource_dirs(self, resource_dirs):
        self.server_app.set_resource_dirs(resource_dirs)

    def set_client_resource_dirs(self, resource_dirs):
        self.client_app.set_resource_dirs(resource_dirs)

    def run(self):
        runner = AppRunner(
            root_dir=self.root_dir,
            experiment_name=self.experiment_name,
            server_app=self.server_app,
            client_app=self.client_app,
            max_workers=self.max_workers,
            num_clients=self.num_clients,
            per_site_props=self.per_site_props,
            # Pass subprocess options
            inprocess=self.inprocess,
            run_cmd=self.run_cmd,
            training_module=self.training_module,
            subprocess_timeout=self.subprocess_timeout,
        )
        return runner.run()
