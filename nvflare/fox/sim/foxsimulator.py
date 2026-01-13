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
import copy
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

from nvflare.apis.signal import Signal
from nvflare.fox.api.app import App, ClientApp, ServerApp
from nvflare.fox.api.constants import MAKE_CLIENT_APP_METHOD, BackendType
from nvflare.fox.api.dec import get_object_collab_interface
from nvflare.fox.api.proxy import Proxy
from nvflare.fox.api.run_server import run_server
from nvflare.fox.sim.backend import SimBackend
from nvflare.fox.sim.ws import SimWorkspace
from nvflare.fox.sys.subprocess_launcher import SubprocessLauncher
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.utils.log_utils import get_obj_logger


class AppRunner:
    """Runner for Fox simulation that manages server and client apps.

    This is the simulation equivalent of FoxExecutor - it manages client-side
    execution and supports both in-process and subprocess modes.

    When inprocess=False, it creates a CoreCell for IPC and SubprocessLauncher
    for each client site to run training in separate processes (e.g., for torchrun).
    """

    def _prepare_app_backends(self, app: App, site_name: str = None):
        """Create SimBackend instances for an app.

        Args:
            app: The app to create backends for.
            site_name: The site name (used to look up subprocess launcher).

        Returns:
            Dictionary of SimBackend instances keyed by object name.
        """
        # Get subprocess launcher for this site if running in subprocess mode
        launcher = self._subprocess_launchers.get(site_name) if site_name else None

        bes = {"": SimBackend("", app, app, self.abort_signal, self.thread_executor, launcher)}
        targets = app.get_collab_objects()
        for name, obj in targets.items():
            bes[name] = SimBackend(name, app, obj, self.abort_signal, self.thread_executor, launcher)
        return bes

    @staticmethod
    def _prepare_proxy(for_app: App, target_app: App, backends: dict):
        app_proxy = Proxy(
            app=for_app,
            target_name=target_app.name,
            target_fqn=target_app.fqn,
            backend=backends[""],
            target_interface=get_object_collab_interface(target_app),
        )
        collab_objs = target_app.get_collab_objects()
        for name, obj in collab_objs.items():
            p = Proxy(
                app=for_app,
                target_name=f"{target_app.name}.{name}",
                target_fqn="",
                backend=backends[name],
                target_interface=get_object_collab_interface(obj),
            )
            app_proxy.add_child(name, p)
        return app_proxy

    def _make_app(self, site_name, fqn):
        """Make a new client app instance for the specified site

        Args:
            site_name: name of the site
            fqn: fully qualified name of the site

        Returns: a new instance of the app

        """
        # If the app contains "make_client_app" method, call it to make the app instance!
        # Otherwise, make the instance by deep copying.
        # If the client_app object cannot be deep-copied, then it must provide the make_client_app method.
        make_client_app_f = getattr(self.client_app, MAKE_CLIENT_APP_METHOD, None)
        if make_client_app_f and callable(make_client_app_f):
            app = make_client_app_f(site_name, BackendType.SIMULATION)
            if not isinstance(app, ClientApp):
                raise RuntimeError(f"result returned by {MAKE_CLIENT_APP_METHOD} must be ClientApp but got {type(app)}")
        else:
            try:
                app = copy.deepcopy(self.client_app)
            except Exception as ex:
                self.logger.error(
                    f"exception occurred {type(ex)} creating client app with deepcopy. "
                    f"Please implement the {MAKE_CLIENT_APP_METHOD} method in the client app class"
                )
                raise ex

        app.name = site_name
        app.set_fqn(fqn)
        app.set_backend_type(BackendType.SIMULATION)
        return app

    def _prepare_proxies(self, for_app: App, server_app: App, client_apps: dict, backends: dict):
        server_proxy = self._prepare_proxy(for_app, server_app, backends[server_app.name])
        client_proxies = []
        for name, app in client_apps.items():
            p = self._prepare_proxy(for_app, app, backends[name])
            client_proxies.append(p)

        return server_proxy, client_proxies

    def __init__(
        self,
        root_dir: str,
        experiment_name: str,
        server_app: ServerApp,
        client_app: ClientApp,
        max_workers: int = 100,
        num_clients: Union[int, Tuple[int, int]] = 2,
        # Subprocess execution options (like FoxExecutor)
        inprocess: bool = True,
        run_cmd: Optional[str] = None,
        training_module: Optional[str] = None,
        subprocess_timeout: float = 300.0,
    ):
        """Initialize AppRunner.

        Args:
            root_dir: Root directory for simulation.
            experiment_name: Name of the experiment.
            server_app: The server application.
            client_app: Template for client applications.
            max_workers: Maximum worker threads.
            num_clients: Number of clients or (height, width) tuple for hierarchy.
            inprocess: If True, execute in-process. If False, use subprocess.
            run_cmd: Command prefix for subprocess (e.g., "torchrun --nproc_per_node=4").
            training_module: Python module containing @fox.collab methods (required when inprocess=False).
            subprocess_timeout: Timeout for subprocess operations.
        """
        if not isinstance(server_app, ServerApp):
            raise ValueError(f"server_app must be ServerApp but got {type(server_app)}")

        if not isinstance(client_app, ClientApp):
            raise ValueError(f"client_app must be ClientApp but got {type(client_app)}")

        # Validate subprocess options
        if not inprocess and not training_module:
            raise ValueError("training_module is required when inprocess=False")

        self.logger = get_obj_logger(self)
        self.abort_signal = Signal()
        server_app.name = "server"
        server_app.set_fqn(server_app.name)
        server_app.set_backend_type(BackendType.SIMULATION)
        self.server_app = server_app
        self.client_app = client_app
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fox_call")

        # Subprocess options
        self.inprocess = inprocess
        self.run_cmd = run_cmd
        self.training_module = training_module
        self.subprocess_timeout = subprocess_timeout

        # Subprocess management
        self._subprocess_launchers: Dict[str, SubprocessLauncher] = {}
        self._ipc_cell: Optional[CoreCell] = None

        # Build client apps
        if isinstance(num_clients, int):
            if num_clients <= 0:
                raise ValueError(f"num_clients must > 0 but got {num_clients}")
            client_apps = {}
            for i in range(num_clients):
                name = f"site-{i + 1}"
                client_apps[name] = self._make_app(name, name)
        elif isinstance(num_clients, tuple):
            if len(num_clients) != 2:
                raise ValueError(f"num_clients must be an int or tuple(int, int) but got {num_clients}")

            # tuple of (height x width)
            height, num_children_per_parent = num_clients
            if not isinstance(height, int) or not isinstance(num_children_per_parent, int):
                raise ValueError(f"num_clients must be an int or tuple(int, int) but got {num_clients}")

            if height <= 0 or num_children_per_parent <= 0:
                raise ValueError(f"num_clients must contain positive ints but got {num_clients}")

            self.logger.info(f"creating clients {height} x {num_children_per_parent}")
            client_apps = self._build_hierarchical_clients(height, num_children_per_parent)
        else:
            raise ValueError(f"num_clients must be an int or tuple(int, int) but got {type(num_clients)}")

        self.logger.info(f"created client apps: {client_apps.keys()}")

        # Create backends for server
        backends = {server_app.name: self._prepare_app_backends(server_app)}

        # Create backends for clients (with subprocess launchers if needed)
        for name, app in client_apps.items():
            backends[name] = self._prepare_app_backends(app, site_name=name)

        exp_id = str(uuid.uuid4())

        for name, app in client_apps.items():
            server_proxy, client_proxies = self._prepare_proxies(app, server_app, client_apps, backends)
            ws = SimWorkspace(root_dir=root_dir, experiment_name=experiment_name, site_name=name, exp_id=exp_id)
            app.setup(ws, server_proxy, client_proxies, self.abort_signal)

        # prepare server
        server_proxy, client_proxies = self._prepare_proxies(server_app, server_app, client_apps, backends)
        ws = SimWorkspace(root_dir=root_dir, experiment_name=experiment_name, site_name=server_app.name, exp_id=exp_id)
        server_app.setup(ws, server_proxy, client_proxies, self.abort_signal)
        self.client_apps = client_apps
        self.exp_dir = ws.get_experiment_dir()

    def _build_hierarchical_clients(self, height: int, num_children_per_parent: int):
        client_apps = {}
        last_client_fqns = {}
        current_client_fqns = {}
        for i in range(height):
            if not last_client_fqns:
                for j in range(num_children_per_parent):
                    name = f"site-{j + 1}"
                    fqn = name
                    app = self._make_app(name, fqn)
                    client_apps[name] = app
                    current_client_fqns[fqn] = app
            else:
                for fqn, parent_app in last_client_fqns.items():
                    # create w clients for each parent
                    for k in range(num_children_per_parent):
                        child_name = f"{parent_app.name}-{k + 1}"
                        child_fqn = f"{fqn}.{child_name}"
                        app = self._make_app(child_name, child_fqn)
                        client_apps[child_name] = app
                        current_client_fqns[child_fqn] = app
            last_client_fqns = current_client_fqns
            current_client_fqns = {}
        return client_apps

    def _setup_ipc_cell(self):
        """Create a CoreCell for subprocess IPC communication."""
        # Create a simple CoreCell for local IPC
        # This acts as the "parent" for subprocess workers to connect to
        self._ipc_cell = CoreCell(
            fqcn="sim.runner",
            root_url="grpc://localhost:0",  # Use port 0 to get an available port
            secure=False,  # Local IPC doesn't need security
            credentials={},
            create_internal_listener=True,  # Create listener for workers to connect
        )
        self._ipc_cell.start()
        self.logger.info(f"IPC Cell started at {self._ipc_cell.get_internal_listener_url()}")

    def _start_subprocesses(self):
        """Start subprocess workers for all client sites."""
        if self.inprocess:
            return

        self.logger.info("Starting subprocess workers...")

        # Create IPC cell for communication
        self._setup_ipc_cell()

        # Create and start subprocess launcher for each client site
        for site_name in self.client_apps.keys():
            self.logger.info(f"Starting subprocess for {site_name}...")

            launcher = SubprocessLauncher(
                site_name=site_name,
                training_module=self.training_module,
                parent_cell=self._ipc_cell,
                run_cmd=self.run_cmd,
                subprocess_timeout=self.subprocess_timeout,
                worker_id=site_name,  # Use site name as worker ID
            )

            if not launcher.start():
                raise RuntimeError(f"Failed to start subprocess for {site_name}")

            self._subprocess_launchers[site_name] = launcher

        self.logger.info(f"All {len(self._subprocess_launchers)} subprocess workers started")

    def _stop_subprocesses(self):
        """Stop all subprocess workers."""
        if not self._subprocess_launchers:
            return

        self.logger.info("Stopping subprocess workers...")

        for site_name, launcher in self._subprocess_launchers.items():
            self.logger.info(f"Stopping subprocess for {site_name}...")
            launcher.stop()

        self._subprocess_launchers.clear()

        # Stop IPC cell
        if self._ipc_cell:
            self._ipc_cell.stop()
            self._ipc_cell = None

        self.logger.info("All subprocess workers stopped")

    def run(self):
        self.logger.debug(f"Server Collab Interface: {self.server_app.get_collab_interface()}")
        self.logger.debug(f"Client Collab Interface: {self.client_app.get_collab_interface()}")

        try:
            # Start subprocesses if needed (before running)
            if not self.inprocess:
                self._start_subprocesses()

            result = self._try_run()
        except KeyboardInterrupt:
            self.logger.info("execution is aborted by user")
            self.abort_signal.trigger(True)
            result = None
        finally:
            # Stop subprocesses
            if not self.inprocess:
                self._stop_subprocesses()

            self.thread_executor.shutdown(wait=True, cancel_futures=True)
        self.logger.info(f"Experiment results are in {self.exp_dir}")
        return result

    def _try_run(self):
        # initialize all apps
        client_ctx = {}
        for n, app in self.client_apps.items():
            ctx = app.new_context(n, n)
            client_ctx[n] = ctx
            self.logger.info(f"initializing client app {n}")
            app.initialize(ctx)

        # run the server
        result = run_server(self.server_app, self.logger)
        for n, app in self.client_apps.items():
            ctx = client_ctx[n]
            self.logger.info(f"finalizing client app {n}")
            app.finalize(ctx)

        return result


class FoxSimulator:
    """High-level Fox simulation runner.

    Provides a simple API to run Fox simulations with server and client objects.
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
        # Subprocess execution options
        inprocess: bool = True,
        run_cmd: Optional[str] = None,
        training_module: Optional[str] = None,
        subprocess_timeout: float = 300.0,
    ):
        """Initialize FoxSimulator.

        Args:
            root_dir: Root directory for simulation output.
            experiment_name: Name of the experiment.
            server: Server object with @fox.algo methods.
            client: Client object with @fox.collab methods.
            server_objects: Additional server collab objects.
            client_objects: Additional client collab objects.
            max_workers: Maximum worker threads.
            num_clients: Number of clients or (height, width) tuple.
            inprocess: If True, execute in-process. If False, use subprocess.
            run_cmd: Command prefix for subprocess (e.g., "torchrun --nproc_per_node=4").
            training_module: Python module containing @fox.collab methods
                            (required when inprocess=False).
            subprocess_timeout: Timeout for subprocess operations.
        """
        server_app: ServerApp = ServerApp(server)
        client_app: ClientApp = ClientApp(client)

        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.max_workers = max_workers
        self.num_clients = num_clients

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
            # Pass subprocess options
            inprocess=self.inprocess,
            run_cmd=self.run_cmd,
            training_module=self.training_module,
            subprocess_timeout=self.subprocess_timeout,
        )
        return runner.run()
