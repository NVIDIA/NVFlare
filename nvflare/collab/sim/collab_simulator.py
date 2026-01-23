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
from nvflare.collab.api.app import App, ClientApp, ServerApp
from nvflare.collab.api.constants import MAKE_CLIENT_APP_METHOD, BackendType
from nvflare.collab.api.dec import get_object_publish_interface
from nvflare.collab.api.proxy_utils import create_proxy_with_children
from nvflare.collab.api.run_server import run_server
from nvflare.collab.api.subprocess_backend import SubprocessBackend
from nvflare.collab.sim.backend import SimBackend
from nvflare.collab.sim.ws import SimWorkspace
from nvflare.collab.sys.subprocess_launcher import SubprocessLauncher
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.utils.log_utils import get_obj_logger


class AppRunner:
    """Runner for Collab simulation that manages server and client apps.

    This is the simulation equivalent of CollabExecutor - it manages client-side
    execution and supports both in-process and subprocess modes.

    When inprocess=False, it creates a CoreCell for IPC and SubprocessLauncher
    for each client site to run training in separate processes (e.g., for torchrun).
    """

    def _prepare_app_backends(self, app: App, site_name: str = None):
        """Create backend instances for an app.

        For in-process mode, creates SimBackend instances.
        For subprocess mode, creates SubprocessBackend instances that route
        calls to the worker subprocess via CellNet.

        Args:
            app: The app to create backends for.
            site_name: The site name (used to look up subprocess launcher).

        Returns:
            Dictionary of Backend instances keyed by object name.
        """
        # Get subprocess launcher for this site if running in subprocess mode
        launcher = self._subprocess_launchers.get(site_name) if site_name else None

        if launcher:
            # Subprocess mode: use SubprocessBackend that forwards to worker
            bes = {"": SubprocessBackend(launcher, self.abort_signal, self.thread_executor, target_name="")}
            targets = app.get_collab_objects()
            for name, obj in targets.items():
                bes[name] = SubprocessBackend(launcher, self.abort_signal, self.thread_executor, target_name=name)
        else:
            # In-process mode: use SimBackend for direct execution
            bes = {"": SimBackend("", app, app, self.abort_signal, self.thread_executor)}
            targets = app.get_collab_objects()
            for name, obj in targets.items():
                bes[name] = SimBackend(name, app, obj, self.abort_signal, self.thread_executor)
        return bes

    def _prepare_proxy(self, for_app: App, target_app: App, backends: dict):
        """Prepare proxy for a target app.

        For both in-process and subprocess modes, uses logical FQN (e.g., "site-1")
        for proxy identification. The SubprocessBackend handles the actual routing
        to workers via the launcher - it doesn't use the proxy FQN for CellNet.
        """
        # Use logical FQN for all modes (app's own FQN)
        # SubprocessBackend uses launcher.call() which handles CellNet routing internally
        target_fqn = target_app.fqn

        # Build child specs with backends and interfaces
        child_specs = {}
        collab_objs = target_app.get_collab_objects()
        for name, obj in collab_objs.items():
            child_specs[name] = {
                "interface": get_object_publish_interface(obj),
                "backend": backends[name],
            }

        # Use shared utility to create proxy with children
        return create_proxy_with_children(
            app=for_app,
            target_name=target_app.name,
            target_fqn=target_fqn,
            main_backend=backends[""],
            main_interface=get_object_publish_interface(target_app),
            child_specs=child_specs,
        )

    def _detect_client_class(self, client_app: ClientApp) -> Optional[str]:
        """Detect the client class name for class-based clients.

        Args:
            client_app: The ClientApp containing the client object.

        Returns:
            The class name if using a class-based client, None for module-based.
        """
        from nvflare.collab.api.module_wrapper import ModuleWrapper

        # Get the underlying client object from ClientApp
        client_obj = client_app.obj

        # ModuleWrapper means module-based, no class name needed
        if isinstance(client_obj, ModuleWrapper):
            return None

        # Get class name for class-based clients
        return client_obj.__class__.__name__

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
        # Check both the ClientApp and its underlying object for the method
        make_client_app_f = getattr(self.client_app, MAKE_CLIENT_APP_METHOD, None)
        if not make_client_app_f:
            # Check the underlying object
            make_client_app_f = getattr(self.client_app.obj, MAKE_CLIENT_APP_METHOD, None)
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
        # Subprocess execution options (like CollabExecutor)
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
            training_module: Python module containing @fox.publish methods (required when inprocess=False).
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

        # Detect client class name for class-based clients
        self.client_class = self._detect_client_class(client_app)

        # Subprocess management
        self._subprocess_launchers: Dict[str, SubprocessLauncher] = {}
        self._server_cell: Optional[CoreCell] = None
        self._client_cells: Dict[str, CoreCell] = {}

        # Store config for deferred setup
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.num_clients = num_clients

        # Build client apps (but defer backend/proxy setup until run())
        self.client_apps = self._build_client_apps(num_clients)
        self.logger.info(f"created client apps: {self.client_apps.keys()}")

        # These will be set up in run() after subprocesses are started
        self.exp_dir = None

    def _build_client_apps(self, num_clients: Union[int, Tuple[int, int]]) -> Dict[str, ClientApp]:
        """Build client app instances."""
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

            height, num_children_per_parent = num_clients
            if not isinstance(height, int) or not isinstance(num_children_per_parent, int):
                raise ValueError(f"num_clients must be an int or tuple(int, int) but got {num_clients}")

            if height <= 0 or num_children_per_parent <= 0:
                raise ValueError(f"num_clients must contain positive ints but got {num_clients}")

            self.logger.info(f"creating clients {height} x {num_children_per_parent}")
            client_apps = self._build_hierarchical_clients(height, num_children_per_parent)
        else:
            raise ValueError(f"num_clients must be an int or tuple(int, int) but got {type(num_clients)}")

        return client_apps

    def _setup_backends_and_proxies(self):
        """Set up backends and proxies after subprocesses are started.

        This must be called after _start_subprocesses() so that launchers are available.
        """
        # Create backends for server (no subprocess launcher for server)
        backends = {self.server_app.name: self._prepare_app_backends(self.server_app)}

        # Create backends for clients (with subprocess launchers if subprocess mode)
        for name, app in self.client_apps.items():
            backends[name] = self._prepare_app_backends(app, site_name=name)

        exp_id = str(uuid.uuid4())

        # Set up client apps with proxies
        for name, app in self.client_apps.items():
            server_proxy, client_proxies = self._prepare_proxies(app, self.server_app, self.client_apps, backends)
            ws = SimWorkspace(
                root_dir=self.root_dir,
                experiment_name=self.experiment_name,
                site_name=name,
                exp_id=exp_id,
            )
            app.setup(ws, server_proxy, client_proxies, self.abort_signal)

        # Set up server with proxies
        server_proxy, client_proxies = self._prepare_proxies(
            self.server_app, self.server_app, self.client_apps, backends
        )
        ws = SimWorkspace(
            root_dir=self.root_dir,
            experiment_name=self.experiment_name,
            site_name=self.server_app.name,
            exp_id=exp_id,
        )
        self.server_app.setup(ws, server_proxy, client_proxies, self.abort_signal)
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
        """Create CoreCells for subprocess IPC communication.

        Follows CellNet established pattern:
        1. Create a local server cell (so client cells don't try external connection)
        2. Create client cells with internal listeners for subprocess workers
        """
        from nvflare.collab.utils.decomposers import register_available_decomposers
        from nvflare.fuel.f3.cellnet.fqcn import FQCN
        from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port

        # Register tensor decomposer for FOBS
        register_available_decomposers()

        # Get an available port
        port = get_open_tcp_port(resources={})

        # Create a local server cell first
        # This registers in ALL_CELLS so client cells won't try to connect externally
        self._server_cell = CoreCell(
            fqcn=FQCN.ROOT_SERVER,
            root_url=f"tcp://127.0.0.1:{port}",
            secure=False,
            credentials={},
            create_internal_listener=True,
        )
        self._server_cell.start()
        self.logger.info(f"Server cell started at {self._server_cell.get_internal_listener_url()}")

    def _start_subprocesses(self):
        """Start subprocess workers for all client sites.

        Follows CellNet established pattern:
        1. Create server cell (so clients don't try external connection)
        2. Create client cell for each site with internal listener
        3. Subprocess workers connect via parent_url
        """
        if self.inprocess:
            return

        self.logger.info("Starting subprocess workers...")

        # Create server cell first (establishes local CellNet)
        self._setup_ipc_cell()

        # Get server's internal listener URL for client cells
        server_url = self._server_cell.get_internal_listener_url()

        # Create and start subprocess launcher for each client site
        for site_index, site_name in enumerate(self.client_apps.keys()):
            self.logger.info(f"Starting subprocess for {site_name}...")

            # Create client cell for this site with internal listener
            from nvflare.fuel.f3.cellnet.fqcn import FQCN

            client_fqcn = FQCN.join([site_name, "app"])
            client_cell = CoreCell(
                fqcn=client_fqcn,
                root_url=self._server_cell.get_root_url_for_child(),
                secure=False,
                credentials={},
                create_internal_listener=True,  # Workers connect here
                parent_url=server_url,
            )
            client_cell.start()
            self._client_cells[site_name] = client_cell

            self.logger.info(f"Client cell {client_fqcn} started at {client_cell.get_internal_listener_url()}")

            # Create subprocess launcher with client cell
            launcher = SubprocessLauncher(
                site_name=site_name,
                training_module=self.training_module,
                parent_cell=client_cell,
                run_cmd=self.run_cmd,
                subprocess_timeout=self.subprocess_timeout,
                worker_id="0",
                site_index=site_index,
                client_class=self.client_class,
            )

            if not launcher.start():
                raise RuntimeError(f"Failed to start subprocess for {site_name}")

            self._subprocess_launchers[site_name] = launcher

        self.logger.info(f"All {len(self._subprocess_launchers)} subprocess workers started")

    def _stop_subprocesses(self):
        """Stop all subprocess workers and cells."""
        if not self._subprocess_launchers:
            return

        self.logger.info("Stopping subprocess workers...")

        # Stop subprocess launchers
        for site_name, launcher in self._subprocess_launchers.items():
            self.logger.info(f"Stopping subprocess for {site_name}...")
            launcher.stop()

        self._subprocess_launchers.clear()

        # Stop client cells
        for site_name, cell in self._client_cells.items():
            self.logger.debug(f"Stopping client cell for {site_name}...")
            cell.stop()
        self._client_cells.clear()

        # Stop server cell
        if self._server_cell:
            self._server_cell.stop()
            self._server_cell = None

        self.logger.info("All subprocess workers stopped")

    def run(self):
        self.logger.debug(f"Server Collab Interface: {self.server_app.get_collab_interface()}")
        self.logger.debug(f"Client Collab Interface: {self.client_app.get_collab_interface()}")

        try:
            # Start subprocesses if needed (before setting up backends/proxies)
            if not self.inprocess:
                self._start_subprocesses()

            # Set up backends and proxies (after subprocesses are started)
            # This ensures subprocess launchers are available for SubprocessBackend
            self._setup_backends_and_proxies()

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


class CollabSimulator:
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
        """Initialize CollabSimulator.

        Args:
            root_dir: Root directory for simulation output.
            experiment_name: Name of the experiment.
            server: Server object with @fox.main methods.
            client: Client object with @fox.publish methods.
            server_objects: Additional server collab objects.
            client_objects: Additional client collab objects.
            max_workers: Maximum worker threads.
            num_clients: Number of clients or (height, width) tuple.
            inprocess: If True, execute in-process. If False, use subprocess.
            run_cmd: Command prefix for subprocess (e.g., "torchrun --nproc_per_node=4").
            training_module: Python module containing @fox.publish methods
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
