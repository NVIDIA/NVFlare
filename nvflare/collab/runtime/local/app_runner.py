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
from typing import Dict, Optional, Tuple, Union

from nvflare.apis.signal import Signal
from nvflare.collab.api.app import App, ClientApp, ServerApp
from nvflare.collab.api.constants import MAKE_CLIENT_APP_METHOD
from nvflare.collab.api.decorators import get_object_publish_interface
from nvflare.collab.api.proxy_utils import create_proxy_with_children
from nvflare.collab.runtime.lifecycle import run_server
from nvflare.collab.runtime.local.direct_dispatcher import DirectDispatcher
from nvflare.collab.runtime.local.local_workspace import LocalWorkspace
from nvflare.fuel.utils.log_utils import get_obj_logger


class AppRunner:
    """Internal direct-call runner used by focused Collab tests."""

    def _prepare_app_backends(self, app: App, site_name: str = None):
        """Create direct invocation dispatchers for an app."""
        dispatchers = {"": DirectDispatcher("", app, app, self.abort_signal, self.thread_executor)}
        for name, obj in app.get_collab_objects().items():
            dispatchers[name] = DirectDispatcher(name, app, obj, self.abort_signal, self.thread_executor)
        return dispatchers

    def _prepare_proxy(self, for_app: App, target_app: App, backends: dict):
        """Prepare a direct-call proxy for a target app."""
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
            app = make_client_app_f(site_name)
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

        # Apply this site's entries from the recipe's per-site config as app
        # props (readable via collab.get_app_prop).
        site_props = self.per_site_props.get(site_name)
        if site_props:
            for name, value in site_props.items():
                app.set_prop(name, value)
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
        per_site_props: Optional[Dict[str, dict]] = None,
    ):
        """Initialize AppRunner.

        Args:
            root_dir: Root directory for simulation.
            experiment_name: Name of the experiment.
            server_app: The server application.
            client_app: Template for client applications.
            max_workers: Maximum worker threads.
            num_clients: Number of clients or (height, width) tuple for hierarchy.
        """
        if not isinstance(server_app, ServerApp):
            raise ValueError(f"server_app must be ServerApp but got {type(server_app)}")

        if not isinstance(client_app, ClientApp):
            raise ValueError(f"client_app must be ClientApp but got {type(client_app)}")

        self.logger = get_obj_logger(self)
        self.abort_signal = Signal()
        server_app.name = "server"
        server_app.set_fqn(server_app.name)
        self.server_app = server_app
        self.client_app = client_app
        self.per_site_props = per_site_props or {}
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="collab_call")

        # Store config for deferred setup
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.num_clients = num_clients

        # Build client apps (but defer dispatcher/proxy setup until run()).
        self.client_apps = self._build_client_apps(num_clients)
        self.logger.info(f"created client apps: {self.client_apps.keys()}")

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
        """Set up direct dispatchers and proxies."""
        backends = {self.server_app.name: self._prepare_app_backends(self.server_app)}

        # Create dispatchers for clients.
        for name, app in self.client_apps.items():
            backends[name] = self._prepare_app_backends(app, site_name=name)

        exp_id = str(uuid.uuid4())

        # Set up client apps with proxies
        for name, app in self.client_apps.items():
            server_proxy, client_proxies = self._prepare_proxies(app, self.server_app, self.client_apps, backends)
            ws = LocalWorkspace(
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
        ws = LocalWorkspace(
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

    def run(self):
        self.logger.debug(f"Server Collab Interface: {self.server_app.get_collab_interface()}")
        self.logger.debug(f"Client Collab Interface: {self.client_app.get_collab_interface()}")

        try:
            self._setup_backends_and_proxies()
            result = self._try_run()
        except KeyboardInterrupt:
            self.logger.info("execution is aborted by user")
            self.abort_signal.trigger(True)
            result = None
        finally:
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
