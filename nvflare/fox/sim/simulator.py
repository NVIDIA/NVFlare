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
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Union

from nvflare.apis.signal import Signal
from nvflare.fox.api.app import App, ClientApp, ClientAppFactory, ServerApp
from nvflare.fox.api.constants import ContextKey, EnvType
from nvflare.fox.api.dec import get_object_collab_interface
from nvflare.fox.api.proxy import Proxy
from nvflare.fox.sim.backend import SimBackend
from nvflare.fuel.utils.log_utils import get_obj_logger


class Simulator:

    def _prepare_app_backends(self, app: App):
        bes = {"": SimBackend("", app, app, self.abort_signal, self.thread_executor)}
        targets = app.get_collab_objects()
        for name, obj in targets.items():
            bes[name] = SimBackend(name, app, obj, self.abort_signal, self.thread_executor)
        return bes

    def _prepare_proxy(self, for_app: App, target_app: App, backends: dict):
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

    def _make_app(self, name, fqn):
        if isinstance(self.client_app, ClientApp):
            app = copy.deepcopy(self.client_app)
        else:
            app = self.client_app.make_client_app(name)
        app.name = name
        app.fqn = fqn
        app.env_type = EnvType.SIMULATION
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
        server_app: ServerApp,
        client_app: Union[ClientAppFactory, ClientApp],
        max_workers: int = 100,
        num_clients: Union[int, Tuple[int, int]] = 2,
    ):
        if not isinstance(server_app, ServerApp):
            raise ValueError(f"server_app must be ServerApp but got {type(server_app)}")

        if not isinstance(client_app, (ClientAppFactory, ClientApp)):
            raise ValueError(f"client_app must be ClientApp or ClientAppFactory but got {type(client_app)}")

        self.logger = get_obj_logger(self)
        self.abort_signal = Signal()
        server_app.name = "server"
        server_app.fqn = server_app.name
        server_app.env_type = EnvType.SIMULATION
        self.server_app = server_app
        self.client_app = client_app
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)

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

            print(f"creating clients {height} x {num_children_per_parent}")
            client_apps = self._build_hierarchical_clients(height, num_children_per_parent)
        else:
            raise ValueError(f"num_clients must be an int or tuple(int, int) but got {type(num_clients)}")

        self.logger.info(f"created client apps: {client_apps.keys()}")

        backends = {server_app.name: self._prepare_app_backends(server_app)}

        for name, app in client_apps.items():
            backends[name] = self._prepare_app_backends(app)

        for name, app in client_apps.items():
            server_proxy, client_proxies = self._prepare_proxies(app, server_app, client_apps, backends)
            app.setup(server_proxy, client_proxies, self.abort_signal)

        # prepare server
        server_proxy, client_proxies = self._prepare_proxies(server_app, server_app, client_apps, backends)
        server_app.setup(server_proxy, client_proxies, self.abort_signal)

        self.client_apps = client_apps

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
        try:
            self._try_run()
        except KeyboardInterrupt:
            self.logger.info("execution is aborted by user")
            self.abort_signal.trigger(True)
        finally:
            self.thread_executor.shutdown(wait=False, cancel_futures=True)

    def _try_run(self):
        # initialize all apps
        server_ctx = self.server_app.new_context(caller=self.server_app.name, callee=self.server_app.name)
        self.logger.info("initializing server app")
        self.server_app.initialize(server_ctx)

        for n, app in self.client_apps.items():
            self.logger.info(f"initializing client app for {n}")
            app.initialize(app.new_context(n, n))

        # run the server
        if not self.server_app.strategies:
            raise RuntimeError("server app does not have any strategies!")

        result = None
        for idx, strategy in enumerate(self.server_app.strategies):
            self.logger.info(f"Running Strategy #{idx+1} - {type(strategy).__name__}")
            self.server_app.current_strategy = strategy
            result = strategy.execute(context=server_ctx)
            server_ctx.set_prop(ContextKey.INPUT, result)
        return result
