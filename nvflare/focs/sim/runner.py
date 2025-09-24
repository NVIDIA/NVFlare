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
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Union

from nvflare.apis.signal import Signal
from nvflare.focs.api.app import SERVER_NAME, App, ClientApp, ClientAppFactory, ServerApp
from nvflare.focs.api.constants import ContextKey
from nvflare.focs.api.proxy import Proxy
from nvflare.focs.sim.backend import SimBackend


class AppRunner:

    def _prepare_app_backends(self, app: App):
        bes = {"": SimBackend(app, "", app, self.abort_signal, self.thread_executor)}
        targets = app.get_target_objects()
        if targets:
            for name, obj in targets:
                bes[name] = SimBackend(app, name, obj, self.abort_signal, self.thread_executor)
        return bes

    def _prepare_app_proxy(self, app_name: str, app: App, caller_name: str, app_backends: dict):
        app_proxy = Proxy(app=app, target_name=app_name, backend=app_backends[""], caller_name=caller_name)
        cos = app.get_target_objects()
        if cos:
            for name, obj in cos:
                p = Proxy(app=app, target_name=name, backend=app_backends[name], caller_name=caller_name)
                setattr(app_proxy, name, p)
        return app_proxy

    def _prepare_proxies(self, server_app: App, client_apps: dict, caller_name, backends: dict):
        server_proxy = self._prepare_app_proxy(SERVER_NAME, server_app, caller_name, backends[SERVER_NAME])
        client_proxies = []
        for name, app in client_apps.items():
            p = self._prepare_app_proxy(name, app, caller_name, backends[name])
            client_proxies.append(p)
        return server_proxy, client_proxies

    def __init__(
        self,
        server_app: ServerApp,
        client_app: Union[ClientAppFactory, ClientApp],
        max_workers: int = 100,
        num_clients: int = 2,
    ):
        if not isinstance(server_app, ServerApp):
            raise ValueError(f"server_app must be ServerApp but got {type(server_app)}")

        if not isinstance(client_app, (ClientAppFactory, ClientApp)):
            raise ValueError(f"client_app must be ClientApp or ClientAppFactory but got {type(client_app)}")

        self.abort_signal = Signal()
        self.server_app = server_app
        self.client_app = client_app
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.num_clients = num_clients

        client_apps = {}
        for i in range(self.num_clients):
            name = f"site-{i + 1}"
            if isinstance(client_app, ClientApp):
                app = copy.deepcopy(client_app)
            else:
                app = client_app.make_client_app(name)
            app.name = name
            client_apps[name] = app

        backends = {SERVER_NAME: self._prepare_app_backends(server_app)}

        for name, app in client_apps.items():
            backends[name] = self._prepare_app_backends(app)

        for name, app in client_apps.items():
            server_proxy, client_proxies = self._prepare_proxies(server_app, client_apps, name, backends)
            app.setup(name, server_proxy, client_proxies, self.abort_signal)

        # prepare server
        server_proxy, client_proxies = self._prepare_proxies(server_app, client_apps, SERVER_NAME, backends)
        server_app.setup(SERVER_NAME, server_proxy, client_proxies, self.abort_signal)

        self.client_apps = client_apps

    def run(self):
        # initialize all apps
        server_ctx = self.server_app.new_context(caller=self.server_app.name, callee=self.server_app.name)
        print("initializing server app")
        self.server_app.initialize(server_ctx)

        for n, app in self.client_apps.items():
            print(f"initializing client app for {n}")
            app.initialize(app.new_context(n, n))

        # run the server
        result = None
        for idx, controller in enumerate(self.server_app.controllers):
            try:
                print(f"Running Controller #{idx+1}")
                self.server_app.current_controller = controller
                result = controller.run(context=server_ctx)
                server_ctx.set_prop(ContextKey.INPUT, result)
            except:
                traceback.print_exc()
                break

        self.thread_executor.shutdown(wait=False, cancel_futures=True)
        return result
