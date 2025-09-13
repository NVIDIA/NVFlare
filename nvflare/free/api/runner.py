import copy
from concurrent.futures import ThreadPoolExecutor

from nvflare.free.api.app import ServerApp, ClientApp, App, SERVER_NAME
from nvflare.free.api.proxy import Proxy
from nvflare.free.api.sim_backend import SimBackend
from nvflare.apis.signal import Signal


class AppRunner:

    def _prepare_app_backends(self, app: App):
        bes = {"": SimBackend(app, self.abort_signal, self.thread_executor)}
        targets = app.get_target_objects()
        if targets:
            for name, obj in targets.items():
                 bes[name] = SimBackend(obj, self.abort_signal, self.thread_executor)
        return bes

    def _prepare_app_proxy(self, app_name: str, app: App, caller_name: str, app_backends: dict):
        app_proxy = Proxy(target_name=app_name, backend=app_backends[""], caller_name=caller_name)
        cos = app.get_target_objects()
        if cos:
            for name, obj in cos.items():
                p = Proxy(target_name=name, backend=app_backends[name], caller_name=caller_name)
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
        client_app: ClientApp,
        num_clients: int,
    ):
        self.abort_signal = Signal()
        self.thread_executor = ThreadPoolExecutor(max_workers=100)

        client_apps = {}
        for i in range(num_clients):
            name = f"site-{i+1}"
            app = copy.deepcopy(client_app)
            app.name = name
            client_apps[name] = app

        backends = {
            SERVER_NAME: self._prepare_app_backends(server_app)
        }

        for name, app in client_apps.items():
            backends[name] = self._prepare_app_backends(app)

        for name, app in client_apps.items():
            server_proxy, client_proxies = self._prepare_proxies(server_app, client_apps, name, backends)
            app.setup(name, server_proxy, client_proxies, self.abort_signal)
            app.initialize()

        # prepare server
        server_proxy, client_proxies = self._prepare_proxies(server_app, client_apps, SERVER_NAME, backends)
        server_app.setup(SERVER_NAME, server_proxy, client_proxies, self.abort_signal)

        self.server_app = server_app

    def run(self):
        # run the server
        self.server_app.run()
        self.thread_executor.shutdown(wait=False, cancel_futures=True)
