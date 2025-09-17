from abc import abstractmethod, ABC
from typing import List, Union
from .proxy import Proxy
from .group import Group

SERVER_NAME = "server"


class App(ABC):

    def __init__(self):
        self.name = None
        self.server = None
        self.clients = None
        self._me = None
        self._target_objs = {}
        self._abort_signal = None

    def add_target_object(self, name: str, obj):
        if name in ['name','server', 'clients', "add_target_object", "get_target_objects", "setup", "get_my_site"]:
            raise ValueError(f"conflict with reserved name {name}")

        # TBD: more name validation needed
        setattr(self, name, obj)
        self._target_objs[name] = obj

    def get_target_objects(self):
        return self._target_objs

    def setup(self, name: str, server: Proxy, clients: List[Proxy], abort_signal):
        self.name = name
        self.server = server
        self._abort_signal = abort_signal

        self.clients = clients
        self._me = None
        if not name or name == "server":
            self._me = server
        else:
            for c in clients:
                if c.name == name:
                    self._me = c
                    break

        if not self._me:
            raise ValueError(f"cannot find site for {name}")

    def get_my_site(self) -> Proxy:
        return self._me

    def initialize(self, **kwargs):
        pass

    def group(
        self,
        proxies: List[Proxy],
        blocking: bool = True,
        timeout: float = None,
        min_resps: int = None,
        wait_after_min_resps: float = None,
        process_resp_cb=None,
        **cb_kwargs,
    ):
        return Group(
            self._abort_signal,
            proxies,
            blocking,
            timeout,
            min_resps,
            wait_after_min_resps,
            process_resp_cb,
            **cb_kwargs
        )


class ServerApp(App):

    @abstractmethod
    def run(self, **kwargs):
        pass


class ClientApp(App):
    pass


class ClientAppFactory(ABC):

    @abstractmethod
    def make_client_app(self, name: str) -> ClientApp:
        pass
