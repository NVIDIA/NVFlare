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
from abc import ABC, abstractmethod
from typing import List

from .controller import Controller
from .ctx import Context
from .proxy import Proxy

SERVER_NAME = "server"


class App(ABC):

    def __init__(self):
        self.name = None
        self.server = None
        self.clients = None
        self._me = None
        self._target_objs = []
        self._abort_signal = None

    def get_default_target(self):
        return None

    def add_target_object(self, name: str, obj):
        if hasattr(obj, name):
            raise ValueError(f"conflict with reserved name {name}")

        setattr(self, name, obj)
        self._target_objs.append((name, obj))

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

    def new_context(self, caller: str, callee: str, props: dict = None):
        ctx = Context(caller, callee, self._abort_signal, props)
        ctx.app = self
        ctx.server = self.server
        ctx.clients = self.clients
        return ctx


class ServerApp(App):

    def __init__(self, controller: Controller):
        super().__init__()
        if not isinstance(controller, Controller):
            raise ValueError(f"controller must be Controller but got {type(controller)}")
        self.controllers = [controller]
        self.current_controller = None

    def add_controller(self, controller):
        if not isinstance(controller, Controller):
            raise ValueError(f"controller must be Controller but got {type(controller)}")
        self.controllers.append(controller)

    def get_default_target(self):
        return self.current_controller


class ClientApp(App):
    pass


class ClientAppFactory(ABC):

    @abstractmethod
    def make_client_app(self, name: str) -> ClientApp:
        pass
