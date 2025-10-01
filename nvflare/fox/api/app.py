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
from abc import ABC, abstractmethod
from typing import List

from .constants import CollabMethodArgName
from .ctx import Context
from .dec import collab, get_object_collab_interface, is_collab
from .proxy import Proxy
from .strategy import Strategy
from .utils import check_context_support


class App:

    def __init__(self):
        self.name = None
        self.server = None
        self.clients = None
        self.env_type = None
        self._me = None
        self._collab_objs = {}
        self._abort_signal = None
        self._props = {}
        self._event_handlers = {}  # event type => list of (cb, kwargs)

    def set_prop(self, name: str, value):
        self._props[name] = value

    def get_prop(self, name: str, default=None):
        return self._props.get(name, default)

    def get_default_collab_object(self):
        return None

    def add_collab_object(self, name: str, obj):
        if name in self._collab_objs:
            raise ValueError(f"conflict with existing collab object '{name}' of {type(self._collab_objs[name])}")

        if hasattr(obj, name):
            raise ValueError(f"conflict with reserved name {name}")

        setattr(self, name, obj)
        self._collab_objs[name] = obj

    def get_collab_objects(self):
        return self._collab_objs

    def setup(self, server: Proxy, clients: List[Proxy], abort_signal):
        self.server = server
        self._abort_signal = abort_signal

        self.clients = clients
        self._me = None
        if not self.name or self.name == "server":
            self._me = server
        else:
            for c in clients:
                if c.name == self.name:
                    self._me = c
                    break

        if not self._me:
            raise ValueError(f"cannot find site for {self.name}")

    def get_my_site(self) -> Proxy:
        return self._me

    def find_method(self, target_obj, method_name):
        m = getattr(target_obj, method_name, None)
        if m:
            return m

        if isinstance(target_obj, App):
            # see whether any targets have this method
            default_target = self.get_default_collab_object()
            if default_target:
                m = getattr(default_target, method_name, None)
                if m:
                    return m

            targets = self.get_collab_objects()
            for _, obj in targets.items():
                m = getattr(obj, method_name, None)
                if m:
                    return m
        return None

    def find_collab_method(self, target_obj, method_name):
        m = self.find_method(target_obj, method_name)
        if m and is_collab(m):
            return m
        return None

    def initialize_app(self, context: Context):
        pass

    def initialize(self, context: Context):
        self.initialize_app(context)

        # initialize target objects
        for name, obj in self._collab_objs.items():
            init_func = getattr(obj, "initialize", None)
            if init_func and callable(init_func):
                print(f"initializing target object {name}")
                kwargs = {CollabMethodArgName.CONTEXT: context}
                check_context_support(init_func, kwargs)
                init_func(**kwargs)

    def new_context(self, caller: str, callee: str, props: dict = None):
        ctx = Context(self.env_type, caller, callee, self._abort_signal, props)
        ctx.app = self
        ctx.server = self.server
        ctx.clients = self.clients
        return ctx

    def register_event_handler(self, event_type: str, handler, **handler_kwargs):
        handlers = self._event_handlers.get(event_type)
        if not handlers:
            handlers = []
            self._event_handlers[event_type] = handlers
        handlers.append((handler, handler_kwargs))

    def get_collab_interface(self):
        result = {"": get_object_collab_interface(self)}
        for name, obj in self._collab_objs.items():
            result[name] = get_object_collab_interface(obj)
        return result

    @collab
    def fire_event(self, event_type: str, data, context: Context):
        for e, handlers in self._event_handlers.items():
            if e == event_type:
                for h, kwargs in handlers:
                    kwargs = copy.copy(kwargs)
                    kwargs.update({CollabMethodArgName.CONTEXT: context})
                    check_context_support(h, kwargs)
                    h(event_type, data, **kwargs)


class ServerApp(App):

    def __init__(self, strategy_name: str = "strategy", strategy: Strategy = None):
        super().__init__()

        if strategy and not isinstance(strategy, Strategy):
            raise ValueError(f"strategy must be Strategy but got {type(strategy)}")

        self.strategies = []
        if strategy:
            if not strategy_name:
                raise ValueError("missing strategy name")
            self.add_strategy(strategy_name, strategy)
        self.current_strategy = None

    def add_strategy(self, strategy_name: str, strategy):
        if not isinstance(strategy, Strategy):
            raise ValueError(f"strategy must be Controller but got {type(strategy)}")
        self.strategies.append(strategy)
        self.add_collab_object(strategy_name, strategy)

    def get_default_collab_object(self):
        return self.current_strategy


class ClientApp(App):
    pass


class ClientAppFactory(ABC):

    @abstractmethod
    def make_client_app(self, name: str) -> ClientApp:
        pass
