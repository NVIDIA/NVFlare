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
import fnmatch
import os
import re
from typing import List

from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.tree_utils import Forest, Node, build_forest

from .constants import BackendType, CollabMethodArgName, ContextKey, FilterDirection
from .ctx import Context, set_call_context
from .dec import (
    publish,
    get_object_main_funcs,
    get_object_publish_interface,
    get_object_final_funcs,
    get_object_init_funcs,
    is_publish,
    supports_context,
)
from .filter import CallFilter, FilterChain, ResultFilter
from .proxy import Proxy
from .utils import check_context_support, get_collab_object_name
from .workspace import Workspace


class App:

    def __init__(self, obj, name: str):
        self.obj = obj
        self.name = name
        self._fqn = None
        self._server_proxy = None
        self._client_proxies = None
        self._client_hierarchy = None
        self._backend_type = None
        self._me = None
        self._collab_objs = {}
        self._abort_signal = None
        self._props = {}
        self._event_handlers = {}  # event type => list of (cb, kwargs)
        self._incoming_call_filter_chains = []
        self._outgoing_call_filter_chains = []
        self._incoming_result_filter_chains = []
        self._outgoing_result_filter_chains = []
        self._workspace = None
        self._resource_dirs = {}
        self._managed_objects = {}  # id => obj
        self.logger = get_obj_logger(self)
        self._collab_interface = {"": get_object_publish_interface(self)}
        self.add_collab_object(name, obj)

    def set_resource_dirs(self, resource_dirs: dict[str, str]):
        if not isinstance(resource_dirs, dict):
            raise TypeError(f"resource_dirs must be a dict but got {type(resource_dirs)}")
        for name, resource_dir in resource_dirs.items():
            if not os.path.isdir(resource_dir):
                raise ValueError(f"Resource dir {resource_dir} does not exist for {name}")
        self._resource_dirs = resource_dirs

    def get_resource_dirs(self):
        return self._resource_dirs

    def _add_managed_object(self, obj):
        self._managed_objects[id(obj)] = obj

    def set_fqn(self, fqn):
        self._fqn = fqn

    @property
    def fqn(self):
        return self._fqn

    @property
    def backend(self):
        if not self._me:
            return None
        else:
            return self._me.backend

    @property
    def backend_type(self):
        return self._backend_type

    def set_backend_type(self, t: str):
        valid_types = [BackendType.SIMULATION, BackendType.FLARE]
        if t not in valid_types:
            raise ValueError(f"bad backend type: {t}: must be one of {valid_types}")
        self._backend_type = t

    @property
    def workspace(self):
        return self._workspace

    @property
    def server_proxy(self):
        return self._server_proxy

    @property
    def client_proxies(self):
        return copy.copy(self._client_proxies)

    @property
    def client_hierarchy(self):
        return self._client_hierarchy

    def _add_filters(self, pattern: str, filters, to_list: list, filter_type, incoming):
        if not filters:
            return

        if not isinstance(filters, list):
            raise ValueError(f"filters must be a list but got {type(filters)}")

        filter_objs = []
        for f in filters:
            if not isinstance(f, filter_type):
                # convert to proper filter type
                filter_obj = filter_type(f, incoming)
            else:
                filter_obj = f

            filter_objs.append(filter_obj)

            # f is a managed object, but the filter_obj (if wrapped) is not!
            self._add_managed_object(f)

        chain = FilterChain(pattern, filter_type)
        chain.add_filters(filter_objs)
        to_list.append(chain)

    def add_incoming_call_filters(self, pattern: str, filters: List[object]):
        self._add_filters(pattern, filters, self._incoming_call_filter_chains, CallFilter, True)

    def get_incoming_call_filters(self):
        return self._incoming_call_filter_chains

    def add_outgoing_call_filters(self, pattern: str, filters: List[object]):
        self._add_filters(pattern, filters, self._outgoing_call_filter_chains, CallFilter, False)

    def get_outgoing_call_filters(self):
        return self._outgoing_call_filter_chains

    def add_incoming_result_filters(self, pattern: str, filters: List[object]):
        self._add_filters(pattern, filters, self._incoming_result_filter_chains, ResultFilter, True)

    def get_incoming_result_filters(self):
        return self._incoming_result_filter_chains

    def add_outgoing_result_filters(self, pattern: str, filters: List[object]):
        self._add_filters(pattern, filters, self._outgoing_result_filter_chains, ResultFilter, False)

    def get_outgoing_result_filters(self):
        return self._outgoing_result_filter_chains

    @staticmethod
    def _find_filter_chain(direction, chains: List[FilterChain], target_name: str, func_name: str, ctx: Context):
        """

        Args:
            chains:
            target_name:
            func_name:

        Returns:

        """
        collab_obj_name = get_collab_object_name(target_name)
        qualified_func_name = f"{collab_obj_name}.{func_name}"
        ctx.set_prop(ContextKey.QUALIFIED_FUNC_NAME, qualified_func_name)
        ctx.set_prop(ContextKey.DIRECTION, direction)

        if not chains:
            return None

        for c in chains:
            if fnmatch.fnmatch(qualified_func_name, c.pattern):
                return c
        return None

    def apply_incoming_call_filters(self, target_name: str, func_name: str, func_kwargs, context: Context):
        filter_chain = self._find_filter_chain(
            FilterDirection.INCOMING, self._incoming_call_filter_chains, target_name, func_name, context
        )
        if filter_chain:
            return filter_chain.apply_filters(func_kwargs, context)
        else:
            return func_kwargs

    def apply_outgoing_call_filters(self, target_name: str, func_name: str, func_kwargs, context: Context):
        filter_chain = self._find_filter_chain(
            FilterDirection.OUTGOING, self._outgoing_call_filter_chains, target_name, func_name, context
        )
        if filter_chain:
            return filter_chain.apply_filters(func_kwargs, context)
        else:
            return func_kwargs

    def apply_incoming_result_filters(self, target_name: str, func_name: str, result, context: Context):
        filter_chain = self._find_filter_chain(
            FilterDirection.INCOMING, self._incoming_result_filter_chains, target_name, func_name, context
        )
        if filter_chain:
            return filter_chain.apply_filters(result, context)
        else:
            return result

    def apply_outgoing_result_filters(self, target_name: str, func_name: str, result, context: Context):
        filter_chain = self._find_filter_chain(
            FilterDirection.OUTGOING, self._outgoing_result_filter_chains, target_name, func_name, context
        )
        if filter_chain:
            return filter_chain.apply_filters(result, context)
        else:
            return result

    def set_prop(self, name: str, value):
        self._props[name] = value

    def get_prop(self, name: str, default=None):
        return self._props.get(name, default)

    def get_props(self):
        return self._props

    def update_props(self, props: dict):
        if isinstance(props, dict):
            self._props.update(props)

    def add_collab_object(self, name: str, obj):
        # name must be acceptable str
        pattern = r"^[A-Za-z][A-Za-z0-9_]*$"
        if not re.match(pattern, name):
            raise ValueError(
                f"invalid name {name} for collab object - must start with a letter, "
                "followed by one or more alphanumeric and/or underscore chars"
            )

        if name in self._collab_objs:
            raise ValueError(f"conflict with existing collab object '{name}' of {type(self._collab_objs[name])}")

        if hasattr(self, name):
            raise ValueError(f"conflict with reserved name {name}")

        setattr(self, name, obj)
        self._collab_objs[name] = obj
        self._collab_interface[name] = get_object_publish_interface(obj)
        self._add_managed_object(obj)

    def get_collab_objects(self):
        return self._collab_objs

    def setup(self, workspace: Workspace, server: Proxy, clients: List[Proxy], abort_signal):
        self._workspace = workspace
        workspace.resource_dirs = self._resource_dirs

        self._server_proxy = server
        self._abort_signal = abort_signal

        self._client_proxies = clients
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

        forest = build_forest(objs=clients, get_fqn_f=lambda c: c.fqn, get_name_f=lambda c: c.name)
        self._client_hierarchy = forest

    @property
    def my_site(self) -> Proxy:
        return self._me

    def find_method(self, target_obj, method_name):
        m = getattr(target_obj, method_name, None)
        if m:
            return m

        if isinstance(target_obj, App):
            # see whether any targets have this method
            default_target = self.obj
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
        if m and is_publish(m):
            return m
        return None

    def _fox_init(self, obj, ctx: Context):
        init_funcs = get_object_init_funcs(obj)
        for name, f in init_funcs:
            self.logger.debug(f"calling init func {name} ...")
            if supports_context(f):
                kwargs = {CollabMethodArgName.CONTEXT: ctx}
            else:
                kwargs = {}
            f(**kwargs)

    def initialize(self, context: Context):
        self._fox_init(self, context)

        # initialize target objects
        for obj in self._managed_objects.values():
            self._fox_init(obj, context)

    def _fox_finalize(self, obj, ctx: Context):
        funcs = get_object_final_funcs(obj)
        for name, f in funcs:
            self.logger.debug(f"calling final func {name} ...")
            if supports_context(f):
                kwargs = {CollabMethodArgName.CONTEXT: ctx}
            else:
                kwargs = {}
            f(**kwargs)

    def finalize(self, context: Context):
        self._fox_finalize(self, context)

        # finalize target objects
        for obj in self._managed_objects.values():
            self._fox_finalize(obj, context)

    def new_context(self, caller: str, callee: str, target_group=None, set_call_ctx=True):
        ctx = Context(self, caller, callee, self._abort_signal, target_group=target_group)
        if set_call_ctx:
            set_call_context(ctx)
        return ctx

    def register_event_handler(self, event_type: str, handler, **handler_kwargs):
        handlers = self._event_handlers.get(event_type)
        if not handlers:
            handlers = []
            self._event_handlers[event_type] = handlers
        handlers.append((handler, handler_kwargs))
        self.logger.debug(f"registered event handler {handler.__qualname__} for {event_type=}")

    def get_collab_interface(self):
        return self._collab_interface

    def get_target_object_publish_interface(self, target_name: str):
        if not target_name or target_name.lower() == "app":
            return self._collab_interface.get("")
        else:
            return self._collab_interface.get(target_name)

    @publish
    def fire_event(self, event_type: str, data, context: Context):
        result = {}
        for e, handlers in self._event_handlers.items():
            if e == event_type:
                for h, kwargs in handlers:
                    kwargs = copy.copy(kwargs)
                    kwargs.update({CollabMethodArgName.CONTEXT: context})
                    check_context_support(h, kwargs)
                    result[h.__qualname__] = h(event_type, data, **kwargs)
        return result

    def get_children(self):
        return []

    def has_children(self):
        return False

    def get_leaf_clients(self):
        if not isinstance(self._client_hierarchy, Forest):
            raise RuntimeError(f"client_hierarchy must be Forest but got {type(self._client_hierarchy)}")
        leaf_nodes = [self._client_hierarchy.nodes[n] for n in self._client_hierarchy.leaves]
        return [node.obj for node in leaf_nodes]


class ServerApp(App):

    def __init__(self, obj, name: str = "server"):
        if not obj:
            raise ValueError("server object must be specified")
        super().__init__(obj, name)
        self.mains = get_object_main_funcs(obj)
        if not self.mains:
            raise ValueError("server object must have at least one algo")

    def get_children(self):
        if not isinstance(self._client_hierarchy, Forest):
            raise RuntimeError(
                f"client_hierarchy in app {self.name} must be Forest but got {type(self._client_hierarchy)}"
            )
        root_nodes = [self._client_hierarchy.nodes[n] for n in self._client_hierarchy.roots]
        return [node.obj for node in root_nodes]

    def has_children(self):
        return True


class ClientApp(App):

    def __init__(self, obj, name: str = "client"):
        if not obj:
            raise ValueError("client object must be specified")
        super().__init__(obj, name)

    def _get_my_node(self):
        if not isinstance(self._client_hierarchy, Forest):
            raise RuntimeError(
                f"client_hierarchy in app {self.name} must be Forest but got {type(self._client_hierarchy)}"
            )

        node = self._client_hierarchy.nodes.get(self.name)
        if not isinstance(node, Node):
            raise RuntimeError(f"node for site {self.name} must be a Node but got {type(node)}")
        return node

    def get_children(self):
        my_node = self._get_my_node()
        if my_node.children:
            return [node.obj for node in my_node.children]
        else:
            return []

    def has_children(self):
        my_node = self._get_my_node()
        return True if my_node.children else False
