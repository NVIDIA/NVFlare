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
import re

from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.tree_utils import Forest, Node, build_forest

from .constants import CollabMethodArgName
from .context import Context, get_call_context, set_call_context
from .decorators import (
    get_object_final_funcs,
    get_object_init_funcs,
    get_object_main_funcs,
    get_object_publish_interface,
    is_publish,
    supports_context,
)
from .module_wrapper import wrap_if_module
from .proxy import Proxy


class App:

    def __init__(self, obj, name: str):
        obj = wrap_if_module(obj)
        self.obj = obj
        self.name = name
        self._fqn = None
        self._server_proxy = None
        self._client_proxies = None
        self._client_hierarchy = None
        self._me = None
        self._collab_objs = {}
        self._abort_signal = None
        self._props = {}
        self._workspace = None
        self._managed_objects = {}  # id => obj
        self.logger = get_obj_logger(self)
        self._collab_interface = {"": get_object_publish_interface(self)}
        self.add_collab_object(name, obj)

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
        obj = wrap_if_module(obj)

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

    def setup(self, workspace: Workspace, server: Proxy, clients: list[Proxy], abort_signal):
        if not isinstance(workspace, Workspace):
            raise TypeError(f"workspace must be a Workspace but got {type(workspace)}")
        self._workspace = workspace

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

    def _collab_init(self, obj, ctx: Context):
        init_funcs = get_object_init_funcs(obj)
        for name, f in init_funcs:
            self.logger.debug(f"calling init func {name} ...")
            if supports_context(f):
                kwargs = {CollabMethodArgName.CONTEXT: ctx}
            else:
                kwargs = {}
            f(**kwargs)

    def initialize(self, context: Context):
        previous_ctx = get_call_context()
        set_call_context(context)
        try:
            self._collab_init(self, context)

            # initialize target objects
            for obj in self._managed_objects.values():
                self._collab_init(obj, context)
        finally:
            set_call_context(previous_ctx)

    def _collab_finalize(self, obj, ctx: Context):
        funcs = get_object_final_funcs(obj)
        for name, f in funcs:
            self.logger.debug(f"calling final func {name} ...")
            if supports_context(f):
                kwargs = {CollabMethodArgName.CONTEXT: ctx}
            else:
                kwargs = {}
            f(**kwargs)

    def finalize(self, context: Context):
        previous_ctx = get_call_context()
        set_call_context(context)
        try:
            self._collab_finalize(self, context)

            # finalize target objects
            for obj in self._managed_objects.values():
                self._collab_finalize(obj, context)
        finally:
            set_call_context(previous_ctx)

    def new_context(self, caller: str, callee: str, target_group=None, set_call_ctx=True):
        ctx = Context(self, caller, callee, self._abort_signal, target_group=target_group)
        if set_call_ctx:
            set_call_context(ctx)
        return ctx

    def get_collab_interface(self) -> dict[str, dict[str, list[str]]]:
        return {name: publish_interface.to_dict() for name, publish_interface in self._collab_interface.items()}

    def get_target_object_publish_interface(self, target_name: str):
        if not target_name or target_name.lower() == "app":
            return self._collab_interface.get("")
        else:
            return self._collab_interface.get(target_name)

    def get_children(self):
        return []

    def has_children(self):
        return False

    def get_leaf_clients(self):
        if not isinstance(self._client_hierarchy, Forest):
            raise RuntimeError(f"client_hierarchy must be Forest but got {type(self._client_hierarchy)}")
        leaf_nodes = [node for node in self._client_hierarchy.nodes.values() if not node.children]
        return [node.obj for node in leaf_nodes]


class ServerApp(App):

    def __init__(self, obj, name: str = "server"):
        if not obj:
            raise ValueError("server object must be specified")
        super().__init__(obj, name)
        self.mains = get_object_main_funcs(self.obj)
        if len(self.mains) != 1:
            raise ValueError(f"server object must have exactly one @collab.main function but got {len(self.mains)}")

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
