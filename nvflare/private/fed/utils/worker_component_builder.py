# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Site-policy-authorized component construction for task workers."""

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.app_common.widgets.component_path_authorizer import ComponentPathAuthorizer
from nvflare.fuel.common.excepts import ComponentNotAuthorized
from nvflare.fuel.utils.class_utils import ModuleScanner
from nvflare.fuel.utils.component_builder import ComponentBuilder
from nvflare.fuel.utils.json_scanner import Node


class WorkerComponentBuilder(ComponentBuilder):
    FL_PACKAGES = ["nvflare"]
    FL_MODULES = ["client", "app"]

    def __init__(self, fl_ctx: FLContext = None, workspace=None) -> None:
        super().__init__()
        self.module_scanner = ModuleScanner(self.FL_PACKAGES, self.FL_MODULES, True)
        self.fl_ctx = fl_ctx
        self.workspace = workspace
        self.component_path_authorizer = ComponentPathAuthorizer()
        self.handlers = []

    def get_module_scanner(self):
        return self.module_scanner

    @staticmethod
    def make_component_node(component_config, index=None):
        """Make a node retaining the component-list position for policy errors."""
        node = Node(component_config)
        node.key = "component"
        node.paths = ["component"]
        if index is not None:
            node.key = f"#{index}"
            node.paths = ["components", node.key]
        return node

    def _authorize_component_config(self, config_dict, node):
        try:
            self.component_path_authorizer.authorize_component_config(
                config_dict, node, fl_ctx=self.fl_ctx, workspace=self.workspace
            )
        except UnsafeComponentError as ex:
            raise ComponentNotAuthorized(f"component not authorized: {ex}")

    @staticmethod
    def _child_node(parent, element, key):
        node = Node(element)
        node.parent = parent
        node.key = str(key)
        node.paths = [*parent.paths, node.key]
        return node

    def authorize_tree(self, element, node=None, force_current=True):
        """Match configurator traversal: component args, ordinary containers and lists.

        Authorize the original graph before builders mutate nested arguments.
        Component-list entries retain their paths, including config_type: dict.
        """
        if node is None:
            node = self.make_component_node(element)
        if isinstance(element, dict) and (force_current or self.is_authorizable_component_config(element, node)):
            self._authorize_component_config(element, node)
            args = element.get("args")
            if not isinstance(args, (dict, list)):
                return
            node = self._child_node(node, args, "args")
            element = args
        if isinstance(element, dict):
            children = element.items()
        elif isinstance(element, list):
            children = ((f"#{i + 1}", item) for i, item in enumerate(element))
        else:
            return
        for key, value in children:
            if isinstance(value, (dict, list)):
                self.authorize_tree(value, self._child_node(node, value, key), force_current=False)

    def build_component(self, config_dict, node=None):
        if node is None:
            node = self.make_component_node(config_dict)
        self.authorize_tree(config_dict, node)
        component = super().build_component(config_dict)
        # Recursive construction returns nested components before their parent,
        # matching ClientJsonConfigurator's lifecycle registration order.
        if isinstance(component, FLComponent):
            self.handlers.append(component)
        return component
