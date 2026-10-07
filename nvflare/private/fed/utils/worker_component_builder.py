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

    def __init__(self, fl_ctx: FLContext = None, workspace=None, enforce_authorization=True) -> None:
        super().__init__()
        self.module_scanner = ModuleScanner(self.FL_PACKAGES, self.FL_MODULES, True)
        self.fl_ctx = fl_ctx
        self.workspace = workspace
        self.enforce_authorization = enforce_authorization
        self.component_path_authorizer = ComponentPathAuthorizer()

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
        if not self.enforce_authorization:
            return
        try:
            self.component_path_authorizer.authorize_component_config(
                config_dict, node, fl_ctx=self.fl_ctx, workspace=self.workspace
            )
        except UnsafeComponentError as ex:
            raise ComponentNotAuthorized(f"component not authorized: {ex}")

    def _authorize_component_config_tree(self, element, node, force_current=False):
        self.authorize_component_config_tree(
            element, node, self._authorize_component_config, force_current=force_current
        )

    def build_component(self, config_dict, node=None):
        if node is None:
            node = self.make_component_node(config_dict)
        self._authorize_component_config_tree(config_dict, node, force_current=True)
        return super().build_component(config_dict)
