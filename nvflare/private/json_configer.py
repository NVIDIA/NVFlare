# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import threading
from typing import List, Union

from nvflare.fuel.common.excepts import ComponentNotAuthorized, ConfigError
from nvflare.fuel.utils.class_loader import load_class
from nvflare.fuel.utils.class_utils import ModuleScanner
from nvflare.fuel.utils.component_builder import ComponentBuilder
from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.dict_utils import augment
from nvflare.fuel.utils.json_scanner import JsonObjectProcessor, JsonScanner, Node
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.wfconf import resolve_var_refs
from nvflare.security.logging import secure_format_exception


class ConfigContext(object):
    def __init__(self):
        """To init thee ConfigContext."""
        self.config_json = None
        self.pass_num = 0


class JsonConfigurator(JsonObjectProcessor, ComponentBuilder):
    def __init__(
        self,
        config_file_name: Union[str, List[str]],
        base_pkgs: List[str],
        module_names: List[str],
        exclude_libs=True,
        num_passes=1,
        sys_vars=None,
    ):
        """To init the JsonConfigurator.

        Args:
            config_file_name: config filename or list of JSON config file names
            base_pkgs: base packages need to be scanned
            module_names: module names need to be scanned
            exclude_libs: True/False to exclude the libs folder
            num_passes: number of passes to parsing the config
            sys_vars: system vars
        """
        JsonObjectProcessor.__init__(self)
        self.logger = get_obj_logger(self)

        if not isinstance(num_passes, int):
            raise TypeError(f"num_passes must be int but got {num_passes}")
        if not num_passes > 0:
            raise ValueError(f"num_passes must > 0 but got {num_passes}")

        if isinstance(config_file_name, str):
            config_files = [config_file_name]
        elif isinstance(config_file_name, list):
            config_files = config_file_name
        else:
            raise TypeError(f"config_file_names must be str or list of strs but got {type(config_file_name)}")

        for f in config_files:
            if not ConfigFactory.has_config(f):
                raise FileNotFoundError(f"config_file_names {f} does not exist or not a file")

        self.config_file_names = config_files
        self.num_passes = num_passes
        self.sys_vars = sys_vars
        self.module_scanner = ModuleScanner(base_pkgs, module_names, exclude_libs)

        config_data = {}
        for f in config_files:
            data = ConfigService.load_config_dict(f)
            try:
                augment(to_dict=config_data, from_dict=data, from_override_to=False)
            except Exception as e:
                raise RuntimeError("Error processing config file {}: {}".format(f, secure_format_exception(e)))

        self.config_data = config_data
        self.config_ctx = self._make_config_ctx()
        self.json_scanner = JsonScanner(config_data, config_files)
        self.build_auth_func = None
        self.build_auth_kwargs = None
        self._build_auth_state = threading.local()

    def _make_config_ctx(self):
        config_ctx = ConfigContext()
        config_ctx.config_json = self.config_data
        return config_ctx

    def _get_config_ctx(self):
        if self.config_ctx is None:
            self.config_ctx = self._make_config_ctx()
        return self.config_ctx

    def _get_build_auth_state(self):
        state = self._build_auth_state
        if not hasattr(state, "build_node_stack"):
            state.build_node_stack = []
            state.authorized_config_ids = None
            state.building_authorized_component_depth = 0
        return state

    @property
    def _build_node_stack(self):
        return self._get_build_auth_state().build_node_stack

    @property
    def _authorized_config_ids(self):
        return self._get_build_auth_state().authorized_config_ids

    @_authorized_config_ids.setter
    def _authorized_config_ids(self, value):
        self._get_build_auth_state().authorized_config_ids = value

    @property
    def _building_authorized_component_depth(self):
        return self._get_build_auth_state().building_authorized_component_depth

    @_building_authorized_component_depth.setter
    def _building_authorized_component_depth(self, value):
        self._get_build_auth_state().building_authorized_component_depth = value

    def set_component_build_authorizer(self, func, **kwargs):
        if not callable(func):
            raise ValueError("authorizer func is not callable")
        self.build_auth_func = func
        self.build_auth_kwargs = kwargs

    def has_component_build_authorizer(self):
        return self.build_auth_func is not None

    def _authorize_component_config(self, config_dict, config_ctx: ConfigContext, node: Node):
        if self.build_auth_func is None:
            return

        if self._authorized_config_ids is not None:
            config_id = id(config_dict)
            if config_id in self._authorized_config_ids:
                return

        err = self.build_auth_func(config_dict, config_ctx, node, **self.build_auth_kwargs)
        if err:
            raise ComponentNotAuthorized(f"component not authorized: {err}")

        if self._authorized_config_ids is not None:
            self._authorized_config_ids.add(id(config_dict))

    def _is_authorizable_component_config(self, config_dict, node=None):
        return self.is_authorizable_component_config(config_dict, node)

    @staticmethod
    def _make_child_node(parent_node, element, key):
        node = Node(element)
        node.processor = parent_node.processor
        node.parent = parent_node
        node.level = parent_node.level + 1
        node.key = str(key)
        node.paths = copy.copy(parent_node.paths)
        node.paths.append(node.key)
        return node

    def _authorize_component_config_tree(self, element, config_ctx: ConfigContext, node: Node, force_current=False):
        if isinstance(element, dict):
            if force_current or self._is_authorizable_component_config(element, node):
                self._authorize_component_config(element, config_ctx, node)
                args = element.get("args")
                if isinstance(args, (dict, list)):
                    self._authorize_component_config_tree(
                        args,
                        config_ctx,
                        self._make_child_node(node, args, "args"),
                    )
                return

            for key, value in element.items():
                if isinstance(value, (dict, list)):
                    self._authorize_component_config_tree(
                        value,
                        config_ctx,
                        self._make_child_node(node, value, key),
                    )
        elif isinstance(element, list):
            for i, item in enumerate(element):
                if isinstance(item, (dict, list)):
                    self._authorize_component_config_tree(
                        item,
                        config_ctx,
                        self._make_child_node(node, item, f"#{i + 1}"),
                    )

    def authorize_and_build_component(self, config_dict, config_ctx: ConfigContext, node: Node):
        new_auth_scope = self._authorized_config_ids is None
        if new_auth_scope:
            self._authorized_config_ids = set()

        try:
            self._authorize_component_config_tree(config_dict, config_ctx, node, force_current=True)

            self._build_node_stack.append((config_ctx, node))
            self._building_authorized_component_depth += 1
            try:
                return self._build_component(config_dict)
            finally:
                self._building_authorized_component_depth -= 1
                self._build_node_stack.pop()
        finally:
            if new_auth_scope:
                self._authorized_config_ids = None

    def _make_nested_component_node(self, config_dict, arg_name):
        node = Node(config_dict)
        node.processor = self
        node.key = str(arg_name)

        if self._build_node_stack and self._build_node_stack[-1][1] is not None:
            _, parent = self._build_node_stack[-1]
            node.parent = parent
            node.processor = parent.processor
            node.level = parent.level + 2
            node.paths = copy.copy(parent.paths)
            node.paths.extend(["args", node.key])

        return node

    def _make_runtime_component_node(self, config_dict):
        node = Node(config_dict)
        node.processor = self
        node.key = "runtime_component"
        node.paths = [node.key]
        return node

    def build_nested_component(self, config_dict, arg_name):
        if self.build_auth_func is None:
            return super().build_nested_component(config_dict, arg_name)

        node = self._make_nested_component_node(config_dict, arg_name)
        config_ctx = self._get_config_ctx()
        if self._build_node_stack:
            config_ctx, _ = self._build_node_stack[-1]
        return self.authorize_and_build_component(config_dict, config_ctx, node)

    def _build_component(self, config_dict):
        return super().build_component(config_dict)

    def build_component(self, config_dict):
        if self.build_auth_func is None or self._building_authorized_component_depth > 0:
            return self._build_component(config_dict)

        node = self._make_runtime_component_node(config_dict)
        return self.authorize_and_build_component(config_dict, self._get_config_ctx(), node)

    def get_module_scanner(self):
        return self.module_scanner

    def _do_configure(self):
        self.config_ctx = self._make_config_ctx()

        # every item could be used as reference
        all_vars = copy.deepcopy(self.config_data)

        # Remove parameterized items from config data since such items could only be used as refs and cannot be
        # part of config. After they are removed from config data, they will not be resolved until they are invoked.
        parameterized_items = []
        for k in self.config_data.keys():
            if isinstance(k, str) and k.startswith("@"):
                parameterized_items.append(k)
        for k in parameterized_items:
            self.config_data.pop(k)

        # Add env_vars to all_vars. If there are conflicts, env_vars take precedence.
        if self.sys_vars:
            all_vars.update(self.sys_vars)

        # resolve var references
        resolve_var_refs(self.json_scanner, all_vars)

        self.start_config(self.config_ctx)

        # scan the config to create components
        for i in range(self.num_passes):
            self.config_ctx.pass_num = i + 1
            self.json_scanner.scan(self)

        # finalize configuration
        self.finalize_config(self.config_ctx)

    def configure(self):
        try:
            self._do_configure()
        except Exception as e:
            print("Error processing config {}: {}".format(self.config_file_names, secure_format_exception(e)))
            raise e

    def process_element(self, node: Node):
        self.process_config_element(self.config_ctx, node)

    def is_configured_subclass(self, config_dict, base_class):
        return issubclass(load_class(self.get_class_path(config_dict)), base_class)

    def start_config(self, config_ctx: ConfigContext):
        pass

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        pass

    def finalize_config(self, config_ctx: ConfigContext):
        pass


def get_component_refs(component):
    if "path" in component:
        name = component["path"]
        key = "path"
    elif "class_path" in component:
        name = component["class_path"]
        key = "class_path"
    elif "name" in component:
        name = component["name"]
        key = "name"
    else:
        raise ConfigError('component has no "path", "class_path", or "name"')

    if name is None or not isinstance(name, str):
        raise ConfigError('component "{}" must be a non-null string, got {}'.format(key, type(name).__name__))
    if len(name) <= 0:
        raise ConfigError('component "{}" must not be empty'.format(key))

    parts = name.split("#")
    component[key] = parts[0]
    return parts
