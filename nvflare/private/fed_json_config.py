# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import re

from nvflare.apis.filter import Filter
from nvflare.fuel.utils.json_scanner import Node
from nvflare.private.json_configer import ConfigContext, ConfigError, JsonConfigurator


class FilterChain(object):
    def __init__(self):
        """To init the FilterChain."""
        self.tasks = []
        self.filters = []


class FedJsonConfigurator(JsonConfigurator):
    def __init__(self, config_file_name: str, base_pkgs: [str], module_names: [str], exclude_libs=True):
        """To init the FedJsonConfigurator.

        Args:
            config_file_name: config filename
            base_pkgs: base packages need to be scanned
            module_names: module names need to be scanned
            exclude_libs: True/False to exclude the libs folder
        """
        JsonConfigurator.__init__(
            self,
            config_file_name=config_file_name,
            base_pkgs=base_pkgs,
            module_names=module_names,
            exclude_libs=exclude_libs,
        )

        self.format_version = None
        self.handlers = []
        self.components = {}  # id => component
        self.task_data_filter_chains = []
        self.task_result_filter_chains = []
        self.current_filter_chain = None
        self.data_filter_table = None
        self.result_filter_table = None

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        element = node.element
        path = node.path()

        if path == "format_version":
            self.format_version = element
            return

        # if re.search(r"^handlers\.#[0-9]+$", path):
        #     h = self.build_component(element)
        #     if not isinstance(h, FLComponent):
        #         raise ConfigError("handler must be a FLComponent object, but got {}".format(type(h)))
        #     # Ensure only add one instance of the handlers for the same component
        #     if type(h).__name__ not in [type(t).__name__ for t in self.handlers]:
        #         self.handlers.append(h)
        #     return

        if re.search(r"^components\.#[0-9]+$", path):
            c = self.build_component(element)
            cid = element.get("id", None)
            if not cid:
                raise ConfigError("missing component id")

            if not isinstance(cid, str):
                raise ConfigError('"id" must be str but got {}'.format(type(cid)))

            if cid in self.components:
                raise ConfigError('duplicate component id "{}"'.format(cid))

            self.components[cid] = c
            return

        # result filters
        if re.search(r"^task_result_filters\.#[0-9]+$", path):
            self.current_filter_chain = FilterChain()
            node.props["data"] = self.current_filter_chain
            node.exit_cb = self._process_result_filter_chain
            return

        if re.search(r"^task_result_filters\.#[0-9]+\.tasks$", path):
            self.current_filter_chain.tasks = element
            return

        if re.search(r"^task_result_filters.#[0-9]+\.filters\.#[0-9]+$", path):
            f = self.build_component(element)
            self.current_filter_chain.filters.append(f)
            return

        # data filters
        if re.search(r"^task_data_filters\.#[0-9]+$", path):
            self.current_filter_chain = FilterChain()
            node.props["data"] = self.current_filter_chain
            node.exit_cb = self._process_data_filter_chain
            return

        if re.search(r"^task_data_filters\.#[0-9]+\.tasks$", path):
            self.current_filter_chain.tasks = element
            return

        if re.search(r"^task_data_filters.#[0-9]+\.filters\.#[0-9]+$", path):
            f = self.build_component(element)
            self.current_filter_chain.filters.append(f)
            return

    def validate_tasks(self, tasks):
        if not isinstance(tasks, list):
            raise ConfigError('"tasks" must be specified as list of task names but got {}'.format(type(tasks)))

        if len(tasks) <= 0:
            raise ConfigError('"tasks" must not be empty')

        for n in tasks:
            if not isinstance(n, str):
                raise ConfigError("task names must be string but got {}".format(type(n)))

    def validate_filter_chain(self, chain: FilterChain):
        self.validate_tasks(chain.tasks)

        if not isinstance(chain.filters, list):
            raise ConfigError('"filters" must be specified as list of filters but got {}'.format(type(chain.filters)))

        if len(chain.filters) <= 0:
            raise ConfigError('"filters" must not be empty')

        for f in chain.filters:
            if not isinstance(f, Filter):
                raise ConfigError('"filters" must contain Filter object but got {}'.format(type(f)))

    def _process_result_filter_chain(self, node: Node):
        filter_chain = node.props["data"]
        self.validate_filter_chain(filter_chain)
        self.task_result_filter_chains.append(filter_chain)

    def _process_data_filter_chain(self, node: Node):
        filter_chain = node.props["data"]
        self.validate_filter_chain(filter_chain)
        self.task_data_filter_chains.append(filter_chain)

    def finalize_config(self, config_ctx: ConfigContext):
        if self.format_version is None:
            raise ConfigError("missing format_version")

        if not isinstance(self.format_version, int):
            raise ConfigError('"format_version" must be int, but got {}'.format(type(self.format_version)))

        if self.format_version != 2:
            raise ConfigError('wrong "format_version" {}: must be 2'.format(self.format_version))

        data_filter_table = {}
        for c in self.task_data_filter_chains:
            if not isinstance(c, FilterChain):
                raise TypeError("chain must be FilterChain but got {}".format(type(c)))
            for t in c.tasks:
                if t in data_filter_table:
                    raise ConfigError("multiple data filter chains defined for task {}".format(t))
                data_filter_table[t] = c.filters
        self.data_filter_table = data_filter_table

        result_filter_table = {}
        for c in self.task_result_filter_chains:
            if not isinstance(c, FilterChain):
                raise TypeError("chain must be FilterChain but got {}".format(type(c)))
            for t in c.tasks:
                if t in result_filter_table:
                    raise ConfigError("multiple data filter chains defined for task {}".format(t))
                result_filter_table[t] = c.filters

        self.result_filter_table = result_filter_table
