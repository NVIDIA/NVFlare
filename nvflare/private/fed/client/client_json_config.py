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

from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.fuel.utils.json_scanner import Node
from nvflare.private.fed_json_config import FedJsonConfigurator
from nvflare.private.json_configer import ConfigContext, ConfigError

from .client_runner import ClientRunnerConfig


class _ExecutorDef(object):
    def __init__(self):
        self.tasks = []
        self.executor = None


FL_PACKAGES = ["nvflare"]
FL_MODULES = ["server", "client", "aggregators", "handlers", "pt", "app", "app_common", "workflows"]


class ClientJsonConfigurator(FedJsonConfigurator):
    def __init__(self, config_file_name: str, exclude_libs=True):
        """To init the ClientJsonConfigurator.

        Args:
            config_file_name: config file name
            exclude_libs: True/False to exclude the libs folder
        """
        base_pkgs = FL_PACKAGES
        module_names = FL_MODULES

        FedJsonConfigurator.__init__(
            self,
            config_file_name=config_file_name,
            base_pkgs=base_pkgs,
            module_names=module_names,
            exclude_libs=exclude_libs,
        )

        self.runner_config = None
        self.executors = []
        self.current_exe = None

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        FedJsonConfigurator.process_config_element(self, config_ctx, node)

        element = node.element
        path = node.path()

        # executors
        if re.search(r"^executors\.#[0-9]+$", path):
            self.current_exe = _ExecutorDef()
            node.props["data"] = self.current_exe
            node.exit_cb = self._process_executor_def
            return

        if re.search(r"^executors\.#[0-9]+\.tasks$", path):
            self.current_exe.tasks = element
            return

        if re.search(r"^executors\.#[0-9]+\.executor$", path):
            self.current_exe.executor = self.build_component(element)
            return

    def build_component(self, config_dict):
        t = super().build_component(config_dict)
        if isinstance(t, FLComponent):
            if type(t).__name__ not in [type(h).__name__ for h in self.handlers]:
                self.handlers.append(t)
        return t

    def _process_executor_def(self, node: Node):
        e = node.props["data"]
        if not isinstance(e, _ExecutorDef):
            raise TypeError("e must be _ExecutorDef but got {}".format(type(e)))
        self.validate_tasks(e.tasks)

        if not isinstance(e.executor, Executor):
            raise ConfigError('"executor" must be an Executor object but got {}'.format(type(e.executor)))

        self.executors.append(e)

    def finalize_config(self, config_ctx: ConfigContext):
        FedJsonConfigurator.finalize_config(self, config_ctx)

        if len(self.executors) <= 0:
            raise ConfigError("executors are not specified")

        task_table = {}
        for e in self.executors:
            for t in e.tasks:
                if t in task_table:
                    raise ConfigError('Multiple executors defined for task "{}"'.format(t))
                task_table[t] = e.executor

        self.runner_config = ClientRunnerConfig(
            task_table=task_table,
            task_data_filters=self.data_filter_table,
            task_result_filters=self.result_filter_table,
            components=self.components,
            handlers=self.handlers,
        )
