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

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.responder import Responder
from nvflare.fuel.utils.json_scanner import Node
from nvflare.private.fed_json_config import FedJsonConfigurator
from nvflare.private.json_configer import ConfigContext, ConfigError

from .server_runner import ServerRunnerConfig

FL_PACKAGES = ["nvflare"]
FL_MODULES = ["server", "client", "aggregators", "handlers", "pt", "app", "app_common", "workflows"]


class WorkFlow:
    def __init__(self, id, responder: Responder):
        """Workflow is a responder with ID.

        Args:
            id: identification
            responder (Responder): A responder
        """
        self.id = id
        self.responder = responder


class ServerJsonConfigurator(FedJsonConfigurator):
    def __init__(self, config_file_name: str, exclude_libs=True):
        """This class parses server config from json file.

        Args:
            config_file_name (str): json file to parse
            exclude_libs (bool): whether to exclude libs
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

        # if server doesn't hear heartbeat from client for this long, we'll consider the client dead
        self.heartbeat_timeout = 60  # default to 1 minute

        # server will ask client to come back for next task after this many secs
        self.task_request_interval = 2  # default to 2 secs

        # workflows to be executed
        self.workflows = []

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        FedJsonConfigurator.process_config_element(self, config_ctx, node)

        element = node.element
        path = node.path()

        if path == "server.heart_beat_timeout":
            self.heartbeat_timeout = element
            if not isinstance(element, int) and not isinstance(element, float):
                raise ConfigError('"heart_beat_timeout" must be a number, but got {}'.format(type(element)))

            if element <= 0.0:
                raise ConfigError('"heart_beat_timeout" must be positive number, but got {}'.format(element))

            return

        if path == "server.task_request_interval":
            self.task_request_interval = element
            if not isinstance(element, int) and not isinstance(element, float):
                raise ConfigError('"task_request_interval" must be a number, but got {}'.format(type(element)))

            if element < 1:
                raise ConfigError('"task_request_interval" must >= 1, but got {}'.format(element))

            return

        if re.search(r"^workflows\.#[0-9]+$", path):
            workflow = self.build_component(element)
            if not isinstance(workflow, Responder):
                raise ConfigError(
                    '"workflow" must be a Responder or Controller object, but got {}'.format(type(workflow))
                )

            cid = element.get("id", None)
            if not cid:
                cid = type(workflow).__name__

            if not isinstance(cid, str):
                raise ConfigError('"id" must be str but got {}'.format(type(cid)))

            if cid in self._get_all_workflows_ids():
                raise ConfigError('duplicate workflow id "{}"'.format(cid))

            if cid in self.components:
                raise ConfigError('duplicate component id "{}"'.format(cid))

            self.workflows.append(WorkFlow(cid, workflow))
            self.components[cid] = workflow
            return

    def _get_all_workflows_ids(self):
        ids = []
        for t in self.workflows:
            ids.append(t.id)
        return ids

    def build_component(self, config_dict):
        t = super().build_component(config_dict)
        if isinstance(t, FLComponent):
            if type(t).__name__ not in [type(h).__name__ for h in self.handlers]:
                self.handlers.append(t)
        return t

    def finalize_config(self, config_ctx: ConfigContext):
        FedJsonConfigurator.finalize_config(self, config_ctx)

        if not self.workflows:
            raise ConfigError("workflows not specified")

        self.runner_config = ServerRunnerConfig(
            heartbeat_timeout=self.heartbeat_timeout,
            task_request_interval=self.task_request_interval,
            workflows=self.workflows,
            task_data_filters=self.data_filter_table,
            task_result_filters=self.result_filter_table,
            components=self.components,
            handlers=self.handlers,
        )
