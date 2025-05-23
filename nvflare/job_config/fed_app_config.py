# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List

from nvflare.apis.executor import Executor
from nvflare.apis.impl.controller import Controller
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.job_config.base_app_config import BaseAppConfig
from nvflare.private.fed.client.client_json_config import _ExecutorDef
from nvflare.private.fed.server.server_json_config import WorkFlow


class ClientAppConfig(BaseAppConfig):
    """ClientAppConfig represents the ClientApp inside the Job. It holds the BaseAppConfig components and the task
    executors components data for the ClientApp.

    """

    def __init__(self) -> None:
        super().__init__()

        self.executors: [_ExecutorDef] = []

    def add_executor(self, tasks: List[str], executor: Executor):
        if not isinstance(executor, Executor):
            raise RuntimeError(f"workflow must be type of Executor, but got {executor.__class__}")

        task_set = set(tasks)
        for item in self.executors:
            b_set = set(item.tasks)
            dup_tasks = task_set.intersection(b_set)
            if len(dup_tasks) > 0:
                raise RuntimeError(f"executor for tasks {dup_tasks} already exist.")

        e = _ExecutorDef()
        e.tasks = tasks
        e.executor = executor
        self.executors.append(e)


class ServerAppConfig(BaseAppConfig):
    """ServerAppConfig represents the ServerApp inside the Job. it holds the BaseAppConfig components and the
    workflow components data for the ServerApp.

    """

    def __init__(self) -> None:
        super().__init__()

        self.workflows: [Controller] = []
        self.ids = []

    def add_workflow(self, cid, controller: Controller):
        if not isinstance(controller, Controller):
            raise RuntimeError(f"workflow must be type of Controller, but got {controller.__class__}")

        # self.add_component(cid, controller)
        if cid in self.components.keys() or cid in self.ids:
            raise RuntimeError(f"Component with ID:{cid} already exist.")

        communicator = WFCommServer()
        self.handlers.append(communicator)
        controller.set_communicator(communicator)

        self.workflows.append(WorkFlow(cid, controller))
        self.ids.append(cid)


class FedAppConfig:
    """FedAppConfig represents the App information inside the Job. It contains either a ServerApp, or a ClientApp, or
    both of them.

    """

    def __init__(self, server_app: ServerAppConfig = None, client_app: ClientAppConfig = None) -> None:
        super().__init__()

        if server_app and not isinstance(server_app, ServerAppConfig):
            raise ValueError(f"server_app must be type of ServerAppConfig, but got {server_app.__class__}")
        if client_app and not isinstance(client_app, ClientAppConfig):
            raise ValueError(f"client_app must be type of ClientAppConfig, but got {client_app.__class__}")

        self.server_app: ServerAppConfig = server_app
        self.client_app: ClientAppConfig = client_app
