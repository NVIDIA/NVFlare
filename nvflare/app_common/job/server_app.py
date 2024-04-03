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

from nvflare.apis.impl.controller import Controller
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.app_common.job.base_app import BaseApp
from nvflare.private.fed.server.server_json_config import WorkFlow


class ServerApp(BaseApp):
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

