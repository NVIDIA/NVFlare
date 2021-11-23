# Copyright (c) 2021, NVIDIA CORPORATION.
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

import time

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import MachineStatus
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.workspace import Workspace
from nvflare.private.event import fire_event
from .client_manager import ClientManager
from .server_aux_runner import ServerAuxRunner


class RunInfo(object):
    def __init__(self, run_num, app_path):
        self.run_number = run_num
        self.start_time = time.time()
        self.app_path = app_path
        self.status = MachineStatus.STOPPED


class RunManager:
    def __init__(
        self,
        server_name,
        engine: ServerEngineSpec,
        run_num,
        workspace: Workspace,
        components: {str: FLComponent},
        client_manager: [ClientManager] = None,
        handlers: [FLComponent] = None,
    ) -> None:
        super().__init__()
        self.server_name = server_name  # should uniquely define a tlt workflow

        self.client_manager = client_manager
        self.handlers = handlers
        self.aux_runner = ServerAuxRunner()
        self.add_handler(self.aux_runner)

        self.fl_ctx_mgr = FLContextManager(
            engine=engine, identity_name=server_name, run_num=run_num,
            public_stickers={}, private_stickers={}
        )

        self.workspace = workspace
        self.run_info = RunInfo(run_num=run_num, app_path=self.workspace.get_app_dir(run_num))

        self.components = components

    def get_server_name(self):
        return self.server_name

    def get_run_info(self):
        return self.run_info

    def get_handlers(self):
        return self.handlers

    def new_context(self) -> FLContext:
        return self.fl_ctx_mgr.new_context()

    def get_workspace(self) -> Workspace:
        return self.workspace

    def get_component(self, component_id: str) -> object:
        return self.components.get(component_id)

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        fire_event(event=event_type, handlers=self.handlers, ctx=fl_ctx)

    def add_handler(self, handler: FLComponent):
        self.handlers.append(handler)
