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

from typing import List, Optional, Tuple

from nvflare.apis.client import Client
from nvflare.apis.engine_spec import EngineSpec
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.workspace import Workspace
from nvflare.private.aux_runner import AuxRunner
from nvflare.private.event import fire_event
from nvflare.private.fed.utils.fed_utils import create_job_processing_context_properties

from .client_manager import ClientManager
from .run_info import RunInfo


class RunManager(EngineSpec):
    def __init__(
        self,
        server_name,
        engine: ServerEngineSpec,
        job_id,
        workspace: Workspace,
        components: {str: FLComponent},
        client_manager: Optional[ClientManager] = None,
        handlers: Optional[List[FLComponent]] = None,
    ):
        """Manage run.

        Args:
            server_name: server name
            engine (ServerEngineSpec): server engine
            job_id: job id
            workspace (Workspace): workspace
            components (dict): A dict of extra python objects {id: object}
            client_manager (ClientManager, optional): client manager
            handlers (List[FLComponent], optional): handlers
        """
        super().__init__()
        self.server_name = server_name

        self.client_manager = client_manager
        self.handlers = handlers
        self.aux_runner = AuxRunner(self)
        self.add_handler(self.aux_runner)

        if job_id:
            job_ctx_props = self.create_job_processing_context_properties(workspace, job_id)
        else:
            job_ctx_props = {}

        self.fl_ctx_mgr = FLContextManager(
            engine=engine, identity_name=server_name, job_id=job_id, public_stickers={}, private_stickers=job_ctx_props
        )

        self.workspace = workspace
        self.run_info = RunInfo(job_id=job_id, app_path=self.workspace.get_app_dir(job_id))

        self.components = components
        self.cell = None

    def get_server_name(self):
        return self.server_name

    def get_run_info(self) -> RunInfo:
        return self.run_info

    def get_handlers(self):
        return self.handlers

    def new_context(self) -> FLContext:
        return self.fl_ctx_mgr.new_context()

    def get_workspace(self) -> Workspace:
        return self.workspace

    def get_component(self, component_id: str) -> object:
        return self.components.get(component_id)

    def add_component(self, component_id: str, component):
        self.components[component_id] = component

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        fire_event(event=event_type, handlers=self.handlers, ctx=fl_ctx)

    def add_handler(self, handler: FLComponent):
        self.handlers.append(handler)

    def get_cell(self):
        return self.cell

    def validate_targets(self, client_names: List[str]) -> Tuple[List[Client], List[str]]:
        return self.client_manager.get_all_clients_from_inputs(client_names)

    def create_job_processing_context_properties(self, workspace, job_id):
        return create_job_processing_context_properties(workspace, job_id)
