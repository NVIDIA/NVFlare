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

import logging
import time
from typing import Dict, List, Optional, Union

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ServerCommandKey, ServerCommandNames, SiteType
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.shareable import Shareable
from nvflare.apis.workspace import Workspace
from nvflare.fuel.f3.cellnet.cell import FQCN
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.utils import fobs
from nvflare.private.aux_runner import AuxRunner
from nvflare.private.defs import CellChannel, CellMessageHeaderKeys, new_cell_message
from nvflare.private.event import fire_event
from nvflare.private.fed.utils.fed_utils import create_job_processing_context_properties
from nvflare.widgets.fed_event import ClientFedEventRunner
from nvflare.widgets.info_collector import InfoCollector
from nvflare.widgets.widget import Widget, WidgetID

from .client_engine_executor_spec import ClientEngineExecutorSpec, TaskAssignment
from .client_json_config import ClientJsonConfigurator
from .client_runner import ClientRunner
from .fed_client import FederatedClient


class ClientRunInfo(object):
    def __init__(self, job_id):
        """To init the ClientRunInfo.

        Args:
            job_id: job id
        """
        self.job_id = job_id
        self.current_task_name = ""
        self.start_time = None


# TODO: make this configurable
#   this is the number of retries for client side child/job process to get clients from server
#   we might need to think of removing the whole get clients from server logic from child process
GET_CLIENTS_RETRY = 300


class ClientRunManager(ClientEngineExecutorSpec):
    """ClientRunManager provides the ClientEngine APIs implementation running in the child process."""

    def __init__(
        self,
        client_name: str,
        job_id: str,
        workspace: Workspace,
        client: FederatedClient,
        components: Dict[str, FLComponent],
        handlers: Optional[List[FLComponent]] = None,
        conf: ClientJsonConfigurator = None,
    ) -> None:
        """To init the ClientRunManager.

        Args:
            client_name: client name
            job_id: job id
            workspace: workspace
            client: FL client object
            components: available FL components
            handlers: available handlers
            conf: ClientJsonConfigurator object
        """
        super().__init__()

        self.client = client
        self.handlers = handlers
        self.workspace = workspace
        self.components = components
        # self.aux_runner = ClientAuxRunner()
        self.aux_runner = AuxRunner(self)
        self.add_handler(self.aux_runner)
        self.conf = conf
        self.cell = None

        self.all_clients = None

        if not components:
            self.components = {}

        if not handlers:
            self.handlers = []

        # get job meta!
        job_ctx_props = self.create_job_processing_context_properties(workspace, job_id)
        self.fl_ctx_mgr = FLContextManager(
            engine=self, identity_name=client_name, job_id=job_id, public_stickers={}, private_stickers=job_ctx_props
        )

        self.run_info = ClientRunInfo(job_id=job_id)

        self.widgets = {WidgetID.INFO_COLLECTOR: InfoCollector(), WidgetID.FED_EVENT_RUNNER: ClientFedEventRunner()}
        for _, widget in self.widgets.items():
            self.handlers.append(widget)

        self.logger = logging.getLogger(self.__class__.__name__)

    def get_task_assignment(self, fl_ctx: FLContext) -> TaskAssignment:
        pull_success, task_name, return_shareable = self.client.fetch_task(fl_ctx)
        task = None
        if pull_success:
            shareable = self.client.extract_shareable(return_shareable, fl_ctx)
            task_id = shareable.get_header(key=FLContextKey.TASK_ID)
            task = TaskAssignment(name=task_name, task_id=task_id, data=shareable)
        return task

    def new_context(self) -> FLContext:
        return self.fl_ctx_mgr.new_context()

    def send_task_result(self, result: Shareable, fl_ctx: FLContext) -> bool:
        push_result = self.client.push_results(result, fl_ctx)  # push task execution results
        if push_result[0] == CellReturnCode.OK:
            return True
        else:
            return False

    def get_workspace(self) -> Workspace:
        return self.workspace

    def get_run_info(self) -> ClientRunInfo:
        return self.run_info

    def show_errors(self) -> ClientRunInfo:
        return self.run_info

    def reset_errors(self) -> ClientRunInfo:
        return self.run_info

    def dispatch(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        return self.aux_runner.dispatch(topic=topic, request=request, fl_ctx=fl_ctx)

    def get_component(self, component_id: str) -> object:
        return self.components.get(component_id)

    def get_all_components(self) -> dict:
        return self.components

    def validate_targets(self, inputs) -> ([], []):
        valid_inputs = []
        invalid_inputs = []
        for item in inputs:
            if item == FQCN.ROOT_SERVER:
                valid_inputs.append(item)
            else:
                client = self.get_client_from_name(item)
                if client:
                    valid_inputs.append(item)
                else:
                    invalid_inputs.append(item)
        return valid_inputs, invalid_inputs

    def get_client_from_name(self, client_name):
        for _, c in self.all_clients.items():
            if client_name == c.name:
                return c
        return None

    def get_widget(self, widget_id: str) -> Widget:
        return self.widgets.get(widget_id)

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        fire_event(event=event_type, handlers=self.handlers, ctx=fl_ctx)

    def add_handler(self, handler: FLComponent):
        self.handlers.append(handler)

    def build_component(self, config_dict):
        if not self.conf:
            raise RuntimeError("No configurator set up.")
        return self.conf.build_component(config_dict)

    def get_cell(self):
        return self.cell

    def send_aux_request(
        self,
        targets: Union[None, str, List[str]],
        topic: str,
        request: Shareable,
        timeout: float,
        fl_ctx: FLContext,
        optional=False,
    ) -> dict:
        if not targets:
            targets = [FQCN.ROOT_SERVER]
        else:
            if isinstance(targets, str):
                if targets == SiteType.ALL:
                    targets = [FQCN.ROOT_SERVER]
                    for _, t in self.all_clients.items():
                        if t.name != self.client.client_name:
                            targets.append(t.name)
                else:
                    targets = [targets]
        if targets:
            return self.aux_runner.send_aux_request(targets, topic, request, timeout, fl_ctx, optional=optional)
        else:
            return {}

    def get_all_clients_from_server(self, fl_ctx, retry=0):
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        get_clients_message = new_cell_message({CellMessageHeaderKeys.JOB_ID: job_id}, fobs.dumps({}))
        return_data = self.client.cell.send_request(
            target=FQCN.ROOT_SERVER,
            channel=CellChannel.SERVER_PARENT_LISTENER,
            topic=ServerCommandNames.GET_CLIENTS,
            request=get_clients_message,
            timeout=5.0,
            optional=True,
        )
        return_code = return_data.get_header(MessageHeaderKey.RETURN_CODE)

        if return_code == CellReturnCode.OK:
            if return_data.payload:
                data = fobs.loads(return_data.payload)
                self.all_clients = data.get(ServerCommandKey.CLIENTS)
            else:
                raise RuntimeError("Empty clients data from server")
        else:
            # retry to handle the server connect has not been established scenario.
            retry += 1
            if retry < GET_CLIENTS_RETRY:
                time.sleep(0.5)
                self.get_all_clients_from_server(fl_ctx, retry)
            else:
                raise RuntimeError("Failed to get the clients from the server.")

    def register_aux_message_handler(self, topic: str, message_handle_func):
        self.aux_runner.register_aux_message_handler(topic, message_handle_func)

    def fire_and_forget_aux_request(self, topic: str, request: Shareable, fl_ctx: FLContext, optional=False) -> dict:
        return self.send_aux_request(
            targets=None, topic=topic, request=request, timeout=0.0, fl_ctx=fl_ctx, optional=optional
        )

    def abort_app(self, job_id: str, fl_ctx: FLContext):
        runner = fl_ctx.get_prop(key=FLContextKey.RUNNER, default=None)
        if isinstance(runner, ClientRunner):
            runner.abort()

    def create_job_processing_context_properties(self, workspace, job_id):
        return create_job_processing_context_properties(workspace, job_id)
