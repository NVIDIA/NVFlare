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

from typing import Dict, List, Optional

from nvflare.apis.client_engine_spec import TaskAssignment
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.workspace import Workspace
from nvflare.private.event import fire_event
from nvflare.widgets.fed_event import ClientFedEventRunner
from nvflare.widgets.info_collector import InfoCollector
from nvflare.widgets.widget import Widget, WidgetID

from .client_aux_runner import ClientAuxRunner
from .client_engine_executor_spec import ClientEngineExecutorSpec
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
        # self.status = MachineStatus.STOPPED


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
            workspace: workspacee
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
        self.aux_runner = ClientAuxRunner()
        self.add_handler(self.aux_runner)
        self.conf = conf

        self.fl_ctx_mgr = FLContextManager(
            engine=self, identity_name=client_name, job_id=job_id, public_stickers={}, private_stickers={}
        )

        self.run_info = ClientRunInfo(job_id=job_id)

        self.widgets = {WidgetID.INFO_COLLECTOR: InfoCollector(), WidgetID.FED_EVENT_RUNNER: ClientFedEventRunner()}
        for _, widget in self.widgets.items():
            self.handlers.append(widget)

    def get_task_assignment(self, fl_ctx: FLContext) -> TaskAssignment:
        pull_success, task_name, remote_tasks = self.client.fetch_task(fl_ctx)
        task = None
        if pull_success:
            shareable = self.client.extract_shareable(remote_tasks, fl_ctx)
            # task_id = fl_ctx.get_peer_context().get_cookie(FLContextKey.TASK_ID)
            task_id = shareable.get_header(key=FLContextKey.TASK_ID)
            task = TaskAssignment(name=task_name, task_id=task_id, data=shareable)
        return task

    def new_context(self) -> FLContext:
        return self.fl_ctx_mgr.new_context()

    def send_task_result(self, result: Shareable, fl_ctx: FLContext) -> bool:
        try:
            self.client.push_results(result, fl_ctx)  # push task execution results
            return True
        except BaseException:
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

    def aux_send(self, topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> Shareable:
        reply = self.client.aux_send(topic, request, timeout, fl_ctx)
        if reply:
            return self.client.extract_shareable(reply, fl_ctx)
        else:
            return make_reply(ReturnCode.COMMUNICATION_ERROR)

    def send_aux_request(self, topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> Shareable:
        return self.aux_runner.send_aux_request(topic, request, timeout, fl_ctx)

    def register_aux_message_handler(self, topic: str, message_handle_func):
        self.aux_runner.register_aux_message_handler(topic, message_handle_func)

    def fire_and_forget_aux_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        return self.send_aux_request(topic, request, 0.0, fl_ctx)

    def abort_app(self, job_id: str, fl_ctx: FLContext):
        runner = fl_ctx.get_prop(key=FLContextKey.RUNNER, default=None)
        if isinstance(runner, ClientRunner):
            runner.abort()
