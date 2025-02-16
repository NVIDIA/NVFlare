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

import time
from typing import Dict, List, Optional, Union

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import (
    FLContextKey,
    ProcessType,
    ReservedKey,
    ServerCommandKey,
    ServerCommandNames,
    SiteType,
)
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.shareable import Shareable
from nvflare.apis.streaming import ConsumerFactory, ObjectProducer, StreamableEngine, StreamContext
from nvflare.apis.workspace import Workspace
from nvflare.fuel.f3.cellnet.core_cell import FQCN
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.aux_runner import AuxMsgTarget, AuxRunner
from nvflare.private.defs import CellChannel, CellMessageHeaderKeys, new_cell_message
from nvflare.private.event import fire_event
from nvflare.private.fed.utils.fed_utils import create_job_processing_context_properties
from nvflare.private.stream_runner import ObjectStreamer
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


class ClientRunManager(ClientEngineExecutorSpec, StreamableEngine):
    """ClientRunManager provides the ClientEngine APIs implementation running in the child process (CJ)."""

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
        self.aux_runner = AuxRunner(self)
        self.object_streamer = ObjectStreamer(self.aux_runner)
        self.add_handler(self.aux_runner)
        self.add_handler(self.object_streamer)
        self.conf = conf
        self.cell = None

        self.all_clients = None
        self.name_to_clients = dict()  # client name => Client

        if not components:
            self.components = {}

        if not handlers:
            self.handlers = []

        # get job meta!
        job_ctx_props = self.create_job_processing_context_properties(workspace, job_id)
        job_ctx_props.update({FLContextKey.PROCESS_TYPE: ProcessType.CLIENT_JOB})

        client_config = client.client_args
        fqsn = client_config.get("fqsn", client.client_name)
        is_leaf = client_config.get("is_leaf", True)

        self.fl_ctx_mgr = FLContextManager(
            engine=self,
            identity_name=client_name,
            job_id=job_id,
            public_stickers={
                ReservedKey.FQSN: fqsn,
                ReservedKey.IS_LEAF: is_leaf,
            },
            private_stickers=job_ctx_props,
        )

        self.run_info = ClientRunInfo(job_id=job_id)

        self.widgets = {WidgetID.INFO_COLLECTOR: InfoCollector(), WidgetID.FED_EVENT_RUNNER: ClientFedEventRunner()}
        for _, widget in self.widgets.items():
            self.handlers.append(widget)

        self.logger = get_obj_logger(self)

    def get_task_assignment(self, fl_ctx: FLContext, timeout=None) -> TaskAssignment:
        pull_success, task_name, return_shareable = self.client.fetch_task(fl_ctx, timeout)
        task = None
        if pull_success:
            shareable = self.client.extract_shareable(return_shareable, fl_ctx)
            task_id = shareable.get_header(key=FLContextKey.TASK_ID)
            task = TaskAssignment(name=task_name, task_id=task_id, data=shareable)
        return task

    def new_context(self) -> FLContext:
        return self.fl_ctx_mgr.new_context()

    def send_task_result(self, result: Shareable, fl_ctx: FLContext, timeout=None) -> bool:
        push_result = self.client.push_results(result, fl_ctx, timeout)  # push task execution results
        if push_result == CellReturnCode.OK:
            return True
        else:
            self.logger.error(f"failed to send task result: {push_result}")
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

    def add_component(self, component_id: str, component):
        self.client.runner_config.add_component(component_id, component)

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
        return self.name_to_clients.get(client_name)

    def get_clients(self):
        return list(self.all_clients.values())

    def persist_components(self, fl_ctx: FLContext, completed: bool):
        self.logger.warning(f"will not persist components, not supported by {self.__class__.__name__}")

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
        secure=False,
    ) -> dict:
        msg_targets = self._to_aux_msg_targets(targets)
        if msg_targets:
            return self.aux_runner.send_aux_request(
                msg_targets,
                topic,
                request,
                timeout,
                fl_ctx,
                optional=optional,
                secure=secure,
            )
        else:
            return {}

    def _get_aux_msg_target(self, name: str):
        if name.lower() == SiteType.SERVER:
            return AuxMsgTarget.server_target()

        c = self.get_client_from_name(name)
        if c:
            return AuxMsgTarget.client_target(c)
        else:
            return None

    def _to_aux_msg_targets(self, target_names: List[str]):
        if not target_names:
            return [AuxMsgTarget.server_target()]

        if isinstance(target_names, str):
            if target_names == SiteType.ALL:
                msg_targets = [AuxMsgTarget.server_target()]
                for _, c in self.all_clients.items():
                    if c.name != self.client.client_name:
                        msg_targets.append(AuxMsgTarget.client_target(c))
                return msg_targets
            else:
                msg_target = self._get_aux_msg_target(target_names)
                if msg_target:
                    return [msg_target]
                else:
                    self.logger.error(f"invalid targe {target_names}")
                    return None

        elif not isinstance(target_names, list):
            raise TypeError(f"invalid target_names {type(target_names)}")

        # targets is a list: make sure every target is valid
        msg_targets = []
        for t in target_names:
            if not isinstance(t, str):
                raise TypeError(f"target name must be str but got {type(t)}")

            msg_target = self._get_aux_msg_target(t)
            if msg_target:
                msg_targets.append(msg_target)
            else:
                self.logger.error(f"invalid target {t}")
                return None
        return msg_targets

    def multicast_aux_requests(
        self,
        topic: str,
        target_requests: Dict[str, Shareable],
        timeout: float,
        fl_ctx: FLContext,
        optional: bool = False,
        secure: bool = False,
    ) -> dict:
        if not target_requests:
            return {}

        msg_targets = []
        for name, req in target_requests.items():
            msg_target = self._get_aux_msg_target(name)
            if not msg_target:
                self.logger.error(f"invalid target {name}")
                return {}
            msg_targets.append((msg_target, req))

        return self.aux_runner.multicast_aux_requests(
            topic, msg_targets, timeout, fl_ctx, optional=optional, secure=secure
        )

    def get_all_clients_from_server(self, fl_ctx, retry=0):
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        get_clients_message = new_cell_message({CellMessageHeaderKeys.JOB_ID: job_id}, {})
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
                data = return_data.payload
                self.all_clients = data.get(ServerCommandKey.CLIENTS)
                for _, c in self.all_clients.items():
                    self.name_to_clients[c.name] = c
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

    def fire_and_forget_aux_request(
        self, topic: str, request: Shareable, fl_ctx: FLContext, optional=False, secure=False
    ) -> dict:
        return self.send_aux_request(
            targets=None, topic=topic, request=request, timeout=0.0, fl_ctx=fl_ctx, optional=optional, secure=secure
        )

    def stream_objects(
        self,
        channel: str,
        topic: str,
        stream_ctx: StreamContext,
        targets: List[str],
        producer: ObjectProducer,
        fl_ctx: FLContext,
        optional=False,
        secure=False,
    ):
        if not self.object_streamer:
            raise RuntimeError("object streamer has not been created")

        return self.object_streamer.stream(
            channel=channel,
            topic=topic,
            stream_ctx=stream_ctx,
            targets=self._to_aux_msg_targets(targets),
            producer=producer,
            fl_ctx=fl_ctx,
            secure=secure,
            optional=optional,
        )

    def register_stream_processing(
        self,
        channel: str,
        topic: str,
        factory: ConsumerFactory,
        stream_done_cb=None,
        **cb_kwargs,
    ):
        if not self.object_streamer:
            raise RuntimeError("object streamer has not been created")

        self.object_streamer.register_stream_processing(channel, topic, factory, stream_done_cb, **cb_kwargs)

    def shutdown_streamer(self):
        if self.object_streamer:
            self.object_streamer.shutdown()

    def abort_app(self, job_id: str, fl_ctx: FLContext):
        runner = fl_ctx.get_prop(key=FLContextKey.RUNNER, default=None)
        if isinstance(runner, ClientRunner):
            runner.abort()

    def create_job_processing_context_properties(self, workspace, job_id):
        return create_job_processing_context_properties(workspace, job_id)
