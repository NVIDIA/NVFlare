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

import os
import re
import shutil
import sys
import threading
from typing import List

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, MachineStatus, ProcessType, ReservedKey, SystemComponents
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.shareable import Shareable
from nvflare.apis.streaming import ConsumerFactory, ObjectProducer, StreamableEngine, StreamContext
from nvflare.apis.utils.fl_context_utils import gen_new_peer_ctx
from nvflare.apis.workspace import Workspace
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.aux_runner import AuxMsgTarget, AuxRunner
from nvflare.private.defs import ERROR_MSG_PREFIX, ClientStatusKey, new_cell_message
from nvflare.private.event import fire_event
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator
from nvflare.private.fed.utils.app_deployer import AppDeployer
from nvflare.private.fed.utils.fed_utils import security_close
from nvflare.private.stream_runner import ObjectStreamer
from nvflare.security.logging import secure_format_exception, secure_log_traceback
from nvflare.widgets.fed_event import ClientFedEventRunner

from .client_engine_internal_spec import ClientEngineInternalSpec
from .client_executor import JobExecutor
from .client_run_manager import ClientRunInfo
from .client_status import ClientStatus
from .fed_client import FederatedClient


def _remove_custom_path():
    regex = re.compile(".*/run_.*/custom")
    custom_paths = list(filter(regex.search, sys.path))
    for path in custom_paths:
        sys.path.remove(path)


class ClientEngine(ClientEngineInternalSpec, StreamableEngine):
    """ClientEngine runs in the client parent process (CP)."""

    def __init__(self, client: FederatedClient, args, rank, workers=5):
        """To init the ClientEngine.

        Args:
            client: FL client object
            args: command args
            rank: local process rank
            workers: number of workers
        """
        super().__init__()
        self.client = client
        self.client_name = client.client_name
        self.args = args
        self.rank = rank
        self.client_executor = JobExecutor(client, os.path.join(args.workspace, "startup"))
        self.admin_agent = None
        self.aux_runner = AuxRunner(self)
        self.object_streamer = ObjectStreamer(self.aux_runner)
        self.cell = None

        client_config = client.client_args
        fqsn = client_config.get("fqsn", client.client_name)
        is_leaf = client_config.get("is_leaf", True)

        self.fl_ctx_mgr = FLContextManager(
            engine=self,
            identity_name=self.client_name,
            job_id="",
            public_stickers={
                ReservedKey.FQSN: fqsn,
                ReservedKey.IS_LEAF: is_leaf,
            },
            private_stickers={
                SystemComponents.DEFAULT_APP_DEPLOYER: AppDeployer(),
                SystemComponents.JOB_META_VALIDATOR: JobMetaValidator(),
                SystemComponents.FED_CLIENT: client,
                FLContextKey.SECURE_MODE: self.client.secure_train,
                FLContextKey.WORKSPACE_ROOT: args.workspace,
                FLContextKey.PROCESS_TYPE: ProcessType.CLIENT_PARENT,
            },
        )

        self.status = MachineStatus.STOPPED

        if workers < 1:
            raise ValueError("workers must >= 1")
        self.logger = get_obj_logger(self)
        self.fl_components = [x for x in self.client.components.values() if isinstance(x, FLComponent)]

        self.fl_components.append(ClientFedEventRunner())

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        fire_event(event=event_type, handlers=self.fl_components, ctx=fl_ctx)

    def get_cell(self):
        """Get the communication cell.
        This method must be implemented since AuxRunner calls to get cell.

        Returns:

        """
        return self.cell

    def initialize_comm(self, cell: Cell):
        """This is called when communication cell has been created.
        We will set up aux message handler here.

        Args:
            cell:

        Returns:

        """
        cell.register_request_cb(
            channel=CellChannel.AUX_COMMUNICATION,
            topic="*",
            cb=self._handle_aux_message,
        )
        self.cell = cell

    def _handle_aux_message(self, request: CellMessage) -> CellMessage:
        assert isinstance(request, CellMessage), "request must be CellMessage but got {}".format(type(request))
        data = request.payload

        topic = request.get_header(MessageHeaderKey.TOPIC)
        with self.new_context() as fl_ctx:
            reply = self.aux_runner.dispatch(topic=topic, request=data, fl_ctx=fl_ctx)
            assert isinstance(reply, Shareable)
            shared_fl_ctx = gen_new_peer_ctx(fl_ctx)
            reply.set_peer_context(shared_fl_ctx)

            if reply is not None:
                return_message = new_cell_message({}, reply)
                return_message.set_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            else:
                return_message = new_cell_message({}, None)
            return return_message

    def register_aux_message_handler(self, topic: str, message_handle_func):
        """Register aux message handling function with specified topics.

        Exception is raised when:
            a handler is already registered for the topic;
            bad topic - must be a non-empty string
            bad message_handle_func - must be callable

        Implementation Note:
            This method should simply call the ServerAuxRunner's register_aux_message_handler method.

        Args:
            topic: the topic to be handled by the func
            message_handle_func: the func to handle the message. Must follow aux_message_handle_func_signature.

        """
        self.aux_runner.register_aux_message_handler(topic, message_handle_func)

    def send_aux_request(
        self,
        topic: str,
        request: Shareable,
        timeout: float,
        fl_ctx: FLContext,
        optional=False,
        secure=False,
    ) -> Shareable:
        """Send a request to the Server via the aux channel.

        Implementation: simply calls the AuxRunner's send_aux_request method.

        Args:
            topic: topic of the request.
            request: request to be sent
            timeout: number of secs to wait for replies. 0 means fire-and-forget.
            fl_ctx: FL context
            optional: whether this message is optional
            secure: send the aux request in a secure way

        Returns: a dict of replies (client name => reply Shareable)

        """
        reply = self.aux_runner.send_aux_request(
            targets=[AuxMsgTarget.server_target()],
            topic=topic,
            request=request,
            timeout=timeout,
            fl_ctx=fl_ctx,
            optional=optional,
            secure=secure,
        )

        if len(reply) > 0:
            self.logger.info(f"got aux reply: {reply}")
            return next(iter(reply.values()))
        else:
            self.logger.debug("no reply from the server aux message response.")
            return Shareable()

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
        """Send a stream of Shareable objects to receivers.

        Args:
            channel: the channel for this stream
            topic: topic of the stream
            stream_ctx: context of the stream
            targets: receiving sites
            producer: the ObjectProducer that can produces the stream of Shareable objects
            fl_ctx: the FLContext object
            optional: whether the stream is optional
            secure: whether to use P2P security

        Returns: result from the generator's reply processing

        """
        if not self.object_streamer:
            raise RuntimeError("object streamer has not been created")

        # We are CP: can only stream to SP
        if targets:
            for t in targets:
                self.logger.debug(f"ignored target: {t}")

        return self.object_streamer.stream(
            channel=channel,
            topic=topic,
            stream_ctx=stream_ctx,
            targets=[AuxMsgTarget.server_target()],
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
        consumed_cb=None,
        **cb_kwargs,
    ):
        """Register a ConsumerFactory for specified app channel and topic.
        Once a new streaming request is received for the channel/topic, the registered factory will be used
        to create an ObjectConsumer object to handle the new stream.

        Note: the factory should generate a new ObjectConsumer every time get_consumer() is called. This is because
        multiple streaming sessions could be going on at the same time. Each streaming session should have its
        own ObjectConsumer.

        Args:
            channel: app channel
            topic: app topic
            factory: the factory to be registered
            stream_done_cb: the callback to be called when streaming is done on receiving side
            consumed_cb: the CB to be called after a chunk is consumed

        Returns: None

        """
        if not self.object_streamer:
            raise RuntimeError("object streamer has not been created")

        self.object_streamer.register_stream_processing(
            topic=topic,
            channel=channel,
            factory=factory,
            stream_done_cb=stream_done_cb,
            consumed_cb=consumed_cb,
            **cb_kwargs,
        )

    def shutdown_streamer(self):
        if self.object_streamer:
            self.object_streamer.shutdown()

    def set_agent(self, admin_agent):
        self.admin_agent = admin_agent

    def new_context(self) -> FLContext:
        return self.fl_ctx_mgr.new_context()

    def add_component(self, component_id: str, component):
        if not isinstance(component_id, str):
            raise TypeError(f"component id must be str but got {type(component_id)}")

        if component_id in self.client.components:
            raise ValueError(f"duplicate component id {component_id}")

        self.client.components[component_id] = component
        if isinstance(component, FLComponent):
            self.fl_components.append(component)

    def get_component(self, component_id: str) -> object:
        return self.client.components.get(component_id)

    def get_engine_status(self):
        running_jobs = []
        for job_id in self.get_all_job_ids():
            ws = Workspace(self.args.workspace)
            run_folder = ws.get_run_dir(job_id)
            app_name = ""
            app_file = os.path.join(run_folder, "fl_app.txt")
            if os.path.exists(app_file):
                with open(app_file, "r") as f:
                    app_name = f.readline().strip()
            job = {
                ClientStatusKey.APP_NAME: app_name,
                ClientStatusKey.JOB_ID: job_id,
                ClientStatusKey.STATUS: self.client_executor.check_status(job_id),
            }
            running_jobs.append(job)

        result = {
            ClientStatusKey.CLIENT_NAME: self.client.client_name,
            ClientStatusKey.RUNNING_JOBS: running_jobs,
        }
        return result

    def start_app(
        self,
        job_id: str,
        job_meta: dict,
        allocated_resource: dict = None,
        token: str = None,
        resource_manager=None,
    ) -> str:
        status = self.client_executor.get_status(job_id)
        if status == ClientStatus.STARTED:
            return "Client app already started."

        workspace = Workspace(
            self.args.workspace,
            site_name=self.client.client_name,
        )
        app_root = workspace.get_app_dir(job_id)
        if not os.path.exists(app_root):
            return f"{ERROR_MSG_PREFIX}: Client app does not exist. Please deploy it before starting client."

        self.logger.info("Starting client app. rank: {}".format(self.rank))

        self.client_executor.start_app(
            self.client,
            job_id,
            job_meta,
            self.args,
            allocated_resource,
            token,
            resource_manager,
            fl_ctx=self.new_context(),
        )

        return "Start the client app..."

    def notify_job_status(self, job_id: str, job_status):
        self.client_executor.notify_job_status(job_id, job_status)

    def get_client_name(self):
        return self.client.client_name

    def abort_app(self, job_id: str) -> str:
        status = self.client_executor.get_status(job_id)
        if status == ClientStatus.STOPPED:
            return "Client app already stopped."

        if status == ClientStatus.NOT_STARTED:
            return "Client app has not started."

        if status == ClientStatus.STARTING:
            return "Client app is starting, please wait for client to have started before abort."

        self.client_executor.abort_app(job_id)

        return "Abort signal has been sent to the client App."

    def send_to_job(
        self,
        job_id,
        channel: str,
        topic: str,
        msg: CellMessage,
        timeout: float,
        optional=False,
    ) -> CellMessage:
        """Send a message to CJ

        Args:
            job_id: id of the job
            channel: message channel
            topic: message topic
            msg: the message to be sent
            timeout: how long to wait for reply
            optional: whether the message is optional

        Returns: reply from CJ

        """
        return self.client_executor.send_to_job(job_id, channel, topic, msg, timeout, optional=optional)

    def abort_task(self, job_id: str) -> str:
        status = self.client_executor.get_status(job_id)
        if status == ClientStatus.NOT_STARTED:
            return "Client app has not started."

        if status == ClientStatus.STARTING:
            return "Client app is starting, please wait for started before abort_task."

        self.client_executor.abort_task(job_id)

        return "Abort signal has been sent to the current task."

    def shutdown(self) -> str:
        self.logger.info("Client shutdown...")
        touch_file = os.path.join(self.args.workspace, "shutdown.fl")
        self.fire_event(EventType.SYSTEM_END, self.new_context())

        thread = threading.Thread(target=shutdown_client, args=(self.client, touch_file))
        thread.start()

        self.shutdown_streamer()
        return "Shutdown the client..."

    def restart(self) -> str:
        self.logger.info("Client shutdown...")
        touch_file = os.path.join(self.args.workspace, "restart.fl")
        self.fire_event(EventType.SYSTEM_END, self.new_context())
        thread = threading.Thread(target=shutdown_client, args=(self.client, touch_file))
        thread.start()

        return "Restart the client..."

    def deploy_app(self, app_name: str, job_id: str, job_meta: dict, client_name: str, app_data) -> str:
        workspace = Workspace(root_dir=self.args.workspace, site_name=client_name)
        app_deployer = self.get_component(SystemComponents.APP_DEPLOYER)
        if not app_deployer:
            # use default deployer
            app_deployer = AppDeployer()

        err = app_deployer.deploy(
            workspace=workspace,
            job_id=job_id,
            job_meta=job_meta,
            app_name=app_name,
            app_data=app_data,
            fl_ctx=self.new_context(),
        )
        if err:
            return f"{ERROR_MSG_PREFIX}: {err}"

        return ""

    def delete_run(self, job_id: str) -> str:
        ws = Workspace(self.args.workspace)
        job_id_folder = ws.get_run_dir(job_id)
        if os.path.exists(job_id_folder):
            shutil.rmtree(job_id_folder)
        return f"Delete run folder: {job_id_folder}."

    def get_current_run_info(self, job_id) -> ClientRunInfo:
        return self.client_executor.get_run_info(job_id)

    def get_errors(self, job_id):
        return self.client_executor.get_errors(job_id)

    def configure_job_log(self, job_id, config):
        return self.client_executor.configure_job_log(job_id, config)

    def reset_errors(self, job_id):
        self.client_executor.reset_errors(job_id)

    def get_all_job_ids(self):
        return self.client_executor.get_run_processes_keys()

    def fire_and_forget_aux_request(
        self, topic: str, request: Shareable, fl_ctx: FLContext, optional=False, secure=False
    ) -> dict:
        return self.send_aux_request(
            topic=topic, request=request, timeout=0.0, fl_ctx=fl_ctx, optional=optional, secure=secure
        )


def shutdown_client(federated_client, touch_file):
    with open(touch_file, "a"):
        os.utime(touch_file, None)

    try:
        print("About to shutdown the client...")
        federated_client.communicator.heartbeat_done = True
        # time.sleep(3)
        federated_client.close()

        federated_client.status = ClientStatus.STOPPED
        # federated_client.cell.stop()
        security_close()
    except Exception as e:
        secure_log_traceback()
        print(f"Failed to shutdown client: {secure_format_exception(e)}")
