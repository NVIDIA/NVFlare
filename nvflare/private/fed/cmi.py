# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.fuel.f3.cellnet.cell import FQCN, Cell, Message, MessageHeaderKey
from nvflare.fuel.f3.cellnet.cell import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.cell import new_message
from nvflare.private.defs import CellMessageHeaderKeys


class CellMessageInterface(FLComponent, ABC):

    HEADER_KEY_PEER_PROPS = "cmi.peer_props"
    HEADER_JOB_ID = "cmi.job_id"
    HEADER_PROJECT_NAME = "cmi.project"
    HEADER_SSID = "cmi.ssid"
    HEADER_CLIENT_TOKEN = "cmi.client_token"
    HEADER_CLIENT_NAME = "cmi.client_name"

    PROP_KEY_CLIENT = "cmi.client"
    PROP_KEY_FL_CTX = "cmi.fl_ctx"
    PROP_KEY_PEER_CTX = "cmi.peer_ctx"

    RC_TABLE = {
        CellReturnCode.TIMEOUT: ReturnCode.COMMUNICATION_ERROR,
        CellReturnCode.COMM_ERROR: ReturnCode.COMMUNICATION_ERROR,
        CellReturnCode.PROCESS_EXCEPTION: ReturnCode.EXECUTION_EXCEPTION,
        CellReturnCode.ABORT_RUN: CellReturnCode.ABORT_RUN,
        CellReturnCode.INVALID_REQUEST: CellReturnCode.INVALID_REQUEST,
        CellReturnCode.INVALID_SESSION: CellReturnCode.INVALID_SESSION,
        CellReturnCode.AUTHENTICATION_ERROR: CellReturnCode.UNAUTHENTICATED,
        CellReturnCode.SERVICE_UNAVAILABLE: CellReturnCode.SERVICE_UNAVAILABLE,
    }

    def __init__(
        self,
        engine,
    ):
        FLComponent.__init__(self)
        self.engine = engine
        self.cell = engine.get_cell()
        self.ready = False

        self.cell.add_incoming_request_filter(channel="*", topic="*", cb=self._filter_incoming_request)

        self.cell.add_outgoing_reply_filter(channel="*", topic="*", cb=self._filter_outgoing_message)

        self.cell.add_outgoing_request_filter(channel="*", topic="*", cb=self._filter_outgoing_message)

        self.cell.add_incoming_reply_filter(channel="*", topic="*", cb=self._filter_incoming_message)

    def new_cmi_message(self, fl_ctx: FLContext, headers=None, payload=None):
        msg = new_message(headers, payload)
        msg.set_prop(self.PROP_KEY_FL_CTX, fl_ctx)
        return msg

    def _filter_incoming_message(self, message: Message):
        public_props = message.get_header(self.HEADER_KEY_PEER_PROPS)
        if public_props:
            peer_ctx = self._make_peer_ctx(public_props)
            message.set_prop(self.PROP_KEY_PEER_CTX, peer_ctx)
        shareable = message.payload
        if isinstance(shareable, Shareable):
            if public_props:
                shareable.set_peer_props(public_props)

    def _filter_incoming_request(self, message: Message):
        self._filter_incoming_message(message)
        fl_ctx = self.engine.new_context()
        peer_ctx = message.get_prop(self.PROP_KEY_PEER_CTX)
        assert isinstance(fl_ctx, FLContext)
        if peer_ctx:
            fl_ctx.set_peer_context(peer_ctx)
        message.set_prop(self.PROP_KEY_FL_CTX, fl_ctx)

    def _filter_outgoing_message(self, message: Message):
        fl_ctx = message.get_prop(self.PROP_KEY_FL_CTX)
        if fl_ctx:
            assert isinstance(fl_ctx, FLContext)
            public_props = fl_ctx.get_all_public_props()
            message.set_header(self.HEADER_KEY_PEER_PROPS, public_props)
            ssid = fl_ctx.get_prop(CellMessageHeaderKeys.SSID)
            if ssid:
                message.set_header(self.HEADER_SSID, ssid)
            project_name = fl_ctx.get_prop(CellMessageHeaderKeys.PROJECT_NAME)
            if project_name:
                message.set_header(self.HEADER_PROJECT_NAME, project_name)
            client_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
            if client_name:
                message.set_header(self.HEADER_CLIENT_NAME, client_name)
            client_token = fl_ctx.get_prop(CellMessageHeaderKeys.TOKEN)
            if client_token:
                message.set_header(self.HEADER_CLIENT_TOKEN, client_token)

    @staticmethod
    def _make_peer_ctx(props: dict) -> FLContext:
        ctx = FLContext()
        ctx.set_public_props(props)
        return ctx

    @staticmethod
    def _convert_return_code(rc: CellReturnCode):
        return CellMessageInterface.RC_TABLE.get(rc, ReturnCode.ERROR)

    @abstractmethod
    def send_to_cell(
        self,
        targets: [],
        channel: str,
        topic: str,
        request: Shareable,
        timeout: float,
        fl_ctx: FLContext,
        bulk_send=False,
    ) -> dict:
        pass


class JobCellMessenger(CellMessageInterface):
    def __init__(self, engine, job_id: str):
        super().__init__(engine)

        self.job_id = job_id

        self.cell.add_incoming_request_filter(channel="*", topic="*", cb=self._filter_incoming)
        self.cell.add_incoming_reply_filter(channel="*", topic="*", cb=self._filter_incoming)
        self.cell.add_outgoing_request_filter(channel="*", topic="*", cb=self._filter_outgoing)
        self.cell.add_outgoing_reply_filter(channel="*", topic="*", cb=self._filter_outgoing)

    def _filter_incoming(self, message: Message):
        job_id = message.get_header(self.HEADER_JOB_ID)
        if job_id and job_id != self.job_id:
            self.logger.error(f"received job id {job_id} != my job id {self.job_id}")

    def _filter_outgoing(self, message: Message):
        message.set_header(self.HEADER_JOB_ID, self.job_id)

    def send_to_cell(
        self,
        targets: [],
        channel: str,
        topic: str,
        request: Shareable,
        timeout: float,
        fl_ctx: FLContext,
        bulk_send=False,
        optional=False,
    ) -> dict:
        """Send request to the job cells of other target sites.
        Args:
            targets (list): list of client names that the request will be sent to
            channel (str): channel of the request
            topic (str): topic of the request
            request (Shareable): request
            timeout (float): how long to wait for result. 0 means fire-and-forget
            fl_ctx (FLContext): the FL context
            bulk_send: whether to bulk send this request (only applies in the fire-and-forget situation)
            optional: whether the request is optional
        Returns:
            A dict of Shareables
        """
        if not isinstance(request, Shareable):
            raise ValueError(f"invalid request type: expect Shareable but got {type(request)}")

        if not targets:
            raise ValueError("targets must be specified")

        if targets is not None and not isinstance(targets, list):
            raise TypeError(f"targets must be a list of str, but got {type(targets)}")

        if not isinstance(topic, str):
            raise TypeError(f"invalid topic '{topic}': expects str but got {type(topic)}")

        if not topic:
            raise ValueError("invalid topic: must not be empty")

        if not isinstance(timeout, float):
            raise TypeError(f"invalid timeout: expects float but got {type(timeout)}")

        if timeout < 0:
            raise ValueError(f"invalid timeout value {timeout}: must >= 0.0")

        if not isinstance(fl_ctx, FLContext):
            raise TypeError(f"invalid fl_ctx: expects FLContext but got {type(fl_ctx)}")

        request.set_header(ReservedHeaderKey.TOPIC, topic)
        job_id = fl_ctx.get_job_id()
        cell = self.engine.get_cell()
        assert isinstance(cell, Cell)

        target_names = []
        for t in targets:
            if not isinstance(t, str):
                raise ValueError(f"invalid target name {t}: expect str but got {type(t)}")
            if t not in target_names:
                target_names.append(t)

        target_fqcns = []
        for name in target_names:
            target_fqcns.append(FQCN.join([name, job_id]))

        cell_msg = self.new_cmi_message(fl_ctx, payload=request)
        if timeout > 0:
            cell_replies = cell.broadcast_request(
                channel=channel, topic=topic, request=cell_msg, targets=target_fqcns, timeout=timeout, optional=optional
            )

            replies = {}
            if cell_replies:
                for k, v in cell_replies.items():
                    assert isinstance(v, Message)
                    rc = v.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
                    client_name = FQCN.get_root(k)
                    if rc == CellReturnCode.OK:
                        result = v.payload
                        if not isinstance(result, Shareable):
                            self.logger.error(f"reply of {channel}:{topic} must be dict but got {type(result)}")
                            result = make_reply(ReturnCode.ERROR)
                        replies[client_name] = result
                    else:
                        src = self._convert_return_code(rc)
                        replies[client_name] = make_reply(src)
            return replies
        else:
            if bulk_send:
                cell.queue_message(channel=channel, topic=topic, message=cell_msg, targets=target_fqcns)
            else:
                cell.fire_and_forget(
                    channel=channel, topic=topic, message=cell_msg, targets=target_fqcns, optional=optional
                )
            return {}
