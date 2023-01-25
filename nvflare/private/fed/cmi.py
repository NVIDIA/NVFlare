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

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_constant import ReturnCode, FLContextKey
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.cell import new_message, Message, ReturnCode as CellReturnCode


class CellMessageInterface(FLComponent):

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
        CellReturnCode.ABORT_RUN: ReturnCode.ABORT_RUN,
        CellReturnCode.INVALID_REQUEST: ReturnCode.INVALID_REQUEST,
        CellReturnCode.INVALID_SESSION: ReturnCode.INVALID_SESSION,
        CellReturnCode.AUTHENTICATION_ERROR: ReturnCode.NOT_AUTHENTICATED,
        CellReturnCode.SERVICE_UNAVAILABLE: ReturnCode.SERVICE_UNAVAILABLE
    }


    def __init__(
            self,
            engine,
    ):
        FLComponent.__init__(self)
        self.engine = engine
        self.cell = engine.get_cell()
        self.ready = False

        self.cell.add_incoming_request_filter(
            channel="*",
            topic="*",
            cb=self._filter_incoming_request
        )

        self.cell.add_outgoing_reply_filter(
            channel="*",
            topic="*",
            cb=self._filter_outgoing_message
        )

        self.cell.add_outgoing_request_filter(
            channel="*",
            topic="*",
            cb=self._filter_outgoing_message
        )

        self.cell.add_incoming_reply_filter(
            channel="*",
            topic="*",
            cb=self._filter_incoming_message
        )

    def new_cmi_message(self, fl_ctx: FLContext, headers=None, payload=None):
        msg = new_message(headers, payload)
        msg.set_prop(self.PROP_KEY_FL_CTX, fl_ctx)
        return msg

    def _filter_incoming_message(
            self,
            message: Message
    ):
        public_props = message.get_header(self.HEADER_KEY_PEER_PROPS)
        if public_props:
            peer_ctx = self._make_peer_ctx(public_props)
            message.set_prop(self.PROP_KEY_PEER_CTX, peer_ctx)
        shareable = message.payload
        if isinstance(shareable, Shareable):
            if public_props:
                shareable.set_peer_props(public_props)

    def _filter_incoming_request(
            self,
            message: Message
    ):
        self._filter_incoming_message(message)
        fl_ctx = self.engine.new_context()
        peer_ctx = message.get_prop(self.PROP_KEY_PEER_CTX)
        assert isinstance(fl_ctx, FLContext)
        if peer_ctx:
            fl_ctx.set_peer_context(peer_ctx)
        message.set_prop(self.PROP_KEY_FL_CTX, fl_ctx)

    def _filter_outgoing_message(
            self,
            message: Message
    ):
        fl_ctx = message.get_prop(self.PROP_KEY_FL_CTX)
        if fl_ctx:
            assert isinstance(fl_ctx, FLContext)
            public_props = fl_ctx.get_all_public_props()
            message.set_header(self.HEADER_KEY_PEER_PROPS, public_props)
            ssid = fl_ctx.get_prop(FLContextKey.SSID)
            if ssid:
                message.set_header(self.HEADER_SSID, ssid)
            project_name = fl_ctx.get_prop(FLContextKey.PROJECT_NAME)
            if project_name:
                message.set_header(self.HEADER_PROJECT_NAME, project_name)
            client_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
            if client_name:
                message.set_header(self.HEADER_CLIENT_NAME, client_name)
            client_token = fl_ctx.get_prop(FLContextKey.CLIENT_TOKEN)
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
