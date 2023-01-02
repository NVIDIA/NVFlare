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

from multiprocessing import Lock

from nvflare.apis.client import Client
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.fuel.f3.cellnet.cell import Cell, Message as CellMessage
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.private.defs import CellChannel
from nvflare.private.fed.jcmi import JobCellMessageInterface


class AuxRunner(FLComponent):

    def __init__(self, engine):
        """To init the AuxRunner."""
        FLComponent.__init__(self)
        self.engine = engine
        self.topic_table = {}  # topic => handler
        self.reg_lock = Lock()

        cell = engine.get_cell()
        assert isinstance(cell, Cell)
        cell.register_request_cb(
            channel=CellChannel.AUX,
            topic="*",
            cb=self._process_cell_request,
        )

    def register_aux_message_handler(self, topic: str, message_handle_func):
        """Register aux message handling function with specified topics.

        This method should be called by ServerEngine's register_aux_message_handler method.

        Args:
            topic: the topic to be handled by the func
            message_handle_func: the func to handle the message. Must follow aux_message_handle_func_signature.

        Returns: N/A

        Exception is raised when:
            a handler is already registered for the topic;
            bad topic - must be a non-empty string
            bad message_handle_func - must be callable
        """
        if not isinstance(topic, str):
            raise TypeError(f"topic must be str, but got {type(topic)}")

        if len(topic) <= 0:
            raise ValueError("topic must not be empty")

        if message_handle_func is None:
            raise ValueError("message handler function is not specified")

        if not callable(message_handle_func):
            raise TypeError("specified message_handle_func {} is not callable".format(message_handle_func))

        with self.reg_lock:
            if topic in self.topic_table:
                raise ValueError(f"handler already registered for topic {topic}")
            self.topic_table[topic] = message_handle_func

    def _process_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """Call to process the request.

        NOTE: peer_ctx props must have been set into the PEER_PROPS header of the request by Engine.

        Args:
            topic: topic of the message
            request: message to be handled
            fl_ctx: fl context

        Returns: reply message

        """
        handler_f = self.topic_table.get(topic, None)
        if handler_f is None:
            self.log_error(fl_ctx, "received unknown aux message topic {}".format(topic))
            return make_reply(ReturnCode.TOPIC_UNKNOWN)

        if not isinstance(request, Shareable):
            self.log_error(fl_ctx,
                           f"received invalid aux request: expects a Shareable but got {type(request)}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        peer_props = request.get_peer_props()
        if not peer_props:
            self.log_error(fl_ctx, "missing peer_ctx from client")
            return make_reply(ReturnCode.MISSING_PEER_CONTEXT)

        if not isinstance(peer_props, dict):
            self.log_error(
                fl_ctx,
                f"bad peer_props from client: expects dict but got {type(peer_props)}",
            )
            return make_reply(ReturnCode.BAD_PEER_CONTEXT)
        try:
            reply = handler_f(topic=topic, request=request, fl_ctx=fl_ctx)
        except BaseException:
            self.log_exception(fl_ctx, "processing error in message handling")
            return make_reply(ReturnCode.HANDLER_EXCEPTION)

        return reply

    def dispatch(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """This method is to be called by the Engine when an aux message is received from peer.

        NOTE: peer_ctx props must have been set into the PEER_PROPS header of the request by Engine.

        Args:
            topic: message topic
            request: request message
            fl_ctx: FLContext

        Returns: reply message

        """
        valid_reply = self._process_request(topic, request, fl_ctx)
        if isinstance(request, Shareable):
            cookie_jar = request.get_cookie_jar()
            if cookie_jar:
                valid_reply.set_cookie_jar(cookie_jar)
        return valid_reply

    def _process_cell_request(
            self,
            request: CellMessage,
    ) -> CellMessage:
        aux_msg = request.payload
        assert isinstance(aux_msg, Shareable)
        topic = request.get_header(MessageHeaderKey.TOPIC)
        fl_ctx = request.get_prop(JobCellMessageInterface.PROP_KEY_FL_CTX)
        assert isinstance(fl_ctx, FLContext)
        cmi = self.engine.get_cmi()
        assert isinstance(cmi, JobCellMessageInterface)
        peer_ctx = request.get_prop(cmi.PROP_KEY_PEER_CTX)
        if peer_ctx:
            fl_ctx.set_peer_context(peer_ctx)
        aux_reply = self.dispatch(
            topic=topic,
            request=aux_msg,
            fl_ctx=fl_ctx
        )
        return cmi.new_cmi_message(fl_ctx, payload=aux_reply)

    def send_aux_request(
            self,
            targets: list,
            topic: str,
            request: Shareable,
            timeout: float,
            fl_ctx: FLContext,
            bulk_send: bool = False
    ) -> dict:
        cmi = self.engine.get_cmi()
        assert isinstance(cmi, JobCellMessageInterface)
        if not targets:
            if cmi.cell.my_info.is_on_server:
                targets = cmi.get_client_names()
            else:
                targets = [FQCN.ROOT_SERVER]

        # validate targets
        target_names = []
        for t in targets:
            if isinstance(t, str):
                name = t
            elif isinstance(t, Client):
                name = t.name
            else:
                raise ValueError(f"invalid target in list: got {type(t)}")

            if not name:
                # ignore empty name
                continue

            if FQCN.is_ancestor(name, cmi.cell.get_fqcn()):
                raise ValueError(f"invalid target '{name}': cannot send to myself")

            if name not in target_names:
                target_names.append(t)

        if not target_names:
            return {}

        clients, invalid_names = cmi.validate_clients(target_names)
        if invalid_names:
            raise ValueError(f"invalid target(s): {invalid_names}")

        return cmi.send_to_job_cell(
            targets=targets,
            channel=CellChannel.AUX,
            topic=topic,
            request=request,
            timeout=timeout,
            fl_ctx=fl_ctx,
            bulk_send=bulk_send
        )
