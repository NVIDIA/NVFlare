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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.fuel.f3.cellnet import Cell, Message as CellMessage, FQCN
from nvflare.private.defs import CellChannel, new_cell_message


class AuxRunner(FLComponent):

    def __init__(self):
        """To init the AuxRunner."""
        FLComponent.__init__(self)
        self.job_id = None
        self.topic_table = {}  # topic => handler
        self.reg_lock = Lock()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.job_id = fl_ctx.get_job_id()
            engine = fl_ctx.get_engine()
            cell = engine.get_cell
            assert isinstance(cell, Cell)
            cell.register_request_cb(
                channel=CellChannel.AUX,
                topic="*",
                cb=self._process_cell_request,
                engine=engine
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

        peer_job_id = peer_props.get(ReservedKey.RUN_NUM)
        if peer_job_id != self.job_id:
            self.log_error(fl_ctx, "invalid aux msg: not for the same job_id")
            return make_reply(ReturnCode.RUN_MISMATCH)

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

        valid_reply.set_peer_props(fl_ctx.get_all_public_props())
        return valid_reply

    def _process_cell_request(
            self,
            cell: Cell,
            channel: str,
            topic: str,
            request: CellMessage,
            engine
    ) -> CellMessage:
        aux_msg = request.payload
        assert isinstance(aux_msg, Shareable)
        with engine.new_context() as fl_ctx:
            aux_reply = self.dispatch(
                topic=topic,
                request=aux_msg,
                fl_ctx=fl_ctx
            )
            return new_cell_message(payload=aux_reply)

    def send_to_job_cell(
            self,
            targets: [],
            topic: str,
            request: Shareable,
            timeout: float,
            fl_ctx: FLContext,
            bulk_send=False
    ) -> dict:
        """Send request through auxiliary channel.

        Args:
            targets (list): list of client names that the request will be sent to
            topic (str): topic of the request
            request (Shareable): request
            timeout (float): how long to wait for result. 0 means fire-and-forget
            fl_ctx (FLContext): the FL context
            bulk_send: whether to bulk send this request

        Returns:
            A dict of results
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

        request.set_peer_props(fl_ctx.get_all_public_props())
        request.set_header(ReservedHeaderKey.TOPIC, topic)

        engine = fl_ctx.get_engine()
        job_id = fl_ctx.get_job_id()
        cell = engine.get_cell()
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

        cell_msg = new_cell_message(payload=request)
        if timeout > 0:
            cell_replies = cell.broadcast_request(
                channel=CellChannel.AUX,
                topic=topic,
                request=cell_msg,
                targets=target_fqcns,
                timeout=timeout
            )

            replies = {}
            if cell_replies:
                for k, v in cell_replies.items():
                    client_name = FQCN.get_root(k)
                    replies[client_name] = v.payload
            return replies
        else:
            if bulk_send:
                cell.queue_message(
                    channel=CellChannel.AUX,
                    topic=topic,
                    message=cell_msg,
                    targets=target_fqcns
                )
            else:
                cell.fire_and_forget(
                    channel=CellChannel.AUX,
                    topic=topic,
                    message=cell_msg,
                    targets=target_fqcns,
                )
            return {}

    def send_aux_request(
            self,
            targets: list,
            topic: str,
            request: Shareable,
            timeout: float,
            fl_ctx: FLContext,
            bulk_send: bool
    ) -> dict:
        pass
