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
from nvflare.apis.signal import Signal


class AuxRunner(FLComponent):

    DATA_KEY_BULK = "bulk_data"
    TOPIC_BULK = "__runner.bulk__"

    def __init__(self):
        """To init the AuxRunner."""
        FLComponent.__init__(self)
        self.job_id = None
        self.topic_table = {
            self.TOPIC_BULK: self._process_bulk_requests,
        }  # topic => handler
        self.reg_lock = Lock()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.job_id = fl_ctx.get_job_id()

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
            raise TypeError("topic must be str, but got {}".format(type(topic)))

        if topic == self.TOPIC_BULK:
            raise ValueError('topic value "{}" is reserved'.format(topic))

        if len(topic) <= 0:
            raise ValueError("topic must not be empty")

        if message_handle_func is None:
            raise ValueError("message handler function is not specified")

        if not callable(message_handle_func):
            raise TypeError("specified message_handle_func {} is not callable".format(message_handle_func))

        with self.reg_lock:
            if topic in self.topic_table:
                raise ValueError("handler already registered for topic {}".format(topic))

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
            self.log_error(fl_ctx, "received invalid aux request: expects a Shareable but got {}".format(type(request)))
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        peer_props = request.get_peer_props()
        if not peer_props:
            self.log_error(fl_ctx, "missing peer_ctx from client")
            return make_reply(ReturnCode.MISSING_PEER_CONTEXT)

        if not isinstance(peer_props, dict):
            self.log_error(
                fl_ctx,
                "bad peer_props from client: expects dict but got {}".format(type(peer_props)),
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

    def _process_bulk_requests(self, topic: str, request: Shareable, fl_ctx: FLContext):
        reqs = request.get(self.DATA_KEY_BULK, None)
        if not isinstance(reqs, list):
            self.log_error(fl_ctx, "invalid bulk request - missing list of requests, got {} instead".format(type(reqs)))
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        abort_signal = fl_ctx.get_run_abort_signal()
        for req in reqs:
            if isinstance(abort_signal, Signal) and abort_signal.triggered:
                break

            if not isinstance(req, Shareable):
                self.log_error(fl_ctx, "invalid request in bulk: expect Shareable but got {}".format(type(req)))
                continue
            req_topic = req.get_header(ReservedHeaderKey.TOPIC, "")
            if not req_topic:
                self.log_error(fl_ctx, "invalid request in bulk: no topic in header")
                continue

            self._process_request(req_topic, req, fl_ctx)
        return make_reply(ReturnCode.OK)
