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
from threading import Lock
from typing import List

from nvflare.apis.client import Client
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.fuel.f3.cellnet.cell import Message, MessageHeaderKey
from nvflare.fuel.f3.cellnet.cell import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.cell import new_message
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.private.defs import CellChannel
from nvflare.security.logging import secure_format_traceback


class AuxRunner(FLComponent):
    def __init__(self, engine):
        """To init the AuxRunner."""
        FLComponent.__init__(self)
        self.engine = engine
        self.topic_table = {}  # topic => handler
        self.reg_lock = Lock()

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

        .. note::

            peer_ctx props must have been set into the PEER_PROPS header of the request by Engine.

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
            self.log_error(fl_ctx, f"received invalid aux request: expects a Shareable but got {type(request)}")
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
        except Exception:
            self.log_exception(fl_ctx, "processing error in message handling")
            return make_reply(ReturnCode.HANDLER_EXCEPTION)

        return reply

    def dispatch(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """This method is to be called by the Engine when an aux message is received from peer.

        .. note::

            peer_ctx props must have been set into the PEER_PROPS header of the request by Engine.

        Args:
            topic: message topic
            request: request message
            fl_ctx: FLContext

        Returns: reply message

        """
        peer_props = request.get_peer_props()
        if peer_props:
            peer_ctx = FLContext()
            peer_ctx.set_public_props(peer_props)
            fl_ctx.set_peer_context(peer_ctx)

        valid_reply = self._process_request(topic, request, fl_ctx)
        if isinstance(request, Shareable):
            cookie_jar = request.get_cookie_jar()
            if cookie_jar:
                valid_reply.set_cookie_jar(cookie_jar)
        return valid_reply

    def send_aux_request(
        self,
        targets: list,
        topic: str,
        request: Shareable,
        timeout: float,
        fl_ctx: FLContext,
        bulk_send: bool = False,
        optional: bool = False,
    ) -> dict:
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

            if name not in target_names:
                target_names.append(t)

        if not target_names:
            return {}

        _, invalid_names = self.engine.validate_targets(target_names)
        if invalid_names:
            raise ValueError(f"invalid target(s): {invalid_names}")

        try:
            return self._send_to_cell(
                targets=targets,
                channel=CellChannel.AUX_COMMUNICATION,
                topic=topic,
                request=request,
                timeout=timeout,
                fl_ctx=fl_ctx,
                bulk_send=bulk_send,
                optional=optional,
            )
        except Exception:
            if optional:
                self.logger.debug(f"Failed to send aux message {topic} to targets: {targets}")
                self.logger.debug(secure_format_traceback())
            else:
                self.logger.error(f"Failed to send aux message {topic} to targets: {targets}")
                self.logger.error(secure_format_traceback())
            return {}

    def _send_to_cell(
        self,
        targets: List[str],
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
        request.set_header(ReservedHeaderKey.TOPIC, topic)
        request.set_peer_props(fl_ctx.get_all_public_props())

        job_id = fl_ctx.get_job_id()
        start = time.time()
        self.logger.debug("waiting for cell...")
        max_wait = 5.0
        while True:
            cell = self.engine.get_cell()
            if cell:
                break
            if time.time() - start > max_wait:
                self.logger.error(f"Cannot get cell after {max_wait} seconds!")
                return {}
            time.sleep(0.01)
        self.logger.debug(f"Got cell in {time.time() - start} secs")

        target_names = []
        for t in targets:
            if not isinstance(t, str):
                raise ValueError(f"invalid target name {t}: expect str but got {type(t)}")
            if t not in target_names:
                target_names.append(t)

        target_fqcns = []
        for name in target_names:
            target_fqcns.append(FQCN.join([name, job_id]))

        cell_msg = new_message(payload=request)
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

    def _convert_return_code(self, rc):
        rc_table = {
            CellReturnCode.TIMEOUT: ReturnCode.COMMUNICATION_ERROR,
            CellReturnCode.COMM_ERROR: ReturnCode.COMMUNICATION_ERROR,
            CellReturnCode.PROCESS_EXCEPTION: ReturnCode.EXECUTION_EXCEPTION,
            CellReturnCode.ABORT_RUN: CellReturnCode.ABORT_RUN,
            CellReturnCode.INVALID_REQUEST: CellReturnCode.INVALID_REQUEST,
            CellReturnCode.INVALID_SESSION: CellReturnCode.INVALID_SESSION,
            CellReturnCode.AUTHENTICATION_ERROR: CellReturnCode.UNAUTHENTICATED,
            CellReturnCode.SERVICE_UNAVAILABLE: CellReturnCode.SERVICE_UNAVAILABLE,
        }
        return rc_table.get(rc, ReturnCode.ERROR)
