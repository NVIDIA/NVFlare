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
from typing import List, Tuple

from nvflare.apis.client import Client
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ConfigVarName, ProcessType, ReturnCode, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.fuel.f3.cellnet.core_cell import Message, MessageHeaderKey
from nvflare.fuel.f3.cellnet.core_cell import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.core_cell import TargetMessage
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.private.defs import CellChannel
from nvflare.security.logging import secure_format_exception, secure_format_traceback


class AuxMsgTarget:
    def __init__(self, name: str, fqcn: str):
        self.name = name
        self.fqcn = fqcn

    @staticmethod
    def server_target():
        return AuxMsgTarget(FQCN.ROOT_SERVER, FQCN.ROOT_SERVER)

    @staticmethod
    def client_target(client: Client):
        return AuxMsgTarget(client.name, client.get_fqcn())

    def __str__(self):
        return f"AuxMsgTarget[name={self.name} fqcn={self.fqcn}]"


class AuxRunner(FLComponent):
    def __init__(self, engine):
        """To init the AuxRunner."""
        FLComponent.__init__(self)
        self.engine = engine
        self.topic_table = {}  # topic => handler
        self.reg_lock = Lock()
        self.cell_wait_timeout = None

    def register_aux_message_handler(self, topic: str, message_handle_func):
        """Register aux message handling function with specified topics.

        This method should be called by Engine's register_aux_message_handler method.

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
            self.logger.info(f"registered aux handler for topic {topic}")

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

        if not isinstance(request, Shareable):
            self.log_error(fl_ctx, f"received invalid aux request: expects a Shareable but got {type(request)}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

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

    def _wait_for_cell(self):
        if self.cell_wait_timeout is None:
            self.cell_wait_timeout = ConfigService.get_float_var(
                name=ConfigVarName.CELL_WAIT_TIMEOUT, conf=SystemConfigs.APPLICATION_CONF, default=5.0
            )

        start = time.time()
        self.logger.debug(f"waiting for cell for {self.cell_wait_timeout} seconds")
        while True:
            cell = self.engine.get_cell()
            if cell:
                self.logger.debug(f"Got cell in {time.time() - start} secs")
                return cell
            if time.time() - start > self.cell_wait_timeout:
                self.logger.error(f"Cannot get cell after {self.cell_wait_timeout} seconds!")
                return None
            time.sleep(0.1)

    def _process_cell_replies(
        self,
        cell_replies: dict,
        topic: str,
        channel: str,
        fqcn_to_name: dict,
    ):
        replies = {}
        if cell_replies:
            for reply_cell_fqcn, v in cell_replies.items():
                assert isinstance(v, Message)
                rc = v.get_header(MessageHeaderKey.RETURN_CODE, CellReturnCode.OK)
                target_name = fqcn_to_name[reply_cell_fqcn]
                if rc == CellReturnCode.OK:
                    result = v.payload
                    if not isinstance(result, Shareable):
                        self.logger.error(f"reply of {channel}: {topic} must be Shareable but got {type(result)}")
                        result = make_reply(ReturnCode.ERROR)
                    replies[target_name] = result
                else:
                    src = self._convert_return_code(rc)
                    replies[target_name] = make_reply(src)
        return replies

    def multicast_aux_requests(
        self,
        topic: str,
        target_requests: List[Tuple[AuxMsgTarget, Shareable]],
        timeout: float,
        fl_ctx: FLContext,
        optional: bool = False,
        secure: bool = False,
    ) -> dict:
        if not target_requests:
            return {}

        targets = []
        for t in target_requests:
            amt, _ = t
            targets.append(amt)

        try:
            return self._send_multi_requests(
                topic=topic,
                target_requests=target_requests,
                timeout=timeout,
                fl_ctx=fl_ctx,
                optional=optional,
                secure=secure,
            )
        except Exception:
            if optional:
                self.logger.debug(f"Failed to send multi requests {topic} to targets: {targets}")
                self.logger.debug(secure_format_traceback())
            else:
                self.logger.error(f"Failed to send multi requests {topic} to targets: {targets}")
                self.logger.error(secure_format_traceback())
            return {}

    def _send_multi_requests(
        self,
        topic: str,
        target_requests: List[Tuple[AuxMsgTarget, Shareable]],
        timeout: float,
        fl_ctx: FLContext,
        optional: bool = False,
        secure: bool = False,
    ) -> dict:
        channel = CellChannel.AUX_COMMUNICATION
        cell = self._wait_for_cell()
        if not cell:
            return {}

        public_props = fl_ctx.get_all_public_props()
        target_messages = {}
        fqcn_to_name = {}
        for t in target_requests:
            msg_target, req = t
            assert isinstance(msg_target, AuxMsgTarget)
            target_name = msg_target.name
            if not isinstance(req, Shareable):
                raise ValueError(f"request of {target_name} should be Shareable but got {type(req)}")

            req.set_header(ReservedHeaderKey.TOPIC, topic)
            req.set_peer_props(public_props)
            cell_fqcn = self._get_target_fqcn(msg_target, fl_ctx)
            self.log_debug(fl_ctx, f"sending multicast aux: {cell_fqcn=}")
            fqcn_to_name[cell_fqcn] = target_name
            target_messages[cell_fqcn] = TargetMessage(
                topic=topic, channel=channel, target=cell_fqcn, message=Message(payload=req)
            )

        if timeout > 0:
            cell_replies = cell.broadcast_multi_requests(
                target_messages, timeout, optional=optional, secure=secure, abort_signal=fl_ctx.get_run_abort_signal()
            )
            return self._process_cell_replies(cell_replies, topic, channel, fqcn_to_name)
        else:
            cell.fire_multi_requests_and_forget(
                target_messages,
                optional=optional,
            )
            return {}

    def send_aux_request(
        self,
        targets: List[AuxMsgTarget],
        topic: str,
        request: Shareable,
        timeout: float,
        fl_ctx: FLContext,
        bulk_send: bool = False,
        optional: bool = False,
        secure: bool = False,
    ) -> dict:
        """Send aux request to specified targets.

        Args:
            targets: a list of AuxMsgTarget(s)
            topic: topic of the message
            request: the request to be sent
            timeout: timeout of the request
            fl_ctx: FL context data
            bulk_send: whether to bulk send
            optional: whether the request is optional
            secure: whether to use P2P message encryption

        Returns: a dict of target_name => reply

        Note: each AuxMsgTarget in "targets" has the target's name and FQCN.
        The returned dict is keyed on the client Name, not client FQCN (which can be multiple levels).

        """
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
                secure=secure,
            )
        except Exception as ex:
            if optional:
                self.logger.debug(
                    f"Failed to send aux message {topic} to targets: {targets}: {secure_format_exception(ex)}"
                )
                self.logger.debug(secure_format_traceback())
            else:
                self.logger.error(f"Failed to send aux message {topic} to targets: {targets}")
                self.logger.error(secure_format_traceback())
            return {}

    def _send_to_cell(
        self,
        targets: List[AuxMsgTarget],
        channel: str,
        topic: str,
        request: Shareable,
        timeout: float,
        fl_ctx: FLContext,
        bulk_send=False,
        optional=False,
        secure=False,
    ) -> dict:
        """Send request to the job cells of other target sites.

        Args:
            targets (list): list of AuxMsgTarget that the request will be sent to
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

        cell = self._wait_for_cell()
        if not cell:
            return {}

        target_fqcns = []
        fqcn_to_name = {}
        for t in targets:
            # targeting job cells!
            cell_fqcn = self._get_target_fqcn(t, fl_ctx)
            target_fqcns.append(cell_fqcn)
            fqcn_to_name[cell_fqcn] = t.name

        cell_msg = Message(payload=request)
        if timeout > 0:
            cell_replies = cell.broadcast_request(
                channel=channel,
                topic=topic,
                request=cell_msg,
                targets=target_fqcns,
                timeout=timeout,
                optional=optional,
                secure=secure,
                abort_signal=fl_ctx.get_run_abort_signal(),
            )
            return self._process_cell_replies(cell_replies, topic, channel, fqcn_to_name)
        else:
            if bulk_send:
                cell.queue_message(channel=channel, topic=topic, message=cell_msg, targets=target_fqcns)
            else:
                cell.fire_and_forget(
                    channel=channel, topic=topic, message=cell_msg, targets=target_fqcns, optional=optional
                )
            return {}

    @staticmethod
    def _get_target_fqcn(target: AuxMsgTarget, fl_ctx: FLContext):
        process_type = fl_ctx.get_process_type()
        if process_type in [ProcessType.CLIENT_PARENT, ProcessType.SERVER_PARENT]:
            # parent process
            return target.fqcn
        elif process_type in [ProcessType.CLIENT_JOB, ProcessType.SERVER_JOB]:
            # job process
            job_id = fl_ctx.get_job_id()
            if not job_id:
                raise RuntimeError("no job ID in fl_ctx in Job Process!")
            return FQCN.join([target.fqcn, job_id])
        else:
            raise RuntimeError(f"invalid process_type {process_type}")

    @staticmethod
    def _convert_return_code(rc):
        rc_table = {
            CellReturnCode.TIMEOUT: ReturnCode.TIMEOUT,
            CellReturnCode.COMM_ERROR: ReturnCode.COMMUNICATION_ERROR,
            CellReturnCode.PROCESS_EXCEPTION: ReturnCode.EXECUTION_EXCEPTION,
            CellReturnCode.ABORT_RUN: CellReturnCode.ABORT_RUN,
            CellReturnCode.INVALID_REQUEST: CellReturnCode.INVALID_REQUEST,
            CellReturnCode.INVALID_SESSION: CellReturnCode.INVALID_SESSION,
            CellReturnCode.AUTHENTICATION_ERROR: CellReturnCode.UNAUTHENTICATED,
            CellReturnCode.SERVICE_UNAVAILABLE: CellReturnCode.SERVICE_UNAVAILABLE,
        }
        return rc_table.get(rc, ReturnCode.ERROR)
