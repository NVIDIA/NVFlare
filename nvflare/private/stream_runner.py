# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import uuid
from threading import Lock
from typing import Any, List, Tuple

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.stream_shareable import (
    StreamShareableGenerator,
    StreamShareableProcessor,
    StreamShareableProcessorFactory,
)
from nvflare.fuel.f3.cellnet.registry import Registry
from nvflare.fuel.utils.obj_utils import get_logger
from nvflare.private.aux_runner import AuxMsgTarget, AuxRunner
from nvflare.security.logging import secure_format_exception

# Topics for Msg Streamer
TOPIC_STREAM_REQUEST = "MsgStream.Request"
TOPIC_STREAM_ABORT = "MsgStream.Abort"


class HeaderKey:
    TX_ID = "MsgStream.TX_ID"
    SEQ = "MsgStream.SEQ"
    TOPIC = "MsgStream.TOPIC"
    CHANNEL = "MsgStream.CHANNEL"


class _ProcessorInfo:
    def __init__(self, processor: StreamShareableProcessor):
        self.processor = processor
        self.stream_start_time = time.time()
        self.last_msg_start_time = None
        self.last_msg_end_time = None

    def process(
        self,
        channel: str,
        topic: str,
        msg: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        self.last_msg_start_time = time.time()
        reply = self.processor.process(channel, topic, msg, fl_ctx, abort_signal)
        self.last_msg_end_time = time.time()
        return reply


class StreamRunner(FLComponent):
    def __init__(self, aux_runner: AuxRunner):
        FLComponent.__init__(self)
        self.aux_runner = aux_runner
        self.registry = Registry()
        self.tx_lock = Lock()
        self.tx_table = {}  # tx_id => _ProcessorInfo
        self.logger = get_logger(self)

        aux_runner.register_aux_message_handler(
            topic=TOPIC_STREAM_REQUEST,
            message_handle_func=self._handle_request,
        )
        aux_runner.register_aux_message_handler(
            topic=TOPIC_STREAM_ABORT,
            message_handle_func=self._handle_abort,
        )

    def register_processor_factory(
        self,
        channel: str,
        topic: str,
        factory: StreamShareableProcessorFactory,
    ):
        """Register a StreamShareableProcessorFactory for specified app channel and topic
        Once a new streaming request is received for the channel/topic, the registered factory will be used
        to create a StreamShareableProcessor object to handle the msg stream.

        Note: the factory should generate a new processor every time get_processor() is called. This is because
        multiple streaming sessions could be going on at the same time. Each streaming session should have its
        own processor.

        Args:
            channel: app channel
            topic: app topic
            factory: the factory to be registered

        Returns: None

        """
        self.registry.set(channel, topic, factory)

    @staticmethod
    def _log_msg(req: Shareable, msg: str):
        topic = req.get_header(HeaderKey.TOPIC)
        channel = req.get_header(HeaderKey.CHANNEL)
        tx_id = req.get_header(HeaderKey.TX_ID)
        seq = req.get_header(HeaderKey.SEQ)
        return f"[{tx_id=} {channel=} {topic=} {seq=}] {msg}"

    def error(self, req: Shareable, msg: str):
        self.logger.error(self._log_msg(req, msg))

    def info(self, req: Shareable, msg: str):
        self.logger.info(self._log_msg(req, msg))

    def _abort_tx(self, tx_id: str):
        with self.tx_lock:
            self.tx_table.pop(tx_id, None)

    def _abort_streaming(self, request: Shareable):
        tx_id = request.get_header(HeaderKey.TX_ID)
        if not tx_id:
            self.logger.error(f"missing header {HeaderKey.TX_ID}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        self._abort_tx(tx_id)
        return make_reply(ReturnCode.OK)

    def _handle_abort(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        tx_id = request.get_header(HeaderKey.TX_ID)
        if not tx_id:
            self.logger.error(f"missing header {HeaderKey.TX_ID}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        self._abort_tx(tx_id)
        return make_reply(ReturnCode.OK)

    def _handle_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        abort_signal = fl_ctx.get_run_abort_signal()
        tx_id = request.get_header(HeaderKey.TX_ID)
        if not tx_id:
            self.logger.error(f"missing header {HeaderKey.TX_ID}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        topic = request.get_header(HeaderKey.TOPIC)
        if not topic:
            self.logger.error(f"missing header {HeaderKey.TOPIC}")
            self._abort_tx(tx_id)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        channel = request.get_header(HeaderKey.CHANNEL)
        if not channel:
            self.logger.error(f"missing header {HeaderKey.CHANNEL}")
            self._abort_tx(tx_id)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        seq = request.get_header(HeaderKey.SEQ)
        if seq is None:
            self.logger.error(f"missing header {HeaderKey.SEQ}")
            self._abort_tx(tx_id)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        if abort_signal and abort_signal.triggered:
            self._abort_tx(tx_id)
            return make_reply(ReturnCode.TASK_ABORTED)

        factory = self.registry.find(channel, topic)
        if not factory:
            self.error(request, f"no StreamShareableProcessorFactory registered for {channel}:{topic}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        with self.tx_lock:
            info = self.tx_table.get(tx_id)
            if info:
                if seq <= 0:
                    # if we already have processor for this tx_id, then the seq number must be > 0
                    self.error(request, f"sequence error: expect > 0 but got {seq}")
                    self._abort_tx(tx_id)
                    return make_reply(ReturnCode.BAD_REQUEST_DATA)
            else:
                if seq != 0:
                    # the sequence number is not 0 - we didn't get the initial msg!
                    self.error(request, f"sequence error: expect 0 but got {seq}")
                    return make_reply(ReturnCode.BAD_REQUEST_DATA)

                try:
                    processor = factory.get_processor(channel, topic, request)
                    if not processor:
                        self.error(request, f"no processor from factory {type(factory)}")
                        return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                    if not isinstance(processor, StreamShareableProcessor):
                        self.error(
                            request,
                            f"bad processor from factory {type(factory)}: "
                            f"expect StreamShareableProcessor but got {type(processor)}",
                        )
                        return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                    info = _ProcessorInfo(processor)
                    self.tx_table[tx_id] = info
                except Exception as ex:
                    self.error(
                        request, f"factory {type(factory)} failed to produce processor: {secure_format_exception(ex)}"
                    )
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # process the message
        try:
            continue_streaming, reply = info.process(channel, topic, request, fl_ctx, abort_signal=abort_signal)
            if not reply:
                # ack
                reply = make_reply(ReturnCode.OK)
        except Exception as ex:
            self.error(
                request, f"processor {type(info.processor)} failed to process msg: {secure_format_exception(ex)}"
            )
            continue_streaming = False
            reply = make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if abort_signal and abort_signal.triggered:
            continue_streaming = False
            reply = make_reply(ReturnCode.TASK_ABORTED)

        if not continue_streaming:
            # remove the tx
            self._abort_tx(tx_id)
        return reply

    def _notify_abort_streaming(
        self,
        targets: List[AuxMsgTarget],
        tx_id: str,
        secure: bool,
        fl_ctx: FLContext,
    ):
        """Notify all targets to stop streaming processing in case they are still waiting.

        Args:
            targets:
            tx_id:
            fl_ctx:
            secure:

        Returns:

        """
        msg = make_reply(ReturnCode.TASK_ABORTED)
        msg.set_header(HeaderKey.TX_ID, tx_id)
        self.aux_runner.send_aux_request(
            targets=targets,
            topic=TOPIC_STREAM_ABORT,
            request=msg,
            timeout=0,  # fire and forget
            fl_ctx=fl_ctx,
            secure=secure,
            optional=True,
        )

    def stream(
        self,
        channel: str,
        topic: str,
        targets: List[AuxMsgTarget],
        generator: StreamShareableGenerator,
        fl_ctx: FLContext,
        secure=False,
        optional=False,
    ) -> Tuple[str, Any]:
        if not isinstance(targets, list):
            raise ValueError(f"targets must be list of AuxMsgTarget but got {type(targets)}")

        # each target must be a valid FQCN
        for t in targets:
            if not isinstance(t, AuxMsgTarget):
                raise ValueError(f"target must be AuxMsgTarget but got {type(t)}")

        tx_id = str(uuid.uuid4())
        seq = 0
        abort_signal = fl_ctx.get_run_abort_signal()
        while True:
            if abort_signal and abort_signal.triggered:
                if seq > 0:
                    self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)
                return ReturnCode.TASK_ABORTED, None

            request, timeout = generator.get_next(channel, topic, fl_ctx, abort_signal)

            if request is None:
                # end of the streaming
                if seq > 0:
                    self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)
                return ReturnCode.OK, None

            if not isinstance(request, Shareable):
                if seq > 0:
                    self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)
                raise ValueError(
                    f"Generator {generator.__class__.__name__} must return Shareable but got {type(request)}"
                )

            if not isinstance(timeout, float):
                if seq > 0:
                    self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)
                raise ValueError(
                    f"Generator {generator.__class__.__name__} must return a float timeout but got {type(timeout)}"
                )

            if timeout <= 0:
                if seq > 0:
                    self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)
                raise ValueError(
                    f"Generator {generator.__class__.__name__} must return a positive float timeout but got {timeout}"
                )

            request.set_header(HeaderKey.CHANNEL, channel)
            request.set_header(HeaderKey.TOPIC, topic)
            request.set_header(HeaderKey.TX_ID, tx_id)
            request.set_header(HeaderKey.SEQ, seq)
            seq += 1

            # broadcast the message to all targets
            replies = self.aux_runner.send_aux_request(
                topic=TOPIC_STREAM_REQUEST,
                targets=targets,
                request=request,
                timeout=timeout,
                secure=secure,
                optional=optional,
                fl_ctx=fl_ctx,
            )

            result = generator.process_replies(replies, fl_ctx, abort_signal)
            if result is not None:
                # this is end of the streaming
                if abort_signal and abort_signal.triggered:
                    rc = ReturnCode.TASK_ABORTED
                else:
                    rc = ReturnCode.OK
                self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)
                return rc, result
