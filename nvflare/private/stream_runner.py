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
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any, List, Tuple

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.streaming import ConsumerFactory, ObjectConsumer, ObjectProducer, StreamContext, StreamContextKey
from nvflare.fuel.f3.cellnet.registry import Registry
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.validation_utils import check_callable, check_object_type, check_str
from nvflare.private.aux_runner import AuxMsgTarget, AuxRunner
from nvflare.security.logging import secure_format_exception

# Topics for streaming messages
PREFIX = "ObjectStreamer."
TOPIC_STREAM_REQUEST = PREFIX + "Request"
TOPIC_STREAM_ABORT = PREFIX + "Abort"


class HeaderKey:
    TX_ID = PREFIX + "TX_ID"
    SEQ = PREFIX + "SEQ"
    TOPIC = PREFIX + "TOPIC"
    CHANNEL = PREFIX + "CHANNEL"
    CTX = PREFIX + "CTX"


class _ConsumerInfo:
    def __init__(
        self,
        logger,
        stream_ctx: StreamContext,
        factory: ConsumerFactory,
        consumer: ObjectConsumer,
        stream_done_cb,
        cb_kwargs,
    ):
        self.logger = logger
        self.factory = factory
        self.stream_ctx = stream_ctx
        self.consumer = consumer
        self.stream_done_cb = stream_done_cb
        self.stream_done_cb_kwargs = cb_kwargs
        self.stream_start_time = time.time()
        self.last_msg_start_time = None
        self.last_msg_end_time = None

    def process(
        self,
        msg: Shareable,
        fl_ctx: FLContext,
    ):
        self.last_msg_start_time = time.time()
        reply = self.consumer.consume(msg, self.stream_ctx, fl_ctx)
        self.last_msg_end_time = time.time()
        return reply

    def stream_done(self, rc: str, fl_ctx: FLContext):
        self.stream_ctx[StreamContextKey.RC] = rc
        try:
            self.consumer.finalize(self.stream_ctx, fl_ctx)
        except Exception as ex:
            self.logger.error(
                f"exception finalizing processor {self.consumer.__class__.__name__}: {secure_format_exception(ex)}"
            )
            self.stream_ctx[StreamContextKey.RC] = ReturnCode.EXECUTION_EXCEPTION

        if self.stream_done_cb:
            try:
                self.stream_done_cb(self.stream_ctx, fl_ctx, **self.stream_done_cb_kwargs)
            except Exception as ex:
                self.logger.error(
                    f"exception from stream_done_cb {self.stream_done_cb.__name__}: {secure_format_exception(ex)}"
                )

        try:
            self.factory.return_consumer(
                consumer=self.consumer,
                stream_ctx=self.stream_ctx,
                fl_ctx=fl_ctx,
            )
        except Exception as ex:
            self.logger.error(
                f"exception returning processor to factory {self.factory.__class__.__name__}: "
                f"{secure_format_exception(ex)}"
            )


class ObjectStreamer(FLComponent):
    def __init__(self, aux_runner: AuxRunner):
        FLComponent.__init__(self)
        self.aux_runner = aux_runner
        self.registry = Registry()
        self.tx_lock = Lock()
        self.tx_table = {}  # tx_id => _ProcessorInfo
        self.logger = get_obj_logger(self)

        # Note: the ConfigService has been initialized
        max_concurrent_streaming_sessions = ConfigService.get_int_var("max_concurrent_streaming_sessions", default=20)
        self.streaming_executor = ThreadPoolExecutor(max_workers=max_concurrent_streaming_sessions)

        aux_runner.register_aux_message_handler(
            topic=TOPIC_STREAM_REQUEST,
            message_handle_func=self._handle_request,
        )
        aux_runner.register_aux_message_handler(
            topic=TOPIC_STREAM_ABORT,
            message_handle_func=self._handle_abort,
        )

    def shutdown(self):
        e = self.streaming_executor
        self.streaming_executor = None
        if e:
            e.shutdown(wait=False, cancel_futures=True)
            self.logger.info("Stream Runer is Shut Down")

    def register_stream_processing(
        self,
        channel: str,
        topic: str,
        factory: ConsumerFactory,
        stream_done_cb=None,
        **cb_kwargs,
    ):
        """Register a ConsumerFactory for specified app channel and topic.
        Once a new streaming request is received for the channel/topic, the registered factory will be used
        to create a new ObjectConsumer object to handle the stream.

        Note: the factory should generate a new ObjectConsumer every time get_consumer() is called. This is because
        multiple streaming sessions could be going on at the same time. Each streaming session should have its
        own ObjectConsumer.

        Args:
            channel: app channel
            topic: app topic
            factory: the factory to be registered
            stream_done_cb: the CB to be called when a stream is done

        Returns: None

        """
        check_str("channel", channel)
        check_str("topic", topic)
        check_object_type("factory", factory, ConsumerFactory)
        if stream_done_cb is not None:
            check_callable("stream_done_cb", stream_done_cb)
        self.registry.set(channel, topic, (factory, stream_done_cb, cb_kwargs))
        self.logger.info(f"registered processor_factory: {channel=} {topic=} {factory.__class__.__name__}")

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

    def debug(self, req: Shareable, msg: str):
        self.logger.debug(self._log_msg(req, msg))

    def _end_tx(self, tx_id: str, rc: str, fl_ctx: FLContext):
        with self.tx_lock:
            info = self.tx_table.pop(tx_id, None)

        if info:
            info.stream_done(rc, fl_ctx)

    def _handle_abort(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        self.logger.debug("abort received")
        tx_id = request.get_header(HeaderKey.TX_ID)
        if not tx_id:
            self.logger.error(f"missing header {HeaderKey.TX_ID}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        self._end_tx(tx_id, ReturnCode.TASK_ABORTED, fl_ctx)
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
            self._end_tx(tx_id, ReturnCode.BAD_REQUEST_DATA, fl_ctx)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        channel = request.get_header(HeaderKey.CHANNEL)
        if not channel:
            self.logger.error(f"missing header {HeaderKey.CHANNEL}")
            self._end_tx(tx_id, ReturnCode.BAD_REQUEST_DATA, fl_ctx)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        seq = request.get_header(HeaderKey.SEQ)
        if seq is None:
            self.logger.error(f"missing header {HeaderKey.SEQ}")
            self._end_tx(tx_id, ReturnCode.BAD_REQUEST_DATA, fl_ctx)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        if abort_signal and abort_signal.triggered:
            self._end_tx(tx_id, ReturnCode.TASK_ABORTED, fl_ctx)
            return make_reply(ReturnCode.TASK_ABORTED)

        factory_info = self.registry.find(channel, topic)
        if not factory_info:
            self.error(request, f"no stream processing info registered for {channel}:{topic}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        factory, stream_done_db, cb_kwargs = factory_info

        self.debug(request, "received stream request")
        with self.tx_lock:
            info = self.tx_table.get(tx_id)
            if info:
                if seq <= 0:
                    # if we already have processor for this tx_id, then the seq number must be > 0
                    self.error(request, f"sequence error: expect > 0 but got {seq}")
                    self._end_tx(tx_id, ReturnCode.BAD_REQUEST_DATA, fl_ctx)
                    return make_reply(ReturnCode.BAD_REQUEST_DATA)
            else:
                if seq != 0:
                    # the sequence number is not 0 - we didn't get the initial msg!
                    self.error(request, f"sequence error: expect 0 but got {seq}")
                    return make_reply(ReturnCode.BAD_REQUEST_DATA)

                try:
                    stream_ctx = request.get_header(HeaderKey.CTX)
                    if not stream_ctx:
                        self.error(request, "missing stream ctx in seq 0")
                        return make_reply(ReturnCode.BAD_REQUEST_DATA)

                    consumer = factory.get_consumer(stream_ctx, fl_ctx)
                    if not consumer:
                        self.error(request, f"no consumer from factory {type(factory)}")
                        return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                    if not isinstance(consumer, ObjectConsumer):
                        self.error(
                            request,
                            f"bad consumer from factory {type(factory)}: "
                            f"expect ObjectConsumer but got {type(consumer)}",
                        )
                        return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                    info = _ConsumerInfo(
                        logger=self.logger,
                        factory=factory,
                        consumer=consumer,
                        stream_ctx=stream_ctx,
                        stream_done_cb=stream_done_db,
                        cb_kwargs=cb_kwargs,
                    )
                    self.tx_table[tx_id] = info
                except Exception as ex:
                    self.error(
                        request, f"factory {type(factory)} failed to produce consumer: {secure_format_exception(ex)}"
                    )
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # process the message
        try:
            continue_streaming, reply = info.process(request, fl_ctx)
            if not reply:
                # ack
                reply = make_reply(ReturnCode.OK)
        except Exception as ex:
            self.error(request, f"consumer {type(info.consumer)} failed to process msg: {secure_format_exception(ex)}")
            continue_streaming = False
            reply = make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if abort_signal and abort_signal.triggered:
            continue_streaming = False
            reply = make_reply(ReturnCode.TASK_ABORTED)

        if not continue_streaming:
            # remove the tx
            self._end_tx(tx_id, rc=reply.get_return_code(), fl_ctx=fl_ctx)

        self.debug(request, f"send reply: {reply}")
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
        stream_ctx: StreamContext,
        targets: List[AuxMsgTarget],
        producer: ObjectProducer,
        fl_ctx: FLContext,
        secure=False,
        optional=False,
    ) -> Tuple[str, Any]:
        if not stream_ctx:
            stream_ctx = StreamContext()

        check_str("channel", channel)
        check_str("topic", topic)
        check_object_type("stream_ctx", stream_ctx, StreamContext)
        check_object_type("producer", producer, ObjectProducer)
        check_object_type("fl_ctx", fl_ctx, FLContext)

        tx_id = str(uuid.uuid4())
        seq = 0
        abort_signal = fl_ctx.get_run_abort_signal()

        stream_ctx[StreamContextKey.TOPIC] = topic
        stream_ctx[StreamContextKey.CHANNEL] = channel

        while True:
            if abort_signal and abort_signal.triggered:
                if seq > 0:
                    self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)
                return ReturnCode.TASK_ABORTED, None

            try:
                request, timeout = producer.produce(stream_ctx, fl_ctx)
                self.logger.debug(f"produce from {producer.__class__.__name__}: {seq=} {timeout=} {tx_id=}")
            except Exception as ex:
                if seq > 0:
                    self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)

                self.logger.error(
                    f"Producer {producer.__class__.__name__} failed to produce next object: "
                    f"{secure_format_exception(ex)}"
                )
                raise ex

            if request is None:
                # end of the streaming
                if seq > 0:
                    self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)
                return ReturnCode.OK, None

            if not isinstance(request, Shareable):
                if seq > 0:
                    self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)
                raise ValueError(
                    f"Producer {producer.__class__.__name__} must produce Shareable but got {type(request)}"
                )

            if not isinstance(timeout, float):
                if seq > 0:
                    self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)
                raise ValueError(
                    f"Producer {producer.__class__.__name__} must return a float timeout but got {type(timeout)}"
                )

            if timeout <= 0:
                if seq > 0:
                    self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)
                raise ValueError(
                    f"Producer {producer.__class__.__name__} must return a positive float timeout but got {timeout}"
                )

            request.set_header(HeaderKey.CHANNEL, channel)
            request.set_header(HeaderKey.TOPIC, topic)
            request.set_header(HeaderKey.TX_ID, tx_id)
            request.set_header(HeaderKey.SEQ, seq)

            if seq == 0:
                # only send meta in 1st request
                request.set_header(HeaderKey.CTX, stream_ctx)

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

            self.logger.debug("got replies from receivers")
            result = producer.process_replies(replies, stream_ctx, fl_ctx)
            self.logger.debug(f"got processed result from producer: {result}")
            if result is not None:
                # this is end of the streaming
                if abort_signal and abort_signal.triggered:
                    rc = ReturnCode.TASK_ABORTED
                else:
                    rc = ReturnCode.OK
                self._notify_abort_streaming(targets, tx_id, secure, fl_ctx)
                self.logger.debug(f"Done streaming: {rc}")
                return rc, result

    def stream_no_wait(
        self,
        channel: str,
        topic: str,
        stream_ctx: StreamContext,
        targets: List[AuxMsgTarget],
        producer: ObjectProducer,
        fl_ctx: FLContext,
        secure=False,
        optional=False,
    ) -> Future:
        if not self.streaming_executor:
            raise RuntimeError("streaming_executor is not available: the streamer has been shut down!")

        return self.streaming_executor.submit(
            self.stream,
            channel=channel,
            topic=topic,
            stream_ctx=stream_ctx,
            targets=targets,
            producer=producer,
            fl_ctx=fl_ctx,
            secure=secure,
            optional=optional,
        )
