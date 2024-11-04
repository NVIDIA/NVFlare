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
import uuid
from threading import Lock
from typing import Any, Dict, List, Tuple

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import CellChannel
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.registry import Registry
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.obj_utils import get_logger
from nvflare.security.logging import secure_format_exception

# Topics for Msg Streamer
TOPIC_STREAM_REQUEST = "MsgStream.Request"
TOPIC_STREAM_REPLY = "MsgStream.Reply"

PROP_KEY_TX_ID = "MsgStream.TX_ID"
PROP_KEY_TOPIC = "MsgStream.TOPIC"
PROP_KEY_CHANNEL = "MsgStream.CHANNEL"

PROP_KEY_DEBUG_INFO = "MsgStream.DEBUG_INFO"


class StreamRC:

    ACK = "ack"
    COMPLETED = "completed"
    ABORTED = "aborted"
    NO_FACTORY = "no_factory"
    NO_PROCESSOR = "no_processor"
    FACTORY_ERROR = "factory_error"
    PROCESSOR_ERROR = "processor_error"


class StreamMsgGenerator:
    def get_next(self) -> Message:
        """Produce the next message to be streamed

        Returns: the message to be streamed; or None if no more message.

        """
        pass

    def process_replies(self, replies: Dict[str, Message]) -> Any:
        """Process response from the receiver.
        If None is returned, the streaming continues.
        If a non-None object is returned, then the streaming stops and the returned object becomes the
        result of the streaming call.

        Args:
            replies: replies from the receiver

        Returns: any object

        Note: the replies is a dict of: target => reply

        """
        pass


class StreamMsgProcessor:
    def process(self, msg: Message, abort_signal: Signal) -> Tuple[bool, Message]:
        pass


class StreamMsgProcessorFactory:
    def get_processor(self, channel: str, topic: str) -> StreamMsgProcessor:
        pass


class StreamMsgReceiver:
    def __init__(self, cell: Cell):
        self.cell = Cell
        self.reg = Registry()
        self.tx_lock = Lock()
        self.tx_table = {}  # tx_id => processor
        self.logger = get_logger(self)
        self.abort_signal = Signal()

        cell.register_request_cb(
            channel=CellChannel.MSG_STREAM,
            topic=TOPIC_STREAM_REQUEST,
            cb=self._process_msg,
        )

    def set_abort_signal(self, abort_signal: Signal):
        if isinstance(abort_signal, Signal):
            self.abort_signal = abort_signal

    def register_processor_factory(
        self,
        channel: str,
        topic: str,
        factory: StreamMsgProcessorFactory,
    ):
        self.reg.set(channel, topic, factory)

    def _process_msg(self, msg: Message):
        if self.abort_signal and self.abort_signal.triggered:
            return make_reply(StreamRC.ABORTED)

        topic = msg.get_header(PROP_KEY_TOPIC)
        channel = msg.get_header(PROP_KEY_CHANNEL)
        tx_id = msg.get_header(PROP_KEY_TX_ID)

        factory = self.reg.find(channel, topic)
        if not factory:
            self.logger.error(
                f"no StreamMsgProcessorFactory registered for {channel}:{topic} in cell {self.cell.get_fqcn()}"
            )
            return make_reply(StreamRC.NO_FACTORY)

        with self.tx_lock:
            processor = self.tx_table.get(tx_id)
            if not processor:
                try:
                    processor = factory.get_processor(channel, topic)
                    if not processor:
                        return make_reply(StreamRC.NO_PROCESSOR)

                    if not isinstance(processor, StreamMsgProcessor):
                        self.logger.error(
                            f"bad processor from factory {type(factory)}: "
                            f"expect StreamMsgProcessor but got {type(processor)}"
                        )
                        return make_reply(StreamRC.FACTORY_ERROR)

                    self.tx_table[tx_id] = processor
                except Exception as ex:
                    self.logger.error(
                        f"factory {type(factory)} failed to produce processor: {secure_format_exception(ex)}"
                    )
                    return make_reply(StreamRC.FACTORY_ERROR)

        # process the message
        try:
            continue_processing, reply = processor.process(msg, abort_signal=self.abort_signal)
            if not reply:
                reply = make_reply(StreamRC.ACK)
        except Exception as ex:
            self.logger.error(f"processor {type(processor)} failed to process msg: {secure_format_exception(ex)}")
            continue_processing = False
            reply = make_reply(StreamRC.PROCESSOR_ERROR)

        if self.abort_signal and self.abort_signal.triggered:
            continue_processing = False
            reply = make_reply(StreamRC.ABORTED)

        if not continue_processing:
            # remove the tx
            with self.tx_lock:
                self.tx_table.pop(tx_id, None)

        return reply


def stream(
    cell: Cell,
    channel: str,
    topic: str,
    targets: List[str],
    generator: StreamMsgGenerator,
    per_msg_timeout: float,
    secure=False,
    optional=False,
    abort_signal: Signal = None,
) -> Tuple[str, Any]:
    if not isinstance(targets, list):
        raise ValueError(f"targets must be list of str but got {type(targets)}")

    # each target must be a valid FQCN
    for t in targets:
        err = FQCN.validate(t)
        if err:
            raise ValueError(f"target {t} is not valid FQCN: {err}")

    tx_id = str(uuid.uuid4())
    while True:
        if abort_signal and abort_signal.triggered:
            return StreamRC.ABORTED, None

        msg = generator.get_next()
        if msg is None:
            # end of the streaming
            return StreamRC.COMPLETED, None

        if not isinstance(msg, Message):
            raise ValueError(f"Generator {generator.__class__.__name__} must return Message but got {type(msg)}")

        msg.add_headers({PROP_KEY_CHANNEL: channel, PROP_KEY_TOPIC: topic, PROP_KEY_TX_ID: tx_id})

        # broadcast the message to all targets
        replies = cell.broadcast_request(
            channel=CellChannel.MSG_STREAM,
            topic=TOPIC_STREAM_REQUEST,
            targets=targets,
            request=msg,
            timeout=per_msg_timeout,
            secure=secure,
            optional=optional,
            abort_signal=abort_signal,
        )

        result = generator.process_replies(replies)
        if result is not None:
            # this is end of the streaming
            if abort_signal and abort_signal.triggered:
                rc = StreamRC.ABORTED
            else:
                rc = StreamRC.COMPLETED
            return rc, result
