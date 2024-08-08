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
import logging
import threading
from typing import Optional

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.fuel.f3.streaming.stream_const import (
    STREAM_ACK_TOPIC,
    STREAM_CHANNEL,
    STREAM_DATA_TOPIC,
    StreamDataType,
    StreamHeaderKey,
)
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamError, StreamFuture
from nvflare.fuel.f3.streaming.stream_utils import (
    ONE_MB,
    gen_stream_id,
    stream_stats_category,
    stream_thread_pool,
    wrap_view,
)

STREAM_CHUNK_SIZE = 1024 * 1024
STREAM_WINDOW_SIZE = 16 * STREAM_CHUNK_SIZE
STREAM_ACK_WAIT = 60

STREAM_TYPE_BYTE = "byte"
STREAM_TYPE_BLOB = "blob"
STREAM_TYPE_FILE = "file"

COUNTER_NAME_SENT = "sent"

log = logging.getLogger(__name__)


class TxTask:
    def __init__(
        self, channel: str, topic: str, target: str, headers: dict, stream: Stream, secure: bool, optional: bool
    ):
        self.sid = gen_stream_id()
        self.buffer = bytearray(ByteStreamer.get_chunk_size())
        # Optimization to send the original buffer without copying
        self.direct_buf: Optional[bytes] = None
        self.buffer_size = 0
        self.channel = channel
        self.topic = topic
        self.target = target
        self.headers = headers
        self.stream = stream
        self.stream_future = None
        self.task_future = None
        self.ack_waiter = threading.Event()
        self.seq = 0
        self.offset = 0
        self.offset_ack = 0
        self.secure = secure
        self.optional = optional

    def __str__(self):
        return f"Tx[SID:{self.sid} to {self.target} for {self.channel}/{self.topic}]"


class ByteStreamer:
    def __init__(self, cell: CoreCell):
        self.cell = cell
        self.cell.register_request_cb(channel=STREAM_CHANNEL, topic=STREAM_ACK_TOPIC, cb=self._ack_handler)
        self.tx_task_map = {}
        self.map_lock = threading.Lock()

        self.sent_stream_counter_pool = StatsPoolManager.add_counter_pool(
            name="Sent_Stream_Counters",
            description="Counters of sent streams",
            counter_names=[COUNTER_NAME_SENT],
            scope=self.cell.my_info.fqcn,
        )

        self.sent_stream_size_pool = StatsPoolManager.add_msg_size_pool(
            "Sent_Stream_Sizes", "Sizes of streams sent (MBs)", scope=self.cell.my_info.fqcn
        )

    @staticmethod
    def get_chunk_size():
        return CommConfigurator().get_streaming_chunk_size(STREAM_CHUNK_SIZE)

    def send(
        self,
        channel: str,
        topic: str,
        target: str,
        headers: dict,
        stream: Stream,
        stream_type=STREAM_TYPE_BYTE,
        secure=False,
        optional=False,
    ) -> StreamFuture:
        tx_task = TxTask(channel, topic, target, headers, stream, secure, optional)
        with self.map_lock:
            self.tx_task_map[tx_task.sid] = tx_task

        future = StreamFuture(tx_task.sid)
        future.set_size(stream.get_size())
        tx_task.stream_future = future
        tx_task.task_future = stream_thread_pool.submit(self._transmit_task, tx_task)

        self.sent_stream_counter_pool.increment(
            category=stream_stats_category(channel, topic, stream_type), counter_name=COUNTER_NAME_SENT
        )

        self.sent_stream_size_pool.record_value(
            category=stream_stats_category(channel, topic, stream_type), value=stream.get_size() / ONE_MB
        )

        return future

    def _transmit_task(self, task: TxTask):

        chunk_size = self.get_chunk_size()
        while True:
            buf = task.stream.read(chunk_size)
            if not buf:
                # End of Stream
                self._transmit(task, final=True)
                self._stop_task(task)
                return

            # Flow control
            window = task.offset - task.offset_ack
            # It may take several ACKs to clear up the window
            window_size = CommConfigurator().get_streaming_window_size(STREAM_WINDOW_SIZE)
            while window > window_size:
                log.debug(f"{task} window size {window} exceeds limit: {window_size}")
                task.ack_waiter.clear()
                ack_wait = CommConfigurator().get_streaming_ack_wait(STREAM_ACK_WAIT)
                if not task.ack_waiter.wait(timeout=ack_wait):
                    self._stop_task(task, StreamError(f"{task} ACK timeouts after {ack_wait} seconds"))
                    return

                window = task.offset - task.offset_ack

            size = len(buf)
            if size > chunk_size:
                raise StreamError(f"Stream returns invalid size: {size} for {task}")
            if size + task.buffer_size > chunk_size:
                self._transmit(task)

            if size == chunk_size:
                task.direct_buf = buf
            else:
                task.buffer[task.buffer_size : task.buffer_size + size] = buf
            task.buffer_size += size

    def _transmit(self, task: TxTask, final=False):

        if task.buffer_size == 0:
            payload = bytes(0)
        elif task.buffer_size == self.get_chunk_size():
            if task.direct_buf:
                payload = task.direct_buf
            else:
                payload = task.buffer
        else:
            payload = wrap_view(task.buffer)[0 : task.buffer_size]

        message = Message(None, payload)

        if task.headers:
            message.add_headers(task.headers)

        message.add_headers(
            {
                StreamHeaderKey.CHANNEL: task.channel,
                StreamHeaderKey.TOPIC: task.topic,
                StreamHeaderKey.SIZE: task.stream.get_size(),
                StreamHeaderKey.STREAM_ID: task.sid,
                StreamHeaderKey.DATA_TYPE: StreamDataType.FINAL if final else StreamDataType.CHUNK,
                StreamHeaderKey.SEQUENCE: task.seq,
                StreamHeaderKey.OFFSET: task.offset,
                StreamHeaderKey.OPTIONAL: task.optional,
            }
        )

        errors = self.cell.fire_and_forget(
            STREAM_CHANNEL, STREAM_DATA_TOPIC, task.target, message, secure=task.secure, optional=task.optional
        )
        error = errors.get(task.target)
        if error:
            msg = f"Message sending error to target {task.target}: {error}"
            log.debug(msg)
            self._stop_task(task, StreamError(msg))
            return

        # Update state
        task.seq += 1
        task.offset += task.buffer_size
        task.buffer_size = 0
        task.direct_buf = None

        # Update future
        task.stream_future.set_progress(task.offset)

    def _stop_task(self, task: TxTask, error: StreamError = None, notify=True):
        with self.map_lock:
            self.tx_task_map.pop(task.sid, None)

        if error:
            log.debug(f"Stream error: {error}")
            if task.stream_future:
                task.stream_future.set_exception(error)

            if notify:
                message = Message(None, None)

                if task.headers:
                    message.add_headers(task.headers)

                message.add_headers(
                    {
                        StreamHeaderKey.STREAM_ID: task.sid,
                        StreamHeaderKey.DATA_TYPE: StreamDataType.ERROR,
                        StreamHeaderKey.OFFSET: task.offset,
                        StreamHeaderKey.ERROR_MSG: str(error),
                    }
                )
                self.cell.fire_and_forget(
                    STREAM_CHANNEL, STREAM_DATA_TOPIC, task.target, message, secure=task.secure, optional=True
                )
        else:
            # Result is the number of bytes streamed
            if task.stream_future:
                task.stream_future.set_result(task.offset)

    def _ack_handler(self, message: Message):
        origin = message.get_header(MessageHeaderKey.ORIGIN)
        sid = message.get_header(StreamHeaderKey.STREAM_ID)
        offset = message.get_header(StreamHeaderKey.OFFSET, None)

        with self.map_lock:
            task = self.tx_task_map.get(sid, None)

        if not task:
            # Last few ACKs always arrive late so this is normal
            log.debug(f"ACK for stream {sid} received late from {origin} with offset {offset}")
            return

        error = message.get_header(StreamHeaderKey.ERROR_MSG, None)
        if error:
            self._stop_task(task, StreamError(f"Received error from {origin}: {error}"), notify=False)
            return

        if offset > task.offset_ack:
            task.offset_ack = offset

        if not task.ack_waiter.is_set():
            task.ack_waiter.set()
