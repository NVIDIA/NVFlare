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

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.stream_const import (
    STREAM_ACK_TOPIC,
    STREAM_CHANNEL,
    STREAM_DATA_TOPIC,
    StreamDataType,
    StreamHeaderKey,
)
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamError, StreamFuture
from nvflare.fuel.f3.streaming.stream_utils import gen_stream_id, stream_thread_pool, wrap_view

STREAM_CHUNK_SIZE = 1024 * 1024
STREAM_WINDOW_SIZE = 16 * STREAM_CHUNK_SIZE
STREAM_ACK_WAIT = 10

log = logging.getLogger(__name__)


class TxTask:
    def __init__(self, channel: str, topic: str, target: str, headers: dict, stream: Stream):
        self.sid = gen_stream_id()
        self.buffer = bytearray(STREAM_CHUNK_SIZE)
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
        self.stop = False

    def __str__(self):
        return f"Tx[{self.sid} to {self.target} for {self.channel}/{self.topic}]"


class ByteStreamer:
    def __init__(self, cell: Cell):
        self.cell = cell
        self.cell.register_request_cb(channel=STREAM_CHANNEL, topic=STREAM_ACK_TOPIC, cb=self._ack_handler)
        self.tx_task_map = {}

    @staticmethod
    def get_chunk_size():
        return STREAM_CHUNK_SIZE

    def send(self, channel: str, topic: str, target: str, headers: dict, stream: Stream) -> StreamFuture:
        tx_task = TxTask(channel, topic, target, headers, stream)
        self.tx_task_map[tx_task.sid] = tx_task

        future = StreamFuture(tx_task.sid)
        future.set_size(stream.get_size())
        tx_task.stream_future = future
        tx_task.task_future = stream_thread_pool.submit(self._transmit_task, tx_task)

        return future

    def _transmit_task(self, task: TxTask):

        while not task.stop:
            buf = task.stream.read(STREAM_CHUNK_SIZE)
            if not buf:
                # End of Stream
                self._transmit(task, final=True)
                self._stop_task(task)
                return

            # Flow control
            window = task.offset - task.offset_ack
            # It may take several ACKs to clear up the window
            while window > STREAM_WINDOW_SIZE:
                log.debug(f"{task} window size {window} exceeds limit: {STREAM_WINDOW_SIZE}")
                task.ack_waiter.clear()
                result = task.ack_waiter.wait(timeout=STREAM_ACK_WAIT)
                if not result:
                    self._stop_task(task, StreamError(f"{task} ACK timeouts after {STREAM_ACK_WAIT} seconds"))
                    return

                if task.stop:
                    return

                window = task.offset - task.offset_ack

            size = len(buf)
            if size > STREAM_CHUNK_SIZE:
                raise StreamError(f"Stream returns invalid size: {size} for {task}")
            if size + task.buffer_size > STREAM_CHUNK_SIZE:
                self._transmit(task)

            if size == STREAM_CHUNK_SIZE:
                task.direct_buf = buf
            else:
                task.buffer[task.buffer_size : task.buffer_size + size] = buf
            task.buffer_size += size

    def _transmit(self, task: TxTask, final=False):

        if task.buffer_size == 0:
            payload = None
        elif task.buffer_size == STREAM_CHUNK_SIZE:
            if task.direct_buf:
                payload = task.direct_buf
            else:
                payload = task.buffer
        else:
            payload = wrap_view(task.buffer)[0 : task.buffer_size]

        message = Message(None, payload)

        if task.offset == 0:
            # User headers are only included in the first chunk
            if task.headers:
                message.add_headers(task.headers)

            message.add_headers(
                {
                    StreamHeaderKey.CHANNEL: task.channel,
                    StreamHeaderKey.TOPIC: task.topic,
                }
            )

        message.add_headers(
            {
                StreamHeaderKey.STREAM_ID: task.sid,
                StreamHeaderKey.DATA_TYPE: StreamDataType.FINAL if final else StreamDataType.CHUNK,
                StreamHeaderKey.SEQUENCE: task.seq,
                StreamHeaderKey.OFFSET: task.offset,
            }
        )

        errors = self.cell.fire_and_forget(STREAM_CHANNEL, STREAM_DATA_TOPIC, task.target, message)
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
        if error:
            log.debug(f"Stream error: {error}")
            task.stream_future.set_exception(error)

            if notify:
                message = Message(None, None)
                message.add_headers(
                    {
                        StreamHeaderKey.STREAM_ID: task.sid,
                        StreamHeaderKey.DATA_TYPE: StreamDataType.ERROR,
                        StreamHeaderKey.OFFSET: task.offset,
                        StreamHeaderKey.ERROR_MSG: str(error),
                    }
                )
                self.cell.fire_and_forget(STREAM_CHANNEL, STREAM_DATA_TOPIC, task.target, message)
        else:
            # Result is the number of bytes streamed
            task.stream_future.set_result(task.offset)
        task.stop = True

    def _ack_handler(self, message: Message):
        origin = message.get_header(MessageHeaderKey.ORIGIN)
        sid = message.get_header(StreamHeaderKey.STREAM_ID)
        task = self.tx_task_map.get(sid, None)
        if not task:
            raise StreamError(f"Unknown stream ID {sid} received from {origin}")

        error = message.get_header(StreamHeaderKey.ERROR_MSG, None)
        if error:
            self._stop_task(task, StreamError(f"Received error from {origin}: {error}"), notify=False)
            return

        offset = message.get_header(StreamHeaderKey.OFFSET, None)
        if offset > task.offset_ack:
            task.offset_ack = offset

        if not task.ack_waiter.is_set():
            task.ack_waiter.set()
