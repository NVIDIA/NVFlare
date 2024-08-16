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
from collections import deque
from typing import Callable, Dict, Tuple

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.registry import Callback, Registry
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.connection import BytesAlike
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.fuel.f3.streaming.stream_const import (
    EOS,
    STREAM_ACK_TOPIC,
    STREAM_CHANNEL,
    STREAM_DATA_TOPIC,
    StreamDataType,
    StreamHeaderKey,
)
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamError, StreamFuture
from nvflare.fuel.f3.streaming.stream_utils import ONE_MB, stream_stats_category, stream_thread_pool

log = logging.getLogger(__name__)

MAX_OUT_SEQ_CHUNKS = 16
# 1/4 of the window size
ACK_INTERVAL = 1024 * 1024 * 4
READ_TIMEOUT = 300
COUNTER_NAME_RECEIVED = "received"


class RxTask:
    """Receiving task for ByteStream"""

    def __init__(self, sid: int, origin: str):
        self.sid = sid
        self.origin = origin
        self.channel = None
        self.topic = None
        self.headers = None
        self.size = 0

        # The reassembled buffer in a double-ended queue
        self.buffers = deque()
        # Out-of-sequence buffers to be assembled
        self.out_seq_buffers: Dict[int, Tuple[bool, BytesAlike]] = {}
        self.stream_future = None
        self.next_seq = 0
        self.offset = 0
        self.offset_ack = 0
        self.eos = False
        self.waiter = threading.Event()
        self.task_lock = threading.Lock()
        self.last_chunk_received = False

    def __str__(self):
        return f"Rx[SID:{self.sid} from {self.origin} for {self.channel}/{self.topic}]"


class RxStream(Stream):
    """A stream that's used to read streams from the buffer"""

    def __init__(self, byte_receiver: "ByteReceiver", task: RxTask):
        super().__init__(task.size, task.headers)
        self.byte_receiver = byte_receiver
        self.task = task

    def read(self, chunk_size: int) -> bytes:
        if self.closed:
            raise StreamError("Read from closed stream")

        if (not self.task.buffers) and self.task.eos:
            return EOS

        # Block if buffers are empty
        count = 0
        while not self.task.buffers:
            if count > 0:
                log.debug(f"Read block is unblocked multiple times: {count}")

            self.task.waiter.clear()
            timeout = CommConfigurator().get_streaming_read_timeout(READ_TIMEOUT)
            if not self.task.waiter.wait(timeout):
                error = StreamError(f"{self.task} read timed out after {timeout} seconds")
                self.byte_receiver.stop_task(self.task, error)
                raise error

            count += 1

        with self.task.task_lock:
            last_chunk, buf = self.task.buffers.popleft()
            if buf is None:
                buf = bytes(0)

            if 0 < chunk_size < len(buf):
                result = buf[0:chunk_size]
                # Put leftover to the head of the queue
                self.task.buffers.appendleft((last_chunk, buf[chunk_size:]))
            else:
                result = buf
                if last_chunk:
                    self.task.eos = True

            self.task.offset += len(result)

            ack_interval = CommConfigurator().get_streaming_ack_interval(ACK_INTERVAL)
            if not self.task.last_chunk_received and (self.task.offset - self.task.offset_ack > ack_interval):
                # Send ACK
                message = Message()
                message.add_headers(
                    {
                        StreamHeaderKey.STREAM_ID: self.task.sid,
                        StreamHeaderKey.DATA_TYPE: StreamDataType.ACK,
                        StreamHeaderKey.OFFSET: self.task.offset,
                    }
                )
                self.byte_receiver.cell.fire_and_forget(STREAM_CHANNEL, STREAM_ACK_TOPIC, self.task.origin, message)
                self.task.offset_ack = self.task.offset

            self.task.stream_future.set_progress(self.task.offset)

            return result

    def close(self):
        if not self.task.stream_future.done():
            self.task.stream_future.set_result(self.task.offset)
        self.closed = True


class ByteReceiver:
    def __init__(self, cell: CoreCell):
        self.cell = cell
        self.cell.register_request_cb(channel=STREAM_CHANNEL, topic=STREAM_DATA_TOPIC, cb=self._data_handler)
        self.registry = Registry()
        self.rx_task_map = {}
        self.map_lock = threading.Lock()

        self.received_stream_counter_pool = StatsPoolManager.add_counter_pool(
            name="Received_Stream_Counters",
            description="Counters of received streams",
            counter_names=[COUNTER_NAME_RECEIVED],
            scope=self.cell.my_info.fqcn,
        )

        self.received_stream_size_pool = StatsPoolManager.add_msg_size_pool(
            "Received_Stream_Sizes", "Sizes of streams received (MBs)", scope=self.cell.my_info.fqcn
        )

    def register_callback(self, channel: str, topic: str, stream_cb: Callable, *args, **kwargs):
        if not callable(stream_cb):
            raise StreamError(f"specified stream_cb {type(stream_cb)} is not callable")

        self.registry.set(channel, topic, Callback(stream_cb, args, kwargs))

    def stop_task(self, task: RxTask, error: StreamError = None, notify=True):

        with self.map_lock:
            self.rx_task_map.pop(task.sid, None)

        if error:
            if task.headers:
                optional = task.headers.get(StreamHeaderKey.OPTIONAL, False)
            else:
                optional = False

            msg = f"Stream error: {error}"
            if optional:
                log.debug(msg)
            else:
                log.error(msg)

            task.stream_future.set_exception(error)

            if notify:
                message = Message()

                message.add_headers(
                    {
                        StreamHeaderKey.STREAM_ID: task.sid,
                        StreamHeaderKey.DATA_TYPE: StreamDataType.ERROR,
                        StreamHeaderKey.ERROR_MSG: str(error),
                    }
                )
                self.cell.fire_and_forget(STREAM_CHANNEL, STREAM_ACK_TOPIC, task.origin, message)

        task.eos = True

    def _data_handler(self, message: Message):

        sid = message.get_header(StreamHeaderKey.STREAM_ID)
        origin = message.get_header(MessageHeaderKey.ORIGIN)
        seq = message.get_header(StreamHeaderKey.SEQUENCE)
        error = message.get_header(StreamHeaderKey.ERROR_MSG, None)

        payload = message.payload

        with self.map_lock:
            task = self.rx_task_map.get(sid, None)
            if not task:
                if error:
                    log.debug(f"Received error for non-existing stream: SID {sid} from {origin}")
                    return

                task = RxTask(sid, origin)
                self.rx_task_map[sid] = task

        if error:
            self.stop_task(task, StreamError(f"Received error from {origin}: {error}"), notify=False)
            return

        if seq == 0:
            # Handle new stream
            task.channel = message.get_header(StreamHeaderKey.CHANNEL)
            task.topic = message.get_header(StreamHeaderKey.TOPIC)
            task.headers = message.headers

            task.stream_future = StreamFuture(sid, message.headers)
            task.size = message.get_header(StreamHeaderKey.SIZE, 0)
            task.stream_future.set_size(task.size)

            # Invoke callback
            callback = self.registry.find(task.channel, task.topic)
            if not callback:
                self.stop_task(task, StreamError(f"No callback is registered for {task.channel}/{task.topic}"))
                return

            self.received_stream_counter_pool.increment(
                category=stream_stats_category(task.channel, task.topic, "stream"), counter_name=COUNTER_NAME_RECEIVED
            )

            self.received_stream_size_pool.record_value(
                category=stream_stats_category(task.channel, task.topic, "stream"), value=task.size / ONE_MB
            )

            stream_thread_pool.submit(self._callback_wrapper, task, callback)

        with task.task_lock:
            data_type = message.get_header(StreamHeaderKey.DATA_TYPE)
            last_chunk = data_type == StreamDataType.FINAL
            if last_chunk:
                task.last_chunk_received = True

            if seq == task.next_seq:
                self._append(task, (last_chunk, payload))
                task.next_seq += 1

                # Try to reassemble out-of-seq buffers
                while task.next_seq in task.out_seq_buffers:
                    chunk = task.out_seq_buffers.pop(task.next_seq)
                    self._append(task, chunk)
                    task.next_seq += 1

            else:
                # Out-of-seq chunk reassembly
                max_out_seq = CommConfigurator().get_streaming_max_out_seq_chunks(MAX_OUT_SEQ_CHUNKS)
                if len(task.out_seq_buffers) >= max_out_seq:
                    self.stop_task(task, StreamError(f"Too many out-of-sequence chunks: {len(task.out_seq_buffers)}"))
                    return
                else:
                    task.out_seq_buffers[seq] = last_chunk, payload

            # If all chunks are lined up, the task can be deleted
            if not task.out_seq_buffers and task.buffers:
                last_chunk, _ = task.buffers[-1]
                if last_chunk:
                    self.stop_task(task)

    def _callback_wrapper(self, task: RxTask, callback: Callback):
        """A wrapper to catch all exceptions in the callback"""
        try:
            stream = RxStream(self, task)
            return callback.cb(task.stream_future, stream, False, *callback.args, **callback.kwargs)
        except Exception as ex:
            msg = f"{task} callback {callback.cb} throws exception: {ex}"
            log.error(msg)
            self.stop_task(task, StreamError(msg))

    @staticmethod
    def _append(task: RxTask, buf: Tuple[bool, BytesAlike]):
        if not buf:
            return

        task.buffers.append(buf)

        # Wake up blocking read()
        if not task.waiter.is_set():
            task.waiter.set()
