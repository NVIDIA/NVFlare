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
from typing import Callable, Deque, Dict, Optional, Tuple

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

# Read result status
RESULT_DATA = 0
RESULT_NO_DATA = 1
RESULT_EOS = 2


class RxTask:
    """Receiving task for ByteStream"""

    rx_task_map = {}
    map_lock = threading.Lock()

    def __init__(self, sid: int, origin: str, cell: CoreCell):
        self.sid = sid
        self.origin = origin
        self.cell = cell

        self.channel = None
        self.topic = None
        self.headers = None
        self.size = 0

        # The reassembled chunks in a double-ended queue
        self.chunks: Deque[Tuple[bool, BytesAlike]] = deque()
        self.chunk_offset = 0  # Start of the remaining data for partially read left-most chunk

        # Out-of-sequence chunks to be assembled
        self.out_seq_chunks: Dict[int, Tuple[bool, BytesAlike]] = {}
        self.stream_future = None
        self.next_seq = 0
        self.offset = 0
        self.offset_ack = 0
        self.waiter = threading.Event()
        self.lock = threading.Lock()
        self.eos = False
        self.last_chunk_received = False

        self.timeout = CommConfigurator().get_streaming_read_timeout(READ_TIMEOUT)
        self.ack_interval = CommConfigurator().get_streaming_ack_interval(ACK_INTERVAL)
        self.max_out_seq = CommConfigurator().get_streaming_max_out_seq_chunks(MAX_OUT_SEQ_CHUNKS)

    def __str__(self):
        return f"Rx[SID:{self.sid} from {self.origin} for {self.channel}/{self.topic} Size: {self.size}]"

    @classmethod
    def find_or_create_task(cls, message: Message, cell: CoreCell) -> Optional["RxTask"]:

        sid = message.get_header(StreamHeaderKey.STREAM_ID)
        origin = message.get_header(MessageHeaderKey.ORIGIN)
        error = message.get_header(StreamHeaderKey.ERROR_MSG, None)

        with cls.map_lock:
            task = cls.rx_task_map.get(sid, None)
            if not task:
                if error:
                    log.warning(f"Received error for non-existing stream: SID {sid} from {origin}")
                    return None

                task = RxTask(sid, origin, cell)
                cls.rx_task_map[sid] = task
            else:
                if error:
                    task.stop(StreamError(f"{task} Received error from {origin}: {error}"), notify=False)
                    return None

        return task

    def read(self, size: int) -> BytesAlike:

        count = 0
        while True:
            result_code, result = self._try_to_read(size)
            if result_code == RESULT_EOS:
                return EOS
            elif result_code == RESULT_DATA:
                return result

            # result_code == RESULT_NO_DATA Block until chunks are received
            if count > 0:
                log.warning(f"{self} Read block is unblocked multiple times: {count}")

            if not self.waiter.wait(self.timeout):
                error = StreamError(f"{self} read timed out after {self.timeout} seconds")
                self.stop(error)
                raise error

            count += 1

    def process_chunk(self, message: Message) -> bool:
        """Returns True if a new stream is created"""

        new_stream = False
        with self.lock:
            seq = message.get_header(StreamHeaderKey.SEQUENCE)
            if seq == 0:
                if self.stream_future:
                    log.warning(f"{self} Received duplicate chunk 0, ignored")
                    return new_stream

                self._handle_new_stream(message)
                new_stream = True

            self._handle_incoming_data(seq, message)
            return new_stream

    def _handle_new_stream(self, message: Message):
        self.channel = message.get_header(StreamHeaderKey.CHANNEL)
        self.topic = message.get_header(StreamHeaderKey.TOPIC)
        self.headers = message.headers
        self.size = message.get_header(StreamHeaderKey.SIZE, 0)

        self.stream_future = StreamFuture(self.sid, self.headers)
        self.stream_future.set_size(self.size)

    def _handle_incoming_data(self, seq: int, message: Message):

        data_type = message.get_header(StreamHeaderKey.DATA_TYPE)

        last_chunk = data_type == StreamDataType.FINAL
        if last_chunk:
            self.last_chunk_received = True

        if seq < self.next_seq:
            log.warning(f"{self} Duplicate chunk ignored {seq=}")
            return

        if seq == self.next_seq:
            self._append((last_chunk, message.payload))

            # Try to reassemble out-of-seq chunks
            while self.next_seq in self.out_seq_chunks:
                chunk = self.out_seq_chunks.pop(self.next_seq)
                self._append(chunk)
        else:
            # Save out-of-seq chunks
            if len(self.out_seq_chunks) >= self.max_out_seq:
                self.stop(StreamError(f"{self} Too many out-of-sequence chunks: {len(self.out_seq_chunks)}"))
                return
            else:
                if seq not in self.out_seq_chunks:
                    self.out_seq_chunks[seq] = last_chunk, message.payload
                else:
                    log.warning(f"{self} Duplicate out-of-seq chunk ignored {seq=}")

        # If all chunks are lined up and last chunk received, the task can be deleted
        if not self.out_seq_chunks and self.chunks:
            last_chunk, _ = self.chunks[-1]
            if last_chunk:
                self.stop()

    def stop(self, error: StreamError = None, notify=True):

        with RxTask.map_lock:
            RxTask.rx_task_map.pop(self.sid, None)

        if not error:
            return

        if self.headers:
            optional = self.headers.get(StreamHeaderKey.OPTIONAL, False)
        else:
            optional = False

        msg = f"Stream error: {error}"
        if optional:
            log.debug(msg)
        else:
            log.error(msg)

        self.stream_future.set_exception(error)

        if notify:
            message = Message()

            message.add_headers(
                {
                    StreamHeaderKey.STREAM_ID: self.sid,
                    StreamHeaderKey.DATA_TYPE: StreamDataType.ERROR,
                    StreamHeaderKey.ERROR_MSG: str(error),
                }
            )
            self.cell.fire_and_forget(STREAM_CHANNEL, STREAM_ACK_TOPIC, self.origin, message)

    def _try_to_read(self, size: int) -> Tuple[int, Optional[BytesAlike]]:

        with self.lock:
            if self.eos:
                return RESULT_EOS, None

            if not self.chunks:
                self.waiter.clear()
                return RESULT_NO_DATA, None

            # Get the left most chunk
            last_chunk, buf = self.chunks[0]
            if buf is None:
                buf = bytes(0)
            end_offset = self.chunk_offset + size
            if 0 < end_offset < len(buf):
                # Partial read
                result = buf[self.chunk_offset : end_offset]
                self.chunk_offset = end_offset
            else:
                # Whole chunk is consumed
                if self.chunk_offset:
                    result = buf[self.chunk_offset :]
                else:
                    result = buf

                self.chunk_offset = 0
                self.chunks.popleft()

                if last_chunk:
                    self.eos = True

            self.offset += len(result)

            if not self.last_chunk_received and (self.offset - self.offset_ack > self.ack_interval):
                # Send ACK
                message = Message()
                message.add_headers(
                    {
                        StreamHeaderKey.STREAM_ID: self.sid,
                        StreamHeaderKey.DATA_TYPE: StreamDataType.ACK,
                        StreamHeaderKey.OFFSET: self.offset,
                    }
                )
                self.cell.fire_and_forget(STREAM_CHANNEL, STREAM_ACK_TOPIC, self.origin, message)
                self.offset_ack = self.offset

            self.stream_future.set_progress(self.offset)

            return RESULT_DATA, result

    def _append(self, buf: Tuple[bool, BytesAlike]):
        if self.eos:
            log.error(f"{self} Data after EOS is ignored")
            return

        self.chunks.append(buf)
        self.next_seq += 1

        # Wake up blocking read()
        if not self.waiter.is_set():
            self.waiter.set()


class RxStream(Stream):
    """A stream that's used to read streams from the streaming task"""

    def __init__(self, task: RxTask):
        super().__init__(task.size, task.headers)
        self.task = task

    def read(self, size: int) -> bytes:
        if self.closed:
            raise StreamError("Read from closed stream")

        return self.task.read(size)

    def close(self):
        if not self.task.stream_future.done():
            self.task.stream_future.set_result(self.task.offset)
        self.closed = True


class ByteReceiver:

    received_stream_counter_pool = StatsPoolManager.add_counter_pool(
        name="Received_Stream_Counters",
        description="Counters of received streams",
        counter_names=[COUNTER_NAME_RECEIVED],
    )

    received_stream_size_pool = StatsPoolManager.add_msg_size_pool(
        "Received_Stream_Sizes", "Sizes of streams received (MBs)"
    )

    def __init__(self, cell: CoreCell):
        self.cell = cell
        self.cell.register_request_cb(channel=STREAM_CHANNEL, topic=STREAM_DATA_TOPIC, cb=self._data_handler)
        self.registry = Registry()

    def register_callback(self, channel: str, topic: str, stream_cb: Callable, *args, **kwargs):
        if not callable(stream_cb):
            raise StreamError(f"specified stream_cb {type(stream_cb)} is not callable")

        self.registry.set(channel, topic, Callback(stream_cb, args, kwargs))

    def _data_handler(self, message: Message):

        task = RxTask.find_or_create_task(message, self.cell)
        if not task:
            return

        new_stream = task.process_chunk(message)
        if new_stream:
            # Invoke callback
            callback = self.registry.find(task.channel, task.topic)
            if not callback:
                task.stop(StreamError(f"{task} No callback is registered for {task.channel}/{task.topic}"))
                return

            fqcn = self.cell.my_info.fqcn
            ByteReceiver.received_stream_counter_pool.increment(
                category=stream_stats_category(fqcn, task.channel, task.topic, "stream"),
                counter_name=COUNTER_NAME_RECEIVED,
            )

            ByteReceiver.received_stream_size_pool.record_value(
                category=stream_stats_category(fqcn, task.channel, task.topic, "stream"),
                value=task.size / ONE_MB,
            )

            stream_thread_pool.submit(self._callback_wrapper, task, callback)

    @staticmethod
    def _callback_wrapper(task: RxTask, callback: Callback):
        """A wrapper to catch all exceptions in the callback"""
        try:
            stream = RxStream(task)
            return callback.cb(task.stream_future, stream, False, *callback.args, **callback.kwargs)
        except Exception as ex:
            msg = f"{task} callback {callback.cb} throws exception: {ex}"
            log.error(msg)
            task.stop(StreamError(msg))
