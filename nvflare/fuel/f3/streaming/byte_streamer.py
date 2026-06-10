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
import time
from typing import Callable, Optional

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.mpm import MainProcessMonitor
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.fuel.f3.streaming.stream_const import (
    STREAM_ACK_TOPIC,
    STREAM_CHANNEL,
    STREAM_DATA_TOPIC,
    StreamDataType,
    StreamHeaderKey,
)
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamError, StreamFuture, StreamTaskSpec
from nvflare.fuel.f3.streaming.stream_utils import (
    ONE_MB,
    CheckedExecutor,
    gen_stream_id,
    stream_stats_category,
    stream_thread_pool,
    wrap_view,
)

STREAM_CHUNK_SIZE = 1024 * 1024
STREAM_WINDOW_SIZE = 16 * STREAM_CHUNK_SIZE
STREAM_ACK_WAIT = 300
STREAM_RETRY_WAIT = 5.0
STREAM_RETRY_TIMEOUT = 60.0
STREAM_RETRY_WORKERS = 32

STREAM_TYPE_BYTE = "byte"
STREAM_TYPE_BLOB = "blob"
STREAM_TYPE_FILE = "file"

COUNTER_NAME_SENT = "sent"

log = logging.getLogger(__name__)


def _payload_size(payload) -> int:
    if payload is None:
        return 0

    if isinstance(payload, list):
        return sum(len(item) for item in payload)

    return len(payload)


def _snapshot_payload(payload):
    if payload is None:
        return None

    if isinstance(payload, list):
        return [bytes(item) for item in payload]

    return bytes(payload)


class ReliableRetryScheduler:
    def __init__(self):
        self.tasks = {}
        self.cv = threading.Condition()
        self.thread = None
        self.stopped = False
        self.generation = 0
        self.retry_task_pool = CheckedExecutor(STREAM_RETRY_WORKERS, "stm_retry")

    def register(self, task):
        with self.cv:
            if self.stopped:
                return

            self.tasks[task.sid] = task
            self.generation += 1
            if not self.thread or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._run, name="stm_retry", daemon=True)
                self.thread.start()
            self.cv.notify()

    def unregister(self, task):
        with self.cv:
            registered = self.tasks.get(task.sid)
            if registered is task:
                self.tasks.pop(task.sid, None)
                self.generation += 1
                self.cv.notify()

    def wakeup(self):
        with self.cv:
            self.generation += 1
            self.cv.notify()

    def shutdown(self):
        with self.cv:
            self.stopped = True
            self.generation += 1
            self.cv.notify()

        thread = self.thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self.retry_task_pool.shutdown(wait=True)

    def _run(self):
        while True:
            with self.cv:
                if self.stopped:
                    return

                tasks = list(self.tasks.values())
                generation = self.generation

            next_wait = None
            futures = [self.retry_task_pool.submit(task.retry_task) for task in tasks]
            for future in futures:
                if future is None:
                    continue

                wait_time = future.result()
                if wait_time is not None:
                    next_wait = wait_time if next_wait is None else min(next_wait, wait_time)

            with self.cv:
                if self.stopped:
                    return

                if self.generation == generation:
                    self.cv.wait(timeout=next_wait)


reliable_retry_scheduler = ReliableRetryScheduler()
MainProcessMonitor.add_cleanup_cb(reliable_retry_scheduler.shutdown)


class TxTask(StreamTaskSpec):
    def __init__(
        self,
        cell: CoreCell,
        chunk_size: int,
        channel: str,
        topic: str,
        target: str,
        headers: dict,
        stream: Stream,
        reliable: Optional[bool],
        secure: bool,
        optional: bool,
    ):
        self.cell = cell
        self.chunk_size = chunk_size
        self.sid = gen_stream_id()
        self.buffer = wrap_view(bytearray(chunk_size))
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
        self.seq_ack = -1
        self.offset = 0
        self.offset_ack = 0
        self.secure = secure
        self.optional = optional
        self.stopped = False
        self.stopping = False
        self.send_lock = threading.RLock()

        self.stream_future = StreamFuture(self.sid, task_handle=self)
        self.stream_future.set_size(stream.get_size())

        config = CommConfigurator()
        self.reliable = config.get_streaming_reliable(False) if reliable is None else reliable
        self.window_size = config.get_streaming_window_size(STREAM_WINDOW_SIZE)
        self.ack_wait = config.get_streaming_ack_wait(STREAM_ACK_WAIT)
        self.ack_progress_timeout = config.get_streaming_ack_progress_timeout(60.0)
        # Guard against zero/negative config to avoid wait(0) busy-spin loops.
        self.ack_progress_check_interval = max(0.01, config.get_streaming_ack_progress_check_interval(5.0))
        self.last_ack_progress_ts = time.monotonic()
        self.retry_wait = max(0.01, config.get_streaming_retry_wait(STREAM_RETRY_WAIT))
        self.retry_timeout = config.get_streaming_retry_timeout(STREAM_RETRY_TIMEOUT)
        self.retry_max_pending_bytes = config.get_streaming_retry_max_pending_bytes(2 * self.window_size)

        if self.reliable:
            self.pending_messages = {}
            self.pending_message_bytes = 0
            self.retry_lock = threading.RLock()
            reliable_retry_scheduler.register(self)
        else:
            self.pending_messages = None
            self.pending_message_bytes = 0
            self.retry_lock = None

    def __str__(self):
        return f"Tx[SID:{self.sid} to {self.target} for {self.channel}/{self.topic}]"

    def send_loop(self):
        """Read/send loop to transmit the whole stream with flow control"""

        while not self.stopped:
            buf = self.stream.read(self.chunk_size)
            if not buf:
                # End of Stream
                if not self.send_pending_buffer(final=True):
                    return
                self.stop()
                return

            # Flow control
            window = self.offset - self.offset_ack
            # It may take several ACKs to clear up the window
            while window > self.window_size:
                log.debug(f"{self} window size {window} exceeds limit: {self.window_size}")
                wait_start = time.monotonic()

                while window > self.window_size:
                    now = time.monotonic()
                    if now - self.last_ack_progress_ts >= self.ack_progress_timeout:
                        self.stop(StreamError(f"{self} ACK made no progress for {self.ack_progress_timeout} seconds"))
                        return

                    elapsed = now - wait_start
                    if elapsed >= self.ack_wait:
                        self.stop(StreamError(f"{self} ACK timeouts after {self.ack_wait} seconds"))
                        return

                    self.ack_waiter.clear()
                    wait_timeout = min(self.ack_progress_check_interval, self.ack_wait - elapsed)
                    self.ack_waiter.wait(timeout=wait_timeout)
                    window = self.offset - self.offset_ack

            size = len(buf)
            if size > self.chunk_size:
                raise StreamError(f"{self} Stream returns invalid size: {size}")

            # Don't push out chunk when it's equal, wait till next round to detect EOS
            # For example, if the stream size is chunk size (1M), this avoids sending two chunks.
            if size + self.buffer_size > self.chunk_size:
                if not self.send_pending_buffer():
                    return

            if size == self.chunk_size:
                self.direct_buf = buf
            else:
                self.buffer[self.buffer_size : self.buffer_size + size] = buf
            self.buffer_size += size

    def send_pending_buffer(self, final=False):

        if self.buffer_size == 0:
            payload = bytes(0)
        elif self.buffer_size == self.chunk_size:
            if self.direct_buf:
                payload = self.direct_buf
            else:
                payload = self.buffer
        else:
            payload = self.buffer[0 : self.buffer_size]

        if self.reliable:
            payload = _snapshot_payload(payload)

        message = Message(None, payload)

        if self.headers:
            message.add_headers(self.headers)

        stream_headers = {
            StreamHeaderKey.CHANNEL: self.channel,
            StreamHeaderKey.TOPIC: self.topic,
            StreamHeaderKey.SIZE: self.stream.get_size(),
            StreamHeaderKey.STREAM_ID: self.sid,
            StreamHeaderKey.DATA_TYPE: StreamDataType.FINAL if final else StreamDataType.CHUNK,
            StreamHeaderKey.SEQUENCE: self.seq,
            StreamHeaderKey.OFFSET: self.offset,
            StreamHeaderKey.RELIABLE: self.reliable,
            StreamHeaderKey.OPTIONAL: self.optional,
        }
        if self.reliable and self.seq == 0:
            stream_headers[StreamHeaderKey.RETRY_WAIT] = self.retry_wait
            stream_headers[StreamHeaderKey.RETRY_TIMEOUT] = self.retry_timeout
        message.add_headers(stream_headers)

        if self.reliable:
            errors = None
            over_limit_error = None
            with self.send_lock:
                curr_time = time.monotonic()
                with self.retry_lock:
                    if self.stopped:
                        return False

                    pending_message_size = _payload_size(message.payload)
                    self.pending_messages[self.seq] = curr_time, curr_time, message
                    self.pending_message_bytes += pending_message_size
                    if self.retry_max_pending_bytes > 0 and self.pending_message_bytes > self.retry_max_pending_bytes:
                        self.pending_messages.pop(self.seq, None)
                        self.pending_message_bytes -= pending_message_size
                        msg = (
                            f"{self} has too many retry messages "
                            f"({self.pending_message_bytes + pending_message_size} > {self.retry_max_pending_bytes})"
                        )
                        over_limit_error = StreamError(msg)

                if not over_limit_error:
                    reliable_retry_scheduler.wakeup()
                    errors = self.cell.fire_and_forget(
                        STREAM_CHANNEL,
                        STREAM_DATA_TOPIC,
                        self.target,
                        message,
                        secure=self.secure,
                        optional=self.optional,
                    )

            if over_limit_error:
                log.error(str(over_limit_error))
                self.stop(over_limit_error)
                return False
        else:
            errors = self.cell.fire_and_forget(
                STREAM_CHANNEL, STREAM_DATA_TOPIC, self.target, message, secure=self.secure, optional=self.optional
            )
        errors = errors or {}
        error = errors.get(self.target)
        if error:
            msg = f"{self} Message sending error to target {self.target}: {error}"
            if self.reliable:
                log.error(f"{msg}, will retry in {self.retry_wait} seconds")
            else:
                self.stop(StreamError(msg))
                return False

        # Update state
        self.seq += 1
        self.offset += self.buffer_size
        self.buffer_size = 0
        self.direct_buf = None

        # Update future
        self.stream_future.set_progress(self.offset)
        return True

    def stop(self, error: Optional[StreamError] = None, notify=True):

        if self.reliable:
            if error:
                with self.send_lock:
                    if not self._prepare_reliable_stop(error):
                        return
            elif not self._prepare_reliable_stop(error):
                return
            reliable_retry_scheduler.unregister(self)
        else:
            if self.stopped:
                return
            self.stopped = True

        self.remove_task()
        if not self.ack_waiter.is_set():
            self.ack_waiter.set()

        if self.task_future:
            self.task_future.cancel()

        if not error:
            # Result is the number of bytes streamed
            if self.stream_future:
                self.stream_future.set_result(self.offset)
            return

        # Error handling
        log.debug(f"{self} Stream error: {error}")
        if self.stream_future:
            self.stream_future.set_exception(error)

        if notify:
            message = Message(None, None)

            if self.headers:
                message.add_headers(self.headers)

            message.add_headers(
                {
                    StreamHeaderKey.STREAM_ID: self.sid,
                    StreamHeaderKey.DATA_TYPE: StreamDataType.ERROR,
                    StreamHeaderKey.OFFSET: self.offset,
                    StreamHeaderKey.ERROR_MSG: str(error),
                }
            )
            try:
                self.cell.fire_and_forget(
                    STREAM_CHANNEL, STREAM_DATA_TOPIC, self.target, message, secure=self.secure, optional=True
                )
            except Exception as ex:
                log.error(f"{self} failed to notify stream error to target {self.target}: {ex}")

    def _prepare_reliable_stop(self, error: Optional[StreamError]) -> bool:
        with self.retry_lock:
            if self.stopped:
                return False

            if not error and self.pending_messages:
                self.stopping = True
                reliable_retry_scheduler.wakeup()
                if not self.ack_waiter.is_set():
                    self.ack_waiter.set()
                return False

            self.stopped = True
            self.stopping = False
            if error:
                self.pending_messages.clear()
                self.pending_message_bytes = 0
            return True

    def handle_ack(self, message: Message):

        origin = message.get_header(MessageHeaderKey.ORIGIN)
        ack_seq = message.get_header(StreamHeaderKey.SEQUENCE, None)
        offset = message.get_header(StreamHeaderKey.OFFSET, None)
        error = message.get_header(StreamHeaderKey.ERROR_MSG, None)

        if error:
            self.stop(StreamError(f"{self} Received error from {origin}: {error}"), notify=False)
            return

        if self.reliable and ack_seq is None:
            self.stop(StreamError(f"{self} receiving end at {origin} doesn't support reliable streaming"), notify=True)
            return

        if self.reliable:
            should_stop = False
            ack_progressed = False
            with self.retry_lock:
                if offset is not None and offset > self.offset_ack:
                    self.offset_ack = offset
                    ack_progressed = True

                if ack_seq is not None and ack_seq > self.seq_ack:
                    self.seq_ack = ack_seq
                    ack_progressed = True

                if ack_progressed:
                    self.last_ack_progress_ts = time.monotonic()

                if self.pending_messages and ack_seq is not None:
                    for seq in list(self.pending_messages):
                        if seq <= ack_seq:
                            _start_time, _last_retry, message = self.pending_messages.pop(seq)
                            self.pending_message_bytes -= _payload_size(message.payload)

                should_stop = self.stopping and not self.pending_messages

            if should_stop:
                self.stop()
        elif offset is not None and offset > self.offset_ack:
            self.offset_ack = offset
            self.last_ack_progress_ts = time.monotonic()

        if not self.ack_waiter.is_set():
            self.ack_waiter.set()

    def start_task_thread(self, task_handler: Callable):
        self.task_future = stream_thread_pool.submit(task_handler, self)

    def cancel(self):
        self.stop(error=StreamError("cancelled"))

    def retry_task(self) -> Optional[float]:
        try:
            return self._retry_task()

        except Exception as ex:
            msg = f"{self} retry thread ended due to error: {ex}"
            log.error(msg)
            self.stop(StreamError(msg), notify=True)
            return None

    def _retry_task(self) -> Optional[float]:
        should_stop = False
        next_wait = None
        messages_to_retry = []
        retry_error = None

        with self.retry_lock:
            if self.stopped:
                return None

            if not self.pending_messages:
                should_stop = self.stopping
            else:
                curr_time = time.monotonic()
                for seq, value in list(self.pending_messages.items()):
                    start_time, last_retry, message = value
                    retry_time = curr_time - start_time
                    if retry_time > self.retry_timeout:
                        msg = f"{self} seq {seq} retry failed after {retry_time:.2f} seconds"
                        log.error(msg)
                        retry_error = StreamError(msg)
                        break

                    wait_time = self.retry_wait - (curr_time - last_retry)
                    if wait_time <= 0:
                        messages_to_retry.append((seq, message))
                        self.pending_messages[seq] = start_time, curr_time, message
                    else:
                        next_wait = wait_time if next_wait is None else min(next_wait, wait_time)

        if retry_error:
            self.stop(error=retry_error)
            return None

        if should_stop:
            self.stop()
            return None

        for seq, message in messages_to_retry:
            errors = self.cell.fire_and_forget(
                STREAM_CHANNEL,
                STREAM_DATA_TOPIC,
                self.target,
                message,
                secure=self.secure,
                optional=self.optional,
            )
            errors = errors or {}
            error = errors.get(self.target)
            if error:
                log.error(
                    f"{self} message retry error for target {self.target} seq {seq}: "
                    f"{error}, will retry again in {self.retry_wait} seconds"
                )

        if messages_to_retry:
            next_wait = self.retry_wait if next_wait is None else min(next_wait, self.retry_wait)

        return next_wait

    def remove_task(self):
        with ByteStreamer.map_lock:
            ByteStreamer.tx_task_map.pop(self.sid, None)
            log.debug(f"{self} is removed")


class ByteStreamer:

    tx_task_map = {}
    map_lock = threading.Lock()

    sent_stream_counter_pool = StatsPoolManager.add_counter_pool(
        name="Sent_Stream_Counters",
        description="Counters of sent streams",
        counter_names=[COUNTER_NAME_SENT],
    )

    sent_stream_size_pool = StatsPoolManager.add_msg_size_pool("Sent_Stream_Sizes", "Sizes of streams sent (MBs)")

    def __init__(self, cell: CoreCell):
        self.cell = cell
        self.cell.register_request_cb(channel=STREAM_CHANNEL, topic=STREAM_ACK_TOPIC, cb=self._ack_handler)
        self.chunk_size = CommConfigurator().get_streaming_chunk_size(STREAM_CHUNK_SIZE)

    def get_chunk_size(self):
        return self.chunk_size

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
        reliable: Optional[bool] = None,
    ) -> StreamFuture:
        tx_task = TxTask(
            self.cell, self.chunk_size, channel, topic, target, headers, stream, reliable, secure, optional
        )
        with ByteStreamer.map_lock:
            ByteStreamer.tx_task_map[tx_task.sid] = tx_task

        tx_task.start_task_thread(self._transmit_task)

        fqcn = self.cell.my_info.fqcn
        ByteStreamer.sent_stream_counter_pool.increment(
            category=stream_stats_category(fqcn, channel, topic, stream_type), counter_name=COUNTER_NAME_SENT
        )

        ByteStreamer.sent_stream_size_pool.record_value(
            category=stream_stats_category(fqcn, channel, topic, stream_type), value=stream.get_size() / ONE_MB
        )

        return tx_task.stream_future

    @staticmethod
    def _transmit_task(task: TxTask):

        try:
            task.send_loop()
        except Exception as ex:
            msg = f"{task} Error while sending: {ex}"
            if task.optional:
                log.debug(msg)
            else:
                log.error(msg)
            task.stop(StreamError(msg), True)

    @staticmethod
    def _ack_handler(message: Message):

        sid = message.get_header(StreamHeaderKey.STREAM_ID)
        with ByteStreamer.map_lock:
            tx_task = ByteStreamer.tx_task_map.get(sid, None)

        if not tx_task:
            origin = message.get_header(MessageHeaderKey.ORIGIN)
            offset = message.get_header(StreamHeaderKey.OFFSET, None)
            seq = message.get_header(StreamHeaderKey.SEQUENCE, None)
            # Last few ACKs always arrive late so this is normal
            log.debug(f"ACK for stream {sid} received late from {origin} with offset {offset} seq {seq}")
            return

        tx_task.handle_ack(message)
