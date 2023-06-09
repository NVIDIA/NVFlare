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
from typing import Any, Callable, Optional

from nvflare.fuel.f3.connection import BytesAlike
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.byte_receiver import ByteReceiver
from nvflare.fuel.f3.streaming.byte_streamer import ByteStreamer
from nvflare.fuel.f3.streaming.stream_const import EOS
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamFuture
from nvflare.fuel.f3.streaming.stream_utils import stream_thread_pool, wrap_view

log = logging.getLogger(__name__)


class BlobStream(Stream):
    def __init__(self, blob: BytesAlike, headers: Optional[dict]):
        super().__init__(len(blob), headers)
        self.blob_view = wrap_view(blob)

    def read(self, chunk_size: int) -> BytesAlike:

        if self.pos >= self.get_size():
            return EOS

        next_pos = self.pos + chunk_size
        if next_pos > self.get_size():
            next_pos = self.get_size()
        buf = self.blob_view[self.pos : next_pos]
        self.pos = next_pos
        return buf


class BlobFuture(StreamFuture):
    """The future used by BLOB callback. Unlike StreamFuture, its result is the blob.
    All other calls are delegated to StreamFuture
    """

    def __init__(self, future: StreamFuture, buffer: BytesAlike):
        super().__init__(future.get_stream_id(), future.get_headers())
        self.future = future
        self.buffer = buffer

    def result(self, timeout=None) -> Any:
        self.future.result()
        return self.buffer

    def get_stream_id(self) -> str:
        return self.future.get_stream_id()

    def get_headers(self) -> Optional[dict]:
        return self.future.get_headers()

    def get_size(self) -> int:
        return self.future.get_size()

    def set_size(self, size: int):
        self.future.set_size(size)

    def get_progress(self) -> int:
        return self.future.get_progress()

    def set_progress(self, progress: int):
        self.future.set_progress(progress)

    def cancel(self):
        self.future.cancel()

    def cancelled(self):
        return self.future.cancelled()

    def running(self):
        return self.future.running()

    def done(self):
        return self.future.done()

    def add_done_callback(self, done_cb: Callable, *args, **kwargs):
        self.future.add_done_callback(done_cb, *args, **kwargs)

    def exception(self, timeout=None):
        return self.future.exception(timeout)

    def set_result(self, value: Any):
        self.future.set_result(value)

    def set_exception(self, exception):
        self.future.set_exception(exception)


class BlobHandler:
    def __init__(self, blob_cb: Callable):
        self.blob_cb = blob_cb
        self.size = 0
        self.buffer = None

    def handle_blob_cb(self, future: StreamFuture, stream: Stream, resume: bool, *args, **kwargs) -> int:

        if resume:
            log.warning("Resume is not supported, ignored")

        self.size = stream.get_size()
        self.buffer = bytearray()

        blob_future = BlobFuture(future, self.buffer)
        stream_thread_pool.submit(self._read_stream, blob_future, stream)

        self.blob_cb(blob_future, *args, **kwargs)

        return 0

    def _read_stream(self, future: BlobFuture, stream: Stream):

        chunk_size = ByteStreamer.get_chunk_size()

        buf_size = 0
        while True:
            buf = stream.read(chunk_size)
            if not buf:
                break

            buf_size += len(buf)
            self.buffer.append(buf)

        if self.size and (self.size != buf_size):
            log.warning(f"Stream size doesn't match: {self.size} <> {buf_size}")

        future.set_result(buf_size)


class BlobStreamer:
    def __init__(self, byte_streamer: ByteStreamer, byte_receiver: ByteReceiver):
        self.byte_streamer = byte_streamer
        self.byte_receiver = byte_receiver

    def send(self, channel: str, topic: str, target: str, headers: dict, blob: BytesAlike) -> StreamFuture:
        blob_stream = BlobStream(blob, headers)
        return self.byte_streamer.send(channel, topic, target, headers, blob_stream)

    def register_blob_callback(self, channel, topic, blob_cb: Callable, *args, **kwargs):
        handler = BlobHandler(blob_cb)
        self.byte_receiver.register_callback(channel, topic, handler.handle_blob_cb, *args, **kwargs)
