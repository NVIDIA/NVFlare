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
from typing import Callable, Optional

from nvflare.fuel.f3.connection import BytesAlike
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.byte_receiver import ByteReceiver
from nvflare.fuel.f3.streaming.byte_streamer import ByteStreamer
from nvflare.fuel.f3.streaming.stream_const import EOS
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamFuture
from nvflare.fuel.f3.streaming.stream_utils import stream_thread_pool, wrap_view
from nvflare.security.logging import secure_format_traceback

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


class BlobHandler:
    def __init__(self, blob_cb: Callable):
        self.blob_cb = blob_cb
        self.size = 0
        self.buffer = None

    def handle_blob_cb(self, future: StreamFuture, stream: Stream, resume: bool, *args, **kwargs) -> int:

        if resume:
            log.warning("Resume is not supported, ignored")

        self.size = stream.get_size()

        if self.size > 0:
            self.buffer = bytearray(self.size)
        else:
            self.buffer = bytes()

        stream_thread_pool.submit(self._read_stream, future, stream)

        self.blob_cb(future, *args, **kwargs)

        return 0

    def _read_stream(self, future: StreamFuture, stream: Stream):

        try:
            chunk_size = ByteStreamer.get_chunk_size()

            buf_size = 0
            while True:
                buf = stream.read(chunk_size)
                if not buf:
                    break

                length = len(buf)
                if self.size > 0:
                    self.buffer[buf_size : buf_size + length] = buf
                else:
                    self.buffer += buf

                buf_size += length

            if self.size and (self.size != buf_size):
                log.warning(f"Stream size doesn't match: {self.size} <> {buf_size}")

            future.set_result(self.buffer)
        except Exception as ex:
            log.error(f"Stream {future.get_stream_id()} read error: {ex}")
            log.debug(secure_format_traceback())
            future.set_exception(ex)


class BlobStreamer:
    def __init__(self, byte_streamer: ByteStreamer, byte_receiver: ByteReceiver):
        self.byte_streamer = byte_streamer
        self.byte_receiver = byte_receiver

    def send(self, channel: str, topic: str, target: str, message: Message) -> StreamFuture:
        blob_stream = BlobStream(message.payload, message.headers)
        return self.byte_streamer.send(channel, topic, target, message.headers, blob_stream)

    def register_blob_callback(self, channel, topic, blob_cb: Callable, *args, **kwargs):
        handler = BlobHandler(blob_cb)
        self.byte_receiver.register_callback(channel, topic, handler.handle_blob_cb, *args, **kwargs)
