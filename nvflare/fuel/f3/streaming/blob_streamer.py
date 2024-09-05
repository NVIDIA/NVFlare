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
from nvflare.fuel.f3.streaming.byte_streamer import STREAM_TYPE_BLOB, ByteStreamer
from nvflare.fuel.f3.streaming.stream_const import EOS
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamError, StreamFuture
from nvflare.fuel.f3.streaming.stream_utils import FastBuffer, stream_thread_pool, wrap_view
from nvflare.fuel.utils.buffer_list import BufferList
from nvflare.security.logging import secure_format_traceback

log = logging.getLogger(__name__)


class BlobStream(Stream):
    def __init__(self, blob: BytesAlike, headers: Optional[dict]):
        size = self.buffer_len(blob)
        super().__init__(size, headers)

        if not isinstance(blob, list):
            self.blob_view = wrap_view(blob)
            self.buffer_list = None
        else:
            self.blob_view = [wrap_view(b) for b in blob]
            self.buffer_list = BufferList(self.blob_view)

    def read(self, chunk_size: int) -> BytesAlike:

        if self.pos >= self.get_size():
            return EOS

        next_pos = self.pos + chunk_size
        if next_pos > self.get_size():
            next_pos = self.get_size()

        if self.buffer_list:
            buf = self.buffer_list.read(self.pos, next_pos)
        else:
            buf = self.blob_view[self.pos : next_pos]

        self.pos = next_pos

        return buf

    @staticmethod
    def buffer_len(buffer: BytesAlike):
        if not isinstance(buffer, list):
            return len(buffer)

        return sum(len(buf) for buf in buffer)


class BlobTask:
    def __init__(self, future: StreamFuture, stream: Stream):
        self.future = future
        self.stream = stream
        self.size = stream.get_size()
        self.pre_allocated = self.size > 0

        if self.pre_allocated:
            self.buffer = wrap_view(bytearray(self.size))
        else:
            self.buffer = FastBuffer()


class BlobHandler:
    def __init__(self, blob_cb: Callable):
        self.blob_cb = blob_cb

    def handle_blob_cb(self, future: StreamFuture, stream: Stream, resume: bool, *args, **kwargs) -> int:

        if resume:
            log.warning("Resume is not supported, ignored")

        blob_task = BlobTask(future, stream)

        stream_thread_pool.submit(self._read_stream, blob_task)

        self.blob_cb(future, *args, **kwargs)

        return 0

    @staticmethod
    def _read_stream(blob_task: BlobTask):

        try:
            # It's most efficient to use the same chunk size as the stream
            chunk_size = ByteStreamer.get_chunk_size()

            buf_size = 0
            while True:
                buf = blob_task.stream.read(chunk_size)
                if not buf:
                    break

                length = len(buf)
                try:
                    if blob_task.pre_allocated:
                        blob_task.buffer[buf_size : buf_size + length] = buf
                    else:
                        blob_task.buffer.append(buf)
                except Exception as ex:
                    log.error(
                        f"memory view error: {ex} "
                        f"Debug info: {length=} {buf_size=} {len(blob_task.pre_allocated)=} {type(buf)=}"
                    )
                    raise ex

                buf_size += length

            if blob_task.size and blob_task.size != buf_size:
                log.warning(
                    f"Stream {blob_task.future.get_stream_id()} size doesn't match: " f"{blob_task.size} <> {buf_size}"
                )

            if blob_task.pre_allocated:
                result = blob_task.buffer
            else:
                result = blob_task.buffer.to_bytes()

            blob_task.future.set_result(result)
        except Exception as ex:
            log.error(f"Stream {blob_task.future.get_stream_id()} read error: {ex}")
            log.error(secure_format_traceback())
            blob_task.future.set_exception(ex)


class BlobStreamer:
    def __init__(self, byte_streamer: ByteStreamer, byte_receiver: ByteReceiver):
        self.byte_streamer = byte_streamer
        self.byte_receiver = byte_receiver

    def send(
        self, channel: str, topic: str, target: str, message: Message, secure: bool, optional: bool
    ) -> StreamFuture:
        if message.payload is None:
            message.payload = bytes(0)

        if not isinstance(message.payload, (bytes, bytearray, memoryview, list)):
            raise StreamError(f"BLOB is invalid type: {type(message.payload)}")

        blob_stream = BlobStream(message.payload, message.headers)
        return self.byte_streamer.send(
            channel, topic, target, message.headers, blob_stream, STREAM_TYPE_BLOB, secure, optional
        )

    def register_blob_callback(self, channel, topic, blob_cb: Callable, *args, **kwargs):
        handler = BlobHandler(blob_cb)
        self.byte_receiver.register_callback(channel, topic, handler.handle_blob_cb, *args, **kwargs)
