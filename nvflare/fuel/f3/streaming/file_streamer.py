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
import os
from pathlib import Path
from typing import Callable, Optional

from nvflare.fuel.f3.connection import BytesAlike
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.byte_receiver import ByteReceiver
from nvflare.fuel.f3.streaming.byte_streamer import STREAM_TYPE_FILE, ByteStreamer
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamFuture
from nvflare.fuel.f3.streaming.stream_utils import stream_thread_pool

log = logging.getLogger(__name__)


class FileStream(Stream):
    def __init__(self, file_name: str, headers: Optional[dict]):
        self.file = open(file_name, "rb")
        size = self.file.seek(0, os.SEEK_END)
        self.file.seek(0, os.SEEK_SET)
        super().__init__(size, headers)

    def read(self, chunk_size: int) -> BytesAlike:
        return self.file.read(chunk_size)

    def close(self):
        self.closed = True
        self.file.close()


class FileHandler:
    def __init__(self, file_cb: Callable):
        self.file_cb = file_cb
        self.size = 0
        self.file_name = None

    def handle_file_cb(self, future: StreamFuture, stream: Stream, resume: bool, *args, **kwargs) -> int:

        if resume:
            log.warning("Resume is not supported, ignored")

        self.size = stream.get_size()
        original_name = future.headers.get(StreamHeaderKey.FILE_NAME)

        file_name = self.file_cb(future, original_name, *args, **kwargs)
        stream_thread_pool.submit(self._write_to_file, file_name, future, stream)

        return 0

    def _write_to_file(self, file_name: str, future: StreamFuture, stream: Stream):

        file = open(file_name, "wb")

        chunk_size = ByteStreamer.get_chunk_size()
        file_size = 0
        while True:
            buf = stream.read(chunk_size)
            if not buf:
                break

            file_size += len(buf)
            file.write(buf)

        file.close()
        if self.size and (self.size != file_size):
            log.warning(f"Size doesn't match: {self.size} <> {file_size}")

        future.set_result(file_name)


class FileStreamer:
    def __init__(self, byte_streamer: ByteStreamer, byte_receiver: ByteReceiver):
        self.byte_streamer = byte_streamer
        self.byte_receiver = byte_receiver

    def send(
        self, channel: str, topic: str, target: str, message: Message, secure=False, optional=False
    ) -> StreamFuture:
        file_name = Path(message.payload).name
        file_stream = FileStream(message.payload, message.headers)

        message.add_headers(
            {
                StreamHeaderKey.SIZE: file_stream.get_size(),
                StreamHeaderKey.FILE_NAME: file_name,
            }
        )

        return self.byte_streamer.send(
            channel, topic, target, message.headers, file_stream, STREAM_TYPE_FILE, secure, optional
        )

    def register_file_callback(self, channel, topic, file_cb: Callable, *args, **kwargs):
        handler = FileHandler(file_cb)
        self.byte_receiver.register_callback(channel, topic, handler.handle_file_cb, *args, **kwargs)
