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
from typing import Any, Callable, Optional

from nvflare.fuel.f3.connection import BytesAlike
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.byte_receiver import ByteReceiver
from nvflare.fuel.f3.streaming.byte_streamer import ByteStreamer
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


class FileFuture(StreamFuture):
    """The future used by File callback. Unlike StreamFuture, its result is the filename.
    All other calls are delegated to StreamFuture
    """

    def __init__(self, future: StreamFuture, file_name: Optional[str]):
        super().__init__(future.get_stream_id(), future.get_headers())
        self.future = future
        self.file_name = file_name

    def result(self, timeout=None) -> Any:
        self.future.result()
        return self.file_name

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


class FileHandler:
    def __init__(self, file_cb: Callable):
        self.file_cb = file_cb
        self.size = 0
        self.file_name = None

    def handle_file_cb(self, future: StreamFuture, stream: Stream, resume: bool, *args, **kwargs) -> int:

        if resume:
            log.warning("Resume is not supported, ignored")

        self.size = stream.get_size()
        file_future = FileFuture(future, None)

        file_name = self.file_cb(file_future, *args, **kwargs)

        file_future.file_name = file_name
        stream_thread_pool.submit(self._write_to_file, file_future, stream)

        return 0

    def _write_to_file(self, future: FileFuture, stream: Stream):

        file = open(future.file_name, "wb")

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
            log.warning(f"Stream size doesn't match: {self.size} <> {file_size}")

        future.set_result(file_size)


class FileStreamer:
    def __init__(self, byte_streamer: ByteStreamer, byte_receiver: ByteReceiver):
        self.byte_streamer = byte_streamer
        self.byte_receiver = byte_receiver

    def send(self, channel: str, topic: str, target: str, headers: dict, file_name: str) -> StreamFuture:
        file_stream = FileStream(file_name, headers)
        return self.byte_streamer.send(channel, topic, target, headers, file_stream)

    def register_file_callback(self, channel, topic, file_cb: Callable, *args, **kwargs):
        handler = FileHandler(file_cb)
        self.byte_receiver.register_callback(channel, topic, handler.handle_file_cb, *args, **kwargs)
