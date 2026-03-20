# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nvflare.fuel.f3.streaming.blob_streamer import BlobHandler, BlobTask
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamError, StreamFuture


class _FakeStream(Stream):
    def __init__(self, declared_size: int, chunks):
        super().__init__(size=declared_size)
        self._chunks = list(chunks)

    def read(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def test_read_stream_fails_on_buffer_overrun():
    handler = BlobHandler(lambda future: None)
    future = StreamFuture(stream_id=1)
    blob_task = BlobTask(future=future, stream=_FakeStream(declared_size=4, chunks=[b"abcdef"]))

    handler._read_stream(blob_task)

    error = future.exception(timeout=0.1)
    assert isinstance(error, StreamError)
    assert "Buffer overrun" in str(error)

    with pytest.raises(StreamError, match="Buffer overrun"):
        future.result(timeout=0.1)


def test_read_stream_fails_on_size_mismatch_underrun():
    handler = BlobHandler(lambda future: None)
    future = StreamFuture(stream_id=2)
    blob_task = BlobTask(future=future, stream=_FakeStream(declared_size=8, chunks=[b"abcd"]))

    handler._read_stream(blob_task)

    error = future.exception(timeout=0.1)
    assert isinstance(error, StreamError)
    assert "Size mismatch" in str(error)

    with pytest.raises(StreamError, match="Size mismatch"):
        future.result(timeout=0.1)
