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

from types import SimpleNamespace

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


def _make_stream_with_task(future):
    """Return a fake stream whose .task.stop() sets an exception on the future."""
    task = SimpleNamespace(stop=lambda err: future.set_exception(err))
    stream = SimpleNamespace(task=task)
    return stream


def test_read_stream_succeeds_with_dynamic_buffer():
    """declared_size=0 exercises the FastBuffer (append) path instead of the pre-allocated path."""
    handler = BlobHandler(lambda future: None)
    future = StreamFuture(stream_id=5)
    blob_task = BlobTask(future=future, stream=_FakeStream(declared_size=0, chunks=[b"hello", b" world"]))

    handler._read_stream(blob_task)

    result = future.result(timeout=0.1)
    assert result == b"hello world"
    assert future.exception(timeout=0.1) is None


def test_run_blob_cb_logs_and_stops_task_when_future_not_failed():
    """blob_cb raises StreamError but future has no error — treated as a genuine bug."""
    future = StreamFuture(stream_id=3)
    stream = _make_stream_with_task(future)

    def bad_blob_cb(f):
        raise StreamError("independent blob_cb error")

    handler = BlobHandler(bad_blob_cb)
    handler._run_blob_cb(future, stream, args=(), kwargs={})

    # task.stop() should have called set_exception on the future
    error = future.exception(timeout=0.1)
    assert isinstance(error, StreamError)
    assert "blob_cb threw" in str(error)


def test_run_blob_cb_suppresses_stream_error_when_future_already_failed(caplog):
    """blob_cb re-raises the StreamError from future.result(); should be suppressed with DEBUG log."""
    import logging

    future = StreamFuture(stream_id=4)
    future.set_exception(StreamError("stream failed"))

    stream = SimpleNamespace()  # no .task — suppression path must not call stop()

    def reraise_blob_cb(f):
        f.result(timeout=0.1)  # raises the stored StreamError

    handler = BlobHandler(reraise_blob_cb)
    with caplog.at_level(logging.DEBUG):
        handler._run_blob_cb(future, stream, args=(), kwargs={})

    assert any("suppressed" in r.message for r in caplog.records)
