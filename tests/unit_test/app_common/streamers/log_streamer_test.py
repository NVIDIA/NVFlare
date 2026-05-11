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

import os
import tempfile
import threading
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.streaming import StreamContextKey
from nvflare.app_common.streamers.log_streamer import (
    KEY_DATA,
    KEY_DATA_SIZE,
    KEY_EOF,
    KEY_FILE_NAME,
    KEY_HEARTBEAT,
    LogChunkConsumerFactory,
    LogStreamer,
    _LogTailProducer,
    dispatch_stream_done,
)


def _make_heartbeat():
    heartbeat = Shareable()
    heartbeat[KEY_HEARTBEAT] = True
    heartbeat[KEY_DATA] = None
    heartbeat[KEY_DATA_SIZE] = 0
    heartbeat[KEY_EOF] = False
    return heartbeat


def test_stream_done_callback_is_scoped_per_stream():
    calls = []

    def stream_done_cb(stream_ctx, fl_ctx, **kwargs):
        calls.append(stream_ctx[KEY_FILE_NAME])

    factory = LogChunkConsumerFactory(
        chunk_received_cb=None,
        idle_timeout=0.0,
        stream_done_cb=stream_done_cb,
        cb_kwargs={},
    )

    stream_ctx_1 = {KEY_FILE_NAME: "one.log"}
    stream_ctx_2 = {KEY_FILE_NAME: "two.log"}

    factory.get_consumer(stream_ctx_1, None)
    factory.get_consumer(stream_ctx_2, None)

    dispatch_stream_done(stream_ctx_1, None)
    dispatch_stream_done(stream_ctx_1, None)
    dispatch_stream_done(stream_ctx_2, None)

    assert calls == ["one.log", "two.log"]


def test_idle_timeout_ends_each_stream_independently():
    ended = []
    fl_ctx = Mock(spec=FLContext)
    fl_ctx.get_run_abort_signal.return_value = None

    def stream_done_cb(stream_ctx, fl_ctx, **kwargs):
        ended.append((stream_ctx[KEY_FILE_NAME], stream_ctx[StreamContextKey.RC]))

    factory = LogChunkConsumerFactory(
        chunk_received_cb=None,
        idle_timeout=0.3,
        stream_done_cb=stream_done_cb,
        cb_kwargs={},
    )

    stream1_event = threading.Event()
    stream2_event = threading.Event()
    stream_ctx_1 = {KEY_FILE_NAME: "one.log"}
    stream_ctx_2 = {KEY_FILE_NAME: "two.log"}

    def end_stream_1(rc, callback_fl_ctx):
        stream_ctx_1[StreamContextKey.RC] = rc
        dispatch_stream_done(stream_ctx_1, callback_fl_ctx)
        stream1_event.set()

    def end_stream_2(rc, callback_fl_ctx):
        stream_ctx_2[StreamContextKey.RC] = rc
        dispatch_stream_done(stream_ctx_2, callback_fl_ctx)
        stream2_event.set()

    stream_ctx_1[StreamContextKey.END_STREAM] = end_stream_1
    stream_ctx_2[StreamContextKey.END_STREAM] = end_stream_2

    consumer_1 = factory.get_consumer(stream_ctx_1, fl_ctx)
    consumer_2 = factory.get_consumer(stream_ctx_2, fl_ctx)

    # Send a heartbeat to consumer_1 only — this resets its idle clock.
    consumer_1.consume(_make_heartbeat(), stream_ctx_1, fl_ctx)

    # After 0.15s, send a heartbeat to consumer_2 to stagger the two clocks.
    # consumer_1's last message is now 0.15s old; consumer_2's is fresh.
    import time

    time.sleep(0.15)
    consumer_2.consume(_make_heartbeat(), stream_ctx_2, fl_ctx)

    # consumer_1 should timeout first (0.3s since its last heartbeat).
    assert stream1_event.wait(timeout=2.0)
    # consumer_2 got a heartbeat 0.15s later, so it should still be alive.
    assert not stream2_event.is_set()

    # Now wait for consumer_2 to timeout as well.
    assert stream2_event.wait(timeout=2.0)

    assert ended == [
        ("one.log", "TIMEOUT"),
        ("two.log", "TIMEOUT"),
    ]


def test_idle_timeout_cleans_up_stream_even_without_fl_ctx():
    ended = []

    def stream_done_cb(stream_ctx, fl_ctx, **kwargs):
        ended.append((stream_ctx[KEY_FILE_NAME], stream_ctx[StreamContextKey.RC], fl_ctx))

    factory = LogChunkConsumerFactory(
        chunk_received_cb=None,
        idle_timeout=0.2,
        stream_done_cb=stream_done_cb,
        cb_kwargs={},
    )

    stream_ctx = {KEY_FILE_NAME: "orphaned.log"}
    stream_done_event = threading.Event()

    def end_stream(rc, callback_fl_ctx):
        stream_ctx[StreamContextKey.RC] = rc
        dispatch_stream_done(stream_ctx, callback_fl_ctx)
        stream_done_event.set()

    stream_ctx[StreamContextKey.END_STREAM] = end_stream
    factory.get_consumer(stream_ctx, None)

    assert stream_done_event.wait(timeout=2.0)
    assert ended == [("orphaned.log", "TIMEOUT", None)]


def test_stream_log_rejects_liveness_interval_not_less_than_idle_timeout():
    fl_ctx = Mock(spec=FLContext)
    fl_ctx.get_engine.return_value = Mock()

    with pytest.raises(ValueError, match="liveness_interval"):
        LogStreamer.stream_log(
            channel="channel",
            topic="topic",
            stream_ctx={},
            targets=["server"],
            file_name="/tmp/test.log",
            fl_ctx=fl_ctx,
            stop_event=threading.Event(),
            liveness_interval=10.0,
            idle_timeout=10.0,
        )


def _make_producer(file_path: str) -> _LogTailProducer:
    return _LogTailProducer(
        file_name=file_path,
        chunk_size=1024,
        chunk_timeout=5.0,
        poll_interval=0.5,
        stop_event=threading.Event(),
        liveness_interval=10.0,
    )


def _produce_data_chunk(producer: _LogTailProducer):
    """Drive produce() once for a data chunk, with abort_signal absent."""
    fl_ctx = Mock(spec=FLContext)
    fl_ctx.get_run_abort_signal.return_value = None
    return producer.produce({}, fl_ctx)


def test_first_chunk_execution_exception_is_transient_and_retries():
    """Bootstrap miss: first reply is EXECUTION_EXCEPTION; producer must roll back
    the file offset and not log an ERROR. The same bytes should be re-emitted on
    the next produce() call."""
    with tempfile.NamedTemporaryFile("wb", suffix=".log", delete=False) as f:
        f.write(b"hello world")
        path = f.name
    try:
        producer = _make_producer(path)
        first_request, _ = _produce_data_chunk(producer)
        assert first_request[KEY_DATA] == b"hello world"

        replies = {"server": make_reply(ReturnCode.EXECUTION_EXCEPTION)}
        result = producer.process_replies(replies, {}, Mock(spec=FLContext))
        assert result is None  # stream must continue
        assert producer._first_ok_received is False
        assert producer._bootstrap_miss_count == 1

        retry_request, _ = _produce_data_chunk(producer)
        assert retry_request[KEY_DATA] == b"hello world"  # bytes re-emitted, not lost
    finally:
        producer.close()
        os.unlink(path)


def test_first_chunk_execution_exception_on_heartbeat_does_not_seek():
    """Heartbeats don't move the file pointer, so the bootstrap-tolerant path
    must not attempt to seek the (possibly None) file handle."""
    producer = _LogTailProducer(
        file_name="/nonexistent/log/path",  # forces self.file = None
        chunk_size=1024,
        chunk_timeout=5.0,
        poll_interval=0.5,
        stop_event=threading.Event(),
        liveness_interval=10.0,
    )
    producer._last_was_data = False  # simulate heartbeat last-emit
    replies = {"server": make_reply(ReturnCode.EXECUTION_EXCEPTION)}
    result = producer.process_replies(replies, {}, Mock(spec=FLContext))
    assert result is None  # transient miss, stream continues
    assert producer._first_ok_received is False
    assert producer._bootstrap_miss_count == 1


def test_execution_exception_after_first_ok_still_ends_stream():
    """Once the receiver has acked at least one chunk OK, subsequent failures
    must surface as real errors (return False to terminate the stream),
    matching the pre-fix behavior for steady-state failures."""
    with tempfile.NamedTemporaryFile("wb", suffix=".log", delete=False) as f:
        f.write(b"x")
        path = f.name
    try:
        producer = _make_producer(path)
        producer._first_ok_received = True  # simulate prior success
        producer._last_was_data = True
        replies = {"server": make_reply(ReturnCode.EXECUTION_EXCEPTION)}
        result = producer.process_replies(replies, {}, Mock(spec=FLContext))
        assert result is False  # stream should tear down
    finally:
        producer.close()
        os.unlink(path)


def test_persistent_bootstrap_execution_exception_is_capped():
    """A receiver that never acks OK should eventually surface as a real
    failure instead of retrying the same first chunk forever."""
    with tempfile.NamedTemporaryFile("wb", suffix=".log", delete=False) as f:
        f.write(b"never acked")
        path = f.name
    try:
        producer = _make_producer(path)
        first_request, _ = _produce_data_chunk(producer)
        assert first_request[KEY_DATA] == b"never acked"

        replies = {"server": make_reply(ReturnCode.EXECUTION_EXCEPTION)}
        for i in range(producer._MAX_BOOTSTRAP_MISSES):
            result = producer.process_replies(replies, {}, Mock(spec=FLContext))
            assert result is None
            assert producer._bootstrap_miss_count == i + 1

        result = producer.process_replies(replies, {}, Mock(spec=FLContext))
        assert result is False
        assert producer._first_ok_received is False
        assert producer._bootstrap_miss_count == producer._MAX_BOOTSTRAP_MISSES
    finally:
        producer.close()
        os.unlink(path)


def test_first_chunk_ok_flips_strict_mode():
    """A first OK reply must flip the producer into strict mode so the next
    failure is treated as a real error rather than another transient miss."""
    with tempfile.NamedTemporaryFile("wb", suffix=".log", delete=False) as f:
        f.write(b"abc")
        path = f.name
    try:
        producer = _make_producer(path)
        producer._last_was_data = True
        producer._bootstrap_miss_count = 3
        ok_replies = {"server": make_reply(ReturnCode.OK)}
        producer.process_replies(ok_replies, {}, Mock(spec=FLContext))
        assert producer._first_ok_received is True
        assert producer._bootstrap_miss_count == 0

        fail_replies = {"server": make_reply(ReturnCode.EXECUTION_EXCEPTION)}
        result = producer.process_replies(fail_replies, {}, Mock(spec=FLContext))
        assert result is False
    finally:
        producer.close()
        os.unlink(path)
