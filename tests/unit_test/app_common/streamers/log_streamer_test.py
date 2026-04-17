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

import threading
from unittest.mock import Mock

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.streaming import StreamContextKey
from nvflare.app_common.streamers.log_streamer import (
    KEY_DATA,
    KEY_DATA_SIZE,
    KEY_EOF,
    KEY_FILE_NAME,
    KEY_HEARTBEAT,
    LogChunkConsumerFactory,
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
