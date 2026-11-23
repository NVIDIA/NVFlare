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

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.streaming import ConsumerFactory, ObjectConsumer, StreamContextKey
from nvflare.private.stream_runner import TOPIC_STREAM_REQUEST, HeaderKey, ObjectStreamer


class _Consumer(ObjectConsumer):
    def consume(self, shareable, stream_ctx, fl_ctx):
        return True, make_reply(ReturnCode.OK)


class _Factory(ConsumerFactory):
    def get_consumer(self, stream_ctx, fl_ctx):
        return _Consumer()


def _request(tx_id, seq):
    request = Shareable()
    request.set_header(HeaderKey.TX_ID, tx_id)
    request.set_header(HeaderKey.SEQ, seq)
    request.set_header(HeaderKey.CHANNEL, "channel")
    request.set_header(HeaderKey.TOPIC, "topic")
    if seq == 0:
        request.set_header(HeaderKey.CTX, {"name": tx_id})
    return request


def test_repeated_start_of_a_live_stream_ends_it_without_blocking_the_streamer():
    streamer = ObjectStreamer(Mock())
    ended = []
    streamer.register_stream_processing(
        "channel",
        "topic",
        _Factory(),
        stream_done_cb=lambda stream_ctx, fl_ctx: ended.append(stream_ctx[StreamContextKey.RC]),
    )
    fl_ctx = FLContext()
    replies = []

    def handle(tx_id, seq):
        reply = streamer._handle_request(TOPIC_STREAM_REQUEST, _request(tx_id, seq), fl_ctx)
        replies.append(reply.get_return_code())

    try:
        handle("live", 0)
        # A handler that ends the stream while holding the transaction lock
        # never returns, so run it where a regression cannot hang the suite.
        repeated = threading.Thread(target=handle, args=("live", 0), daemon=True)
        repeated.start()
        repeated.join(timeout=2)
        assert not repeated.is_alive()

        assert replies == [ReturnCode.OK, ReturnCode.BAD_REQUEST_DATA]
        assert ended == [ReturnCode.BAD_REQUEST_DATA]
        assert not streamer.tx_table
        handle("next", 0)
        assert replies[-1] == ReturnCode.OK
    finally:
        streamer.shutdown()
