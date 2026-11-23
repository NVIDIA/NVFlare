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
from pathlib import Path
from unittest.mock import Mock

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.streaming import ConsumerFactory, ObjectConsumer, StreamContextKey
from nvflare.app_common.streamers.file_streamer import KEY_DATA, KEY_DATA_SIZE, KEY_FILE_SIZE, FileStreamer
from nvflare.app_common.streamers.file_streamer import _ChunkConsumerFactory as FileConsumerFactory
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.hci.proto import StreamChannel
from nvflare.fuel.hci.server.hci import AdminServer
from nvflare.fuel.hci.server.sess import SessionManager
from nvflare.private.defs import InternalFLContextKey
from nvflare.private.stream_runner import TOPIC_STREAM_REQUEST, HeaderKey, ObjectStreamer
from tests.admin_session_test_utils import register_admin_transport


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


def test_admin_upload_rejects_other_owners_and_removes_partial_file_on_revocation(tmp_path, monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.sess.threading.Thread.start", lambda _: None)
    manager = SessionManager(Mock())
    streamer = ObjectStreamer(Mock())
    admin = object.__new__(AdminServer)
    admin.cmd_reg = Mock()
    streamer.register_stream_processing(
        StreamChannel.UPLOAD,
        "topic",
        FileConsumerFactory(str(tmp_path)),
        stream_done_cb=admin._process_upload,
    )

    def create_owner(user):
        _, grant = register_admin_transport(manager, user=user)
        session, _ = manager.create_session(grant, "org", "lead")
        fl_ctx = FLContext()
        fl_ctx.set_prop(InternalFLContextKey.ADMIN_UPLOAD_OWNER, (manager, session), private=True, sticky=False)
        fl_ctx.set_prop(InternalFLContextKey.ADMIN_TRANSPORT_GRANT, grant, private=True, sticky=False)
        return session, fl_ctx

    def chunk(seq, data):
        request = _request("upload-id", seq)
        request.set_header(HeaderKey.CHANNEL, StreamChannel.UPLOAD)
        if seq == 0:
            request.set_header(HeaderKey.CTX, {KEY_FILE_SIZE: 2})
        request[KEY_DATA] = data
        request[KEY_DATA_SIZE] = len(data)
        return request

    try:
        owner, owner_ctx = create_owner("lead@example.com")
        _, other_ctx = create_owner("other@example.com")
        open_reply = streamer._handle_request(TOPIC_STREAM_REQUEST, chunk(0, b"a"), owner_ctx)
        assert open_reply.get_return_code() == ReturnCode.OK
        info = streamer.tx_table["upload-id"]
        file = info.consumer.file
        file_location = Path(FileStreamer.get_file_location(info.stream_ctx))
        assert file_location.exists()

        other_chunk = streamer._handle_request(TOPIC_STREAM_REQUEST, chunk(1, b"b"), other_ctx)
        other_abort = streamer._handle_abort("abort", _request("upload-id", 1), other_ctx)
        assert other_chunk.get_return_code() == other_abort.get_return_code() == CellReturnCode.UNAUTHENTICATED
        assert streamer.tx_table["upload-id"] is info
        file.flush()
        assert file_location.read_bytes() == b"a"

        manager.end_session_by_id(owner.sess_id)
        assert not streamer.tx_table
        assert not manager.uploads
        assert file.closed
        assert not file_location.exists()
        admin.cmd_reg.process_command.assert_not_called()
    finally:
        manager.shutdown()
        streamer.shutdown()
