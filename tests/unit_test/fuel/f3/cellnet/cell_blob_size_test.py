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
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.f3.cellnet import cell as cell_module
from nvflare.fuel.f3.cellnet.cell import Adapter, Cell
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.blob_streamer import BlobStreamer
from nvflare.fuel.f3.streaming.byte_receiver import ByteReceiver, RxTask
from nvflare.fuel.f3.streaming.byte_streamer import ByteStreamer
from nvflare.fuel.f3.streaming.stream_const import STREAM_ERROR_TOPIC, StreamDataType, StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import BlobSizeError, StreamError, StreamFuture


def _oversized_response_cell():
    def send_blob(*args, **kwargs):
        raise BlobSizeError("Blob size 5 exceeds configured limit 4 (streaming_max_blob_size)")

    return SimpleNamespace(send_blob=send_blob, get_fobs_context=lambda: {})


def _async_response_cell(reply_future):
    return SimpleNamespace(send_blob=lambda *args, **kwargs: reply_future, get_fobs_context=lambda: {})


def test_oversized_server_job_response_exits_job_process(monkeypatch):
    adapter = Adapter(None, SimpleNamespace(fqcn="server.job-id"), _oversized_response_cell())
    monkeypatch.setattr(cell_module, "encode_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(cell_module.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exc_info:
        adapter._send_response(
            Message(payload=b"payload"), "stream-id", "request-id", "channel", "topic", "client", False, False
        )

    assert exc_info.value.code == 1


def test_oversized_non_server_job_response_propagates_error(monkeypatch):
    adapter = Adapter(None, SimpleNamespace(fqcn="site-1.job-id"), _oversized_response_cell())
    monkeypatch.setattr(cell_module, "encode_payload", lambda *args, **kwargs: None)

    with pytest.raises(BlobSizeError, match=r"limit 4 \(streaming_max_blob_size\)"):
        adapter._send_response(
            Message(payload=b"payload"), "stream-id", "request-id", "channel", "topic", "peer", False, False
        )


def test_asymmetric_receiver_limit_exits_server_job_after_async_rejection(monkeypatch):
    reply_future = StreamFuture(stream_id=1)
    adapter = Adapter(None, SimpleNamespace(fqcn="server.job-id"), _async_response_cell(reply_future))
    monkeypatch.setattr(cell_module, "encode_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(cell_module.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    adapter._send_response(
        Message(payload=b"payload"), "stream-id", "request-id", "channel", "topic", "client", False, False
    )

    with pytest.raises(SystemExit) as exc_info:
        reply_future.set_exception(BlobSizeError("receiver limit 4 is smaller than response size 5"))

    assert exc_info.value.code == 1


def test_async_stream_error_does_not_exit_non_server_job(monkeypatch):
    reply_future = StreamFuture(stream_id=2)
    adapter = Adapter(None, SimpleNamespace(fqcn="site-1.job-id"), _async_response_cell(reply_future))
    monkeypatch.setattr(cell_module, "encode_payload", lambda *args, **kwargs: None)
    exit_calls = []
    monkeypatch.setattr(cell_module.os, "_exit", exit_calls.append)

    adapter._send_response(
        Message(payload=b"payload"), "stream-id", "request-id", "channel", "topic", "peer", False, False
    )
    reply_future.set_exception(StreamError("receiver stopped stream"))

    assert exit_calls == []


def test_single_frame_receiver_rejection_reports_generic_error_after_receive_completion(monkeypatch):
    import nvflare.fuel.f3.streaming.blob_streamer as blob_streamer_module
    import nvflare.fuel.f3.streaming.byte_receiver as byte_receiver_module

    monkeypatch.setattr(CommConfigurator, "get_streaming_max_blob_size", lambda self: 4)
    monkeypatch.setattr(cell_module.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    pending_callbacks = []
    monkeypatch.setattr(
        byte_receiver_module.stream_thread_pool,
        "submit",
        lambda fn, *args: pending_callbacks.append((fn, args)),
    )
    monkeypatch.setattr(blob_streamer_module.callback_thread_pool, "submit", lambda fn, *args: fn(*args))

    sender_cell = MagicMock()
    sender_cell.my_info.fqcn = "server.job-id"
    byte_streamer = ByteStreamer(sender_cell)
    server = Cell.__new__(Cell)
    server.core_cell = sender_cell
    server.requests_dict = {}
    server.logger = MagicMock()
    byte_streamer.register_error_callback(server._process_stream_error)

    receiver_cell = MagicMock()
    receiver_cell.my_info.fqcn = "client"
    error_messages = []

    def route_receiver_message(_channel, topic, _target, message, **_kwargs):
        if topic == STREAM_ERROR_TOPIC:
            error_messages.append(message)
            message.set_header(MessageHeaderKey.ORIGIN, "client")
            byte_streamer._error_handler(message)
        return {}

    receiver_cell.fire_and_forget.side_effect = route_receiver_message
    byte_receiver = ByteReceiver(receiver_cell)
    BlobStreamer(SimpleNamespace(), byte_receiver).register_blob_callback(
        CellChannel.RETURN_ONLY,
        "channel:topic",
        lambda _future: pytest.fail("oversized blob callback must not run"),
    )
    incoming = Message(
        {
            MessageHeaderKey.ORIGIN: "server.job-id",
            StreamHeaderKey.STREAM_ID: 101,
            StreamHeaderKey.CHANNEL: CellChannel.RETURN_ONLY,
            StreamHeaderKey.TOPIC: "channel:topic",
            StreamHeaderKey.STREAM_REQ_ID: "request-id",
            StreamHeaderKey.SIZE: 5,
            StreamHeaderKey.SEQUENCE: 0,
            StreamHeaderKey.OFFSET: 0,
            StreamHeaderKey.DATA_TYPE: StreamDataType.FINAL,
            StreamHeaderKey.RELIABLE: True,
            StreamHeaderKey.CHUNK_SIZE: 8,
            StreamHeaderKey.WINDOW_SIZE: 8,
        },
        b"abcde",
    )

    try:
        byte_receiver._data_handler(incoming)

        with RxTask.map_lock:
            rx_task = RxTask.rx_task_map[("server.job-id", 101)]
        assert rx_task.completed is True
        assert rx_task.failed is False
        assert rx_task.stream_future.done() is False
        assert len(pending_callbacks) == 1

        with pytest.raises(SystemExit) as exc_info:
            fn, args = pending_callbacks.pop()
            fn(*args)

        assert exc_info.value.code == 1
        receiver_error = rx_task.stream_future.exception(timeout=0.1)
        assert isinstance(receiver_error, BlobSizeError)
        assert rx_task.failed is True
        assert rx_task.completed is True
        assert len(error_messages) == 1
        assert error_messages[0].get_header(StreamHeaderKey.ERROR_TYPE) == BlobSizeError.__name__
    finally:
        with RxTask.map_lock:
            rx_task = RxTask.rx_task_map.pop(("server.job-id", 101), None)
        if rx_task and rx_task.cleanup_timer:
            rx_task.cleanup_timer.cancel()
