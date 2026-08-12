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
from nvflare.fuel.f3.cellnet.cell import Adapter
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.blob_streamer import BlobStream, BlobStreamer
from nvflare.fuel.f3.streaming.byte_receiver import ByteReceiver, RxTask
from nvflare.fuel.f3.streaming.byte_streamer import TxTask
from nvflare.fuel.f3.streaming.stream_const import StreamDataType, StreamHeaderKey
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


def test_single_frame_receiver_rejection_fails_server_reply_and_receiver_future(monkeypatch):
    monkeypatch.setattr(CommConfigurator, "get_streaming_max_blob_size", lambda self: 4)
    monkeypatch.setattr(cell_module, "encode_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(cell_module.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    sender_cell = MagicMock()
    tx_task = TxTask(
        cell=sender_cell,
        chunk_size=8,
        channel=CellChannel.RETURN_ONLY,
        topic="channel:topic",
        target="client",
        headers={},
        stream=BlobStream(b"abcde", {}),
        reliable=True,
        secure=False,
        optional=False,
    )
    send_kwargs = {}

    def send_blob(*args, **kwargs):
        send_kwargs.update(kwargs)
        return tx_task.stream_future

    adapter_cell = SimpleNamespace(send_blob=send_blob, get_fobs_context=lambda: {})
    adapter = Adapter(None, SimpleNamespace(fqcn="server.job-id"), adapter_cell)
    adapter._send_response(
        Message(payload=b"abcde"), "stream-id", "request-id", "channel", "topic", "client", False, False
    )
    assert send_kwargs["reliable"] is True

    receiver_cell = MagicMock()
    receiver_cell.my_info.fqcn = "client"
    error_messages = []

    def return_error(_channel, _topic, _target, message, **_kwargs):
        error_messages.append(message)
        message.set_header(MessageHeaderKey.ORIGIN, "client")
        tx_task.handle_ack(message)
        return {}

    receiver_cell.fire_and_forget.side_effect = return_error
    byte_receiver = ByteReceiver(receiver_cell)
    BlobStreamer(SimpleNamespace(), byte_receiver).register_blob_callback(
        CellChannel.RETURN_ONLY,
        "channel:topic",
        lambda _future: pytest.fail("oversized blob callback must not run"),
    )
    incoming = Message(
        {
            MessageHeaderKey.ORIGIN: "server.job-id",
            StreamHeaderKey.STREAM_ID: tx_task.sid,
            StreamHeaderKey.CHANNEL: CellChannel.RETURN_ONLY,
            StreamHeaderKey.TOPIC: "channel:topic",
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
        with pytest.raises(SystemExit) as exc_info:
            byte_receiver._data_handler(incoming)

        assert exc_info.value.code == 1
        with RxTask.map_lock:
            rx_task = RxTask.rx_task_map[("server.job-id", tx_task.sid)]
        receiver_error = rx_task.stream_future.exception(timeout=0.1)
        assert isinstance(receiver_error, BlobSizeError)
        assert rx_task.failed is True
        assert rx_task.completed is False
        assert len(error_messages) == 1
        assert error_messages[0].get_header(StreamHeaderKey.ERROR_TYPE) == BlobSizeError.__name__
        assert isinstance(tx_task.stream_future.exception(timeout=0.1), BlobSizeError)
    finally:
        with RxTask.map_lock:
            rx_task = RxTask.rx_task_map.pop(("server.job-id", tx_task.sid), None)
        if rx_task and rx_task.cleanup_timer:
            rx_task.cleanup_timer.cancel()
