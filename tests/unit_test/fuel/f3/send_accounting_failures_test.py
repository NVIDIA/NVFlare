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

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.f3.cellnet.cell import Adapter, Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.send_accounting import attach_logical_send_context
from nvflare.fuel.f3.streaming.blob_streamer import BlobStream
from nvflare.fuel.f3.streaming.byte_streamer import ByteStreamer, TxTask
from nvflare.private.fed.resource_stats.f3_counter import F3Counter, F3TrafficClass

ORIGIN = "site-1.job"
DESTINATION = "server.job"


def _pre_admitted_message():
    counter = F3Counter()
    message = Message(headers={MessageHeaderKey.DESTINATION: DESTINATION}, payload=b"payload")
    context = attach_logical_send_context(message, counter, F3TrafficClass.TASK_RESULT, recipient_name="server")
    assert context.pre_admit(ORIGIN, DESTINATION)
    assert counter.pending_count == 1
    return message, context, counter


def _assert_abandoned(counter):
    assert counter.pending_count == 0
    assert counter.close_and_drain(0.0)
    assert counter.freeze() == {"sent_to": []}


def _adapter():
    cell = SimpleNamespace(get_fobs_context=MagicMock(return_value={}), send_blob=MagicMock())
    return Adapter(None, FqcnInfo(ORIGIN), cell)


def _send_response(adapter, response):
    adapter._send_response(response, "stream-request", "request", "channel", "topic", DESTINATION, False)


@pytest.mark.parametrize("accounted", [False, True])
@pytest.mark.parametrize("stage", ["context", "encoding", "blob_setup"])
def test_response_pre_send_failure_preserves_error_and_releases_pre_admission(monkeypatch, stage, accounted):
    response, _context, counter = _pre_admitted_message() if accounted else (Message(payload=b"payload"), None, None)
    adapter = _adapter()
    error = RuntimeError(stage)
    encode = MagicMock(return_value=7)
    monkeypatch.setattr("nvflare.fuel.f3.cellnet.cell.encode_payload", encode)
    {
        "context": adapter.cell.get_fobs_context,
        "encoding": encode,
        "blob_setup": adapter.cell.send_blob,
    }[stage].side_effect = error

    with pytest.raises(RuntimeError) as caught:
        _send_response(adapter, response)

    assert caught.value is error
    if stage != "blob_setup":
        adapter.cell.send_blob.assert_not_called()
    if accounted:
        _assert_abandoned(counter)


@pytest.mark.parametrize(
    "context", [SimpleNamespace(), SimpleNamespace(abandon_before_send=MagicMock(side_effect=ValueError))]
)
def test_optional_accounting_cleanup_cannot_replace_response_error(monkeypatch, context):
    response = Message(payload=b"payload")
    response.set_logical_send_context(context)
    error = RuntimeError("original encode error")
    monkeypatch.setattr("nvflare.fuel.f3.cellnet.cell.encode_payload", MagicMock(side_effect=error))

    with pytest.raises(RuntimeError) as caught:
        _send_response(_adapter(), response)

    assert caught.value is error


def test_response_setup_failure_after_transport_begins_cannot_abandon_transport(monkeypatch):
    response, context, counter = _pre_admitted_message()
    adapter = _adapter()
    error = RuntimeError("after transport admission")
    attempts = []
    monkeypatch.setattr("nvflare.fuel.f3.cellnet.cell.encode_payload", lambda *_args, **_kwargs: 7)

    def begin_then_fail(*_args, **_kwargs):
        attempts.append(context.try_begin(ORIGIN, DESTINATION))
        raise error

    adapter.cell.send_blob.side_effect = begin_then_fail
    with pytest.raises(RuntimeError) as caught:
        _send_response(adapter, response)

    assert caught.value is error
    assert counter.pending_count == 1
    attempts[0].accepted(7)
    assert counter.freeze() == {"sent_to": [{"participant_name": "server", "payload_bytes": "7", "messages": "1"}]}


def test_response_without_accounting_preserves_payload_and_send_arguments():
    response = Message(payload=b"payload")
    adapter = _adapter()

    _send_response(adapter, response)

    args, kwargs = adapter.cell.send_blob.call_args
    assert args[2:] == (DESTINATION, response, False)
    assert kwargs == {"optional": True, "reliable": True}
    assert response.payload == b"payload"
    assert response.get_logical_send_context() is None
    adapter.cell.send_blob.return_value.add_done_callback.assert_called_once()


def test_response_without_reply_id_releases_pre_admission_without_sending():
    response, _context, counter = _pre_admitted_message()
    adapter = _adapter()

    adapter._send_response(response, "", "request", "channel", "topic", DESTINATION, False)

    adapter.cell.get_fobs_context.assert_not_called()
    adapter.cell.send_blob.assert_not_called()
    _assert_abandoned(counter)


@pytest.mark.parametrize("accounted", [False, True])
@pytest.mark.parametrize("stage", ["encoding", "request_setup"])
def test_request_pre_send_failure_preserves_error_and_releases_pre_admission(stage, accounted):
    request, _context, counter = _pre_admitted_message() if accounted else (Message(payload=b"payload"), None, None)
    cell = Cell.__new__(Cell)
    cell.get_fqcn = MagicMock(return_value=ORIGIN)
    cell._encode_message = MagicMock()
    cell._send_one_request = MagicMock()
    error = RuntimeError(stage)
    (cell._encode_message if stage == "encoding" else cell._send_one_request).side_effect = error

    with pytest.raises(RuntimeError) as caught:
        cell._send_request("channel", DESTINATION, "topic", request)

    assert caught.value is error
    if stage == "encoding":
        cell._send_one_request.assert_not_called()
    if accounted:
        _assert_abandoned(counter)


def _core_cell():
    cell = CoreCell.__new__(CoreCell)
    cell.my_info = FqcnInfo(ORIGIN)
    cell.logger = logging.getLogger(__name__)
    cell.fobs_ctx = {}
    cell.max_msg_size = 1024
    cell.ALL_CELLS = {}
    cell.communicator = MagicMock()
    cell.sent_msg_size_pool = MagicMock()
    cell._stats_category = MagicMock(return_value="test")
    cell.log_error = MagicMock()
    cell.encrypt_payload = MagicMock()
    return cell


@pytest.mark.parametrize("accounted", [False, True])
@pytest.mark.parametrize("stage", ["context", "encoding", "encrypt"])
def test_core_send_failure_keeps_comm_error_and_releases_pre_admission(monkeypatch, stage, accounted):
    message, _context, counter = _pre_admitted_message() if accounted else (Message(payload=b"payload"), None, None)
    cell = _core_cell()
    cell.get_fobs_context = MagicMock(return_value={})
    encode = MagicMock(return_value=7)
    monkeypatch.setattr("nvflare.fuel.f3.cellnet.core_cell.encode_payload", encode)
    {
        "context": cell.get_fobs_context,
        "encoding": encode,
        "encrypt": cell.encrypt_payload,
    }[
        stage
    ].side_effect = RuntimeError(stage)

    assert cell._send_to_endpoint(Endpoint(DESTINATION), message) == ReturnCode.COMM_ERROR

    cell.communicator.send.assert_not_called()
    if accounted:
        _assert_abandoned(counter)


@pytest.mark.parametrize("stage", ["prepare", "start"])
def test_byte_streamer_existing_failure_path_abandons_pre_admission(monkeypatch, stage):
    _message, context, counter = _pre_admitted_message()
    streamer = ByteStreamer.__new__(ByteStreamer)
    streamer.cell = SimpleNamespace(my_info=FqcnInfo(ORIGIN), ALL_CELLS={})
    streamer.chunk_size = 4
    error = RuntimeError(stage)
    tasks = []
    if stage == "prepare":
        monkeypatch.setattr("nvflare.fuel.f3.streaming.byte_streamer.TxTask", MagicMock(side_effect=error))
    else:

        def fail_start(task, _handler):
            tasks.append(task)
            raise error

        monkeypatch.setattr(TxTask, "start_task_thread", fail_start)

    try:
        with pytest.raises(RuntimeError) as caught:
            streamer.send(
                "channel",
                "topic",
                DESTINATION,
                {},
                BlobStream(b"payload", {}),
                reliable=False,
                accounting_context=context,
            )
        assert caught.value is error
        _assert_abandoned(counter)
    finally:
        # Do not leave a transport task created by this controlled submit failure.
        with ByteStreamer.map_lock:
            for task in tasks:
                ByteStreamer.tx_task_map.pop(task.sid, None)
