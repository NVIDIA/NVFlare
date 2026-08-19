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
import time
from unittest.mock import MagicMock

import pytest

import nvflare.fuel.f3.cellnet.cell as cell_module
import nvflare.fuel.f3.streaming.byte_streamer as byte_streamer_module
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.byte_streamer import ByteStreamer, TxTask
from nvflare.fuel.f3.streaming.stream_types import Stream
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY
from nvflare.fuel.utils.waiter_utils import WaiterRC


class _ReplyFuture:
    headers = {}
    error = None

    def result(self):
        return None


class _OneByteStream(Stream):
    def __init__(self):
        super().__init__(size=1, headers={})
        self.data = b"x"

    def read(self, _size):
        data, self.data = self.data, b""
        return data


def _make_cell():
    cell = Cell.__new__(Cell)
    cell.requests_dict = {}
    cell.logger = MagicMock()
    cell.send_blob = MagicMock(return_value=object())
    cell._future_wait = MagicMock(return_value=True)
    cell.decode_pass_through_channels = set()
    cell.decode_pass_through_topics = set()
    cell.get_fobs_context = MagicMock(return_value={})
    return cell


def test_unknown_reply_warning_does_not_log_headers(caplog):
    cell = _make_cell()
    cell.logger = logging.getLogger("test_unknown_reply_warning")
    future = MagicMock()
    future.headers = {
        cell_module.StreamHeaderKey.STREAM_REQ_ID: "late-request",
        "__token__": "secret-token-sentinel",
        "__token_signature__": "secret-signature-sentinel",
        "other": "other-secret-sentinel",
    }

    with caplog.at_level(logging.WARNING, logger=cell.logger.name):
        cell._process_reply(future)

    assert "Receiving unknown req_id='late-request'" in caplog.text
    assert "secret-token-sentinel" not in caplog.text
    assert "secret-signature-sentinel" not in caplog.text
    assert "other-secret-sentinel" not in caplog.text


def test_encode_message_can_stamp_receiver_ids_for_multi_receiver_download_refs(monkeypatch):
    cell = _make_cell()
    captured = {}
    cell.get_fobs_context.side_effect = lambda props=None: props

    def _capture_encode(_msg, _encoding_key, fobs_ctx):
        captured.update(fobs_ctx)
        return 0

    monkeypatch.setattr(cell_module, "encode_payload", _capture_encode)

    cell._encode_message(
        Message(headers={}, payload=None),
        abort_signal=Signal(),
        num_receivers=2,
        receiver_ids=["receiver-a", "receiver-b"],
    )

    assert captured[FOBSContextKey.NUM_RECEIVERS] == 2
    assert captured[FOBSContextKey.RECEIVER_IDS] == ["receiver-a", "receiver-b"]


def _broadcast_and_capture_encode(pass_through: bool):
    cell = _make_cell()
    cell._encode_message = MagicMock()
    cell._send_one_request = MagicMock(return_value=Message(headers={}, payload=None))
    headers = {MessageHeaderKey.PASS_THROUGH: True} if pass_through else {}
    request = Message(headers=headers, payload=None)
    targets = ["site-1", "site-2"]

    cell._broadcast_request(
        channel="task",
        topic="train",
        targets=targets,
        request=request,
        abort_signal=Signal(),
    )

    return cell._encode_message.call_args.kwargs


def test_broadcast_stamps_receiver_ids_for_direct_downloads():
    encode_args = _broadcast_and_capture_encode(pass_through=False)

    assert encode_args["num_receivers"] == 2
    assert encode_args["receiver_ids"] == ["site-1", "site-2"]


def test_broadcast_uses_count_based_completion_for_pass_through_downloads():
    encode_args = _broadcast_and_capture_encode(pass_through=True)

    assert encode_args["num_receivers"] == 2
    assert encode_args["receiver_ids"] is None


def test_encode_message_applies_call_scoped_fobs_props_without_mutating_them(monkeypatch):
    cell = _make_cell()
    captured = {}
    cell.get_fobs_context.side_effect = lambda props=None: props
    callback = MagicMock()
    call_props = {RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY: callback}

    monkeypatch.setattr(
        cell_module,
        "encode_payload",
        lambda _msg, _encoding_key, fobs_ctx: captured.update(fobs_ctx) or 0,
    )

    cell._encode_message(
        Message(headers={}, payload=None),
        abort_signal=Signal(),
        receiver_ids=("receiver-a",),
        fobs_ctx_props=call_props,
    )

    assert captured[RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY] is callback
    assert captured[FOBSContextKey.RECEIVER_IDS] == ("receiver-a",)
    assert call_props == {RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY: callback}


def test_remote_processing_wait_continues_without_resend_while_progress_callback_is_true(monkeypatch):
    cell = _make_cell()
    waits = []

    def _conditional_wait(_event, _timeout, _abort_signal):
        waits.append(1)
        if len(waits) == 1:
            return WaiterRC.TIMEOUT
        waiter = next(iter(cell.requests_dict.values()))
        waiter.receiving_future = _ReplyFuture()
        return WaiterRC.IS_SET

    monkeypatch.setattr(cell_module, "conditional_wait", _conditional_wait)
    progress_wait_cb = MagicMock(return_value=True)

    result = cell._send_one_request(
        channel="task",
        target="site-1",
        topic="train",
        request=Message(headers={}, payload=None),
        timeout=1.0,
        abort_signal=Signal(),
        progress_wait_cb=progress_wait_cb,
    )

    assert isinstance(result, Message)
    assert cell.send_blob.call_count == 1
    assert progress_wait_cb.call_count == 1
    assert len(waits) == 2


def test_remote_processing_wait_handles_many_progress_timeouts_without_resend(monkeypatch):
    """Simulate congested large transfer progress lasting beyond many old fixed-timeout periods."""
    cell = _make_cell()
    waits = []

    def _conditional_wait(_event, _timeout, _abort_signal):
        waits.append(1)
        if len(waits) <= 16:
            return WaiterRC.TIMEOUT
        waiter = next(iter(cell.requests_dict.values()))
        waiter.receiving_future = _ReplyFuture()
        return WaiterRC.IS_SET

    monkeypatch.setattr(cell_module, "conditional_wait", _conditional_wait)
    progress_wait_cb = MagicMock(return_value=True)

    result = cell._send_one_request(
        channel="task",
        target="site-1",
        topic="train",
        request=Message(headers={}, payload=None),
        timeout=1.0,
        abort_signal=Signal(),
        progress_wait_cb=progress_wait_cb,
        reliable=True,
    )

    assert isinstance(result, Message)
    assert cell.send_blob.call_count == 1
    assert cell.send_blob.call_args.kwargs["reliable"] is True
    assert progress_wait_cb.call_count == 16
    assert len(waits) == 17


def test_receiving_wait_continues_without_resend_while_progress_callback_is_true(monkeypatch):
    cell = _make_cell()

    def _conditional_wait(_event, _timeout, _abort_signal):
        waiter = next(iter(cell.requests_dict.values()))
        waiter.receiving_future = _ReplyFuture()
        return WaiterRC.IS_SET

    monkeypatch.setattr(cell_module, "conditional_wait", _conditional_wait)
    cell._future_wait.side_effect = [True, False, True]
    progress_wait_cb = MagicMock(return_value=True)

    result = cell._send_one_request(
        channel="task",
        target="site-1",
        topic="train",
        request=Message(headers={}, payload=None),
        timeout=1.0,
        abort_signal=Signal(),
        progress_wait_cb=progress_wait_cb,
    )

    assert isinstance(result, Message)
    assert cell.send_blob.call_count == 1
    assert cell._future_wait.call_count == 3
    assert progress_wait_cb.call_count == 1


def test_receiving_wait_does_not_extend_stream_future_error(monkeypatch):
    cell = _make_cell()

    class _ErrorReplyFuture(_ReplyFuture):
        error = RuntimeError("stream failed")

    def _conditional_wait(_event, _timeout, _abort_signal):
        waiter = next(iter(cell.requests_dict.values()))
        waiter.receiving_future = _ErrorReplyFuture()
        return WaiterRC.IS_SET

    monkeypatch.setattr(cell_module, "conditional_wait", _conditional_wait)
    cell._future_wait.side_effect = [True, False]
    progress_wait_cb = MagicMock(return_value=True)

    result = cell._send_one_request(
        channel="task",
        target="site-1",
        topic="train",
        request=Message(headers={}, payload=None),
        timeout=1.0,
        abort_signal=Signal(),
        progress_wait_cb=progress_wait_cb,
    )

    assert result.get_header(cell_module.MessageHeaderKey.RETURN_CODE) == ReturnCode.TIMEOUT
    progress_wait_cb.assert_not_called()


def test_remote_processing_wait_returns_timeout_when_progress_callback_is_false(monkeypatch):
    cell = _make_cell()
    monkeypatch.setattr(cell_module, "conditional_wait", lambda _event, _timeout, _abort_signal: WaiterRC.TIMEOUT)
    progress_wait_cb = MagicMock(return_value=False)

    result = cell._send_one_request(
        channel="task",
        target="site-1",
        topic="train",
        request=Message(headers={}, payload=None),
        timeout=1.0,
        abort_signal=Signal(),
        progress_wait_cb=progress_wait_cb,
    )

    assert result.get_header(cell_module.MessageHeaderKey.RETURN_CODE) == ReturnCode.TIMEOUT
    assert cell.send_blob.call_count == 1
    assert progress_wait_cb.call_count == 1
    assert cell.requests_dict == {}


def test_remote_processing_wait_treats_progress_callback_exception_as_no_progress(monkeypatch):
    cell = _make_cell()
    monkeypatch.setattr(cell_module, "conditional_wait", lambda _event, _timeout, _abort_signal: WaiterRC.TIMEOUT)

    def progress_wait_cb():
        raise RuntimeError("callback failed")

    result = cell._send_one_request(
        channel="task",
        target="site-1",
        topic="train",
        request=Message(headers={}, payload=None),
        timeout=1.0,
        abort_signal=Signal(),
        progress_wait_cb=progress_wait_cb,
    )

    assert result.get_header(cell_module.MessageHeaderKey.RETURN_CODE) == ReturnCode.TIMEOUT
    assert cell.send_blob.call_count == 1
    assert cell.requests_dict == {}
    assert cell.logger.warning.called


def test_remote_processing_wait_aborted_does_not_call_progress_callback(monkeypatch):
    cell = _make_cell()
    monkeypatch.setattr(cell_module, "conditional_wait", lambda _event, _timeout, _abort_signal: WaiterRC.ABORTED)
    progress_wait_cb = MagicMock(return_value=True)

    result = cell._send_one_request(
        channel="task",
        target="site-1",
        topic="train",
        request=Message(headers={}, payload=None),
        timeout=1.0,
        abort_signal=Signal(),
        progress_wait_cb=progress_wait_cb,
    )

    assert result.get_header(cell_module.MessageHeaderKey.RETURN_CODE) == ReturnCode.TIMEOUT
    assert progress_wait_cb.call_count == 0
    assert cell.requests_dict == {}


@pytest.mark.parametrize("abort", [False, True], ids=["timeout", "abort"])
def test_unfinished_reliable_request_is_terminal_before_caller_returns(monkeypatch, abort):
    monkeypatch.setattr(byte_streamer_module.reliable_retry_scheduler, "register", MagicMock())
    unregister = MagicMock()
    monkeypatch.setattr(byte_streamer_module.reliable_retry_scheduler, "unregister", unregister)
    monkeypatch.setattr(byte_streamer_module.reliable_retry_scheduler, "wakeup", MagicMock())
    monkeypatch.setattr(CommConfigurator, "get_streaming_retry_wait", lambda self, default: 0.01)

    wire_cell = MagicMock()
    wire_cell.fire_and_forget.return_value = {}
    task = TxTask(
        cell=wire_cell,
        chunk_size=4,
        channel="task",
        topic="result_ready",
        target="site-1",
        headers={},
        stream=_OneByteStream(),
        reliable=True,
        secure=False,
        optional=False,
    )
    with ByteStreamer.map_lock:
        ByteStreamer.tx_task_map[task.sid] = task

    try:
        task.send_loop()
        assert not task.stream_future.done()
        assert wire_cell.fire_and_forget.call_count == 1

        cell = _make_cell()
        cell.send_blob.return_value = task.stream_future
        cell._future_wait = Cell._future_wait.__get__(cell, Cell)
        original_cancel = task.cancel

        def cancel_while_waiter_is_owned():
            if not task.stopped:
                assert cell.requests_dict
            original_cancel()

        task.cancel = cancel_while_waiter_is_owned
        abort_signal = Signal()
        timeout = 0.01
        if abort:
            abort_signal.trigger("stop")
            timeout = 1.0

        result = cell._send_one_request(
            channel="task",
            target="site-1",
            topic="result_ready",
            request=Message(headers={}, payload=None),
            timeout=timeout,
            abort_signal=abort_signal,
            reliable=True,
        )

        assert result.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.TIMEOUT
        assert task.stopped is True
        assert task.stream_future.cancelled() is True
        assert task.pending_messages == {}
        assert cell.requests_dict == {}
        with ByteStreamer.map_lock:
            assert task.sid not in ByteStreamer.tx_task_map
        unregister.assert_called_once_with(task)

        wire_sends_at_return = wire_cell.fire_and_forget.call_count
        time.sleep(0.02)
        assert task.retry_task() is None
        assert wire_cell.fire_and_forget.call_count == wire_sends_at_return
    finally:
        task.cancel()
        with ByteStreamer.map_lock:
            ByteStreamer.tx_task_map.pop(task.sid, None)
