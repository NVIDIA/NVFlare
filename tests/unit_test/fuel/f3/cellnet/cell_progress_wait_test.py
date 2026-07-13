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

from unittest.mock import MagicMock

import nvflare.fuel.f3.cellnet.cell as cell_module
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.waiter_utils import WaiterRC


class _ReplyFuture:
    headers = {}
    error = None

    def result(self):
        return None


def _make_cell():
    cell = Cell.__new__(Cell)
    cell.requests_dict = {}
    cell.logger = MagicMock()
    cell.send_blob = MagicMock(return_value=object())
    cell._future_wait = MagicMock(return_value=True)
    cell.decode_pass_through_channels = set()
    cell.get_fobs_context = MagicMock(return_value={})
    return cell


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
    )

    assert isinstance(result, Message)
    assert cell.send_blob.call_count == 1
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

    assert result.get_header(cell_module.MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
    assert "stream failed" in result.get_header(cell_module.MessageHeaderKey.ERROR)
    assert result.get_header(MessageHeaderKey.PAYLOAD_PROCESSING_ERROR)
    progress_wait_cb.assert_not_called()


def test_sending_stream_future_error_returns_payload_processing_error():
    cell = _make_cell()
    future = MagicMock(error=RuntimeError("request stream failed"))
    cell.send_blob.return_value = future
    cell._future_wait.return_value = False

    result = cell._send_one_request(
        channel="task",
        target="site-1",
        topic="train",
        request=Message(headers={}, payload=None),
        timeout=1.0,
        abort_signal=Signal(),
    )

    assert result.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
    assert "request stream failed" in result.get_header(MessageHeaderKey.ERROR)
    assert result.get_header(MessageHeaderKey.PAYLOAD_PROCESSING_ERROR)
    assert cell.requests_dict == {}


def test_sending_timeout_without_stream_error_remains_timeout():
    cell = _make_cell()
    cell.send_blob.return_value = MagicMock(error=None)
    cell._future_wait.return_value = False

    result = cell._send_one_request(
        channel="task",
        target="site-1",
        topic="train",
        request=Message(headers={}, payload=None),
        timeout=1.0,
        abort_signal=Signal(),
    )

    assert result.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.TIMEOUT
    assert result.get_header(MessageHeaderKey.PAYLOAD_PROCESSING_ERROR) is None
    assert cell.requests_dict == {}


def test_reply_decode_error_returns_payload_processing_error(monkeypatch):
    cell = _make_cell()

    def _conditional_wait(_event, _timeout, _abort_signal):
        waiter = next(iter(cell.requests_dict.values()))
        waiter.receiving_future = _ReplyFuture()
        return WaiterRC.IS_SET

    monkeypatch.setattr(cell_module, "conditional_wait", _conditional_wait)
    monkeypatch.setattr(cell_module, "decode_payload", MagicMock(side_effect=RuntimeError("decode failed")))

    result = cell._send_one_request(
        channel="task",
        target="site-1",
        topic="train",
        request=Message(headers={}, payload=None),
        timeout=1.0,
        abort_signal=Signal(),
    )

    assert result.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
    assert "decode failed" in result.get_header(MessageHeaderKey.ERROR)
    assert result.get_header(MessageHeaderKey.PAYLOAD_PROCESSING_ERROR)
    assert cell.requests_dict == {}


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
