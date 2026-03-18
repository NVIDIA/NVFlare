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

"""Unit tests for CellPipe (Fixes 1, 2, 6).

Fix 1 — CellMessage caching on retries:
    Root Cause: CellPipe.send() called _to_cell_message() on every retry,
    causing the FOBS layer to re-serialize msg.data and create a new
    ArrayDownloadable transaction per retry.  With a 5 GiB model and 14+
    retries this produced 70–135 GiB of live transactions simultaneously (OOM).
    Fix: Cache the resulting CellMessage on the Message object after the first
    serialization.

Fix 2 — release_send_cache():
    PipeHandler must call pipe.release_send_cache() after the retry loop so
    the cached CellMessage (and its encoded payload bytes) are freed promptly.

Fix 6 — ACK before Message object creation:
    Root Cause: _receive_message() previously called _from_cell_message() to
    convert the raw CellMessage to a pipe Message BEFORE returning ACK to the
    sender.  With reverse PASS_THROUGH absent, the FOBS decode (tensor
    download) already happened in Adapter.call() before the callback was
    invoked, but Message object allocation still occurred inside the callback.
    Fix: Queue the raw CellMessage in _receive_message() and defer the
    _from_cell_message() conversion to receive() time.  This makes the ACK
    path perform the absolute minimum work.  True "ACK before decode" for
    large tensors is achieved by Fix 8 (PASS_THROUGH on the pipe cell).
"""

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.f3.cellnet.cell import Message as CellMessage  # f3-layer message
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.cell_pipe import (
    _HEADER_MSG_ID,
    _HEADER_MSG_TYPE,
    _HEADER_REQ_ID,
    CellPipe,
    _from_cell_message,
    _to_cell_message,
)
from nvflare.fuel.utils.pipe.pipe import Message, Topic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipe():
    """Return a CellPipe with all network infrastructure mocked out."""
    pipe = object.__new__(CellPipe)
    pipe.mode = Mode.ACTIVE
    pipe.logger = MagicMock()
    pipe.pipe_lock = threading.Lock()
    pipe.closed = False
    pipe.channel = "cell_pipe.test_channel"
    pipe.peer_fqcn = "test_peer"
    pipe.hb_seq = 1
    pipe.received_msgs = queue.Queue()
    pipe.pass_through_on_send = False

    # Mock the cell so send_request returns a successful reply.
    ok_reply = MagicMock()
    ok_reply.get_header.return_value = ReturnCode.OK
    pipe.cell = MagicMock()
    pipe.cell.send_request.return_value = ok_reply
    pipe.cell.fire_and_forget.return_value = None
    return pipe


def _make_cell_message(
    msg_id="mid1", msg_type=Message.REQUEST, topic="train", req_id=None, payload="data", origin="test_peer"
):
    """Build a minimal CellMessage (f3 Message) as the cell network would deliver."""
    headers = {
        _HEADER_MSG_ID: msg_id,
        _HEADER_MSG_TYPE: msg_type,
        MessageHeaderKey.TOPIC: topic,
        MessageHeaderKey.ORIGIN: origin,
    }
    if req_id is not None:
        headers[_HEADER_REQ_ID] = req_id
    return CellMessage(headers=headers, payload=payload)


def _make_msg(topic="train", data="payload"):
    return Message.new_request(topic=topic, data=data)


# ---------------------------------------------------------------------------
# Fix 1: CellMessage caching on retries
# ---------------------------------------------------------------------------


class TestCellMessageCaching:
    """Verify that _to_cell_message is called only once per Message, regardless
    of how many times send() is retried (Fix 1)."""

    def test_cell_message_cached_after_first_send(self):
        """After the first send(), msg._cached_cell_msg must be set."""
        pipe = _make_pipe()
        msg = _make_msg()

        assert not hasattr(msg, "_cached_cell_msg"), "cache must be absent before any send"

        with patch("nvflare.fuel.utils.pipe.cell_pipe._to_cell_message", wraps=_to_cell_message) as mock_build:
            pipe.send(msg, timeout=1.0)

        assert mock_build.call_count == 1
        assert hasattr(msg, "_cached_cell_msg"), "cache must be present after first send"

    def test_to_cell_message_called_only_once_on_retries(self):
        """_to_cell_message must NOT be called again on the 2nd and 3rd send().

        Before Fix 1: each send() call created a fresh CellMessage → FOBS
        re-serialized msg.data → new ArrayDownloadable transaction per retry.
        After Fix 1: the cached CellMessage is reused; no re-serialization.
        """
        pipe = _make_pipe()
        msg = _make_msg()

        with patch("nvflare.fuel.utils.pipe.cell_pipe._to_cell_message", wraps=_to_cell_message) as mock_build:
            pipe.send(msg, timeout=1.0)  # first send — should call _to_cell_message
            pipe.send(msg, timeout=1.0)  # retry 1 — must NOT call _to_cell_message
            pipe.send(msg, timeout=1.0)  # retry 2 — must NOT call _to_cell_message

        assert mock_build.call_count == 1, (
            f"_to_cell_message called {mock_build.call_count} times; " "expected exactly 1 (cached after first send)"
        )

    def test_same_cell_message_object_reused_on_retry(self):
        """send_request() must receive the identical CellMessage object on every retry."""
        pipe = _make_pipe()
        msg = _make_msg()

        pipe.send(msg, timeout=1.0)
        cached = msg._cached_cell_msg
        pipe.send(msg, timeout=1.0)

        calls = pipe.cell.send_request.call_args_list
        assert len(calls) == 2
        # Both calls must pass the exact same CellMessage instance.
        request_first = calls[0][1].get("request") or calls[0][0][3]
        request_second = calls[1][1].get("request") or calls[1][0][3]
        assert request_first is cached
        assert request_second is cached

    def test_msg_root_id_set_consistently(self):
        """MSG_ROOT_ID header must equal msg.msg_id on every send() call."""
        pipe = _make_pipe()
        msg = _make_msg()

        pipe.send(msg, timeout=1.0)
        pipe.send(msg, timeout=1.0)

        assert hasattr(msg, "_cached_cell_msg")
        assert msg._cached_cell_msg.get_header(MessageHeaderKey.MSG_ROOT_ID) == msg.msg_id

    def test_different_messages_get_independent_caches(self):
        """Two different Message objects must each get their own cached CellMessage."""
        pipe = _make_pipe()
        msg_a = _make_msg(data="payload_a")
        msg_b = _make_msg(data="payload_b")

        with patch("nvflare.fuel.utils.pipe.cell_pipe._to_cell_message", wraps=_to_cell_message) as mock_build:
            pipe.send(msg_a, timeout=1.0)
            pipe.send(msg_b, timeout=1.0)
            pipe.send(msg_a, timeout=1.0)  # retry of msg_a — no new CellMessage
            pipe.send(msg_b, timeout=1.0)  # retry of msg_b — no new CellMessage

        # One call per distinct Message object, regardless of retry count.
        assert mock_build.call_count == 2

        assert msg_a._cached_cell_msg is not msg_b._cached_cell_msg


# ---------------------------------------------------------------------------
# Heartbeat messages must NOT be cached
# ---------------------------------------------------------------------------


class TestHeartbeatNotCached:
    """Heartbeats are fire-and-forget; each must get a fresh CellMessage so
    the timestamp header reflects the actual send time.  They must not be
    accidentally cached on the Message object."""

    def test_heartbeat_not_cached(self):
        """HEARTBEAT must never set _cached_cell_msg."""
        pipe = _make_pipe()
        msg = _make_msg(topic=Topic.HEARTBEAT, data="")

        with patch("nvflare.fuel.utils.pipe.cell_pipe._to_cell_message", wraps=_to_cell_message) as mock_build:
            pipe.send(msg)
            pipe.send(msg)
            pipe.send(msg)

        assert not hasattr(msg, "_cached_cell_msg"), "heartbeat must not populate the cache"
        # A fresh CellMessage must be built each time for heartbeats.
        assert mock_build.call_count == 3

    def test_heartbeat_uses_fire_and_forget(self):
        """HEARTBEAT must go through cell.fire_and_forget, not send_request."""
        pipe = _make_pipe()
        msg = _make_msg(topic=Topic.HEARTBEAT, data="")

        pipe.send(msg)

        pipe.cell.fire_and_forget.assert_called_once()
        pipe.cell.send_request.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 2: release_send_cache() — explicit cache teardown
# ---------------------------------------------------------------------------


class TestReleaseSendCache:
    """Verify that release_send_cache() removes the cached CellMessage and that
    PipeHandler calls it after the retry loop exits (Fix 2)."""

    def test_release_clears_cache(self):
        """After release_send_cache(), _cached_cell_msg must be absent."""
        pipe = _make_pipe()
        msg = _make_msg()

        pipe.send(msg, timeout=1.0)
        assert hasattr(msg, "_cached_cell_msg")

        pipe.release_send_cache(msg)
        assert not hasattr(msg, "_cached_cell_msg")

    def test_release_is_idempotent(self):
        """release_send_cache() must not raise when called on a message with no cache."""
        pipe = _make_pipe()
        msg = _make_msg()

        # No prior send — no cache present; must not raise.
        pipe.release_send_cache(msg)
        # Second call also safe.
        pipe.release_send_cache(msg)

    def test_pipe_handler_calls_release_on_success(self, monkeypatch):
        """PipeHandler must call pipe.release_send_cache() after a successful send."""
        from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler

        pipe = _make_pipe()
        released = []
        monkeypatch.setattr(type(pipe), "release_send_cache", lambda self, m: released.append(m))

        handler = PipeHandler(pipe, resend_interval=0.01, heartbeat_timeout=0)
        msg = _make_msg()
        handler._send_to_pipe(msg, timeout=1.0)

        assert len(released) == 1
        assert released[0] is msg

    def test_pipe_handler_calls_release_on_max_resends_exceeded(self, monkeypatch):
        """PipeHandler must call release_send_cache() even when max_resends is exceeded."""
        from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler

        # Make all sends fail.
        pipe = _make_pipe()
        pipe.cell.send_request.return_value = None  # send() returns False

        released = []
        monkeypatch.setattr(type(pipe), "release_send_cache", lambda self, m: released.append(m))
        monkeypatch.setattr(type(pipe), "can_resend", lambda self: True)

        handler = PipeHandler(pipe, resend_interval=0.01, max_resends=1, heartbeat_timeout=0)
        msg = _make_msg()
        handler._send_to_pipe(msg, timeout=0.05)

        assert len(released) == 1
        assert released[0] is msg


# ---------------------------------------------------------------------------
# Closed pipe still raises
# ---------------------------------------------------------------------------


def test_send_raises_on_closed_pipe():
    pipe = _make_pipe()
    pipe.closed = True
    msg = _make_msg()

    with pytest.raises(BrokenPipeError):
        pipe.send(msg, timeout=1.0)


# ---------------------------------------------------------------------------
# Fix 6: _receive_message() queues raw CellMessage; receive() converts
# ---------------------------------------------------------------------------


class TestReceiveMessageFix6:
    """Verify that _receive_message() defers _from_cell_message() to receive().

    Before Fix 6: _receive_message() called _from_cell_message() inline,
    creating the pipe Message object before returning ACK.  While _from_cell_message()
    itself is fast (just header extraction + payload-ref copy), queueing it
    inside the callback kept work in the ACK path.  With Fix 6 the queue now
    holds raw CellMessage objects and the conversion is deferred to receive().
    """

    def test_receive_message_queues_raw_cell_message(self):
        """_receive_message() must enqueue the raw CellMessage, not a pipe Message."""
        pipe = _make_pipe()
        cm = _make_cell_message()

        pipe._receive_message(cm)

        assert pipe.received_msgs.qsize() == 1
        queued = pipe.received_msgs.get_nowait()
        # Must be the original CellMessage (f3 Message), not a pipe Message.
        assert queued is cm

    def test_receive_message_returns_ok(self):
        """_receive_message() must return a ReturnCode.OK reply."""
        pipe = _make_pipe()
        cm = _make_cell_message()

        reply = pipe._receive_message(cm)

        assert reply is not None
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK

    def test_receive_converts_cell_message_to_pipe_message(self):
        """receive() must convert the queued CellMessage to a pipe Message."""
        pipe = _make_pipe()
        cm = _make_cell_message(msg_id="m1", topic="train", payload="tensor_data")
        pipe._receive_message(cm)

        msg = pipe.receive()

        assert msg is not None
        assert isinstance(msg, Message)
        assert msg.topic == "train"
        assert msg.data == "tensor_data"

    def test_receive_returns_none_when_empty(self):
        """receive() with no timeout must return None when queue is empty."""
        pipe = _make_pipe()

        result = pipe.receive()

        assert result is None

    def test_receive_preserves_fifo_order(self):
        """Messages must be returned in the order they arrived."""
        pipe = _make_pipe()
        cm1 = _make_cell_message(msg_id="m1", topic="train")
        cm2 = _make_cell_message(msg_id="m2", topic="validate")
        cm3 = _make_cell_message(msg_id="m3", topic="submit_model")

        pipe._receive_message(cm1)
        pipe._receive_message(cm2)
        pipe._receive_message(cm3)

        msg1 = pipe.receive()
        msg2 = pipe.receive()
        msg3 = pipe.receive()

        assert msg1.topic == "train"
        assert msg2.topic == "validate"
        assert msg3.topic == "submit_model"

    def test_from_cell_message_not_called_in_receive_message(self, monkeypatch):
        """Before Fix 6: _from_cell_message() was called inside _receive_message().
        After Fix 6: it must NOT be called there — only in receive().

        This test demonstrates the pre-fix bug and verifies it is resolved.
        """
        pipe = _make_pipe()
        cm = _make_cell_message()

        calls_in_callback = []

        original_from_cm = _from_cell_message

        def tracking_from_cm(cell_msg):
            calls_in_callback.append(cell_msg)
            return original_from_cm(cell_msg)

        monkeypatch.setattr("nvflare.fuel.utils.pipe.cell_pipe._from_cell_message", tracking_from_cm)

        pipe._receive_message(cm)
        assert len(calls_in_callback) == 0, "_from_cell_message must NOT be called inside _receive_message() (Fix 6)"

        # Conversion must happen in receive().
        pipe.receive()
        assert len(calls_in_callback) == 1, "_from_cell_message must be called in receive()"

    def test_receive_message_raises_on_fqcn_mismatch(self):
        """_receive_message() must raise when the sender FQCN does not match peer_fqcn."""
        pipe = _make_pipe()
        pipe.peer_fqcn = "expected_peer"
        cm = _make_cell_message(origin="wrong_peer")

        with pytest.raises(RuntimeError, match="peer FQCN mismatch"):
            pipe._receive_message(cm)

        # Nothing must have been queued.
        assert pipe.received_msgs.empty()

    def test_clear_empties_queued_cell_messages(self):
        """clear() must drain the queue even when it holds raw CellMessages."""
        pipe = _make_pipe()
        for i in range(3):
            pipe._receive_message(_make_cell_message(msg_id=f"m{i}"))

        assert pipe.received_msgs.qsize() == 3
        pipe.clear()
        assert pipe.received_msgs.empty()


# ---------------------------------------------------------------------------
# Fix 8: MSG_ROOT_TTL on REPLY messages
# ---------------------------------------------------------------------------


class TestMsgRootTtlOnReply:
    """Verify that send() stamps MSG_ROOT_TTL on REPLY messages (Fix 8).

    When the subprocess sends its result back through the CellPipe and PASS_THROUGH
    is enabled on the pipe cell, the server needs to download tensors directly from
    the subprocess.  MSG_ROOT_TTL tells the ViaDownloader how long to keep the
    transaction alive.  We use float(timeout) = submit_result_timeout as the TTL,
    matching the value already configurable via Fix 3.

    CONTRACT:
    - REPLY messages WITH a positive timeout → MSG_ROOT_TTL set to float(timeout)
    - REQUEST messages → MSG_ROOT_TTL NOT set
    - REPLY messages WITH timeout=None → MSG_ROOT_TTL NOT set
    - REPLY messages WITH timeout=0 → MSG_ROOT_TTL NOT set
    """

    def _get_sent_request(self, pipe):
        """Return the CellMessage passed to cell.send_request on the last call."""
        assert pipe.cell.send_request.called, "send_request was not called"
        # send_request is called with keyword args: channel=, topic=, target=, request=, ...
        return pipe.cell.send_request.call_args.kwargs["request"]

    def test_reply_with_timeout_gets_msg_root_ttl(self):
        """REPLY message with positive timeout must have MSG_ROOT_TTL set."""
        pipe = _make_pipe()
        msg = Message.new_reply(topic="train", data="result", req_msg_id="r1")

        pipe.send(msg, timeout=300.0)

        cell_msg = self._get_sent_request(pipe)
        ttl = cell_msg.get_header(MessageHeaderKey.MSG_ROOT_TTL)
        assert ttl == 300.0, f"Expected MSG_ROOT_TTL=300.0, got {ttl!r}"

    def test_request_does_not_get_msg_root_ttl(self):
        """REQUEST messages must NOT have MSG_ROOT_TTL set."""
        pipe = _make_pipe()
        msg = _make_msg(topic="train", data="params")  # msg_type=REQUEST

        pipe.send(msg, timeout=300.0)

        cell_msg = self._get_sent_request(pipe)
        ttl = cell_msg.get_header(MessageHeaderKey.MSG_ROOT_TTL)
        assert ttl is None, f"REQUEST must not carry MSG_ROOT_TTL, got {ttl!r}"

    def test_reply_with_none_timeout_no_msg_root_ttl(self):
        """REPLY with timeout=None must NOT have MSG_ROOT_TTL set."""
        pipe = _make_pipe()
        msg = Message.new_reply(topic="train", data="result", req_msg_id="r1")

        pipe.send(msg, timeout=None)

        cell_msg = self._get_sent_request(pipe)
        ttl = cell_msg.get_header(MessageHeaderKey.MSG_ROOT_TTL)
        assert ttl is None, f"REPLY with timeout=None must not carry MSG_ROOT_TTL, got {ttl!r}"

    def test_reply_with_zero_timeout_no_msg_root_ttl(self):
        """REPLY with timeout=0 must NOT have MSG_ROOT_TTL set."""
        pipe = _make_pipe()
        msg = Message.new_reply(topic="train", data="result", req_msg_id="r1")

        pipe.send(msg, timeout=0)

        cell_msg = self._get_sent_request(pipe)
        ttl = cell_msg.get_header(MessageHeaderKey.MSG_ROOT_TTL)
        assert ttl is None, f"REPLY with timeout=0 must not carry MSG_ROOT_TTL, got {ttl!r}"

    def test_reply_msg_root_ttl_equals_timeout(self):
        """MSG_ROOT_TTL value must exactly equal float(timeout)."""
        pipe = _make_pipe()
        msg = Message.new_reply(topic="train", data="result", req_msg_id="r1")

        pipe.send(msg, timeout=42)

        cell_msg = self._get_sent_request(pipe)
        ttl = cell_msg.get_header(MessageHeaderKey.MSG_ROOT_TTL)
        assert ttl == 42.0
        assert isinstance(ttl, float)
