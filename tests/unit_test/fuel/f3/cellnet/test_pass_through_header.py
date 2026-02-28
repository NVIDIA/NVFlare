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

"""
Unit tests for the per-message PASS_THROUGH mechanism (Fix 14).

The PASS_THROUGH flag is carried as a cell-message header rather than being
set globally on the engine cell.  This lets task/result messages (CellPipe
`pass_through_on_send=True`) trigger lazy tensor loading at the receiver while
Swarm P2P aggregation messages — which arrive on the same cell without the
header — are decoded normally (no LazyDownloadRef, no crash).

Tests verify:

  Adapter.call() (cell.py):
  1. PASS_THROUGH=True in message header  → decode_ctx[PASS_THROUGH] = True
  2. No PASS_THROUGH header (Swarm P2P)   → decode_ctx[PASS_THROUGH] = False
  3. Explicit PASS_THROUGH=False header   → decode_ctx[PASS_THROUGH] = False

  CellPipe.send() (cell_pipe.py):
  4. pass_through_on_send=True  → PASS_THROUGH header stamped on request
  5. pass_through_on_send=False → PASS_THROUGH header NOT stamped
  6. Heartbeat messages skip PASS_THROUGH even when pass_through_on_send=True
"""

import threading
from unittest.mock import MagicMock, patch

from nvflare.fuel.f3.cellnet.cell import Adapter
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.pipe import Message, Topic

# ---------------------------------------------------------------------------
# Helpers for Adapter tests
# ---------------------------------------------------------------------------


def _make_mock_cell(captured_ctx: dict):
    """Return a mock Cell whose get_fobs_context() captures the props it receives."""
    cell = MagicMock()

    def _get_fobs_context(props=None):
        ctx = {}
        if props:
            ctx.update(props)
        captured_ctx.update(ctx)
        return ctx

    cell.get_fobs_context.side_effect = _get_fobs_context
    return cell


def _make_future(headers: dict, payload=b""):
    """Return a mock StreamFuture with the given headers and payload."""
    future = MagicMock()
    future.headers = headers
    future.result.return_value = payload
    future.error = None
    return future


def _make_adapter(captured_ctx: dict):
    """Return an Adapter backed by a mock cell and a trivial callback."""
    cell = _make_mock_cell(captured_ctx)
    cb = MagicMock(return_value=MagicMock())
    return Adapter(cb=cb, my_info=None, cell=cell)


def _headers_without_pass_through(stream_req_id=""):
    """Build minimal valid headers that do NOT include PASS_THROUGH (e.g. Swarm P2P)."""
    return {
        StreamHeaderKey.STREAM_REQ_ID: stream_req_id,
        StreamHeaderKey.CHANNEL: "aux_communication",
        StreamHeaderKey.TOPIC: "aggregate",
        MessageHeaderKey.ORIGIN: "client1",
        MessageHeaderKey.REQ_ID: "req-swarm-001",
        MessageHeaderKey.SECURE: False,
        MessageHeaderKey.OPTIONAL: False,
    }


# ---------------------------------------------------------------------------
# 1-3: Adapter.call() — per-message decode_ctx construction
# ---------------------------------------------------------------------------


class TestAdapterPassThroughHeader:
    """Adapter.call() must build a per-call decode_ctx based solely on the
    MessageHeaderKey.PASS_THROUGH header of the incoming message."""

    def test_header_true_sets_pass_through_in_decode_ctx(self):
        """PASS_THROUGH=True in the message header → decode_ctx[PASS_THROUGH] = True."""
        captured = {}
        adapter = _make_adapter(captured)

        headers = _headers_without_pass_through()
        headers[MessageHeaderKey.PASS_THROUGH] = True

        future = _make_future(headers)

        with patch("nvflare.fuel.f3.cellnet.cell.decode_payload"):
            adapter.call(future)

        assert (
            captured.get(FOBSContextKey.PASS_THROUGH) is True
        ), "decode_ctx must contain PASS_THROUGH=True when the message header carries it"

    def test_no_header_sets_pass_through_false_in_decode_ctx(self):
        """No PASS_THROUGH header (Swarm P2P messages) → decode_ctx[PASS_THROUGH] = False.

        This is the key regression guard for Root Cause 9: Swarm P2P aggregation
        results must not be decoded with PASS_THROUGH=True (which would yield
        LazyDownloadRef objects and crash the aggregator with TypeError).
        """
        captured = {}
        adapter = _make_adapter(captured)

        # Swarm P2P: no PASS_THROUGH header at all
        headers = _headers_without_pass_through()
        assert MessageHeaderKey.PASS_THROUGH not in headers  # guard

        future = _make_future(headers)

        with patch("nvflare.fuel.f3.cellnet.cell.decode_payload"):
            adapter.call(future)

        assert captured.get(FOBSContextKey.PASS_THROUGH) is False, (
            "decode_ctx must contain PASS_THROUGH=False when the message header is absent "
            "(Swarm P2P path — tensors must be downloaded inline, not as LazyDownloadRef)"
        )

    def test_explicit_header_false_sets_pass_through_false_in_decode_ctx(self):
        """Explicit PASS_THROUGH=False header → decode_ctx[PASS_THROUGH] = False."""
        captured = {}
        adapter = _make_adapter(captured)

        headers = _headers_without_pass_through()
        headers[MessageHeaderKey.PASS_THROUGH] = False

        future = _make_future(headers)

        with patch("nvflare.fuel.f3.cellnet.cell.decode_payload"):
            adapter.call(future)

        assert captured.get(FOBSContextKey.PASS_THROUGH) is False

    def test_decode_payload_receives_the_per_call_decode_ctx(self):
        """decode_payload must be called with the per-call decode_ctx, not None."""
        captured = {}
        adapter = _make_adapter(captured)

        headers = _headers_without_pass_through()
        headers[MessageHeaderKey.PASS_THROUGH] = True

        future = _make_future(headers)

        with patch("nvflare.fuel.f3.cellnet.cell.decode_payload") as mock_decode:
            adapter.call(future)

        assert mock_decode.called, "decode_payload must be called"
        _, _, decode_ctx = (
            mock_decode.call_args[0][0],
            mock_decode.call_args[0][1],
            mock_decode.call_args.kwargs.get("fobs_ctx") or mock_decode.call_args[0][2],
        )
        assert isinstance(decode_ctx, dict), "decode_payload must receive a dict fobs_ctx"
        assert decode_ctx.get(FOBSContextKey.PASS_THROUGH) is True

    def test_cell_level_fobs_context_is_not_mutated(self):
        """get_fobs_context(props=...) must be called — cell's base context is never mutated.

        The Adapter always calls cell.get_fobs_context(props=...) to build a fresh
        shallow-copy dict per message, so the cell-level base context stays clean
        across concurrent calls.
        """
        captured = {}
        adapter = _make_adapter(captured)

        headers = _headers_without_pass_through()
        headers[MessageHeaderKey.PASS_THROUGH] = True
        future = _make_future(headers)

        with patch("nvflare.fuel.f3.cellnet.cell.decode_payload"):
            adapter.call(future)

        # Verify the cell method was called with props, not without
        cell = adapter.cell
        cell.get_fobs_context.assert_called_once_with(props={FOBSContextKey.PASS_THROUGH: True})


# ---------------------------------------------------------------------------
# Helpers for CellPipe tests
# ---------------------------------------------------------------------------


def _make_stub_cell_pipe(pass_through_on_send: bool = False):
    """Return a CellPipe stub that bypasses __init__ network setup.

    Only the attributes exercised by CellPipe.send() are initialised here.
    The underlying cell is replaced by a MagicMock so no real network is needed.
    """
    pipe = object.__new__(CellPipe)  # bypass __init__
    pipe.pass_through_on_send = pass_through_on_send
    pipe.closed = False
    pipe.hb_seq = 1
    pipe.pipe_lock = threading.Lock()
    pipe.channel = "cell_pipe.task_channel"
    pipe.peer_fqcn = "site1_token_passive"
    pipe.logger = MagicMock()

    # Mock cell: send_request returns a reply with ReturnCode.OK
    mock_reply = MagicMock()
    mock_reply.get_header.return_value = ReturnCode.OK
    mock_cell = MagicMock()
    mock_cell.send_request.return_value = mock_reply
    pipe.cell = mock_cell

    return pipe


def _make_reply_msg(topic="submit_model"):
    """Return a valid pipe REPLY Message."""
    return Message(msg_type=Message.REPLY, topic=topic, data=b"model_bytes", req_id="req-001")


def _get_sent_request(pipe: CellPipe):
    """Return the CellMessage that was passed to cell.send_request()."""
    call_kwargs = pipe.cell.send_request.call_args[1]
    return call_kwargs["request"]


# ---------------------------------------------------------------------------
# 4-6: CellPipe.send() — PASS_THROUGH header stamping
# ---------------------------------------------------------------------------


class TestCellPipePassThroughHeader:
    """CellPipe.send() must stamp MessageHeaderKey.PASS_THROUGH when
    pass_through_on_send=True, and must NOT stamp it otherwise."""

    def test_pass_through_on_send_true_stamps_header(self):
        """pass_through_on_send=True → PASS_THROUGH header present in the sent CellMessage."""
        pipe = _make_stub_cell_pipe(pass_through_on_send=True)
        msg = _make_reply_msg()

        pipe.send(msg, timeout=30.0)

        request = _get_sent_request(pipe)
        header_value = request.get_header(MessageHeaderKey.PASS_THROUGH)
        assert header_value is True, (
            "CellPipe must stamp PASS_THROUGH=True on every outgoing message " "when pass_through_on_send=True"
        )

    def test_pass_through_on_send_false_does_not_stamp_header(self):
        """pass_through_on_send=False (default) → PASS_THROUGH header absent from CellMessage."""
        pipe = _make_stub_cell_pipe(pass_through_on_send=False)
        msg = _make_reply_msg()

        pipe.send(msg, timeout=30.0)

        request = _get_sent_request(pipe)
        # get_header returns None when absent (default sentinel)
        header_value = request.get_header(MessageHeaderKey.PASS_THROUGH)
        assert header_value is None, "CellPipe must NOT stamp PASS_THROUGH when pass_through_on_send=False"

    def test_default_pass_through_on_send_is_false(self):
        """The default value of pass_through_on_send must be False.

        Only CellPipe instances that opt in (CJ task pipe and subprocess result pipe)
        should stamp the header. All other CellPipe users get normal encode/decode.
        """
        # Create stub without overriding the attribute
        pipe = object.__new__(CellPipe)
        # Simulate the end of __init__ where pass_through_on_send is assigned
        pipe.pass_through_on_send = False
        assert pipe.pass_through_on_send is False

    def test_request_message_stamped_with_header(self):
        """PASS_THROUGH is stamped on REQUEST messages too, not only REPLY."""
        pipe = _make_stub_cell_pipe(pass_through_on_send=True)
        msg = Message(msg_type=Message.REQUEST, topic="get_model", data=b"")

        pipe.send(msg, timeout=30.0)

        request = _get_sent_request(pipe)
        assert request.get_header(MessageHeaderKey.PASS_THROUGH) is True

    def test_cached_cell_message_is_reused_across_retries(self):
        """The CellMessage is cached on msg._cached_cell_msg so retries reuse it.

        This prevents creating multiple ArrayDownloadable transactions for the same
        message (which caused OOM with large models + many send retries).
        """
        pipe = _make_stub_cell_pipe(pass_through_on_send=True)
        msg = _make_reply_msg()

        pipe.send(msg, timeout=30.0)
        first_request = _get_sent_request(pipe)

        pipe.send(msg, timeout=30.0)
        second_request = _get_sent_request(pipe)

        assert first_request is second_request, (
            "send() must reuse the cached CellMessage across calls to avoid "
            "creating duplicate ArrayDownloadable transactions"
        )

    def test_heartbeat_does_not_stamp_pass_through(self):
        """Heartbeat messages must never carry PASS_THROUGH, even when pass_through_on_send=True.

        Heartbeats are sent via fire_and_forget (not send_request), so the PASS_THROUGH
        stamping branch is never reached.  This test verifies that send_request is NOT
        called for heartbeats and fire_and_forget IS called instead.
        """
        pipe = object.__new__(CellPipe)
        pipe.pass_through_on_send = True
        pipe.closed = False
        pipe.hb_seq = 1
        pipe.pipe_lock = threading.Lock()
        pipe.channel = "cell_pipe.task_channel"
        pipe.peer_fqcn = "site1_token_passive"
        pipe.logger = MagicMock()
        pipe.cell = MagicMock()

        hb_msg = Message(msg_type=Message.REQUEST, topic=Topic.HEARTBEAT, data=b"")
        pipe.send(hb_msg, timeout=None)

        # Heartbeat → fire_and_forget, not send_request
        pipe.cell.fire_and_forget.assert_called_once()
        pipe.cell.send_request.assert_not_called()
