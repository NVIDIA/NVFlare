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
Unit tests for the PASS_THROUGH mechanism (Design 1: receiver-side opt-in).

PASS_THROUGH activates LazyDownloadRef decode so tensors are not downloaded at
an intermediate hop.  It can be triggered by two independent sources:

  1. Per-message header: MessageHeaderKey.PASS_THROUGH stamped by the sender
     (e.g. subprocess-side CellPipe with pass_through_on_send=True stamps this
     on result messages so CJ decodes them as LazyDownloadRef for the reverse
     path: subprocess → CJ → server direct download).

  2. Per-channel receiver opt-in: the channel name is added to
     cell.decode_pass_through_channels by ClientAPILauncherExecutor.initialize().
     When a message arrives on that channel it is decoded with PASS_THROUGH=True
     regardless of the sender header, enabling the forward path
     (server/aggregator → CJ → subprocess direct download) without the server
     needing to stamp anything.  Each job registers only its own pipe channel,
     so concurrent jobs with different channels are unaffected.

Either source activates PASS_THROUGH; both can coexist.  For channels not in
decode_pass_through_channels (the default — in-process executors, subprocess),
only an explicit sender header triggers PASS_THROUGH.

Tests verify:

  Adapter.call() (cell.py) — channel NOT in decode_pass_through_channels:
  1. PASS_THROUGH=True in message header  → decode_ctx[PASS_THROUGH] = True
  2. No PASS_THROUGH header               → decode_ctx[PASS_THROUGH] = False
  3. Explicit PASS_THROUGH=False header   → decode_ctx[PASS_THROUGH] = False

  Adapter.call() — channel IN decode_pass_through_channels (ext-process CJ):
  4. No header, channel registered        → decode_ctx[PASS_THROUGH] = True

  CellPipe.send() (cell_pipe.py):
  5. pass_through_on_send=True  → PASS_THROUGH header stamped on request
  6. pass_through_on_send=False → PASS_THROUGH header NOT stamped
  7. Heartbeat messages skip PASS_THROUGH even when pass_through_on_send=True
"""

import threading
from unittest.mock import MagicMock, patch

from nvflare.fuel.f3.cellnet.cell import Adapter, Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message as F3Message
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.pipe import Message, Topic
from nvflare.fuel.utils.waiter_utils import WaiterRC

# ---------------------------------------------------------------------------
# Helpers for Adapter tests
# ---------------------------------------------------------------------------

# Channel used by all Adapter test headers — must match _headers_without_pass_through().
_TEST_CHANNEL = "aux_communication"


def _make_mock_cell(captured_ctx: dict, decode_pass_through_channels: set = None):
    """Return a mock Cell whose get_fobs_context() captures the props it receives.

    decode_pass_through_channels mirrors Cell.decode_pass_through_channels.
    Pass a set containing _TEST_CHANNEL to simulate an ext-process CJ cell
    (receiver-side per-channel opt-in).
    """
    cell = MagicMock()
    cell.decode_pass_through_channels = (
        decode_pass_through_channels if decode_pass_through_channels is not None else set()
    )

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


def _make_adapter(captured_ctx: dict, decode_pass_through_channels: set = None):
    """Return an Adapter backed by a mock cell and a trivial callback."""
    cell = _make_mock_cell(captured_ctx, decode_pass_through_channels=decode_pass_through_channels)
    cb = MagicMock(return_value=MagicMock())
    return Adapter(cb=cb, my_info=None, cell=cell)


def _headers_without_pass_through(stream_req_id=""):
    """Build minimal valid headers that do NOT include PASS_THROUGH (e.g. Swarm P2P)."""
    return {
        StreamHeaderKey.STREAM_REQ_ID: stream_req_id,
        StreamHeaderKey.CHANNEL: _TEST_CHANNEL,
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
    """Adapter.call() must build a per-call decode_ctx from both the
    MessageHeaderKey.PASS_THROUGH header and cell.decode_pass_through_channels.
    Either source alone is sufficient to activate PASS_THROUGH=True."""

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

    def test_channel_in_decode_pass_through_channels_overrides_absent_header(self):
        """Channel registered in decode_pass_through_channels → PASS_THROUGH=True even without a header.

        This is the receiver-side per-channel opt-in for ext-process CJ cells (Design 1).
        ClientAPILauncherExecutor.initialize() adds the job's pipe channel to
        decode_pass_through_channels so incoming server/aggregator task messages on
        that channel are decoded with PASS_THROUGH=True regardless of whether the
        sender stamped the header.
        """
        captured = {}
        adapter = _make_adapter(captured, decode_pass_through_channels={_TEST_CHANNEL})

        # No PASS_THROUGH header — simulates server sending task without any stamp
        headers = _headers_without_pass_through()
        assert MessageHeaderKey.PASS_THROUGH not in headers  # guard

        future = _make_future(headers)

        with patch("nvflare.fuel.f3.cellnet.cell.decode_payload"):
            adapter.call(future)

        assert captured.get(FOBSContextKey.PASS_THROUGH) is True, (
            "Channel in decode_pass_through_channels must activate PASS_THROUGH=True in the "
            "decode context even when the sender did not stamp the header."
        )

    def test_unregistered_channel_does_not_activate_pass_through(self):
        """A different channel registered in decode_pass_through_channels does not affect this message.

        Concurrent-job safety: only the exact channel name opts in.
        """
        captured = {}
        # Register a different channel, not the one in the test headers
        adapter = _make_adapter(captured, decode_pass_through_channels={"some_other_channel"})

        headers = _headers_without_pass_through()  # channel = _TEST_CHANNEL
        assert MessageHeaderKey.PASS_THROUGH not in headers  # guard

        future = _make_future(headers)

        with patch("nvflare.fuel.f3.cellnet.cell.decode_payload"):
            adapter.call(future)

        assert (
            captured.get(FOBSContextKey.PASS_THROUGH) is False
        ), "A channel not in decode_pass_through_channels must not activate PASS_THROUGH."

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

        Only the subprocess-side CellPipe (subprocess result pipe, set in
        ExProcessClientAPI.init()) should opt in.  CJ's own pipe must NOT
        stamp the header on outgoing task messages.
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


# ---------------------------------------------------------------------------
# 7-8: Combined directional contract (forward path vs reverse path)
# ---------------------------------------------------------------------------


class TestPassThroughDirectionContract:
    """Documents the two-direction PASS_THROUGH contract (Design 1).

    Forward path (server/aggregator → CJ → subprocess):
      CJ cell has the job's pipe channel in decode_pass_through_channels
      (registered by ClientAPILauncherExecutor.initialize()).
      CJ Adapter.call() builds PASS_THROUGH=True for that channel regardless of sender header.
      ViaDownloaderDecomposer creates LazyDownloadRef(fqcn=server) at CJ.
      CJ re-encodes via LazyDownloadRefDecomposer (re-emits original datum).
      Subprocess Adapter.call() (channel not registered) downloads tensors
      directly from the server's DownloadService → real tensors at subprocess.

      Subprocess cell has empty decode_pass_through_channels (never populated).
      CJ pipe has pass_through_on_send=False → no PASS_THROUGH header on task msgs.
      Subprocess Adapter.call() builds PASS_THROUGH=False → downloads normally.

    Reverse path (subprocess → CJ → server):
      Subprocess-side CellPipe has pass_through_on_send=True (ExProcessClientAPI).
      PASS_THROUGH header is stamped on result messages.
      CJ Adapter.call() builds PASS_THROUGH=True (from header OR channel registration).
      ViaDownloaderDecomposer stores _LazyBatchInfo → recompose() returns
      LazyDownloadRef → CJ forwards the reference to the server for direct download.
    """

    def test_forward_path_cj_pipe_sends_task_without_pass_through_header(self):
        """Forward path step 1: CJ's pipe (default False) sends task → no header.

        Combined test connecting CellPipe.send() to the absence of the header that
        would otherwise trigger LazyDownloadRef creation at the subprocess.
        """
        cj_pipe = _make_stub_cell_pipe(pass_through_on_send=False)
        task_msg = Message(msg_type=Message.REQUEST, topic="train", data=b"model")
        cj_pipe.send(task_msg, timeout=30.0)

        sent = _get_sent_request(cj_pipe)
        header = sent.get_header(MessageHeaderKey.PASS_THROUGH)
        assert header is None, (
            "CJ's pipe (pass_through_on_send=False) must not stamp PASS_THROUGH.\n"
            "Presence of this header on task messages would give the subprocess\n"
            "PASS_THROUGH=True decode context → LazyDownloadRef → crash."
        )

    def test_forward_path_no_header_gives_subprocess_pass_through_false_decode_ctx(self):
        """Forward path: subprocess (channel not registered) + no header → PASS_THROUGH=False.

        The subprocess cell never has any channel in decode_pass_through_channels.
        CJ's pipe has pass_through_on_send=False → no PASS_THROUGH header on task msgs.
        Result: subprocess Adapter.call() builds PASS_THROUGH=False, so
        ViaDownloaderDecomposer downloads tensors normally and torch.as_tensor() succeeds.
        """
        captured = {}
        # Subprocess cell: empty decode_pass_through_channels (never populated)
        adapter = _make_adapter(captured)
        headers = _headers_without_pass_through()
        future = _make_future(headers)

        with patch("nvflare.fuel.f3.cellnet.cell.decode_payload"):
            adapter.call(future)

        assert captured.get(FOBSContextKey.PASS_THROUGH) is False, (
            "No PASS_THROUGH header + empty decode_pass_through_channels → "
            "subprocess decode ctx must have PASS_THROUGH=False.\n"
            "ViaDownloaderDecomposer.process_datum() then downloads tensors normally;\n"
            "torch.as_tensor() on the result works without error."
        )

    def test_reverse_path_subprocess_pipe_stamps_pass_through_on_result(self):
        """Reverse path step 1: subprocess pipe (pass_through_on_send=True) stamps header.

        Confirms the subprocess result pipe correctly stamps PASS_THROUGH so CJ
        creates LazyDownloadRef and forwards the reference to the server.
        """
        subprocess_pipe = _make_stub_cell_pipe(pass_through_on_send=True)
        result_msg = Message(msg_type=Message.REPLY, topic="submit_model", data=b"result")
        subprocess_pipe.send(result_msg, timeout=30.0)

        sent = _get_sent_request(subprocess_pipe)
        assert sent.get_header(MessageHeaderKey.PASS_THROUGH) is True, (
            "Subprocess pipe (pass_through_on_send=True) must stamp PASS_THROUGH=True "
            "on result messages so CJ's decode context is PASS_THROUGH=True."
        )

    def test_reverse_path_pass_through_header_gives_cj_pass_through_true_decode_ctx(self):
        """Reverse path step 2: header present → CJ Adapter builds PASS_THROUGH=True.

        Confirms that when the subprocess pipe stamps the header (reverse path),
        CJ's FOBS decode context has PASS_THROUGH=True, so ViaDownloaderDecomposer
        stores _LazyBatchInfo and recompose() returns LazyDownloadRef — CJ then
        forwards the reference to the server without materialising any tensor data.
        """
        captured = {}
        adapter = _make_adapter(captured)
        headers = _headers_without_pass_through()
        headers[MessageHeaderKey.PASS_THROUGH] = True  # stamped by subprocess pipe
        future = _make_future(headers)

        with patch("nvflare.fuel.f3.cellnet.cell.decode_payload"):
            adapter.call(future)

        assert captured.get(FOBSContextKey.PASS_THROUGH) is True, (
            "PASS_THROUGH=True header → CJ decode ctx must have PASS_THROUGH=True.\n"
            "ViaDownloaderDecomposer.process_datum() then stores _LazyBatchInfo;\n"
            "recompose() returns LazyDownloadRef that CJ forwards to the server."
        )


# ---------------------------------------------------------------------------
# Tests for Cell._send_one_request() PASS_THROUGH on the REPLY decode path
# (FedAvg GET_TASK response and any other request/reply exchange)
# ---------------------------------------------------------------------------

# Channel passed to _run_send_one_request — must match the literal below.
_SEND_REQ_CHANNEL = "ch"


def _make_cell_stub(captured_ctx: dict, decode_pass_through_channels: set = None):
    """Minimal stub for Cell._send_one_request() — only the attributes that
    method touches are initialised; everything else stays as MagicMock."""
    cell = MagicMock(spec=Cell)
    cell.requests_dict = {}
    cell.decode_pass_through_channels = (
        decode_pass_through_channels if decode_pass_through_channels is not None else set()
    )
    cell.logger = MagicMock()
    cell._future_wait.return_value = True  # both sending-complete and receiving-complete
    cell._get_result.return_value = MagicMock()

    def _get_fobs_context(props=None):
        ctx = {}
        if props:
            ctx.update(props)
        captured_ctx.update(ctx)
        return ctx

    cell.get_fobs_context.side_effect = _get_fobs_context
    return cell


def _run_send_one_request(cell_stub, reply_headers: dict):
    """Drive Cell._send_one_request() with a fake reply carrying reply_headers.

    Patches SimpleWaiter so we can inject reply_headers into the receiving
    future without wiring up real threading primitives, and patches
    conditional_wait to skip the real event wait.
    """
    r_future = MagicMock()
    r_future.headers = reply_headers
    r_future.result.return_value = b""

    mock_waiter = MagicMock()
    mock_waiter.receiving_future = r_future

    request = F3Message(headers={}, payload=b"")

    with (
        patch("nvflare.fuel.f3.cellnet.cell.SimpleWaiter", return_value=mock_waiter),
        patch("nvflare.fuel.f3.cellnet.cell.conditional_wait", return_value=WaiterRC.IS_SET),
        patch("nvflare.fuel.f3.cellnet.cell.decode_payload"),
    ):
        Cell._send_one_request(cell_stub, _SEND_REQ_CHANNEL, "target", "topic", request, timeout=5.0)


class TestSendOneRequestPassThrough:
    """Cell._send_one_request() must inject the correct PASS_THROUGH value into
    the fobs_ctx passed to decode_payload() when decoding the reply.

    This path covers the FedAvg GET_TASK reply (and any other send_request
    exchange) — distinct from Adapter.call() which handles incoming REQUESTs.
    """

    def test_channel_registered_no_header_gives_pass_through_true(self):
        """Channel in decode_pass_through_channels + no reply header → PASS_THROUGH=True in fobs_ctx.

        This is the FedAvg forward path: CJ (ext-process) sends GET_TASK,
        server replies without stamping PASS_THROUGH.  The registered channel
        activates PASS_THROUGH so tensors arrive as LazyDownloadRef(server)
        rather than being downloaded inline.
        """
        captured = {}
        cell = _make_cell_stub(captured, decode_pass_through_channels={_SEND_REQ_CHANNEL})

        _run_send_one_request(cell, reply_headers={})

        assert captured.get(FOBSContextKey.PASS_THROUGH) is True, (
            "Channel in decode_pass_through_channels + no reply header must give PASS_THROUGH=True "
            "so CJ creates LazyDownloadRef from the server reply."
        )

    def test_channel_not_registered_no_header_gives_pass_through_false(self):
        """Channel not in decode_pass_through_channels + no reply header → PASS_THROUGH=False.

        Default cell (in-process executor, subprocess): tensors are downloaded
        immediately at this hop, not deferred as LazyDownloadRef.
        """
        captured = {}
        cell = _make_cell_stub(captured)

        _run_send_one_request(cell, reply_headers={})

        assert (
            captured.get(FOBSContextKey.PASS_THROUGH) is False
        ), "Empty decode_pass_through_channels + no reply header must give PASS_THROUGH=False."

    def test_reply_header_true_with_channel_not_registered_gives_pass_through_true(self):
        """Reply carries PASS_THROUGH=True header + channel not registered → PASS_THROUGH=True.

        The per-message header alone is sufficient to activate PASS_THROUGH;
        channel registration is not required.
        """
        captured = {}
        cell = _make_cell_stub(captured)

        _run_send_one_request(cell, reply_headers={MessageHeaderKey.PASS_THROUGH: True})

        assert captured.get(FOBSContextKey.PASS_THROUGH) is True

    def test_reply_header_false_with_channel_not_registered_gives_pass_through_false(self):
        """Explicit PASS_THROUGH=False in reply header + channel not registered → PASS_THROUGH=False."""
        captured = {}
        cell = _make_cell_stub(captured)

        _run_send_one_request(cell, reply_headers={MessageHeaderKey.PASS_THROUGH: False})

        assert captured.get(FOBSContextKey.PASS_THROUGH) is False
