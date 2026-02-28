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
Unit tests for Fix 16: subprocess exit gating via DOWNLOAD_COMPLETE_CB.

Without Fix 16, FlareAgent._do_submit_result() returns as soon as
send_to_peer() ACKs the pipe message.  With reverse PASS_THROUGH active, CJ
ACKs immediately (LazyDownloadRef creation is microseconds) while the server
downloads tensors asynchronously from the subprocess DownloadService.  If the
subprocess exits right after ACK, the DownloadService disappears and the server
gets "no ref found".

Fix 16 wires a threading.Event through FOBSContextKey.DOWNLOAD_COMPLETE_CB so
that _create_downloader() sets it as the ObjectDownloader transaction_done_cb.
When the transaction completes (all receivers finished), the event is set and
_do_submit_result() unblocks and returns.

Tests verify:

  FlareAgent._do_submit_result():
  1. CellPipe + pass_through_on_send=True → DOWNLOAD_COMPLETE_CB registered in
     FOBS context before send_to_peer().
  2. CellPipe + pass_through_on_send=True → waits for DOWNLOAD_COMPLETE_CB to
     fire; returns True when it fires within timeout.
  3. Timeout path → warning logged and True returned (non-fatal: server may
     still be downloading when we proceed).
  4. send_to_peer() fails → returns False without waiting.
  5. CellPipe + pass_through_on_send=False → falls back to plain send_to_peer()
     (no DOWNLOAD_COMPLETE_CB, no wait).
  6. Non-CellPipe (FilePipe) → falls back to plain send_to_peer().
  7. DOWNLOAD_COMPLETE_CB is cleared from FOBS context after the wait (cleanup).

  via_downloader._create_downloader():
  8. DOWNLOAD_COMPLETE_CB present in fobs_ctx → passed as transaction_done_cb.
  9. DOWNLOAD_COMPLETE_CB absent → transaction_done_cb=None (no gating).
  10. _on_tx_done function removed → GC callback no longer exists.

  ClientConfig:
  11. get_download_complete_timeout() returns configured value.
  12. get_download_complete_timeout() returns 1800.0 when not set.
"""

import threading
from unittest.mock import MagicMock, patch

from nvflare.client.flare_agent import FlareAgent, _TaskContext
from nvflare.fuel.utils.fobs import FOBSContextKey

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cell_pipe(pass_through_on_send: bool = True):
    """Return a mock CellPipe with a trackable cell.update_fobs_context."""
    from nvflare.fuel.utils.pipe.cell_pipe import CellPipe

    pipe = MagicMock(spec=CellPipe)
    pipe.pass_through_on_send = pass_through_on_send
    # cell.update_fobs_context() captures the props dict for inspection
    pipe.cell = MagicMock()
    return pipe


def _make_non_cell_pipe():
    """Return a mock FilePipe (not a CellPipe subclass)."""
    from nvflare.fuel.utils.pipe.file_pipe import FilePipe

    pipe = MagicMock(spec=FilePipe)
    return pipe


def _make_agent(pipe, download_complete_timeout: float = 5.0):
    """Return a FlareAgent stub backed by the given pipe.

    Bypasses __init__ network setup by constructing manually.
    """
    agent = FlareAgent.__new__(FlareAgent)
    agent.logger = MagicMock()
    agent.pipe = pipe
    agent.submit_result_timeout = 30.0
    agent._download_complete_timeout = download_complete_timeout
    agent._close_pipe = False
    agent._close_metric_pipe = False
    agent.task_lock = threading.Lock()
    agent.asked_to_stop = False
    agent.current_task = None

    # pipe_handler.send_to_peer returns True by default
    agent.pipe_handler = MagicMock()
    agent.pipe_handler.send_to_peer.return_value = True

    return agent


def _make_task_ctx():
    return _TaskContext(task_id="tid-1", task_name="train", msg_id="msg-1")


# ---------------------------------------------------------------------------
# 1-7: FlareAgent._do_submit_result()
# ---------------------------------------------------------------------------


class TestDoSubmitResultGating:
    """FlareAgent._do_submit_result() gating behaviour."""

    def _patch_shareable(self, agent):
        agent.task_result_to_shareable = MagicMock(return_value=MagicMock())

    def test_download_complete_cb_registered_before_send(self):
        """DOWNLOAD_COMPLETE_CB must be in cell FOBS context before send_to_peer() is called."""
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe)
        self._patch_shareable(agent)

        registered_before_send = {}

        def capture_on_send(reply, timeout):
            # Inspect what was registered on the cell at send time
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_before_send["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            # Fire the callback to unblock the wait
            if registered_before_send.get("cb"):
                registered_before_send["cb"]("tid", "FINISHED", [])
            return True

        agent.pipe_handler.send_to_peer.side_effect = capture_on_send

        agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert "cb" in registered_before_send, "DOWNLOAD_COMPLETE_CB must be registered before send_to_peer()"
        assert callable(registered_before_send["cb"])

    def test_waits_for_cb_and_returns_true(self):
        """Returns True when DOWNLOAD_COMPLETE_CB fires within timeout."""
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=5.0)
        self._patch_shareable(agent)

        # Fire the callback from a background thread shortly after send_to_peer()
        registered_cb = {}

        def fire_cb_on_send(reply, timeout):
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_cb["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]

            def _fire():
                if registered_cb.get("cb"):
                    registered_cb["cb"]("tid", "FINISHED", [])

            threading.Thread(target=_fire, daemon=True).start()
            return True

        agent.pipe_handler.send_to_peer.side_effect = fire_cb_on_send

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True

    def test_timeout_logs_warning_and_returns_true(self):
        """When DOWNLOAD_COMPLETE_CB never fires, a warning is logged and True is returned.

        Subprocess exit is non-fatal even on timeout — the server may still be
        downloading, but blocking forever would hang the entire training job.
        """
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=0.05)  # very short timeout
        self._patch_shareable(agent)
        # send_to_peer succeeds but callback never fires

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True, "Non-fatal timeout must still return True"
        agent.logger.warning.assert_called_once()
        warning_msg = agent.logger.warning.call_args[0][0]
        assert "0.05" in warning_msg or "Download completion" in warning_msg

    def test_send_fails_returns_false_without_waiting(self):
        """When send_to_peer() fails, returns False and does NOT wait for the callback."""
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=60.0)
        self._patch_shareable(agent)
        agent.pipe_handler.send_to_peer.return_value = False

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is False

    def test_pass_through_false_uses_plain_send(self):
        """pass_through_on_send=False → plain send_to_peer() without event gating."""
        pipe = _make_cell_pipe(pass_through_on_send=False)
        agent = _make_agent(pipe)
        self._patch_shareable(agent)
        agent.pipe_handler.send_to_peer.return_value = True

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True
        # DOWNLOAD_COMPLETE_CB must NOT be registered on the cell
        for c in pipe.cell.update_fobs_context.call_args_list:
            assert (
                FOBSContextKey.DOWNLOAD_COMPLETE_CB not in c[0][0]
            ), "DOWNLOAD_COMPLETE_CB must not be set when pass_through_on_send=False"

    def test_non_cell_pipe_uses_plain_send(self):
        """Non-CellPipe (e.g. FilePipe) → plain send_to_peer(), no gating.

        FilePipe has no cell attribute; the code must not attempt to access
        pipe.cell.update_fobs_context.  We verify by confirming the result is
        True (send_to_peer succeeded) and that no AttributeError was raised.
        """
        pipe = _make_non_cell_pipe()
        agent = _make_agent(pipe)
        self._patch_shareable(agent)
        agent.pipe_handler.send_to_peer.return_value = True

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True
        agent.pipe_handler.send_to_peer.assert_called_once()

    def test_download_complete_cb_cleared_after_wait(self):
        """DOWNLOAD_COMPLETE_CB is set to None in FOBS context after the wait.

        Stale callbacks accumulate across rounds if not cleared and could fire
        for a later transaction, corrupting the gating Event for that round.
        """
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=0.05)
        self._patch_shareable(agent)

        agent._do_submit_result(_make_task_ctx(), None, "OK")

        # The last update_fobs_context call must clear DOWNLOAD_COMPLETE_CB
        last_call = pipe.cell.update_fobs_context.call_args_list[-1]
        last_props = last_call[0][0]
        assert FOBSContextKey.DOWNLOAD_COMPLETE_CB in last_props
        assert (
            last_props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is None
        ), "DOWNLOAD_COMPLETE_CB must be set to None after the wait"


# ---------------------------------------------------------------------------
# 8-9: via_downloader._create_downloader() — DOWNLOAD_COMPLETE_CB wiring
# ---------------------------------------------------------------------------


class TestCreateDownloaderCallback:
    """_create_downloader() must wire DOWNLOAD_COMPLETE_CB as transaction_done_cb."""

    def _make_fobs_ctx(self, cb=None):
        mock_cell = MagicMock()
        mock_cell.get_fqcn.return_value = "site1/job1"
        ctx = {
            FOBSContextKey.CELL: mock_cell,
            FOBSContextKey.NUM_RECEIVERS: 1,
        }
        if cb is not None:
            ctx[FOBSContextKey.DOWNLOAD_COMPLETE_CB] = cb
        return ctx

    def _make_decomposer(self):
        """Return a NumpyArrayDecomposer — the simplest concrete ViaDownloaderDecomposer."""
        from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer

        return NumpyArrayDecomposer()

    def test_download_complete_cb_passed_as_transaction_done_cb(self):
        """When DOWNLOAD_COMPLETE_CB is in fobs_ctx, it is wired as transaction_done_cb."""
        sentinel_cb = MagicMock()
        fobs_ctx = self._make_fobs_ctx(cb=sentinel_cb)

        with (
            patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader") as MockOD,
            patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.subscribe_to_msg_root"),
        ):
            MockOD.return_value = MagicMock()
            self._make_decomposer()._create_downloader(fobs_ctx)

        MockOD.assert_called_once()
        _, kwargs = MockOD.call_args
        assert (
            kwargs.get("transaction_done_cb") is sentinel_cb
        ), "DOWNLOAD_COMPLETE_CB must be wired as transaction_done_cb"

    def test_no_download_complete_cb_gives_none_transaction_done_cb(self):
        """Without DOWNLOAD_COMPLETE_CB in fobs_ctx, transaction_done_cb=None."""
        fobs_ctx = self._make_fobs_ctx(cb=None)

        with (
            patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader") as MockOD,
            patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.subscribe_to_msg_root"),
        ):
            MockOD.return_value = MagicMock()
            self._make_decomposer()._create_downloader(fobs_ctx)

        _, kwargs = MockOD.call_args
        assert (
            kwargs.get("transaction_done_cb") is None
        ), "transaction_done_cb must be None when DOWNLOAD_COMPLETE_CB is absent"

    def test_gc_callback_removed(self):
        """_on_tx_done (GC transaction_done_cb) must no longer exist in via_downloader."""
        import nvflare.fuel.utils.fobs.decomposers.via_downloader as vd

        assert not hasattr(vd, "_on_tx_done"), (
            "_on_tx_done must be removed; GC via transaction_done_cb was discarded "
            "(Fix 4 already frees base_obj synchronously)"
        )


# ---------------------------------------------------------------------------
# 11-12: ClientConfig.get_download_complete_timeout()
# ---------------------------------------------------------------------------


class TestClientConfigDownloadCompleteTimeout:
    """ClientConfig.get_download_complete_timeout() returns the configured value."""

    def test_returns_configured_value(self):
        """get_download_complete_timeout() returns the value from TASK_EXCHANGE section."""
        from nvflare.client.config import ClientConfig, ConfigKey

        cfg = ClientConfig(config={ConfigKey.TASK_EXCHANGE: {ConfigKey.DOWNLOAD_COMPLETE_TIMEOUT: 3600.0}})
        assert cfg.get_download_complete_timeout() == 3600.0

    def test_returns_default_when_not_set(self):
        """get_download_complete_timeout() returns 1800.0 when not configured."""
        from nvflare.client.config import ClientConfig

        cfg = ClientConfig(config={})
        assert cfg.get_download_complete_timeout() == 1800.0
