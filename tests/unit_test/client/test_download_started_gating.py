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

"""Unit tests for Bug 3 fix: DOWNLOAD_STARTED_CB skip-wait for validate results.

Bug 3 — FlareAgent download gating blocks validate tasks in CSE rounds:
    Root Cause: When pass_through_on_send is enabled, _do_submit_result()
    unconditionally waits for DOWNLOAD_COMPLETE_CB after sending any result.
    For validate results (metrics only, no tensors), ViaDownloader never
    creates a download transaction, so the callback is never fired.
    The subprocess blocks until timeout (1800s), missing subsequent tasks.

    Fix: Add a DOWNLOAD_STARTED_CB mechanism.  ViaDownloader invokes it
    when a download transaction is actually created.  FlareAgent checks
    download_started.is_set() after send; if not set, proceeds immediately.

Tests verify:

  FlareAgent._do_submit_result():
  1. DOWNLOAD_STARTED_CB is registered in FOBS context before send.
  2. When download_started is NOT set after send → returns True immediately
     without waiting for DOWNLOAD_COMPLETE_CB (validate metrics path).
  3. When download_started IS set after send → waits for DOWNLOAD_COMPLETE_CB
     as before (train tensors path).
  4. DOWNLOAD_STARTED_CB is cleared from FOBS context after the call.

  ViaDownloaderDecomposer._finalize_download_tx():
  5. When downloadable_objs exist → DOWNLOAD_STARTED_CB is invoked.
  6. When downloadable_objs is empty/None → DOWNLOAD_STARTED_CB is NOT invoked.
  7. When DOWNLOAD_STARTED_CB is absent in fobs_ctx → no error raised.
"""

import threading
from unittest.mock import MagicMock, patch

from nvflare.client.flare_agent import FlareAgent, _TaskContext
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import ViaDownloaderDecomposer, _CtxKey

# ---------------------------------------------------------------------------
# Helpers — FlareAgent
# ---------------------------------------------------------------------------


def _make_cell_pipe(pass_through_on_send: bool = True):
    """Return a mock CellPipe with a trackable cell.update_fobs_context."""
    from nvflare.fuel.utils.pipe.cell_pipe import CellPipe

    pipe = MagicMock(spec=CellPipe)
    pipe.pass_through_on_send = pass_through_on_send
    pipe.cell = MagicMock()
    return pipe


def _make_agent(pipe, download_complete_timeout: float = 5.0):
    """Return a FlareAgent stub backed by the given pipe."""
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

    agent.pipe_handler = MagicMock()
    agent.pipe_handler.send_to_peer.return_value = True

    return agent


def _make_task_ctx():
    return _TaskContext(task_id="tid-1", task_name="validate", msg_id="msg-1")


# ---------------------------------------------------------------------------
# Helpers — ViaDownloaderDecomposer
# ---------------------------------------------------------------------------


class _FakeDecomposer(ViaDownloaderDecomposer):
    """Concrete stub for testing _finalize_download_tx."""

    def __init__(self):
        super().__init__(max_chunk_size=1024 * 1024, config_var_prefix="np_")

    def to_downloadable(self, items, max_chunk_size, fobs_ctx):
        return MagicMock()

    def download(self, from_fqcn, ref_id, per_request_timeout, cell, secure=False, optional=False, abort_signal=None):
        return None, {}

    def get_download_dot(self):
        return 99

    def native_decompose(self, target, manager=None):
        return b""

    def native_recompose(self, data, manager=None):
        return data

    def supported_type(self):
        return object


def _make_fobs_ctx_with_objects(objects=None, started_cb=None):
    """Create a fobs_ctx dict with optional downloadable objects and DOWNLOAD_STARTED_CB."""
    ctx = {
        FOBSContextKey.CELL: MagicMock(),
        FOBSContextKey.NUM_RECEIVERS: 1,
    }
    if objects is not None:
        ctx[_CtxKey.OBJECTS] = objects
    if started_cb is not None:
        ctx[FOBSContextKey.DOWNLOAD_STARTED_CB] = started_cb
    return ctx


# ---------------------------------------------------------------------------
# 1-4: FlareAgent._do_submit_result() — DOWNLOAD_STARTED_CB behaviour
# ---------------------------------------------------------------------------


class TestDownloadStartedGating:
    """FlareAgent._do_submit_result() must skip download wait for validate results."""

    def _patch_shareable(self, agent):
        agent.task_result_to_shareable = MagicMock(return_value=MagicMock())

    def test_download_started_cb_registered_before_send(self):
        """DOWNLOAD_STARTED_CB must be in FOBS context before send_to_peer()."""
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe)
        self._patch_shareable(agent)

        registered_before_send = {}

        def capture_on_send(reply, timeout):
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_STARTED_CB in props
                    and props[FOBSContextKey.DOWNLOAD_STARTED_CB] is not None
                ):
                    registered_before_send["cb"] = props[FOBSContextKey.DOWNLOAD_STARTED_CB]
            # Fire download_complete to unblock if needed
            if FOBSContextKey.DOWNLOAD_COMPLETE_CB in props and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None:
                props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]("tid", "FINISHED", [])
            return True

        agent.pipe_handler.send_to_peer.side_effect = capture_on_send

        agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert "cb" in registered_before_send, "DOWNLOAD_STARTED_CB must be registered before send_to_peer()"
        assert callable(registered_before_send["cb"])

    def test_no_download_started_returns_true_immediately(self):
        """When download_started is NOT set, returns True without waiting for download_done."""
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=60.0)
        self._patch_shareable(agent)

        # send_to_peer succeeds but no DOWNLOAD_STARTED_CB fired (validate metrics)
        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True
        # Verify info log mentions "no download transaction"
        info_msgs = [str(c) for c in agent.logger.info.call_args_list]
        assert any(
            "no download transaction" in m for m in info_msgs
        ), f"Must log 'no download transaction' for validate path. Got: {info_msgs}"

    def test_download_started_waits_for_download_complete(self):
        """When download_started IS set, waits for DOWNLOAD_COMPLETE_CB."""
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=5.0)
        self._patch_shareable(agent)

        def fire_both_cbs(reply, timeout):
            # Extract registered callbacks
            started_cb = None
            complete_cb = None
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if FOBSContextKey.DOWNLOAD_STARTED_CB in props:
                    started_cb = props[FOBSContextKey.DOWNLOAD_STARTED_CB]
                if FOBSContextKey.DOWNLOAD_COMPLETE_CB in props:
                    complete_cb = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]

            # Simulate ViaDownloader creating a download transaction (train path)
            if started_cb:
                started_cb()
            # Simulate server finishing download
            if complete_cb:
                threading.Thread(target=lambda: complete_cb("tid", "FINISHED", []), daemon=True).start()
            return True

        agent.pipe_handler.send_to_peer.side_effect = fire_both_cbs

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True
        # Verify info log mentions "waiting" (not "no download transaction")
        info_msgs = [str(c) for c in agent.logger.info.call_args_list]
        assert any("waiting" in m for m in info_msgs), f"Must log 'waiting' for train path. Got: {info_msgs}"

    def test_download_started_cb_cleared_after_call(self):
        """DOWNLOAD_STARTED_CB must be set to None after the call completes."""
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=0.05)
        self._patch_shareable(agent)

        agent._do_submit_result(_make_task_ctx(), None, "OK")

        # The last update_fobs_context call must clear DOWNLOAD_STARTED_CB
        last_call = pipe.cell.update_fobs_context.call_args_list[-1]
        last_props = last_call[0][0]
        assert FOBSContextKey.DOWNLOAD_STARTED_CB in last_props
        assert (
            last_props[FOBSContextKey.DOWNLOAD_STARTED_CB] is None
        ), "DOWNLOAD_STARTED_CB must be set to None after the call"


# ---------------------------------------------------------------------------
# 5-7: ViaDownloaderDecomposer._finalize_download_tx() — DOWNLOAD_STARTED_CB
# ---------------------------------------------------------------------------


class TestFinalizeDownloadTxStartedCb:
    """_finalize_download_tx() must invoke DOWNLOAD_STARTED_CB when objects exist."""

    def test_started_cb_invoked_when_objects_exist(self):
        """DOWNLOAD_STARTED_CB is called when downloadable_objs is non-empty."""
        decomposer = _FakeDecomposer()
        started_cb = MagicMock()

        objects = [("ref-1", MagicMock())]
        fobs_ctx = _make_fobs_ctx_with_objects(objects=objects, started_cb=started_cb)

        mgr = MagicMock()
        mgr.fobs_ctx = fobs_ctx

        with (
            patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader") as MockOD,
            patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.subscribe_to_msg_root"),
        ):
            MockOD.return_value = MagicMock()
            decomposer._finalize_download_tx(mgr)

        started_cb.assert_called_once()

    def test_started_cb_not_invoked_when_no_objects(self):
        """DOWNLOAD_STARTED_CB is NOT called when downloadable_objs is None."""
        decomposer = _FakeDecomposer()
        started_cb = MagicMock()

        fobs_ctx = _make_fobs_ctx_with_objects(objects=None, started_cb=started_cb)

        mgr = MagicMock()
        mgr.fobs_ctx = fobs_ctx

        decomposer._finalize_download_tx(mgr)

        started_cb.assert_not_called()

    def test_started_cb_not_invoked_when_empty_objects(self):
        """DOWNLOAD_STARTED_CB is NOT called when downloadable_objs is empty list."""
        decomposer = _FakeDecomposer()
        started_cb = MagicMock()

        fobs_ctx = _make_fobs_ctx_with_objects(objects=[], started_cb=started_cb)

        mgr = MagicMock()
        mgr.fobs_ctx = fobs_ctx

        decomposer._finalize_download_tx(mgr)

        started_cb.assert_not_called()

    def test_no_error_when_started_cb_absent(self):
        """No error when DOWNLOAD_STARTED_CB is not in fobs_ctx."""
        decomposer = _FakeDecomposer()

        objects = [("ref-1", MagicMock())]
        fobs_ctx = _make_fobs_ctx_with_objects(objects=objects, started_cb=None)
        # Remove the key entirely
        fobs_ctx.pop(FOBSContextKey.DOWNLOAD_STARTED_CB, None)

        mgr = MagicMock()
        mgr.fobs_ctx = fobs_ctx

        with (
            patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader") as MockOD,
            patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.subscribe_to_msg_root"),
        ):
            MockOD.return_value = MagicMock()
            # Should not raise
            decomposer._finalize_download_tx(mgr)
