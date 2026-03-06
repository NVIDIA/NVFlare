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

"""Unit tests for Bug 3 fix: thread-local download-initiation detection.

Root cause: FlareAgent._do_submit_result() unconditionally waits 1800s for
DOWNLOAD_COMPLETE_CB after every result send when pass_through_on_send=True.
For validate results (metrics only, no tensors), _finalize_download_tx() finds
no downloadable objects and creates no download transaction — DOWNLOAD_COMPLETE_CB
never fires — subprocess blocks for 1800s.

Fix: Use thread-local was_download_initiated() / clear_download_initiated() in
via_downloader.py.  _finalize_download_tx() sets _tls.download_initiated=True
only when downloadable objects exist.  FlareAgent checks this flag after
send_to_peer() returns (synchronous, same thread) and returns immediately if
False (no download transaction created — validate result).

Why thread-local: task pipe and metric pipe share the same CoreCell
(same FQCN -> same _CellInfo cache), so a plain fobs_ctx flag could be clobbered
by concurrent metric serialisation from a different thread.

Tests verify:
  1. _finalize_download_tx() with downloadable objects sets was_download_initiated()=True.
  2. _finalize_download_tx() with no downloadable objects -> was_download_initiated()=False.
  3. clear_download_initiated() resets the flag -> was_download_initiated() returns False.
  4. Thread isolation: another thread setting _tls.download_initiated=True does NOT
     affect the main thread's was_download_initiated().
  5. FlareAgent: was_download_initiated()=False -> returns immediately (validate path).
  6. FlareAgent: was_download_initiated()=True -> waits for DOWNLOAD_COMPLETE_CB (train path).
  7. clear_download_initiated() called before send_to_peer() in FlareAgent.
"""

import threading
from unittest.mock import MagicMock

from nvflare.client.flare_agent import FlareAgent, _TaskContext
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import _tls, clear_download_initiated, was_download_initiated

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cell_pipe(pass_through_on_send=True):
    from nvflare.fuel.utils.pipe.cell_pipe import CellPipe

    pipe = MagicMock(spec=CellPipe)
    pipe.pass_through_on_send = pass_through_on_send
    pipe.cell = MagicMock()
    return pipe


def _make_agent(pipe, download_complete_timeout=5.0):
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
# 1-4: Thread-local helpers
# ---------------------------------------------------------------------------


class TestThreadLocal:
    """was_download_initiated() / clear_download_initiated() contract."""

    def setup_method(self):
        # Ensure clean state before each test
        clear_download_initiated()

    def test_default_is_false(self):
        """was_download_initiated() returns False when never set."""
        clear_download_initiated()
        assert was_download_initiated() is False

    def test_set_true_returns_true(self):
        """Setting _tls.download_initiated=True -> was_download_initiated()=True."""
        _tls.download_initiated = True
        assert was_download_initiated() is True

    def test_clear_resets_to_false(self):
        """clear_download_initiated() resets to False after True was set."""
        _tls.download_initiated = True
        clear_download_initiated()
        assert was_download_initiated() is False

    def test_thread_isolation(self):
        """Another thread's download_initiated=True does NOT affect the main thread.

        This is the core correctness property: task pipe and metric pipe share the
        same CoreCell — concurrent metric serialisation (different thread) must not
        set the main thread's detection flag.
        """
        clear_download_initiated()  # main thread starts as False

        other_thread_started = threading.Event()
        other_thread_done = threading.Event()

        def _other():
            # Simulate metric thread's _finalize_download_tx setting the flag
            _tls.download_initiated = True
            other_thread_started.set()
            other_thread_done.wait(timeout=1.0)

        t = threading.Thread(target=_other, daemon=True)
        t.start()
        other_thread_started.wait(timeout=1.0)

        # Main thread must still see False
        result = was_download_initiated()
        other_thread_done.set()
        t.join(timeout=1.0)

        assert result is False, (
            "Thread isolation failed: other thread's _tls.download_initiated=True "
            "must not affect the main thread's was_download_initiated()"
        )


# ---------------------------------------------------------------------------
# 5-7: FlareAgent._do_submit_result() with thread-local gating
# ---------------------------------------------------------------------------


class TestDoSubmitResultGatingWithThreadLocal:
    """FlareAgent._do_submit_result() skips wait when no download transaction."""

    def _patch_shareable(self, agent):
        agent.task_result_to_shareable = MagicMock(return_value=MagicMock())

    def setup_method(self):
        clear_download_initiated()

    def test_validate_result_returns_immediately(self):
        """validate result (no tensors): was_download_initiated()=False -> returns True immediately.

        Before Bug 3 fix, the subprocess would block for 1800s waiting for a
        DOWNLOAD_COMPLETE_CB that never fires (no download transaction created).
        After fix, it returns immediately and the next validate task can be consumed.
        """
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=1800.0)
        self._patch_shareable(agent)

        # _finalize_download_tx() did NOT run (no tensors) -> download_initiated stays False
        # This simulates the validate result path.
        clear_download_initiated()

        import time

        start = time.time()
        result = agent._do_submit_result(_make_task_ctx(), None, "OK")
        elapsed = time.time() - start

        assert result is True
        # Must return in well under 1s — NOT wait for download_complete_timeout
        assert elapsed < 1.0, (
            f"validate result must return immediately (got {elapsed:.2f}s); "
            "Bug 3 not fixed: subprocess would hang for 1800s on validate round 2+"
        )

    def test_train_result_waits_for_download_complete(self):
        """train result (has tensors): was_download_initiated()=True -> waits for DOWNLOAD_COMPLETE_CB.

        The existing gating behaviour (Fix 16) must be preserved for training results.
        """
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=5.0)
        self._patch_shareable(agent)

        registered_cb = {}

        def simulate_send_with_tensors(reply, timeout):
            # Simulate _finalize_download_tx() setting the flag (train result has tensors)
            _tls.download_initiated = True
            # Capture the registered callback
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_cb["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            # Fire the callback from a background thread to unblock the wait

            def _fire():
                if registered_cb.get("cb"):
                    registered_cb["cb"]("tid", "FINISHED", [])

            threading.Thread(target=_fire, daemon=True).start()
            return True

        agent.pipe_handler.send_to_peer.side_effect = simulate_send_with_tensors

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True
        assert "cb" in registered_cb, "DOWNLOAD_COMPLETE_CB must be registered for train result"

    def test_clear_called_before_send(self):
        """clear_download_initiated() must be called before send_to_peer() to reset stale flag.

        Without this, a True from a previous training round would carry over to the
        current validate round, causing a false-positive wait.
        """
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=0.05)
        self._patch_shareable(agent)

        # Pre-set a stale True (simulating previous training round)
        _tls.download_initiated = True

        cleared_before_send = {}

        def capture_state_at_send(reply, timeout):
            # At this point, clear_download_initiated() should have run
            cleared_before_send["value"] = was_download_initiated()
            return True

        agent.pipe_handler.send_to_peer.side_effect = capture_state_at_send

        agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert cleared_before_send.get("value") is False, (
            "clear_download_initiated() must be called before send_to_peer() "
            "to prevent stale True from previous training round carrying over to validate"
        )
