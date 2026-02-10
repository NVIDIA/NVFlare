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
Tests that assert the CORRECT behavior for streaming pool starvation scenarios.
These tests FAIL against the current code, reproducing the bug.

Root cause chain:
  1. stream_thread_pool is shared for blob read, callback execution, and tx tasks.
  2. BlobHandler.handle_blob_cb submits blob read, then runs callback synchronously —
     the callback (Adapter.call) blocks on future.result(), runs expensive produce(),
     and sends the reply blob all on the same worker thread.
  3. Under load the pool is saturated → reply processing is delayed.
  4. Request side times out → waiter is popped from requests_dict.
  5. Late reply arrives → _process_reply finds no waiter → silently discarded.
  6. download_object treats timeout as terminal (no retry) → transfer fails.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.f3.cellnet.cell import Cell, SimpleWaiter
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.streaming.blob_streamer import BlobHandler
from nvflare.fuel.f3.streaming.download_service import Consumer, ProduceRC, download_object
from nvflare.fuel.f3.streaming.stream_types import StreamFuture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class TrackingConsumer(Consumer):
    """Tracks download_completed / download_failed calls."""

    def __init__(self):
        super().__init__()
        self.completed = False
        self.failed = False
        self.failure_reason = None

    def consume(self, ref_id, state, data):
        return state

    def download_completed(self, ref_id):
        self.completed = True

    def download_failed(self, ref_id, reason):
        self.failed = True
        self.failure_reason = reason


# ---------------------------------------------------------------------------
# Test 1 — BlobHandler should not block pool worker with synchronous callback
# ---------------------------------------------------------------------------
class TestBlobHandlerPoolBlocking:
    """BlobHandler.handle_blob_cb runs blob_cb synchronously on the pool worker,
    which prevents other tasks from being scheduled on that worker.

    EXPECTED (correct): handle_blob_cb should not block the pool worker —
    the callback should be decoupled so the worker is freed promptly.
    ACTUAL (bug): blob_cb runs synchronously, blocking the worker for the
    entire duration of produce() + send_blob()."""

    def test_blob_handler_should_not_block_pool_worker(self):
        """Pool worker should be free to handle other tasks while blob_cb executes."""
        callback_started = threading.Event()
        callback_release = threading.Event()

        def slow_blob_cb(future, *args, **kwargs):
            callback_started.set()
            callback_release.wait()  # simulates expensive produce + send_blob

        handler = BlobHandler(slow_blob_cb)
        mock_future = MagicMock(spec=StreamFuture)
        mock_future.headers = {}
        mock_stream = MagicMock()
        mock_stream.get_size.return_value = 0

        pool = ThreadPoolExecutor(max_workers=1)

        def wrapper():
            handler.handle_blob_cb(mock_future, mock_stream, False)

        pool.submit(wrapper)
        assert callback_started.wait(timeout=2.0)

        # With correct behavior, the pool worker should be free to run other tasks
        # even while blob_cb is still executing.
        second_ran = threading.Event()
        pool.submit(lambda: second_ran.set())
        ran = second_ran.wait(timeout=0.5)

        callback_release.set()
        pool.shutdown(wait=True)

        assert ran, (
            "BUG: pool worker is blocked by synchronous blob_cb execution. "
            "handle_blob_cb should not hold the worker while the callback runs."
        )


# ---------------------------------------------------------------------------
# Test 2 — Late reply should still be accepted after timeout
# ---------------------------------------------------------------------------
class TestLateReplyHandling:
    """When the request side times out and a reply arrives late, the reply
    should still be recoverable.

    EXPECTED (correct): _process_reply should be able to deliver a late reply
    so the caller can use it (or at least not silently discard valid data).
    ACTUAL (bug): _get_result pops the waiter, so _process_reply hits KeyError
    and discards the reply."""

    def test_late_reply_should_be_accepted(self):
        """A reply arriving after timeout should still be deliverable, not silently discarded."""
        cell = object.__new__(Cell)
        cell.requests_dict = {}
        cell.logger = MagicMock()

        req_id = "test-req-late"
        waiter = SimpleWaiter(req_id=req_id, result=make_reply(ReturnCode.TIMEOUT))
        cell.requests_dict[req_id] = waiter

        # Simulate timeout — this pops the waiter
        cell._get_result(req_id)

        # Now a late (but valid) reply arrives
        late_future = MagicMock(spec=StreamFuture)
        late_future.headers = {"stream_req_id": req_id}

        cell._process_reply(late_future)

        # With correct behavior, the reply should NOT be discarded.
        # The warning about "discarded" means we lost valid data.
        if cell.logger.warning.called:
            warning_msg = cell.logger.warning.call_args[0][0].lower()
            assert "discarded" not in warning_msg, (
                "BUG: late reply was silently discarded because waiter was already "
                "popped by timeout. Late replies should be recoverable."
            )

    def test_full_starvation_cascade_should_recover(self):
        """Simulate pool starvation → timeout → late reply. The reply should
        still reach the caller."""
        cell = object.__new__(Cell)
        cell.requests_dict = {}
        cell.logger = MagicMock()

        req_id = "cascade-req"
        waiter = SimpleWaiter(req_id=req_id, result=make_reply(ReturnCode.TIMEOUT))
        cell.requests_dict[req_id] = waiter

        pool = ThreadPoolExecutor(max_workers=1)
        blocker = threading.Event()
        reply_delivered = threading.Event()
        reply_discarded = threading.Event()

        def delayed_reply():
            blocker.wait()
            late_future = MagicMock(spec=StreamFuture)
            late_future.headers = {"stream_req_id": req_id}
            cell._process_reply(late_future)
            # Check if the reply was actually delivered vs discarded
            if cell.logger.warning.called and "discarded" in cell.logger.warning.call_args[0][0].lower():
                reply_discarded.set()
            else:
                reply_delivered.set()

        pool.submit(delayed_reply)

        # Request side times out and removes waiter
        time.sleep(0.1)
        cell._get_result(req_id)

        # Reply handler finally runs
        blocker.set()

        # Wait for outcome
        delivered = reply_delivered.wait(timeout=2.0)
        discarded = reply_discarded.wait(timeout=0.1)

        pool.shutdown(wait=True)

        assert not discarded, (
            "BUG: pool starvation caused reply to arrive after waiter was popped, "
            "and the reply was silently discarded."
        )
        assert delivered, "Reply should have been delivered despite arriving late."


# ---------------------------------------------------------------------------
# Test 3 — download_object should retry on transient timeout
# ---------------------------------------------------------------------------
class TestDownloadObjectRetry:
    """download_object calls download_failed and exits immediately on any
    non-OK return code, including transient timeouts.

    EXPECTED (correct): transient timeouts should be retried — a single slow
    response should not kill the entire transfer.
    ACTUAL (bug): timeout is treated as terminal with no retry."""

    def test_transient_timeout_should_be_retried(self):
        """A transient timeout followed by success should complete the download."""
        consumer = TrackingConsumer()
        mock_cell = MagicMock()

        # First call: transient TIMEOUT. Second call: success with EOF.
        timeout_reply = make_reply(ReturnCode.TIMEOUT)
        ok_reply = make_reply(ReturnCode.OK, body={"status": ProduceRC.EOF})
        mock_cell.send_request.side_effect = [timeout_reply, ok_reply]

        download_object(
            from_fqcn="server",
            ref_id="ref-retry",
            per_request_timeout=1.0,
            cell=mock_cell,
            consumer=consumer,
        )

        retries = mock_cell.send_request.call_count - 1
        print(f"retries={retries} (total calls={mock_cell.send_request.call_count})")

        assert consumer.completed, (
            "BUG: download_object treated a transient timeout as terminal. "
            f"Got download_failed with reason: {consumer.failure_reason!r}. "
            "It should have retried and succeeded on the second attempt."
        )
        assert not consumer.failed
        assert mock_cell.send_request.call_count == 2, (
            f"BUG: send_request called {mock_cell.send_request.call_count} time(s), "
            "expected 2 (initial + retry)."
        )

    def test_single_timeout_should_not_fail_download(self):
        """A single timeout from send_request should not permanently fail the download."""
        consumer = TrackingConsumer()
        mock_cell = MagicMock()

        timeout_reply = make_reply(ReturnCode.TIMEOUT)
        ok_reply = make_reply(ReturnCode.OK, body={"status": ProduceRC.EOF})
        mock_cell.send_request.side_effect = [timeout_reply, ok_reply]

        download_object(
            from_fqcn="server",
            ref_id="ref-single-timeout",
            per_request_timeout=1.0,
            cell=mock_cell,
            consumer=consumer,
        )

        retries = mock_cell.send_request.call_count - 1
        print(f"retries={retries} (total calls={mock_cell.send_request.call_count})")

        assert not consumer.failed, (
            "BUG: download_object failed permanently on a single transient timeout. "
            f"Reason: {consumer.failure_reason!r}. "
            "Transient errors should be retried before giving up."
        )
