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
End-to-end download starvation test using real Cell objects and blob streaming.

Exercises the FULL download_object() flow:
  download_object() -> Cell.send_request() -> send_blob() ->
  BlobHandler -> stream_thread_pool / callback_thread_pool -> Adapter.call() ->
  DownloadService._handle_download() -> Downloadable.produce() -> Consumer.consume()

Two test classes, each with their own Cell pair:

1. TestDownloadWithFix (SHOULD PASS):
   Real code: blob_cb dispatched to callback_thread_pool (async).

2. TestDownloadPreFixStarvation (SHOULD FAIL/timeout):
   Before creating cells, patches BlobHandler.handle_blob_cb to call blob_cb
   SYNCHRONOUSLY + slows _read_stream with 0.2s delay. Creates cells AFTER patch
   so the bound methods capture the patched behavior.
   With a 4-worker pool and 8 concurrent downloads -> deadlock.
"""

import threading
import time
from typing import Any, Optional, Tuple

import pytest

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.blob_streamer import BlobHandler, BlobTask
from nvflare.fuel.f3.streaming.download_service import Consumer, Downloadable, ProduceRC, download_object
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader
from nvflare.fuel.f3.streaming.stream_utils import CheckedExecutor
from nvflare.fuel.utils.network_utils import get_open_ports

SERVER_CELL = "server"
CLIENT_CELL = "client"
SERVER2_CELL = "server2"
CLIENT2_CELL = "client2"

CHUNK_SIZE = 256
TOTAL_SIZE = 50 * 1024  # 50 KB -> ~200 chunks per download
NUM_PARALLEL = 8
PER_REQ_TIMEOUT = 5.0
TX_TIMEOUT = 120.0
CELL_CONNECT_TIMEOUT = 2.0  # seconds to wait for TCP cell connection

# Event used to cancel the slow _read_stream delay during teardown,
# so deadlocked workers unblock and the process exits cleanly.
_stop_delay = threading.Event()


class ChunkedDownloadable(Downloadable):
    def __init__(self, data: bytes, chunk_size: int):
        super().__init__(data)
        self.chunk_size = chunk_size

    def produce(self, state: dict, requester: str) -> Tuple[str, Any, dict]:
        data = self.base_obj
        offset = state.get("offset", 0) if state else 0
        if offset >= len(data):
            return ProduceRC.EOF, None, {}
        end = min(offset + self.chunk_size, len(data))
        return ProduceRC.OK, data[offset:end], {"offset": end}


class ChunkedConsumer(Consumer):
    def __init__(self):
        super().__init__()
        self.received = bytearray()
        self.completed = threading.Event()
        self.error: Optional[str] = None

    def consume(self, ref_id: str, state: dict, data: Any) -> dict:
        if isinstance(data, (bytes, bytearray)):
            self.received.extend(data)
        return state

    def download_completed(self, ref_id: str):
        self.completed.set()

    def download_failed(self, ref_id: str, reason: str):
        self.error = reason
        self.completed.set()


def _make_test_data() -> bytes:
    return bytes(range(256)) * (TOTAL_SIZE // 256)


def _run_parallel_downloads(
    server_name, server, client, data, num_parallel, per_req_timeout, wait_secs=60.0, max_retries=0
):
    consumers = []
    threads = []

    for _ in range(num_parallel):
        downloadable = ChunkedDownloadable(data, CHUNK_SIZE)
        downloader = ObjectDownloader(cell=server, timeout=TX_TIMEOUT, num_receivers=1)
        ref_id = downloader.add_object(downloadable)

        consumer = ChunkedConsumer()
        consumers.append(consumer)

        t = threading.Thread(
            target=download_object,
            kwargs=dict(
                from_fqcn=server_name,
                ref_id=ref_id,
                per_request_timeout=per_req_timeout,
                cell=client,
                consumer=consumer,
                max_retries=max_retries,
            ),
            daemon=True,  # prevent download threads from blocking test teardown
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=wait_secs)

    succeeded = failed = timed_out = 0
    for c in consumers:
        if not c.completed.is_set():
            timed_out += 1
        elif c.error:
            failed += 1
        elif bytes(c.received) == data:
            succeeded += 1
        else:
            failed += 1

    return succeeded, failed, timed_out


# ---------------------------------------------------------------------------
# Pre-fix BlobHandler.handle_blob_cb reproducer
# ---------------------------------------------------------------------------


def _slow_read_stream(original_read_stream, delay):
    """Wraps _read_stream with a cancellable delay simulating large blob transfer.

    Uses _stop_delay event so the delay can be cancelled during teardown,
    allowing deadlocked workers to unblock and the process to exit cleanly.
    """

    def wrapper(blob_task):
        _stop_delay.wait(timeout=delay)  # cancellable via _stop_delay.set()
        return original_read_stream(blob_task)

    return wrapper


def _pre_fix_handle_blob_cb(self, future, stream, resume, *args, **kwargs):
    """Pre-fix behavior: blob_cb runs SYNCHRONOUSLY on the pool worker.

    Combined with slow _read_stream (0.2s), blob_cb blocks on future.result()
    long enough for concurrent requests to fill the pool. With 4 workers and 8
    concurrent requests, all workers end up stuck in blob_cb, and the queued
    _read_stream tasks can't run -> deadlock.
    """
    import nvflare.fuel.f3.streaming.blob_streamer as blob_mod

    blob_task = BlobTask(future, stream)

    slow_reader = _slow_read_stream(self._read_stream, delay=0.2)
    blob_mod.stream_thread_pool.submit(slow_reader, blob_task)

    # SYNCHRONOUS blob_cb -- the pre-fix behavior
    self.blob_cb(future, *args, **kwargs)

    return 0


# ======================================================================== #
# Test 1: With the fix (real code) -- should PASS
# ======================================================================== #
@pytest.mark.timeout(60)
class TestDownloadWithFix:

    @pytest.fixture(scope="class")
    def cells(self):
        port = get_open_ports(1)[0]
        server = Cell(SERVER_CELL, f"tcp://localhost:{port}", secure=False, credentials={})
        server.core_cell.start()
        client = Cell(CLIENT_CELL, f"tcp://localhost:{port}", secure=False, credentials={})
        client.core_cell.start()
        time.sleep(CELL_CONNECT_TIMEOUT)
        try:
            yield server, client
        finally:
            client.core_cell.stop()
            server.core_cell.stop()

    def test_parallel_downloads_with_fix(self, cells):
        """All parallel downloads complete when callback_thread_pool is separate."""
        server, client = cells
        data = _make_test_data()

        succeeded, failed, timed_out = _run_parallel_downloads(
            SERVER_CELL, server, client, data, NUM_PARALLEL, PER_REQ_TIMEOUT
        )
        print(f"\n[WITH FIX] {succeeded} ok, {failed} failed, {timed_out} timed out / {NUM_PARALLEL}")

        assert succeeded == NUM_PARALLEL, (
            f"[WITH FIX] Only {succeeded}/{NUM_PARALLEL} succeeded " f"({failed} failed, {timed_out} timed out)"
        )


# ======================================================================== #
# Test 2: Simulate pre-fix -- should FAIL with starvation
# ======================================================================== #
@pytest.mark.timeout(120)
class TestDownloadPreFixStarvation:

    @pytest.fixture(scope="class")
    def patched_cells(self):
        """Create cells AFTER patching BlobHandler + stream_thread_pool.

        The BlobHandler instances are created during Cell.__init__ -> register_blob_cb.
        The bound method handle_blob_cb is stored in ByteReceiver's callback registry.
        We must patch BEFORE cell creation so the registered callbacks use our
        pre-fix behavior.
        """
        import nvflare.fuel.f3.streaming.blob_streamer as blob_mod
        import nvflare.fuel.f3.streaming.byte_receiver as recv_mod
        import nvflare.fuel.f3.streaming.byte_streamer as send_mod

        # Save originals
        orig_handle = BlobHandler.handle_blob_cb
        orig_blob_stp = blob_mod.stream_thread_pool
        orig_recv_stp = recv_mod.stream_thread_pool
        orig_send_stp = send_mod.stream_thread_pool

        # Tiny shared pool
        tiny_pool = CheckedExecutor(4, "tiny_shared")

        # Reset the stop flag for this test run
        _stop_delay.clear()

        # Patch BEFORE cell creation
        BlobHandler.handle_blob_cb = _pre_fix_handle_blob_cb
        blob_mod.stream_thread_pool = tiny_pool
        recv_mod.stream_thread_pool = tiny_pool
        send_mod.stream_thread_pool = tiny_pool

        port = get_open_ports(1)[0]
        server = Cell(SERVER2_CELL, f"tcp://localhost:{port}", secure=False, credentials={})
        server.core_cell.start()
        client = Cell(CLIENT2_CELL, f"tcp://localhost:{port}", secure=False, credentials={})
        client.core_cell.start()
        time.sleep(CELL_CONNECT_TIMEOUT)

        try:
            yield server, client
        finally:
            # Restore original pools and class method FIRST
            BlobHandler.handle_blob_cb = orig_handle
            blob_mod.stream_thread_pool = orig_blob_stp
            recv_mod.stream_thread_pool = orig_recv_stp
            send_mod.stream_thread_pool = orig_send_stp

            # Signal the slow _read_stream delays to cancel immediately,
            # unblocking any deadlocked workers still waiting on future.result()
            _stop_delay.set()

            # Shut down the tiny pool (don't wait -- workers may be deadlocked)
            tiny_pool.shutdown(wait=False)

            # Remove the tiny pool's deadlocked threads from Python's internal
            # atexit tracking so they don't block process exit.
            # _threads_queues is a CPython implementation detail; guard for portability.
            try:
                from concurrent.futures import thread as _thread_mod

                for t in list(_thread_mod._threads_queues):
                    if getattr(t, "name", "").startswith("tiny_shared"):
                        _thread_mod._threads_queues.pop(t, None)
            except (ImportError, AttributeError):
                pass  # non-CPython or future Python version without this internal

            client.core_cell.stop()
            server.core_cell.stop()

    def test_parallel_downloads_simulating_pre_fix_starvation(self, patched_cells):
        """Simulates the pre-fix bug: blob_cb runs synchronously on pool worker,
        _read_stream has 0.2s delay (simulating large blob transfer).

        With pool_size=4 and 8 concurrent downloads:
          - ByteReceiver._callback_wrapper runs on tiny pool
          - handle_blob_cb submits slow _read_stream + calls blob_cb synchronously
          - blob_cb (Adapter.call) blocks on future.result() for 0.2s
          - _callback_wrapper holds the pool worker the whole time
          - 4 workers all stuck -> _read_stream queued but can't run -> deadlock
          - Remaining 4 requests can't even start
          - per_request_timeout=3s fires -> download_failed
        """
        server, client = patched_cells
        data = _make_test_data()

        succeeded, failed, timed_out = _run_parallel_downloads(
            SERVER2_CELL,
            server,
            client,
            data,
            NUM_PARALLEL,
            per_req_timeout=3.0,
            wait_secs=40.0,
            max_retries=0,
        )
        print(f"\n[PRE-FIX SIM] {succeeded} ok, {failed} failed, {timed_out} timed out / {NUM_PARALLEL}")

        assert succeeded < NUM_PARALLEL, (
            f"[PRE-FIX SIM] Unexpectedly all {succeeded}/{NUM_PARALLEL} succeeded. "
            f"The synchronous blob_cb + slow _read_stream should cause deadlock."
        )
        print(f"[PRE-FIX SIM] Confirmed starvation: only {succeeded}/{NUM_PARALLEL} succeeded")
