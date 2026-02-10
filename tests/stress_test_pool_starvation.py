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
Stress test: reproduces pool-starvation → timeout → lost reply in download_object.

Setup:
  - One server Cell (listener) with DownloadService and a slow Downloadable
  - Multiple client Cells that call download_object concurrently
  - stream_thread_pool patched to a SMALL size to force starvation
  - Short per_request_timeout to trigger the timeout path

Expected (correct): all downloads should complete.
Actual (bug): some/all downloads fail with TIMEOUT because:
  1. Pool workers are blocked in Adapter.call (produce + send_blob)
  2. No workers left to process incoming replies
  3. Request side times out, pops waiter
  4. Late reply is discarded
  5. download_object treats timeout as terminal

Usage:
  python tests/stress_test_pool_starvation.py
  python tests/stress_test_pool_starvation.py --clients 8 --pool-size 128 --produce-delay 0.3 --timeout 2.0  # passes
"""

import logging
import threading
import time
from typing import Any, Tuple
from unittest.mock import patch

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.download_service import (
    Consumer,
    Downloadable,
    DownloadService,
    ProduceRC,
)
from nvflare.fuel.f3.streaming.stream_utils import CheckedExecutor
from nvflare.fuel.utils.network_utils import get_open_ports

logging.basicConfig(
    level=logging.INFO,
    format="%(relativeCreated)6d [%(threadName)-16s] %(levelname)-5s %(name)s: %(message)s",
)
log = logging.getLogger("stress_test")


class RetryCounter(logging.Handler):
    """Counts retry warning messages from download_service."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        if "retryable error" in record.getMessage():
            self.count += 1


# ---------------------------------------------------------------------------
# Slow Downloadable — simulates large-model produce() overhead
# ---------------------------------------------------------------------------
class SlowDownloadable(Downloadable):
    """Returns NUM_CHUNKS chunks, each with a configurable delay to simulate
    expensive produce() (e.g. serialising a large model shard)."""

    NUM_CHUNKS = 3
    CHUNK_DATA = b"x" * 1024  # 1 KB per chunk

    def __init__(self, produce_delay: float = 0.5):
        super().__init__(None)
        self.produce_delay = produce_delay

    def produce(self, state: dict, requester: str) -> Tuple[str, Any, dict]:
        if not state:
            idx = 0
        else:
            idx = state.get("idx", 0)

        if idx >= self.NUM_CHUNKS:
            return ProduceRC.EOF, None, {}

        time.sleep(self.produce_delay)
        return ProduceRC.OK, self.CHUNK_DATA, {"idx": idx + 1}


# ---------------------------------------------------------------------------
# Tracking Consumer
# ---------------------------------------------------------------------------
class TrackingConsumer(Consumer):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.completed = False
        self.failed = False
        self.failure_reason = None
        self.chunks_received = 0

    def consume(self, ref_id: str, state: dict, data: Any) -> dict:
        self.chunks_received += 1
        return state

    def download_completed(self, ref_id: str):
        self.completed = True
        log.info(f"[{self.name}] download COMPLETED ({self.chunks_received} chunks)")

    def download_failed(self, ref_id: str, reason: str):
        self.failed = True
        self.failure_reason = reason
        log.warning(f"[{self.name}] download FAILED: {reason}")


def _download_worker(cell, ref_id, timeout, consumer):
    from nvflare.fuel.f3.streaming.download_service import download_object

    download_object(
        from_fqcn="server",
        ref_id=ref_id,
        per_request_timeout=timeout,
        cell=cell,
        consumer=consumer,
    )


# ---------------------------------------------------------------------------
# Main stress test
# ---------------------------------------------------------------------------
def run_stress_test(
    num_clients: int = 8,
    pool_size: int = 2,
    produce_delay: float = 0.3,
    per_request_timeout: float = 2.0,
):
    port = get_open_ports(1)[0]
    listen_url = f"tcp://localhost:{port}"
    connect_url = f"tcp://localhost:{port}"

    small_pool = CheckedExecutor(pool_size, "stm-stress")
    retry_counter = RetryCounter()
    logging.getLogger("nvflare.fuel.f3.streaming.download_service").addHandler(retry_counter)

    server_cell = None
    client_cells = []
    tx_id = None

    with patch("nvflare.fuel.f3.streaming.stream_utils.stream_thread_pool", small_pool), \
         patch("nvflare.fuel.f3.streaming.stream_utils.callback_thread_pool", small_pool), \
         patch("nvflare.fuel.f3.streaming.byte_receiver.callback_thread_pool", small_pool), \
         patch("nvflare.fuel.f3.streaming.byte_streamer.stream_thread_pool", small_pool), \
         patch("nvflare.fuel.f3.streaming.blob_streamer.stream_thread_pool", small_pool), \
         patch("nvflare.fuel.f3.streaming.blob_streamer.callback_thread_pool", small_pool):

        try:
            server_cell = Cell("server", listen_url, secure=False, credentials={})
            server_cell.core_cell.start()

            downloadable = SlowDownloadable(produce_delay=produce_delay)
            tx_id = DownloadService.new_transaction(
                cell=server_cell, timeout=60.0, num_receivers=num_clients
            )
            ref_id = DownloadService.add_object(tx_id, downloadable)
            log.info(f"Server ready. {pool_size=}, {num_clients=}, "
                     f"produce_delay={produce_delay}s, timeout={per_request_timeout}s")

            for i in range(num_clients):
                c = Cell(f"client_{i}", connect_url, secure=False, credentials={})
                c.core_cell.start()
                client_cells.append(c)

            time.sleep(1.0)

            # launch concurrent downloads
            consumers = []
            threads = []
            for i, client_cell in enumerate(client_cells):
                consumer = TrackingConsumer(f"client_{i}")
                consumers.append(consumer)
                t = threading.Thread(
                    target=_download_worker,
                    args=(client_cell, ref_id, per_request_timeout, consumer),
                    name=f"dl-client-{i}",
                    daemon=True,
                )
                threads.append(t)

            start = time.time()
            for t in threads:
                t.start()

            # wait with a hard deadline to prevent hangs
            deadline = per_request_timeout * SlowDownloadable.NUM_CHUNKS * 2 + 10
            for t in threads:
                remaining = max(0.1, deadline - (time.time() - start))
                t.join(timeout=remaining)

            elapsed = time.time() - start
        finally:
            # Suppress noisy errors from in-flight pool tasks during teardown
            logging.getLogger("nvflare.fuel.f3.streaming").setLevel(logging.CRITICAL)

            # Stop pool first so no new tasks are accepted
            small_pool.stopped = True

            if tx_id:
                try:
                    DownloadService.delete_transaction(tx_id)
                except Exception:
                    pass
            for c in client_cells:
                try:
                    c.core_cell.stop()
                except Exception:
                    pass
            if server_cell:
                try:
                    server_cell.core_cell.stop()
                except Exception:
                    pass

            small_pool.shutdown(wait=False)
            time.sleep(0.5)  # let in-flight tasks drain

            logging.getLogger("nvflare.fuel.f3.streaming").setLevel(logging.INFO)

    # --- report ---
    succeeded = sum(1 for c in consumers if c.completed)
    failed = sum(1 for c in consumers if c.failed)
    timed_out = sum(1 for t in threads if t.is_alive())

    log.info(f"\n{'='*60}")
    log.info(f"Completed in {elapsed:.1f}s")
    log.info(f"Pool size: {pool_size}, Clients: {num_clients}, "
             f"Produce delay: {produce_delay}s, Timeout: {per_request_timeout}s")
    for c in consumers:
        status = "OK" if c.completed else f"FAIL ({c.failure_reason})"
        log.info(f"  {c.name}: {status}  chunks={c.chunks_received}")
    log.info(f"Result: {succeeded}/{num_clients} succeeded, "
             f"{failed}/{num_clients} failed, {timed_out} still stuck, "
             f"{retry_counter.count} retries")
    log.info(f"{'='*60}")

    assert failed == 0, (
        f"BUG REPRODUCED: {failed}/{num_clients} downloads failed due to pool starvation.\n"
        f"Failures:\n"
        + "\n".join(f"  {c.name}: {c.failure_reason}" for c in consumers if c.failed)
    )
    log.info("All downloads succeeded (bug NOT triggered).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pool starvation stress test")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--produce-delay", type=float, default=0.3)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    run_stress_test(
        num_clients=args.clients,
        pool_size=args.pool_size,
        produce_delay=args.produce_delay,
        per_request_timeout=args.timeout,
    )
