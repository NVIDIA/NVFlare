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

"""Parallel downloads over real Cells, isolated from the pytest process."""

import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

import pytest

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.download_service import Consumer, Downloadable, ProduceRC, download_object
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader
from nvflare.fuel.utils.network_utils import get_open_ports
from tests.timing_utils import ThreadGroup, wait_for


class _ChunkedDownloadable(Downloadable):
    def produce(self, state, requester):
        offset = state.get("offset", 0) if state else 0
        if offset >= len(self.base_obj):
            return ProduceRC.EOF, None, {}
        end = min(offset + 256, len(self.base_obj))
        return ProduceRC.OK, self.base_obj[offset:end], {"offset": end}


class _Consumer(Consumer):
    def __init__(self):
        super().__init__()
        self.received = bytearray()
        self.completed = threading.Event()
        self.error = None

    def consume(self, ref_id, state, data):
        self.received.extend(data)
        return state

    def download_completed(self, ref_id):
        self.completed.set()

    def download_failed(self, ref_id, reason):
        self.error = reason
        self.completed.set()


def _parallel_download_worker():
    identity = uuid.uuid4().hex[:8]
    data = bytes(range(256)) * 16
    abort = Signal()
    root_url = f"tcp://localhost:{get_open_ports(1)[0]}"
    with ExitStack() as cleanup:
        server = Cell(f"download-server-{identity}", root_url, secure=False, credentials={})
        cleanup.callback(server.core_cell.stop)
        server.core_cell.start()
        client = Cell(f"download-client-{identity}", root_url, secure=False, credentials={})
        cleanup.callback(client.core_cell.stop)
        client.core_cell.start()
        wait_for(
            lambda: client.core_cell.is_cell_connected(server.get_fqcn())
            and server.core_cell.is_cell_connected(client.get_fqcn()),
            message="parallel download cells did not become ready",
        )
        consumers = []
        with ThreadGroup() as workers:
            try:
                for _ in range(8):
                    downloader = ObjectDownloader(cell=server, timeout=120.0, num_receivers=1)
                    cleanup.callback(downloader.delete_transaction)
                    ref_id = downloader.add_object(_ChunkedDownloadable(data))
                    consumer = _Consumer()
                    consumers.append(consumer)
                    workers.start(
                        download_object,
                        from_fqcn=server.get_fqcn(),
                        ref_id=ref_id,
                        per_request_timeout=10.0,
                        cell=client,
                        consumer=consumer,
                        max_retries=0,
                        abort_signal=abort,
                    )
                deadline = time.monotonic() + 45.0
                for consumer in consumers:
                    assert consumer.completed.wait(max(0, deadline - time.monotonic())), "download did not finish"
                    assert consumer.error is None, consumer.error
                    assert bytes(consumer.received) == data
            finally:
                abort.trigger("parallel download test finished")


@pytest.mark.timeout(90)
def test_parallel_downloads_complete_with_real_streaming():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])['_parallel_download_worker']()",
            str(Path(__file__).resolve()),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        capture_output=True,
        text=True,
        timeout=75,
    )
    assert result.returncode == 0, result.stdout + result.stderr
