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

"""Parallel download worker kept separate from pytest's interpreter and pools."""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.download_service import Consumer, Downloadable, ProduceRC, download_object
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader
from nvflare.fuel.utils.network_utils import get_open_ports


class ChunkedDownloadable(Downloadable):
    def produce(self, state, requester):
        offset = state.get("offset", 0) if state else 0
        if offset >= len(self.base_obj):
            return ProduceRC.EOF, None, {}
        end = min(offset + 256, len(self.base_obj))
        return ProduceRC.OK, self.base_obj[offset:end], {"offset": end}


class ChunkedConsumer(Consumer):
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


def run_parallel_downloads():
    identity = uuid.uuid4().hex[:8]
    data = bytes(range(256)) * 16
    abort = Signal()
    url = f"tcp://localhost:{get_open_ports(1)[0]}"
    with ExitStack() as cleanup:
        server = Cell(f"download-server-{identity}", url, secure=False, credentials={})
        cleanup.callback(server.stop)
        server.start()
        client = Cell(f"download-client-{identity}", url, secure=False, credentials={})
        cleanup.callback(client.stop)
        client.start()
        deadline = time.monotonic() + 15
        while not (server.is_cell_connected(client.get_fqcn()) and client.is_cell_connected(server.get_fqcn())):
            assert time.monotonic() < deadline, "parallel download cells did not connect"
            time.sleep(0.02)
        print("parallel download cells ready", flush=True)
        workers = ThreadPoolExecutor(max_workers=8)
        try:
            consumers, pending = [], []
            for _ in range(8):
                downloader = ObjectDownloader(cell=server, timeout=120, num_receivers=1)
                cleanup.callback(downloader.delete_transaction)
                ref_id = downloader.add_object(ChunkedDownloadable(data))
                consumer = ChunkedConsumer()
                consumers.append(consumer)
                pending.append(
                    workers.submit(
                        download_object,
                        from_fqcn=server.get_fqcn(),
                        ref_id=ref_id,
                        per_request_timeout=10,
                        cell=client,
                        consumer=consumer,
                        max_retries=0,
                        abort_signal=abort,
                    )
                )
            deadline = time.monotonic() + 45
            for future in pending:
                future.result(timeout=max(0, deadline - time.monotonic()))
            for consumer in consumers:
                assert consumer.completed.is_set(), "download did not finish"
                assert consumer.error is None, consumer.error
                assert bytes(consumer.received) == data
            print("all eight downloads completed", flush=True)
        finally:
            abort.trigger("parallel download test finished")
            workers.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    run_parallel_downloads()
