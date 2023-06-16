# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import time

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.stream_types import StreamFuture
from nvflare.fuel.f3.streaming.tools.utils import RX_CELL, TEST_CHANNEL, TEST_TOPIC, make_buffer, BUF_SIZE

logging.basicConfig(level=logging.DEBUG)
formatter = logging.Formatter(fmt="%(relativeCreated)6d [%(threadName)-12s] [%(levelname)-5s] %(name)s: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
root_log = logging.getLogger()
root_log.handlers.clear()
root_log.addHandler(handler)


class Receiver:

    def __init__(self, listening_url: str):
        cell = Cell(RX_CELL, listening_url, secure=False, credentials={})
        cell.start()
        self.stream_cell = StreamCell(cell)
        self.stream_cell.register_blob_cb(TEST_CHANNEL, TEST_TOPIC, self.blob_cb)
        self.futures = {}

    def get_futures(self) -> dict:
        return self.futures

    def blob_cb(self, stream_future: StreamFuture, *args, **kwargs):
        sid = stream_future.get_stream_id()
        print(f"Stream {sid} received")
        self.futures[sid] = stream_future


url = "tcp://localhost:1234"
receiver = Receiver(url)
time.sleep(2)
result = None
while True:
    if receiver.get_futures:
        for sid, fut in receiver.get_futures().items():
            if fut.done():
                result = fut.result()
                break
            else:
                print(f"{sid} Progress: {fut.get_progress()}")
    time.sleep(1)
    if result:
        break

buffer = make_buffer(BUF_SIZE)

if buffer == result:
    print("Result is correct")
