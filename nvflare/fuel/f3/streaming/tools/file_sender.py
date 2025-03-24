# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import argparse
import logging
import os
import threading
import time

from nvflare.fuel.f3.cellnet.core_cell import CellAgent, CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.tools.utils import RX_CELL, TEST_CHANNEL, TEST_TOPIC, TIMESTAMP, TX_CELL, setup_log

log = logging.getLogger("file_sender")


class FileSender:
    """Utility to send a file to another cell"""

    def __init__(self, url: str):
        core_cell = CoreCell(TX_CELL, url, secure=False, credentials={})
        core_cell.set_cell_connected_cb(self.cell_connected)
        core_cell.start()

        self.core_cell = core_cell
        self.stream_cell = StreamCell(core_cell)
        self.ready = threading.Event()
        self.end_time = None

    def stop(self):
        self.stream_cell.cell.stop()

    def wait(self):
        self.ready.wait()

    def on_done(self):
        self.end_time = time.time()

    def cell_connected(self, agent: CellAgent):
        if agent.get_fqcn() == RX_CELL:
            self.ready.set()
            log.info("Connected to file receiver")

    def send(self, file_to_send: str):

        start_time = time.time()
        future = self.stream_cell.send_file(
            TEST_CHANNEL, TEST_TOPIC, RX_CELL, Message({TIMESTAMP: start_time}, file_to_send)
        )

        future.add_done_callback(self.on_done)

        file_size = os.path.getsize(file_to_send)
        delta = file_size / 100.0
        last_progress = 0
        while True:
            if future.done():
                break

            progress = future.get_progress()
            if progress - last_progress > delta:
                last_progress = progress
                percent = progress * 100.0 / future.get_size()
                log.info(f"Sent {progress} bytes {percent:.2f}% done")

        size = future.result()
        log.info(f"Total {size} bytes sent for file {file_to_send}, took {time.time() - self.end_time} seconds:.3f")


def send_file(url: str, file_to_send: str):
    sender = FileSender(url)
    sender.wait()
    sender.send(file_to_send)
    time.sleep(1)
    sender.stop()


if __name__ == "__main__":
    setup_log(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str, help="Connecting URL")
    parser.add_argument("file_name", type=str, help="Name of the file to send")
    args = parser.parse_args()

    send_file(args.url, args.file_name)
