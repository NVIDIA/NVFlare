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
import sys
import threading
import time

from nvflare.fuel.f3.cellnet.core_cell import CellAgent, CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.tools.utils import RX_CELL, TEST_CHANNEL, TEST_TOPIC, TX_CELL, setup_log


class FileSender:
    """Utility to send a file to another cell"""

    def __init__(self, url: str):
        core_cell = CoreCell(TX_CELL, url, secure=False, credentials={})
        self.stream_cell = StreamCell(core_cell)
        core_cell.set_cell_connected_cb(self.cell_connected)
        core_cell.start()
        self.cell = core_cell
        self.ready = threading.Event()

    def stop(self):
        self.cell.stop()

    def wait(self):
        self.ready.wait()

    def send(self, file_to_send: str):
        future = self.stream_cell.send_file(TEST_CHANNEL, TEST_TOPIC, RX_CELL, Message(None, file_to_send))

        while True:
            if future.done():
                break

            time.sleep(1)
            progress = future.get_progress()
            percent = progress * 100.0 / future.get_size()
            print(f"Sent {progress} bytes {percent:.2f}% done")

        size = future.result()
        print(f"Total {size} bytes sent for file {file_to_send}")

    def cell_connected(self, agent: CellAgent):
        if agent.get_fqcn() == RX_CELL:
            self.ready.set()


if __name__ == "__main__":
    setup_log(logging.INFO)
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} connect_url file_name")
        sys.exit(1)

    connect_url = sys.argv[1]
    file_name = sys.argv[2]
    sender = FileSender(connect_url)
    print("Waiting for receiver to be online ...")
    sender.wait()
    print(f"Sending file {file_name} ...")

    start = time.time()
    sender.send(file_name)
    print(f"Time elapsed: {(time.time()-start):.3f} seconds")

    sender.stop()
    print("Done")
