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
import os
import sys
import time

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.stream_types import StreamFuture
from nvflare.fuel.f3.streaming.stream_utils import stream_thread_pool
from nvflare.fuel.f3.streaming.tools.utils import RX_CELL, TEST_CHANNEL, TEST_TOPIC, setup_log


class FileReceiver:
    """Utility to receive files sent from another cell"""

    def __init__(self, listening_url: str, out_folder: str):
        self.cell = CoreCell(RX_CELL, listening_url, secure=False, credentials={})
        self.stream_cell = StreamCell(self.cell)
        self.stream_cell.register_file_cb(TEST_CHANNEL, TEST_TOPIC, self.file_cb)
        self.cell.start()
        self.out_folder = out_folder
        self.file_received = 0

    def stop(self):
        self.cell.stop()

    def file_cb(self, future: StreamFuture, original_name: str):
        out_file = os.path.join(self.out_folder, original_name)
        stream_thread_pool.submit(self.monitor_status, future)
        print(f"Received file {original_name}, writing to {out_file} ...")
        return out_file

    def monitor_status(self, future: StreamFuture):

        start = time.time()

        while True:
            if future.done():
                break

            progress = future.get_progress()
            percent = progress * 100.0 / future.get_size()
            print(f"Received {progress} bytes {percent:.2f}% done")
            time.sleep(1)

        name = future.result()
        print(f"Time elapsed: {(time.time() - start):.3f} seconds")
        print(f"File {name} is sent")
        self.file_received += 1

        return name


if __name__ == "__main__":
    setup_log(logging.INFO)
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} listening_url out_folder")
        sys.exit(1)

    listening_url = sys.argv[1]
    out_folder = sys.argv[2]

    receiver = FileReceiver(listening_url, out_folder)

    while True:
        if receiver.file_received >= 1:
            break
        time.sleep(1)

    receiver.stop()
    print(f"Done. Files received: {receiver.file_received}")
