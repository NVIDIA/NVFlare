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

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.stream_types import StreamFuture
from nvflare.fuel.f3.streaming.tools.utils import RX_CELL, TEST_CHANNEL, TEST_TOPIC, TIMESTAMP, setup_log

log = logging.getLogger("file_receiver")


class FileReceiver:
    """Utility to receive files sent from another cell"""

    def __init__(self, listening_url: str, out_folder: str):
        cell = CoreCell(RX_CELL, listening_url, secure=False, credentials={})
        cell.start()
        self.stream_cell = StreamCell(cell)
        self.stream_cell.register_file_cb(TEST_CHANNEL, TEST_TOPIC, self.file_cb)
        self.out_folder = out_folder
        self.end_time = None
        self.file_arrival = threading.Event()
        self.future = None

    def stop(self):
        self.stream_cell.cell.stop()

    def wait_for_file(self) -> StreamFuture:
        self.file_arrival.wait()
        return self.future

    def on_done(self):
        self.end_time = time.time()

    def file_cb(self, future: StreamFuture, original_name: str):
        self.future = future
        future.add_done_callback(self.on_done)
        out_file = os.path.join(self.out_folder, original_name)
        self.file_arrival.set()
        return out_file

    def show_progress(self):
        future = self.wait_for_file()
        sender_start_time = future.get_headers().get(TIMESTAMP)
        start_time = time.time()
        size = future.get_size()
        delta = size / 100.0
        last_progress = 0
        while True:
            if future.done():
                break

            progress = future.get_progress()
            if (progress - last_progress) >= delta:
                percent = progress * 100.0 / future.get_size()
                last_progress = progress
                log.info(f"Received {progress} bytes {percent:.2f}% done")

            time.sleep(1)

        name = future.result()
        log.info(
            f"File {name} received in {self.end_time - start_time:.3f} seconds "
            f"Total time: {self.end_time - sender_start_time:.3f}"
        )


def receive_file(listening_url: str, out_folder: str):
    receiver = FileReceiver(listening_url, out_folder)
    receiver.show_progress()
    receiver.stop()


if __name__ == "__main__":
    setup_log(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str, help="Listening URL")
    parser.add_argument("folder", type=str, help="Folder to save received files")
    args = parser.parse_args()

    receive_file(args.url, args.folder)
