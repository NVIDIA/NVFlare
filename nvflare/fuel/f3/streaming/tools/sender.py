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
import threading
import time

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CellAgent
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.tools.utils import (
    BUF_SIZE,
    RX_CELL,
    TEST_CHANNEL,
    TEST_TOPIC,
    TIMESTAMP,
    TX_CELL,
    make_buffer,
    setup_log,
)

log = logging.getLogger("sender")
receiver_ready = threading.Event()


def cell_connected(connected_cell: CellAgent):
    global receiver_ready
    log.info(f"{connected_cell.get_fqcn()} is online")
    receiver_ready.set()


def create_sender_cell(url: str):
    cell = Cell(fqcn=TX_CELL, root_url=url, secure=False, credentials={})
    log.info(f"Sender is trying to connect to {url}")
    cell.core_cell.set_cell_connected_cb(cell_connected)
    cell.start()
    return cell


def send_blob(url: str, buf_size: int):
    global receiver_ready

    sender = create_sender_cell(url)
    log.info(f"Creating buffer with size {buf_size} ...")
    buffer = make_buffer(buf_size)

    try:
        log.info("Waiting for receiver to go online")
        receiver_ready.wait()

        log.info("Starting sending buffer ...")
        start_time = time.time()
        headers = {TIMESTAMP: start_time}
        result = sender.send_request(
            channel=TEST_CHANNEL, topic=TEST_TOPIC, target=RX_CELL, request=Message(headers, buffer)
        )

        if result and result.payload:
            log.info(f"Headers: {result.headers} Payload: {result.payload}")
            return_start = result.get_header(TIMESTAMP)
            curr_time = time.time()
            log.info(
                f"Total time: {curr_time - start_time} seconds Return time: {curr_time - return_start:.3f} seconds"
            )
        else:
            log.error(f"Error sending message to {RX_CELL}")
    except Exception as e:
        log.exception(e)

    sender.stop()


if __name__ == "__main__":
    setup_log(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str, help="Connecting URL")
    parser.add_argument("-s", type=int, metavar="size", help="Buffer size", default=BUF_SIZE)
    args = parser.parse_args()

    send_blob(args.url, args.s)
