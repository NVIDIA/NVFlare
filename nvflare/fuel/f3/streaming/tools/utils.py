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

from nvflare.fuel.f3.streaming.stream_utils import wrap_view

BUF_SIZE = 64 * 1024 * 1024 + 1
TEST_CHANNEL = "stream"
TEST_TOPIC = "test"
TX_CELL = "sender"
RX_CELL = "server"


def make_buffer(size: int) -> bytearray:

    buf = wrap_view(bytearray(size))
    buf_len = 0
    n = 0
    while True:
        temp = n.to_bytes(8, "big", signed=False)
        temp_len = len(temp)
        if (buf_len + temp_len) > size:
            temp_len = size - buf_len
        buf[buf_len : buf_len + temp_len] = temp[0:temp_len]
        buf_len += temp_len
        n += 1
        if buf_len >= size:
            break

    return buf


def setup_log(level):
    logging.basicConfig(level=level)
    formatter = logging.Formatter(
        fmt="%(relativeCreated)6d [%(threadName)-12s] [%(levelname)-5s] %(name)s: %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_log = logging.getLogger()
    root_log.handlers.clear()
    root_log.addHandler(handler)
