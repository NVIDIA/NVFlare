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

import os
import random

import pytest

from nvflare.fuel.hci.chunk import MAX_CHUNK_SIZE, Receiver, Sender, get_slice


class DataCollector:
    def __init__(self):
        self.buffer = bytearray()

    def collect(self, data):
        self.buffer.extend(data)

    def collect_from(self, data, start: int, length: int):
        self.buffer.extend(get_slice(data, start, length))


class TestChunkSendReceive:
    def do_send_receive(self):
        data = bytearray()
        coll = DataCollector()
        sender = Sender(send_data_func=coll.collect)
        c1 = os.urandom(1000)
        data.extend(c1)
        sender.send(c1)

        c2 = os.urandom(random.randint(1, 1024))
        data.extend(c2)
        sender.send(c2)

        c3 = os.urandom(random.randint(1, 2048))
        data.extend(c3)
        sender.send(c3)

        sender.close()

        buffer = coll.buffer
        num_stops = 4
        stops = random.sample(range(1, len(buffer) - 1), num_stops)
        stops.sort()

        coll2 = DataCollector()
        receiver = Receiver(receive_data_func=coll2.collect_from)
        start = 0
        for i in range(num_stops + 1):
            if i < num_stops:
                end = stops[i]
            else:
                end = len(buffer)
            buf = buffer[start:end]
            receiver.receive(buf)
            start += end - start

        assert coll2.buffer == data

    def test_send_random(self):
        for _ in range(1000):
            self.do_send_receive()

    def send_one_chunk(self, size):
        coll = DataCollector()
        sender = Sender(send_data_func=coll.collect)
        if size == 0:
            data = b""
        else:
            data = os.urandom(size)
        sender.send(data)
        buffer = coll.buffer

        coll2 = DataCollector()
        receiver = Receiver(receive_data_func=coll2.collect_from)
        receiver.receive(buffer)
        assert coll2.buffer == data

    def test_send_one_byte(self):
        self.send_one_chunk(1)

    def test_send_zero_byte(self):
        self.send_one_chunk(0)

    def test_send_max_bytes(self):
        self.send_one_chunk(MAX_CHUNK_SIZE)

    def test_max_size_error(self):
        with pytest.raises(RuntimeError):
            self.send_one_chunk(MAX_CHUNK_SIZE + 1)
