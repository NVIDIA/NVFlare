# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile
import uuid

from nvflare.fuel.hci.binary_proto import CT_BINARY, Receiver, Sender, receive_all, send_binary_file


class MySender(Sender):
    def __init__(self):
        self.buf = bytearray()

    def sendall(self, data):
        self.buf.extend(data)


class MyReceiver(Receiver):
    def __init__(self, buf):
        self.buf = buf
        self.bytes_received = 0

    def recv(self, size: int) -> bytes:
        remaining = len(self.buf) - self.bytes_received
        if remaining == 0:
            return None
        bytes_to_rcv = random.randint(1, 1024)
        if bytes_to_rcv > remaining:
            bytes_to_rcv = remaining
        if bytes_to_rcv > size:
            bytes_to_rcv = size
        data = self.buf[self.bytes_received : self.bytes_received + bytes_to_rcv]
        self.bytes_received += len(data)
        return data


class TestBinaryProto:
    def send_and_receive(self, meta: str, size: int):
        if size == 0:
            body = b""
        else:
            body = os.urandom(size)
        td = tempfile.gettempdir()
        tf = os.path.join(td, str(uuid.uuid4()))
        with open(tf, "w+b") as f:
            f.write(body)

        s = MySender()
        send_binary_file(s, tf, meta)

        r = MyReceiver(s.buf)
        ct, received_meta, file_name = receive_all(r)
        if file_name:
            file_stats = os.stat(file_name)
            received_size = file_stats.st_size
            with open(file_name, "rb") as f:
                data = f.read()
                assert data == body
            os.remove(file_name)
        else:
            received_size = 0
        assert ct == CT_BINARY
        assert received_meta == meta
        assert received_size == len(body)
        os.remove(tf)

    def test_no_meta(self):
        self.send_and_receive("", 100)

    def test_ascii_meta(self):
        self.send_and_receive("this is a normal text for meta", 100)

    def test_non_ascii_meta(self):
        self.send_and_receive("this is 中文信息", 100)

    def test_send_one_byte(self):
        self.send_and_receive("meta", 1)

    def test_send_zero_byte(self):
        self.send_and_receive("meta", 0)

    def test_send_1m(self):
        self.send_and_receive("meta", 1024 * 1024)

    def test_send_1g(self):
        self.send_and_receive("meta", 1024 * 1024 * 1024)
