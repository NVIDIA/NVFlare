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
import struct
import tempfile
import uuid

from .proto import ALL_END, LINE_END, MAX_BLOCK_SIZE

CT_TEXT = 0
CT_BINARY = 1
BINARY_MARKER = 1
HEADER_STRUCT = struct.Struct(">II")  # meta_size(4), body_size(4)
HEADER_LEN = HEADER_STRUCT.size


class Receiver:
    def recv(self, size) -> bytes:
        pass


class DataProcessor:
    def process(self, data, content_type: int):
        pass

    def finalize(self):
        pass


class MsgReceiver:
    def __init__(self, receiver: Receiver, processor: DataProcessor):
        self.receiver = receiver
        self.processor = processor
        self.meta = None

    def parse(self):
        # get marker
        data = self.receiver.recv(1)
        if not data:
            raise RuntimeError("no data type marker received")

        marker = data[0]
        body_size = 0

        if marker == BINARY_MARKER:
            content_type = CT_BINARY
            buffer = self.receiver.recv(HEADER_LEN)
            if not buffer:
                raise RuntimeError(f"cannot get header buffer of {HEADER_LEN} bytes")
            if len(buffer) != HEADER_LEN:
                raise RuntimeError(f"expect {HEADER_LEN} header bytes but only got {len(buffer)}")

            meta_size, body_size = HEADER_STRUCT.unpack_from(buffer, 0)
            if meta_size < 0:
                raise RuntimeError(f"invalid binary data meta size {meta_size}")

            # get headers
            meta_bytes = self.receiver.recv(meta_size)
            if not data:
                raise RuntimeError("no meta data received")
            if len(meta_bytes) != meta_size:
                raise RuntimeError(f"expect {meta_size} meta bytes but got {len(meta_bytes)}")

            self.meta = str(meta_bytes, "utf-8")
        else:
            # text content - the 1st byte is part of the data!
            content_type = CT_TEXT
            self.processor.process(data, content_type)

        total_size = 0
        while True:
            data = self.receiver.recv(MAX_BLOCK_SIZE)
            if not data:
                break

            total_size += len(data)
            should_stop = self.processor.process(data, content_type)
            if should_stop:
                break

            if content_type == CT_BINARY and total_size >= body_size:
                break

        if content_type == CT_BINARY and total_size != body_size:
            raise RuntimeError(f"expect to get {body_size} body bytes but got {total_size}")

        self.processor.finalize()


class MsgDataProcessor(DataProcessor):
    def __init__(self):
        self.file = None
        self.file_name = None
        self.text_segs = []
        self.total_text = None

    def process(self, data, content_type: int):
        if content_type == CT_TEXT:
            data = str(data, "utf-8")
            end_idx = data.find(ALL_END)
            if end_idx >= 0:
                self.text_segs.append(data[:end_idx])
                return True  # all received
            else:
                self.text_segs.append(data)
        else:
            # binary - write to file
            if not self.file:
                self.file_name = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
                self.file = open(self.file_name, "w+b")
            self.file.write(data)
        return False

    def finalize(self):
        if self.text_segs:
            total_text = "".join(self.text_segs)
            self.total_text = total_text.replace(LINE_END, "")

        if self.file:
            self.file.close()


def receive_all(sock):
    """Receive all data from the peer.

    Args:
        sock: the socket

    Returns: a tuple of (content_type, request_text, additional_data)

    """
    p = MsgDataProcessor()
    parser = MsgReceiver(receiver=sock, processor=p)
    parser.parse()
    if p.total_text:
        return CT_TEXT, p.total_text, None
    else:
        # binary
        return (
            CT_BINARY,
            parser.meta,
            p.file_name,
        )


def binary_header(meta_size: int, body_size: int):
    return HEADER_STRUCT.pack(meta_size, body_size)


class DataGenerator:
    def data_size(self) -> int:
        pass

    def generate(self) -> bytes:
        pass


class Sender:
    def sendall(self, data):
        pass


def send_binary_data(sender: Sender, generator: DataGenerator, meta: str):
    body_size = generator.data_size()
    meta_size = 0
    meta_bytes = None
    if meta:
        meta_bytes = bytes(meta, "utf-8")
        meta_size = len(meta_bytes)

    header_bytes = binary_header(meta_size, body_size)
    sender.sendall(bytes([BINARY_MARKER]))  # add binary marker at the beginning!
    sender.sendall(header_bytes)

    if meta_bytes:
        sender.sendall(meta_bytes)

    sent_body_size = 0
    while True:
        data = generator.generate()
        if not data:
            break
        sent_body_size += len(data)
        sender.sendall(data)
    if sent_body_size != body_size:
        raise RuntimeError(f"generated body size {sent_body_size} != expected body size {body_size}")


class GenerateDataFromFile(DataGenerator):
    def __init__(self, file_name: str):
        file_stats = os.stat(file_name)
        self.size = file_stats.st_size
        self.file = open(file_name, "rb")

    def data_size(self) -> int:
        return self.size

    def generate(self) -> bytes:
        data = self.file.read(MAX_BLOCK_SIZE)
        if not data:
            self.file.close()
        return data


def send_binary_file(sock, file_name: str, meta: str):
    gen = GenerateDataFromFile(file_name)
    send_binary_data(sock, gen, meta)
