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

"""
This package implements a protocol for Admin client and server to exchange data in binary protocol. This is mainly
used for large data exchanges such as job submission and download.

The format of a binary exchange has 4 sections (header, meta, body, and footer), as follows:

    [Header] [Meta] [Body] [Footer]

Header section: {binary_marker:1} {meta_size:4} {body_size:8}
Meta section: {meta:meta_size}
Body section: {body:body_size}
Footer section: {end_marker:4} {checksum:4}

The 1-byte binary_marker in Header signifies that the exchange is binary. If this marker is missing,
the exchange is text.
A binary exchange can optionally include text-encoded meta information (e.g. a JSON string).
At the end of the exchange is the footer that contains end-of-data marker (four bytes of 0) and the checksum computed
over the body bytes.

"""

import os
import struct
import tempfile
import uuid

from .checksum import Checksum
from .proto import ALL_END, LINE_END, MAX_BLOCK_SIZE

CT_TEXT = 0
CT_BINARY = 1
BINARY_MARKER = 1

HEADER_STRUCT = struct.Struct(">IQ")  # meta_size(4), body_size(8)
HEADER_LEN = HEADER_STRUCT.size

FOOTER_STRUCT = struct.Struct(">II")  # end_marker(four 0s), checksum(4)
FOOTER_LEN = FOOTER_STRUCT.size


class Receiver:
    """
    A Receiver must be able to receive bytes from the peer.
    """

    def recv(self, size: int) -> bytes:
        """Receive bytes of up to the specified size.

        Args:
            size: the number of bytes to receive.

        Returns: bytes of no more than size

        """
        pass


class DataProcessor:
    """
    A DataProcessor is used to process data received from the peer.
    """

    def process(self, data: bytes, content_type: int):
        """Process the data received from peer.

        Args:
            data: the data to be processed
            content_type: the content type: CT_TEXT or CT_BINARY

        Returns: None

        """
        pass

    def finalize(self):
        pass


class MsgReceiver:
    """
    A MsgReceiver is used to receive and parse exchange from the peer.
    It uses the provided Receiver to receive data from the peer, parses the data according to the echange protocl,
    and calls the provided DataProcessor to process the data.
    """

    def __init__(self, receiver: Receiver, processor: DataProcessor):
        """Constructor of MsgReceiver

        Args:
            receiver: the data receiver object to be used to receive data from the peer
            processor: the data processor for data processing
        """
        self.receiver = receiver
        self.processor = processor
        self.meta = None

    def _must_recv(self, num_bytes: int):
        """Must receive specified number of bytes.

        Note that the receiver's recv method can return any number of bytes. We keep calling it until the
        specified number of bytes are received.

        Args:
            num_bytes: number of bytes to receive

        Returns: received bytes

        """
        buffer = bytearray()
        received = 0
        while received < num_bytes:
            size_to_recv = num_bytes - received
            data = self.receiver.recv(size_to_recv)
            if not data:
                return None
            if len(data) == num_bytes:
                # most case
                return data
            else:
                buffer.extend(data)
            received += len(data)
        return bytes(buffer)

    def _parse_text(self):
        while True:
            data = self.receiver.recv(MAX_BLOCK_SIZE)
            if not data:
                break

            should_stop = self.processor.process(data, CT_TEXT)
            if should_stop:
                break

    def _parse_binary(self, body_size):
        received_size = 0
        checksum = Checksum()

        # receive the body data of the exchange
        # note that we do not receive the footer in this loop!
        while True:
            remaining = body_size - received_size

            if remaining == 0:
                break

            rcv_size = MAX_BLOCK_SIZE
            if remaining < MAX_BLOCK_SIZE:
                rcv_size = remaining

            data = self.receiver.recv(rcv_size)
            if not data:
                raise RuntimeError(f"failed to receive {rcv_size} bytes")

            received_size += len(data)
            checksum.update(data)
            self.processor.process(data, CT_BINARY)

        # receive the footer and validate: check end-of-data marker, and compare checksum
        buffer = self._must_recv(FOOTER_LEN)
        if not buffer:
            raise RuntimeError(f"cannot get footer buffer of {FOOTER_LEN} bytes")
        if len(buffer) != FOOTER_LEN:
            raise RuntimeError(f"expect {FOOTER_LEN} footer bytes but only got {len(buffer)}")

        footer_marker, checksum_received = FOOTER_STRUCT.unpack_from(buffer, 0)
        if footer_marker != 0:
            raise RuntimeError(f"footer marker must be 0 but got {footer_marker}")
        computed_checksum = checksum.result()
        if checksum_received != computed_checksum:
            raise RuntimeError(f"checksum mismatch: received {checksum_received} != {computed_checksum}")

    def parse(self):
        # get marker
        data = self.receiver.recv(1)
        if not data:
            raise RuntimeError("no data type marker received")

        marker = data[0]
        if marker == BINARY_MARKER:
            buffer = self._must_recv(HEADER_LEN)
            if not buffer:
                raise RuntimeError(f"cannot get header buffer of {HEADER_LEN} bytes")
            if len(buffer) != HEADER_LEN:
                raise RuntimeError(f"expect {HEADER_LEN} header bytes but only got {len(buffer)}")

            meta_size, body_size = HEADER_STRUCT.unpack_from(buffer, 0)
            if meta_size < 0:
                raise RuntimeError(f"invalid binary data meta size {meta_size}")

            # get headers
            meta_bytes = self._must_recv(meta_size)
            if not data:
                raise RuntimeError("no meta data received")
            if len(meta_bytes) != meta_size:
                raise RuntimeError(f"expect {meta_size} meta bytes but got {len(meta_bytes)}")

            self.meta = str(meta_bytes, "utf-8")
            self._parse_binary(body_size)
        else:
            # text content - the 1st byte is part of the data!
            content_type = CT_TEXT
            self.processor.process(data, content_type)
            self._parse_text()

        self.processor.finalize()


class MsgDataProcessor(DataProcessor):
    """The MsgDataProcessor is a special DataProcessor that can handle both TEXT and BINARY content type.
    For TEXT, it collects all received text segments and assemble them into one text string;
    For BINARY, it saves received data into a temporary file so that no memory is used to collect the data. This
    is necessary to support extremely large data exchanges (e.g. large job submission and download).

    """

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
    """Receive all data from the peer via the specified communication socket.
    This function uses the MsgDataProcessor for memory saving during data exchange.

    Args:
        sock: the communication socket

    Returns: a tuple of (content_type, request_text, additional_data)

    When content_type is CT_TEXT, the request_text is the request body, and additional_data is None;
    When content_type is CT_BINARY, the request_text is the meta info, and additional_data is the name of a
    temporary file that holds the received body data.
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
    """Create bytes for the exchange header.

    Args:
        meta_size: size of the meta info
        body_size: size of the message body

    Returns:

    """
    return HEADER_STRUCT.pack(meta_size, body_size)


def binary_footer(checksum: int):
    """Create bytes for the exchange footer.

    Args:
        checksum: the checksum value.

    Returns:

    """
    # the value of the end-of-data marker is always 0!
    return FOOTER_STRUCT.pack(0, checksum)


class DataGenerator:
    def data_size(self) -> int:
        pass

    def generate(self) -> bytes:
        pass


class Sender:
    def sendall(self, data):
        pass


def send_binary_data(sender: Sender, generator: DataGenerator, meta: str) -> int:
    """Send data in binary exchange protocol.

    Args:
        sender: the sender that is capable of sending bytes
        generator: the generator that is capable of generating data to be sent
        meta: the meta info to be included in the exchange

    Returns: number of body bytes sent

    """
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
    checksum = Checksum()
    while True:
        data = generator.generate()
        if not data:
            break
        sent_body_size += len(data)
        checksum.update(data)
        sender.sendall(data)
    if sent_body_size != body_size:
        raise RuntimeError(f"generated body size {sent_body_size} != expected body size {body_size}")

    # add footer
    footer_bytes = binary_footer(checksum.result())
    sender.sendall(footer_bytes)
    return sent_body_size


class GenerateDataFromFile(DataGenerator):
    """
    This is a special DataGenerator that generates bytes from a file.
    """

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


def send_binary_file(sock, file_name: str, meta: str) -> int:
    """Send a file using binary protocol.

    Args:
        sock: the communication socket. Note that the socket is a Sender (duck typing).
        file_name: the file to be sent
        meta: the meta info to be sent

    Returns: number of bytes sent (the same as size of the file)

    """
    gen = GenerateDataFromFile(file_name)
    return send_binary_data(sock, gen, meta)
