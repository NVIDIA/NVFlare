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
import struct

from .checksum import Checksum

HEADER_STRUCT = struct.Struct(">BII")  # marker(1), seq_num(4), size(4)
HEADER_LEN = HEADER_STRUCT.size

MARKER_DATA = 101
MARKER_END = 102

MAX_CHUNK_SIZE = 1024 * 1024


def get_slice(buf, start: int, length: int):
    view = memoryview(buf)
    return view[start : start + length]


class Header:
    def __init__(self, marker, num1, num2):
        self.marker = marker
        self.checksum = 0
        self.seq = 0
        self.size = 0
        if marker == MARKER_DATA:
            self.seq = num1
            self.size = num2
        elif marker == MARKER_END:
            if num1 != 0:
                raise ValueError(f"num1 must be 0 for checksum but got {num1}")
            self.checksum = num2
        else:
            raise ValueError(f"invalid chunk marker {marker}")

    def __str__(self):
        d = {
            "marker": self.marker,
            "seq": self.seq,
            "size": self.size,
            "checksum": self.checksum,
        }
        return f"{d}"

    @classmethod
    def from_bytes(cls, buffer: bytes):
        if len(buffer) < HEADER_LEN:
            raise ValueError("Prefix too short")

        marker, num1, num2 = HEADER_STRUCT.unpack_from(buffer, 0)
        return Header(marker, num1, num2)

    def to_bytes(self):
        if self.marker == MARKER_DATA:
            num1 = self.seq
            num2 = self.size
        else:
            num1 = 0
            num2 = self.checksum

        return HEADER_STRUCT.pack(self.marker, num1, num2)


class ChunkState:
    def __init__(self, expect_seq=1):
        self.header_bytes = bytearray()
        self.header = None
        self.received = 0
        self.expect_seq = expect_seq

    def __str__(self):
        d = {
            "header": f"{self.header}",
            "header_bytes": f"{self.header_bytes}",
            "received": self.received,
            "expect_seq": self.expect_seq,
        }
        return f"{d}"

    def unpack_header(self):
        self.header = Header.from_bytes(self.header_bytes)
        if self.header.marker == MARKER_DATA:
            if self.header.seq != self.expect_seq:
                raise RuntimeError(
                    f"Protocol Error: received seq {self.header.seq} does not match expected seq {self.expect_seq}"
                )

            if self.header.size < 0 or self.header.size > MAX_CHUNK_SIZE:
                raise RuntimeError(f"Protocol Error: received size {self.header.size} is not in [0, {MAX_CHUNK_SIZE}]")

    def is_last(self):
        return self.header and self.header.marker == MARKER_END


class Receiver:
    def __init__(self, receive_data_func):
        self.receive_data_func = receive_data_func
        self.checksum = Checksum()
        self.current_state = ChunkState()
        self.done = False

    def receive(self, data) -> bool:
        if self.done:
            raise RuntimeError("this receiver is already done")
        s = chunk_it(self.current_state, data, 0, self._process_chunk)
        self.current_state = s
        done = s.is_last()
        if done:
            self.done = True
            # compare checksum
            expected_checksum = self.checksum.result()
            if expected_checksum != s.header.checksum:
                raise RuntimeError(f"checksum mismatch: expect {expected_checksum} but received {s.header.checksum}")
        return done

    def _process_chunk(self, c: ChunkState, data, start: int, length: int):
        self.checksum.update(get_slice(data, start, length))
        if self.receive_data_func:
            self.receive_data_func(data, start, length)


class Sender:
    def __init__(self, send_data_func):
        self.send_data_func = send_data_func
        self.checksum = Checksum()
        self.next_seq = 1
        self.closed = False

    def send(self, data):
        if self.closed:
            raise RuntimeError("this sender is already closed")
        if data is None:
            data = b""
        header = Header(MARKER_DATA, self.next_seq, len(data))
        self.next_seq += 1
        self.checksum.update(data)
        header_bytes = header.to_bytes()
        self.send_data_func(header_bytes)
        self.send_data_func(data)

    def close(self):
        if self.closed:
            raise RuntimeError("this sender is already closed")
        self.closed = True
        cs = self.checksum.result()
        header = Header(MARKER_END, 0, cs)
        header_bytes = header.to_bytes()
        self.send_data_func(header_bytes)


def chunk_it(c: ChunkState, data, cursor: int, process_chunk_func) -> ChunkState:
    if not isinstance(data, (bytearray, bytes)):
        raise ValueError(f"can only chunk bytes data but got {type(data)}")

    data_len = len(data)
    if data_len <= 0:
        return c

    if cursor < 0 or cursor >= data_len:
        raise ValueError(f"cursor {cursor} is out of data range [0, {data_len-1}]")
    data_len -= cursor

    header_bytes_len = len(c.header_bytes)
    if header_bytes_len < HEADER_LEN:
        # header not completed yet
        num_bytes_needed = HEADER_LEN - header_bytes_len
        # need this many bytes for header
        if data_len >= num_bytes_needed:
            # data has enough bytes
            c.header_bytes.extend(get_slice(data, cursor, num_bytes_needed))
            cursor += num_bytes_needed
            data_len -= num_bytes_needed
            c.unpack_header()  # header bytes are ready
        else:
            c.header_bytes.extend(get_slice(data, cursor, data_len))
            return c

    if data_len == 0 or c.is_last():
        return c

    lack = c.header.size - c.received
    if data_len <= lack:
        # remaining data is part of the chunk
        c.received += data_len
        process_chunk_func(c, data, cursor, data_len)
        if c.received == c.header.size:
            # this chunk is completed: start a new chunk
            return ChunkState(c.header.seq + 1)
        else:
            # this chunk is not done
            return c
    else:
        # some remaining data is part of the chunk, but after that belongs to next chunk
        c.received += lack
        process_chunk_func(c, data, cursor, lack)
        cursor += lack
        next_chunk = ChunkState(c.header.seq + 1)
        return chunk_it(next_chunk, data, cursor, process_chunk_func)
