# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import asyncio
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.drivers.aio_conn import AioConnection
from nvflare.fuel.f3.drivers.socket_conn import SocketConnection
from nvflare.fuel.f3.sfm.prefix import PREFIX_LEN, Prefix


class ChunkedSocket:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def recv_into(self, view, _length):
        chunk = next(self.chunks)
        view[: len(chunk)] = chunk
        return len(chunk)


class ChunkedReader:
    def __init__(self, data, chunk_size):
        self.data = bytearray(data)
        self.chunk_size = chunk_size

    async def readexactly(self, length):
        result = bytes(self.data[:length])
        del self.data[:length]
        assert len(result) == length
        return result

    async def read(self, length):
        count = min(length, self.chunk_size)
        result = bytes(self.data[:count])
        del self.data[:count]
        return result


def test_socket_reader_reports_each_received_chunk():
    receiver = Mock()
    connection = SocketConnection.__new__(SocketConnection)
    connection.sock = ChunkedSocket([b"abc", b"de"])
    connection.frame_receiver = receiver

    target = bytearray(5)
    connection.read_into(target, 0, len(target))

    assert target == b"abcde"
    assert receiver.on_receive_progress.call_count == 2


def test_async_reader_reports_progress_during_a_large_frame():
    payload = b"x" * (PREFIX_LEN + 11)
    frame = bytearray(len(payload))
    Prefix(length=len(frame)).to_buffer(frame, 0)
    frame[PREFIX_LEN:] = payload[PREFIX_LEN:]

    connection = AioConnection.__new__(AioConnection)
    connection.reader = ChunkedReader(frame, chunk_size=3)
    receiver = Mock()
    connection.frame_receiver = receiver

    result = asyncio.run(connection._async_read_frame())

    assert result == frame
    assert receiver.on_receive_progress.call_count > 2


def test_async_reader_reports_incomplete_payload_progress_before_eof():
    frame = bytearray(PREFIX_LEN + 9)
    Prefix(length=len(frame)).to_buffer(frame, 0)
    connection = AioConnection.__new__(AioConnection)
    connection.reader = ChunkedReader(frame[: PREFIX_LEN + 4], chunk_size=2)
    receiver = Mock()
    connection.frame_receiver = receiver

    async def read_frame():
        with pytest.raises(asyncio.IncompleteReadError):
            await connection._async_read_frame()

    asyncio.run(read_frame())

    assert receiver.on_receive_progress.call_count == 3
