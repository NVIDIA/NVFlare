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

import pytest

from nvflare.fuel.utils.buffer_list import BufferList


@pytest.mark.parametrize(
    "buffers",
    [
        [b"abc", b"def", b"ghi"],
        [bytearray(b"abc"), bytearray(b"def"), bytearray(b"ghi")],
        [memoryview(b"abc"), memoryview(b"def"), memoryview(b"ghi")],
    ],
)
def test_read_bytes_across_buffers_returns_bytes(buffers):
    result = BufferList(buffers).read_bytes(1, 8)

    assert type(result) is bytes
    assert result == b"bcdefgh"


def test_read_bytes_within_one_buffer_returns_bytes():
    result = BufferList([memoryview(b"abcdef")]).read_bytes(1, 5)

    assert type(result) is bytes
    assert result == b"bcde"


def test_read_bytes_rejects_end_past_available_data():
    with pytest.raises(ValueError, match="exceeds available data"):
        BufferList([b"abc"]).read_bytes(0, 4)


def test_discard_before_releases_complete_buffers_and_keeps_absolute_offsets():
    buffers = [b"abc", b"def", b"ghi"]
    buffer_list = BufferList(buffers)

    buffer_list.discard_before(6)

    assert buffer_list.get_list() == [b"ghi"]
    assert buffer_list.start_offset == 6
    assert buffer_list.read_bytes(6, 9) == b"ghi"
    with pytest.raises(ValueError, match="precedes discarded data"):
        buffer_list.read_bytes(0, 1)


@pytest.mark.parametrize(
    "start,end,error",
    [
        (-1, 1, "start must be non-negative"),
        (2, 1, "must not be less than start"),
    ],
)
def test_read_bytes_rejects_invalid_range(start, end, error):
    with pytest.raises(ValueError, match=error):
        BufferList([b"abc"]).read_bytes(start, end)
