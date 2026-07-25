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


class BufferList:
    """A buffer list that can be treated as a single buffer"""

    def __init__(self, buf_list: list):
        self.buf_list = buf_list

    def get_size(self):

        if self.buf_list:
            size = sum(len(buf) for buf in self.buf_list)
        else:
            size = 0

        return size

    def get_list(self):
        return self.buf_list

    def append(self, buf: bytes):
        if not self.buf_list:
            self.buf_list = []

        self.buf_list.append(buf)

    def read(self, start: int, end: int):

        buffer = None
        view_start = 0
        pos = 0
        for view in self.buf_list:

            view_end = view_start + len(view)

            if view_start <= start < view_end and end <= view_end:
                return view[start - view_start : end - view_start]

            buf_start = start + pos

            if buf_start < view_end:
                if not buffer:
                    buffer = bytearray(end - start)

                remaining = min(end, view_end) - buf_start
                view_pos = buf_start - view_start
                buffer[pos : pos + remaining] = view[view_pos : view_pos + remaining]
                pos = pos + remaining

                if view_end >= end:
                    break

            view_start = view_end

        return buffer

    def read_bytes(self, start: int, end: int) -> bytes:
        """Read a range into one immutable bytes allocation.

        Unlike ``read()``, this method does not first assemble a multi-buffer
        range into a bytearray. This matters for large streamed FOBS sections:
        callers that require bytes would otherwise copy the complete range
        once into a bytearray and a second time into bytes.
        """
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        if end < start:
            raise ValueError(f"end {end} must not be less than start {start}")
        if start == end:
            return b""

        parts = []
        view_start = 0
        for buffer in self.buf_list or []:
            view_end = view_start + len(buffer)
            if view_end <= start:
                view_start = view_end
                continue
            if view_start >= end:
                break

            part_start = max(start, view_start) - view_start
            part_end = min(end, view_end) - view_start
            parts.append(memoryview(buffer)[part_start:part_end])
            view_start = view_end

        if not parts:
            return b""
        if len(parts) == 1:
            return parts[0].tobytes()
        return b"".join(parts)

    def flatten(self):

        size = self.get_size()
        if not size:
            return None

        result = bytearray(size)
        start = 0
        for b in self.buf_list:
            size = len(b)
            result[start : start + size] = b
            start += size

        return result
