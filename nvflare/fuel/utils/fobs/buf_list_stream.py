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
from nvflare.fuel.utils.buffer_list import BufferList, ConsumableBufferList


class BufListStream:
    def __init__(self, buf_list: list = None):
        self.buffer_list = BufferList(buf_list)
        self.release_consumed = isinstance(buf_list, ConsumableBufferList)
        self.pos = 0
        self.size = self.buffer_list.get_size()

    def getvalue(self):
        return self.buffer_list.get_list()

    def write(self, buf: bytes):
        self.buffer_list.append(buf)

    def read(self, n: int):
        end = self.pos + n
        if end > self.size:
            end = self.size

        result = self.buffer_list.read_bytes(self.pos, end)
        self.pos = end
        # FOBS reads forward only. Once a section has been copied into its
        # immutable bytes result, release complete network chunks behind it so
        # a large section does not coexist with the entire original blob.
        if self.release_consumed:
            self.buffer_list.discard_before(self.pos)

        return result
