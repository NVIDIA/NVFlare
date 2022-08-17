# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import array
from abc import ABC, abstractmethod

from nvflare.apis.collective_comm_constants import CollectiveCommHandleError, CollectiveCommShareableHeader


class CollectiveFunctor(ABC):
    @abstractmethod
    def __call__(self, request, world_size, buffer):
        pass


def _add(a, b):
    return a + b


def _array_func(a, b, func):
    c = []
    for i in range(len(a)):
        c.append(func(a[i], b[i]))
    return array.array(a.typecode, c)


def _init_array(typecode, length):
    if typecode in ["b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"]:
        default = 0
    elif typecode in ["f", "d"]:
        default = 0.0
    else:
        default = ""
    return array.array(typecode, [default] * length)


class AllReduceFunctor(CollectiveFunctor):
    def __call__(self, request, world_size, buffer):
        reduce_function = request.get_header(CollectiveCommShareableHeader.REDUCE_FUNCTION)
        if reduce_function is None:
            raise CollectiveCommHandleError("missing reduce_function in incoming All Reduce request")
        if reduce_function not in ["MAX", "MIN", "SUM"]:
            raise CollectiveCommHandleError(f"reduce_function {reduce_function} is not supported")

        buffer_in = request.get_header(CollectiveCommShareableHeader.BUFFER)
        if buffer is None:
            buffer = buffer_in
        else:
            if len(buffer_in) != len(buffer):
                raise CollectiveCommHandleError("buffer length mismatch!")
            elif buffer_in.typecode != buffer.typecode:
                raise CollectiveCommHandleError("buffer typecode mismatch!")
            if reduce_function == "MAX":
                buffer = _array_func(buffer, buffer_in, max)
            elif reduce_function == "MIN":
                buffer = _array_func(buffer, buffer_in, min)
            elif reduce_function == "SUM":
                buffer = _array_func(buffer, buffer_in, _add)
        return buffer


class AllGatherFunctor(CollectiveFunctor):
    def __call__(self, request, world_size, buffer):
        rank = request.get_header(CollectiveCommShareableHeader.RANK)
        buffer_in: array.array = request.get_header(CollectiveCommShareableHeader.BUFFER)
        if buffer is None:
            buffer = _init_array(buffer_in.typecode, len(buffer_in) * world_size)
        start_ind = rank * len(buffer_in)
        end_ind = (rank + 1) * len(buffer_in)
        buffer[start_ind:end_ind] = buffer_in
        return buffer


class BroadcastFunctor(CollectiveFunctor):
    def __call__(self, request, world_size, buffer):
        rank = request.get_header(CollectiveCommShareableHeader.RANK)
        root = request.get_header(CollectiveCommShareableHeader.ROOT)
        if root is None:
            raise CollectiveCommHandleError("missing root in incoming Broadcast request")
        if rank == root:
            return request.get_header(CollectiveCommShareableHeader.BUFFER)
        return None
