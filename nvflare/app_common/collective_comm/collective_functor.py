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

from abc import ABC, abstractmethod

from nvflare.apis.collective_comm_constants import CollectiveCommHandleError, CollectiveCommShareableHeader


class CollectiveFunctor(ABC):
    @abstractmethod
    def __call__(self, request, world_size, buffer):
        pass


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
            if reduce_function == "MAX":
                buffer = max(buffer, buffer_in)
            elif reduce_function == "MIN":
                buffer = min(buffer, buffer_in)
            elif reduce_function == "SUM":
                buffer += buffer_in
        return buffer


class AllGatherFunctor(CollectiveFunctor):
    def __call__(self, request, world_size, buffer):
        rank = request.get_header(CollectiveCommShareableHeader.RANK)
        if buffer is None:
            buffer = [None] * world_size
        buffer[rank] = request.get_header(CollectiveCommShareableHeader.BUFFER)
        return buffer


class BroadcastFunctor(CollectiveFunctor):
    def __call__(self, request, world_size, buffer):
        rank = request.get_header(CollectiveCommShareableHeader.RANK)
        root = request.get_header(CollectiveCommShareableHeader.ROOT)
        if root is None:
            raise CollectiveCommHandleError("missing root in incoming Broadcast request")
        if rank == root:
            return request.get_header(CollectiveCommShareableHeader.BUFFER)
