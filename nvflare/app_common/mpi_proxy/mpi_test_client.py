#  Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import grpc

from nvflare.app_common.mpi_proxy import mpi_pb2_grpc
from nvflare.app_common.mpi_proxy.mpi_pb2 import (
    AllgatherRequest,
    AllreduceRequest,
    BroadcastRequest,
    DataType,
    ReduceOperation,
)


def all_gather(stub):
    request = AllgatherRequest()
    request.sequence_number = 123
    request.rank = 9
    request.send_buffer = "Test buffer".encode()
    reply = stub.Allgather(request)
    print(reply.receive_buffer.decode())


def all_reduce(stub):
    request = AllreduceRequest()
    request.sequence_number = 123
    request.rank = 9
    request.send_buffer = "Test buffer".encode()
    request.data_type = DataType.DOUBLE
    request.reduce_operation = ReduceOperation.SUM

    reply = stub.Allreduce(request)
    print(reply.receive_buffer.decode())


def broadcast(stub):
    request = BroadcastRequest()
    request.sequence_number = 123
    request.rank = 9
    request.root = 1
    request.send_buffer = "Test buffer".encode()
    reply = stub.Broadcast(request)
    print(reply.receive_buffer.decode())


if __name__ == "__main__":
    channel = grpc.insecure_channel("localhost:55442")
    grpc_stub = mpi_pb2_grpc.FederatedStub(channel)
    broadcast(grpc_stub)
