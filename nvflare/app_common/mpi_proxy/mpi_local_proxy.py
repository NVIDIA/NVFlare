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
import logging
from array import array
from concurrent import futures

import grpc

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.common_utils import get_open_ports
from nvflare.app_common.mpi_proxy import mpi_pb2_grpc, mpi_pb2
from nvflare.app_common.mpi_proxy.mpi_constants import MpiFields, MpiFunctions, MPI_PROXY_TOPIC
from nvflare.app_common.mpi_proxy.mpi_pb2 import AllgatherReply, AllreduceReply, DataType, ReduceOperation, \
    BroadcastReply

logger = logging.getLogger(__name__)


class MpiLocalProxy(mpi_pb2_grpc.FederatedServicer):

    NUM_THREADS = 8
    GRPC_OPTIONS = [
        ("grpc.max_send_message_length", 1024 * 1024 * 1024),
        ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
    ]

    # DATA_TYPE defined in proto
    # CHAR = 0
    # UCHAR = 1
    # INT = 2
    # UINT = 3
    # LONG = 4
    # ULONG = 5
    # FLOAT = 6
    # DOUBLE = 7
    # LONGLONG = 8
    # ULONGLONG = 9
    TYPE_CODE_MAP = ["b", "B", "i", "I", "l", "L", "f", "d", "q", "Q"]

    def __init__(self, fl_context: FLContext):
        self.fl_context = fl_context
        self.port = get_open_ports(1)[0]
        self.server_future = None
        logger.info(f"MPI Proxy port: {self.port}")

    def start(self):
        """ Start MPI proxy server on localhost"""

        executor = futures.ThreadPoolExecutor(max_workers=MpiLocalProxy.NUM_THREADS)
        self.server_future = executor.submit(self._run_grpc_server, executor=executor)

        logger.debug("GRPC server started")

    def stop(self):
        """Stop the proxy server"""
        self.server.stop()
        logger.info(f"MPI Proxy on port {self.port} has stopped")

    def wait(self):
        """Wait till GRPC server ends"""
        if self.server_future:
            self.server_future.result()

    # Servicer implementation
    def Allgather(self, request, context):
        shareable = Shareable()
        shareable[MpiFields.MPI_FUNC] = MpiFunctions.ALL_GATHER
        shareable[MpiFields.SEQUENCE_NUMBER] = request.sequence_number
        shareable[MpiFields.WORLD_RANK] = request.rank
        shareable[MpiFields.BUFFER] = array("B", request.send_buffer)

        engine = self.fl_context.get_engine()
        result = engine.send_aux_request(
            topic=MPI_PROXY_TOPIC, request=shareable, timeout=30.0, fl_ctx=self.fl_context
        )
        buffer = result[MpiFields.BUFFER]

        return AllgatherReply(receive_buffer=buffer.tobytes())

    def Allreduce(self, request, context):
        shareable = Shareable()
        shareable[MpiFields.MPI_FUNC] = MpiFunctions.ALL_REDUCE
        shareable[MpiFields.SEQUENCE_NUMBER] = request.sequence_number
        shareable[MpiFields.RANK] = request.rank

        type_code = MpiLocalProxy.TYPE_CODE_MAP[request.data_type]

        shareable[MpiFields.BUFFER] = array(type_code, request.send_buffer)
        shareable[MpiFields.DATA_TYPE] = DataType.Name(request.data_type)
        shareable[MpiFields.REDUCE_OPERATION] = ReduceOperation.Name(request.reduce_operation)

        engine = self.fl_context.get_engine()
        result = engine.send_aux_request(
            topic=MPI_PROXY_TOPIC, request=shareable, timeout=30.0, fl_ctx=self.fl_context
        )
        buffer = result[MpiFields.BUFFER]

        return AllreduceReply(receive_buffer=buffer.tobytes())

    def Broadcast(self, request, context):
        shareable = Shareable()
        shareable[MpiFields.MPI_FUNC] = MpiFunctions.BROADCAST
        shareable[MpiFields.SEQUENCE_NUMBER] = request.sequence_number
        shareable[MpiFields.BUFFER] = array("B", request.send_buffer)
        shareable[MpiFields.RANK] = request.rank
        shareable[MpiFields.ROOT] = request.root

        engine = self.fl_context.get_engine()
        result = engine.send_aux_request(
            topic=MPI_PROXY_TOPIC, request=shareable, timeout=30.0, fl_ctx=self.fl_context
        )
        buffer = result[MpiFields.BUFFER]

        return BroadcastReply(receive_buffer=buffer.tobytes())

    def _run_grpc_server(self, executor: futures.ThreadPoolExecutor):
        server = grpc.server(
            executor,
            options=MpiLocalProxy.GRPC_OPTIONS,
            compression=grpc.Compression.Gzip)

        mpi_pb2_grpc.add_FederatedServicer_to_server(self, server)
        local_port = "localhost:" + str(self.port)
        server.add_insecure_port(local_port)
        server.start()
        logger.info(f"MPI Proxy is started on {local_port}")
        server.wait_for_termination()
        return server
