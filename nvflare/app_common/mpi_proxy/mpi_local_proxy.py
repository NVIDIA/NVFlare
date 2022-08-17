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

from nvflare.apis.collective_comm_constants import CollectiveCommRequestTopic, CollectiveCommShareableHeader
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.common_utils import get_open_ports
from nvflare.app_common.mpi_proxy import mpi_pb2_grpc
from nvflare.app_common.mpi_proxy.mpi_pb2 import AllgatherReply, AllreduceReply, BroadcastReply, ReduceOperation


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

    def __init__(self, server_key_path, server_cert_path, client_cert_path, fl_context: FLContext):
        self._logger = logging.getLogger(__name__)
        self._server_key_path = server_key_path
        self._server_cert_path = server_cert_path
        self._client_cert_path = client_cert_path
        self.fl_context = fl_context
        self.port = get_open_ports(1)[0]
        self.grpc_server = None
        self._logger.info(f"MPI Proxy port: {self.port}")

    def start(self):
        """Start MPI proxy server on localhost"""

        executor = futures.ThreadPoolExecutor(max_workers=MpiLocalProxy.NUM_THREADS)
        self.grpc_server = grpc.server(executor, options=MpiLocalProxy.GRPC_OPTIONS, compression=grpc.Compression.Gzip)
        mpi_pb2_grpc.add_FederatedServicer_to_server(self, self.grpc_server)
        local_address = "0.0.0.0:" + str(self.port)

        with open(self._server_key_path, "rb") as f:
            private_key = f.read()
        with open(self._server_cert_path, "rb") as f:
            certificate_chain = f.read()
        with open(self._client_cert_path, "rb") as f:
            root_ca = f.read()

        server_credentials = grpc.ssl_server_credentials(
            (
                (
                    private_key,
                    certificate_chain,
                ),
            ),
            root_certificates=root_ca,
            require_client_auth=True,
        )
        self.grpc_server.add_secure_port(local_address, server_credentials)
        self.grpc_server.start()

        self._logger.info(f"MPI Proxy on port {self.port} has started")

    def stop(self):
        """Stop the proxy server"""
        if self.grpc_server:
            self.grpc_server.stop(0)
        self._logger.info(f"MPI Proxy on port {self.port} has stopped")

    # Servicer implementation
    def Allgather(self, request, context):
        shareable = Shareable()
        shareable.set_header(CollectiveCommShareableHeader.IS_COLLECTIVE_AUX, True)
        shareable.set_header(CollectiveCommShareableHeader.SEQUENCE_NUMBER, request.sequence_number)
        shareable.set_header(CollectiveCommShareableHeader.RANK, request.rank)
        shareable.set_header(CollectiveCommShareableHeader.BUFFER, array("B", request.send_buffer))

        engine = self.fl_context.get_engine()
        result: Shareable = engine.send_aux_request(
            topic=CollectiveCommRequestTopic.ALL_GATHER, request=shareable, timeout=30.0, fl_ctx=self.fl_context
        )
        buffer = result.get_header(CollectiveCommShareableHeader.BUFFER)

        return AllgatherReply(receive_buffer=buffer.tobytes())

    def Allreduce(self, request, context):
        shareable = Shareable()
        shareable.set_header(CollectiveCommShareableHeader.IS_COLLECTIVE_AUX, True)
        shareable.set_header(CollectiveCommShareableHeader.SEQUENCE_NUMBER, request.sequence_number)
        shareable.set_header(CollectiveCommShareableHeader.RANK, request.rank)

        type_code = MpiLocalProxy.TYPE_CODE_MAP[request.data_type]
        shareable.set_header(CollectiveCommShareableHeader.BUFFER, array(type_code, request.send_buffer))
        shareable.set_header(
            CollectiveCommShareableHeader.REDUCE_FUNCTION, ReduceOperation.Name(request.reduce_operation)
        )

        engine = self.fl_context.get_engine()
        result = engine.send_aux_request(
            topic=CollectiveCommRequestTopic.ALL_REDUCE, request=shareable, timeout=30.0, fl_ctx=self.fl_context
        )
        buffer = result.get_header(CollectiveCommShareableHeader.BUFFER)

        return AllreduceReply(receive_buffer=buffer.tobytes())

    def Broadcast(self, request, context):
        shareable = Shareable()
        shareable.set_header(CollectiveCommShareableHeader.IS_COLLECTIVE_AUX, True)
        shareable.set_header(CollectiveCommShareableHeader.SEQUENCE_NUMBER, request.sequence_number)
        shareable.set_header(CollectiveCommShareableHeader.RANK, request.rank)
        shareable.set_header(CollectiveCommShareableHeader.BUFFER, array("B", request.send_buffer))
        shareable.set_header(CollectiveCommShareableHeader.ROOT, request.root)

        engine = self.fl_context.get_engine()
        result = engine.send_aux_request(
            topic=CollectiveCommRequestTopic.BROADCAST, request=shareable, timeout=30.0, fl_ctx=self.fl_context
        )
        buffer = result.get_header(CollectiveCommShareableHeader.BUFFER)

        return BroadcastReply(receive_buffer=buffer.tobytes())

    def _print_request(self, func_name, request, type_code):
        buffer_array = array(type_code, request.send_buffer).tolist()
        self._logger.info(
            f"{func_name}: seq: {request.sequence_number}, rank: {request.rank}," f"buffer to list: {buffer_array}"
        )
