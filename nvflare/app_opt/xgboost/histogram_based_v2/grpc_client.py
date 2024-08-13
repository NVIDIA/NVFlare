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

import grpc

import nvflare.app_opt.xgboost.histogram_based_v2.proto.federated_pb2 as pb2
from nvflare.app_opt.xgboost.histogram_based_v2.defs import GRPC_DEFAULT_OPTIONS
from nvflare.app_opt.xgboost.histogram_based_v2.proto.federated_pb2_grpc import FederatedStub
from nvflare.fuel.utils.obj_utils import get_logger


class GrpcClient:
    """This class implements a gRPC XGB Client that is capable of sending XGB operations to a gRPC XGB Server."""

    def __init__(self, server_addr, grpc_options=None):
        """Constructor

        Args:
            server_addr: address of the gRPC server to connect to
            grpc_options: An optional list of key-value pairs (`channel_arguments`
                in gRPC Core runtime) to configure the gRPC channel.
        """
        if not grpc_options:
            grpc_options = GRPC_DEFAULT_OPTIONS

        self.stub = None
        self.channel = None
        self.server_addr = server_addr
        self.grpc_options = grpc_options
        self.started = False
        self.logger = get_logger(self)

    def start(self, ready_timeout=10):
        """Start the gRPC client and wait for the server to be ready.

        Args:
            ready_timeout: how long to wait for the server to be ready

        Returns: None

        """
        if self.started:
            return

        self.started = True

        self.channel = grpc.insecure_channel(self.server_addr, options=self.grpc_options)
        self.stub = FederatedStub(self.channel)

        # wait for channel ready
        try:
            grpc.channel_ready_future(self.channel).result(timeout=ready_timeout)
        except grpc.FutureTimeoutError:
            raise RuntimeError(f"cannot connect to server after {ready_timeout} seconds")

    def send_allgather(self, seq_num, rank, data: bytes):
        """Send Allgather request to gRPC server

        Args:
            seq_num: sequence number
            rank: rank of the client
            data: the send_buf data

        Returns: an AllgatherReply object; or None if processing error is encountered

        """
        req = pb2.AllgatherRequest(
            sequence_number=seq_num,
            rank=rank,
            send_buffer=data,
        )

        self.logger.info(f"Allgather is sending {len(data)} bytes Rank: {rank} Seq: {seq_num}")
        result = self.stub.Allgather(req)

        if not isinstance(result, pb2.AllgatherReply):
            self.logger.error(f"expect reply to be pb2.AllgatherReply but got {type(result)}")
            return None
        return result

    def send_allgatherv(self, seq_num, rank, data: bytes):
        """Send AllgatherV request to gRPC server

        Args:
            seq_num: sequence number
            rank: rank of the client
            data: the send_buf data

        Returns: an AllgatherVReply object; or None if processing error is encountered

        """
        req = pb2.AllgatherVRequest(
            sequence_number=seq_num,
            rank=rank,
            send_buffer=data,
        )

        result = self.stub.AllgatherV(req)
        if not isinstance(result, pb2.AllgatherVReply):
            self.logger.error(f"expect reply to be pb2.AllgatherVReply but got {type(result)}")
            return None
        return result

    def send_allreduce(self, seq_num, rank, data: bytes, data_type, reduce_op):
        """Send Allreduce request to gRPC server

        Args:
            seq_num: sequence number
            rank: rank of the client
            data: the send_buf data
            data_type: data type of the input
            reduce_op: reduce op to be performed

        Returns: an AllreduceReply object; or None if processing error is encountered

        """
        req = pb2.AllreduceRequest(
            sequence_number=seq_num,
            rank=rank,
            send_buffer=data,
            data_type=data_type,
            reduce_operation=reduce_op,
        )

        result = self.stub.Allreduce(req)
        if not isinstance(result, pb2.AllreduceReply):
            self.logger.error(f"expect reply to be pb2.AllreduceReply but got {type(result)}")
            return None
        return result

    def send_broadcast(self, seq_num, rank, data: bytes, root):
        """Send Broadcast request to gRPC server

        Args:
            seq_num: sequence number
            rank: rank of the client
            data: the send_buf data
            root: rank of the root

        Returns: a BroadcastReply object; or None if processing error is encountered

        """
        req = pb2.BroadcastRequest(
            sequence_number=seq_num,
            rank=rank,
            send_buffer=data,
            root=root,
        )

        result = self.stub.Broadcast(req)
        if not isinstance(result, pb2.BroadcastReply):
            self.logger.error(f"expect reply to be pb2.BroadcastReply but got {type(result)}")
            return None
        return result

    def stop(self):
        """Stop the gRPC client

        Returns: None

        """
        ch = self.channel
        self.channel = None  # set to None in case another thread also tries to close.
        if ch:
            try:
                ch.close()
            except:
                # ignore errors when closing the channel
                pass
