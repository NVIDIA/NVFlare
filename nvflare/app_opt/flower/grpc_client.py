# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import flwr.proto.grpcadapter_pb2 as pb2
from flwr.proto.grpcadapter_pb2_grpc import GrpcAdapterStub

from nvflare.app_opt.flower.defs import GRPC_DEFAULT_OPTIONS
from nvflare.fuel.utils.grpc_utils import create_channel
from nvflare.fuel.utils.obj_utils import get_logger

from .utils import reply_should_exit


class GrpcClient:
    """This class implements a gRPC Client that is capable of sending Flower requests to a Flower gRPC Server."""

    def __init__(self, server_addr, grpc_options=None):
        """Constructor

        Args:
            server_addr: address of the gRPC server to connect to
            grpc_options: gRPC options for the gRPC client
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

        self.channel = create_channel(
            server_addr=self.server_addr,
            grpc_options=self.grpc_options,
            ready_timeout=ready_timeout,
            test_only=False,
        )
        self.stub = GrpcAdapterStub(self.channel)

    def send_request(self, request: pb2.MessageContainer):
        """Send Flower request to gRPC server

        Args:
            request: grpc request

        Returns: a pb2.MessageContainer object

        """
        self.logger.info(f"sending {len(request.grpc_message_content)} bytes: {request.grpc_message_name=}")
        try:
            result = self.stub.SendReceive(request)
        except Exception as ex:
            self.logger.warning(f"exception occurred communicating to Flower server: {ex}")
            return reply_should_exit()

        if not isinstance(result, pb2.MessageContainer):
            self.logger.error(f"expect reply to be pb2.MessageContainer but got {type(result)}")
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
