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

import grpc

import nvflare.app_opt.flower.proto.fleet_pb2 as pb2
from nvflare.app_opt.flower.defs import GRPC_DEFAULT_OPTIONS
from nvflare.app_opt.flower.proto.fleet_pb2_grpc import NvFlowerStub
from nvflare.fuel.utils.obj_utils import get_logger


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

        self.channel = grpc.insecure_channel(self.server_addr, options=self.grpc_options)
        self.stub = NvFlowerStub(self.channel)

        # wait for channel ready
        try:
            grpc.channel_ready_future(self.channel).result(timeout=ready_timeout)
        except grpc.FutureTimeoutError:
            raise RuntimeError(f"cannot connect to server after {ready_timeout} seconds")

    def send_request(self, request: pb2.MessageContainer):
        """Send Flower request to gRPC server

        Args:
            request: grpc request

        Returns: a pb2.MessageContainer object

        """
        self.logger.info(f"sending {len(request.grpc_message_content)} bytes: {request.grpc_message_name=}")
        result = self.stub.SendReceive(request)
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
