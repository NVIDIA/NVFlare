# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.fuel.utils.log_utils import get_obj_logger

from .edge_api_pb2 import Reply, Request
from .edge_api_pb2_grpc import EdgeApiStub


class EdgeApiClient:

    def __init__(self, ssl_credentials=None, grpc_options=None):
        self.grpc_options = grpc_options
        self.ssl_credentials = ssl_credentials
        self.logger = get_obj_logger(self)
        if ssl_credentials:
            self.logger.info("SSL Credentials provided - will connect via secure channel")
        else:
            self.logger.info("SSL Credentials not provided - will connect via insecure channel")

    def query(self, address: str, request: Request) -> Reply:
        if self.ssl_credentials:
            # SSL
            channel = grpc.secure_channel(address, options=self.grpc_options, credentials=self.ssl_credentials)
        else:
            channel = grpc.insecure_channel(address, options=self.grpc_options)

        stub = EdgeApiStub(channel)
        return stub.Query(request)
