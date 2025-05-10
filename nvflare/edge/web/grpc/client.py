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
import threading

import grpc

from .edge_api_pb2 import Reply, Request
from .edge_api_pb2_grpc import EdgeApiStub


class Client:

    def __init__(self, grpc_options=None):
        self.grpc_options = grpc_options
        self.stubs = {}  # address => Stub
        self._stub_lock = threading.Lock()

    def _get_stub(self, address):
        with self._stub_lock:
            stub = self.stubs.get(address)
            if not stub:
                channel = grpc.insecure_channel(address, options=self.grpc_options)
                stub = EdgeApiStub(channel)
                self.stubs[address] = stub
            return stub

    def query(self, address: str, request: Request) -> Reply:
        stub = self._get_stub(address)
        return stub.Query(request)
