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

from nvflare.app_common.xgb.adaptors.grpc.mock.aggr_servicer import AggrServicer
from nvflare.app_common.xgb.adaptors.grpc.server import XGBServer
from nvflare.app_common.xgb.adaptors.grpc.server_adaptor import GrpcServerAdaptor
from nvflare.app_common.xgb.defs import Constant


class MockServerAdaptor(GrpcServerAdaptor):
    def __init__(self, max_workers=10, xgb_server_ready_timeout=Constant.XGB_SERVER_READY_TIMEOUT):
        GrpcServerAdaptor.__init__(self, xgb_server_ready_timeout)
        self.max_workers = max_workers
        self.server = None
        self.server_stopped = False

    def start_server(self, addr: str, port: int, world_size: int):
        self.server = XGBServer(
            addr,
            max_workers=self.max_workers,
            options=None,
            servicer=AggrServicer(num_clients=world_size),
        )
        self.server.start(no_blocking=True)

    def stop_server(self):
        s = self.server
        self.server = None
        if s:
            s.shutdown()
        self.server_stopped = True

    def is_server_stopped(self) -> (bool, int):
        return self.server_stopped, 0
