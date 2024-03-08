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

from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.grpc.grpc_server import GrpcServer
from nvflare.app_opt.xgboost.histogram_based_v2.runner import XGBRunner

from .aggr_servicer import AggrServicer


class MockServerRunner(XGBRunner):
    def __init__(self, server_max_workers=10):
        self.server_max_workers = server_max_workers
        self._stopped = False
        self._server = None

    def run(self, ctx: dict):
        world_size = ctx.get(Constant.RUNNER_CTX_WORLD_SIZE)
        addr = ctx.get(Constant.RUNNER_CTX_SERVER_ADDR)

        self._server = GrpcServer(
            addr,
            max_workers=self.server_max_workers,
            grpc_options=None,
            servicer=AggrServicer(num_clients=world_size),
        )
        self._server.start(no_blocking=False)

    def stop(self):
        s = self._server
        self._server = None
        if s:
            s.shutdown()
        self._stopped = True

    def is_stopped(self) -> (bool, int):
        return self._stopped, 0
