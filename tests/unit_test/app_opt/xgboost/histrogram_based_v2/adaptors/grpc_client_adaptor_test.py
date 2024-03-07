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

from unittest.mock import patch

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.grpc_client_adaptor import GrpcClientAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant

from .mock_runner import MockRunner, wait_for_status


class TestGrpcClientAdaptor:
    def test_start_and_stop(self):
        runner = MockRunner()
        adaptor = GrpcClientAdaptor(in_process=True)
        config = {Constant.CONF_KEY_WORLD_SIZE: 66, Constant.CONF_KEY_RANK: 44, Constant.CONF_KEY_NUM_ROUNDS: 100}
        ctx = FLContext()
        adaptor.configure(config, ctx)

        adaptor.set_runner(runner)
        with patch("nvflare.app_opt.xgboost.histogram_based_v2.grpc.grpc_server.GrpcServer.start") as mock_method:
            mock_method.return_value = True
            adaptor.start(ctx)
            assert wait_for_status(runner, True)

            adaptor.stop(ctx)
            assert wait_for_status(runner, False)
