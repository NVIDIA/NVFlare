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
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.grpc_client_adaptor import GrpcClientAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.executor import XGBExecutor
from nvflare.app_opt.xgboost.histogram_based_v2.mock.mock_secure_client_runner import MockSecureClientRunner


class MockSecureXGBExecutor(XGBExecutor):
    def __init__(
        self,
        int_server_grpc_options=None,
        req_timeout=10.0,
        in_process=True,
    ):
        XGBExecutor.__init__(
            self,
            adaptor_component_id="",
            req_timeout=req_timeout,
        )
        self.int_server_grpc_options = int_server_grpc_options
        self.in_process = in_process

    def get_adaptor(self, fl_ctx: FLContext):
        runner = MockSecureClientRunner()
        runner.initialize(fl_ctx)
        adaptor = GrpcClientAdaptor(
            int_server_grpc_options=self.int_server_grpc_options,
            in_process=self.in_process,
        )
        adaptor.set_runner(runner)
        return adaptor
