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
import uuid

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.grpc_client_adaptor import GrpcClientAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_client_runner import XGBClientRunner

from .executor import XGBExecutor
from .sec.client_handler import ClientSecurityHandler


class FedXGBHistogramExecutor(XGBExecutor):
    def __init__(
        self,
        data_loader_id: str,
        per_msg_timeout=60.0,
        tx_timeout=600.0,
        model_file_name="model.json",
        metrics_writer_id: str = None,
        in_process=True,
    ):
        XGBExecutor.__init__(
            self,
            adaptor_component_id="",
            per_msg_timeout=per_msg_timeout,
            tx_timeout=tx_timeout,
        )
        self.data_loader_id = data_loader_id
        # do not let use specify int_server_grpc_options in this version - always use default
        self.int_server_grpc_options = None
        self.model_file_name = model_file_name
        self.metrics_writer_id = metrics_writer_id
        self.in_process = in_process

    def get_adaptor(self, fl_ctx: FLContext):

        engine = fl_ctx.get_engine()
        handler = ClientSecurityHandler()
        engine.add_component(str(uuid.uuid4()), handler)

        runner = XGBClientRunner(
            data_loader_id=self.data_loader_id,
            model_file_name=self.model_file_name,
            metrics_writer_id=self.metrics_writer_id,
        )
        runner.initialize(fl_ctx)
        adaptor = GrpcClientAdaptor(
            int_server_grpc_options=self.int_server_grpc_options,
            in_process=self.in_process,
            per_msg_timeout=self.per_msg_timeout,
            tx_timeout=self.tx_timeout,
        )
        adaptor.set_runner(runner)
        return adaptor
