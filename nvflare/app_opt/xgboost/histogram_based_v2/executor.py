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

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.adaptor_executor import XGBExecutor
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.grpc_client_adaptor import GrpcClientAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.runners.client_runner import XGBClientRunner


class FedXGBHistogramExecutor(XGBExecutor):
    def __init__(
        self,
        early_stopping_rounds,
        xgb_params: dict,
        data_loader_id: str,
        verbose_eval=False,
        use_gpus=False,
        int_server_grpc_options=None,
        req_timeout=100.0,
        model_file_name="model.json",
        metrics_writer_id: str = None,
        in_process=True,
    ):
        XGBExecutor.__init__(
            self,
            adaptor_component_id="",
            req_timeout=req_timeout,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.xgb_params = xgb_params
        self.data_loader_id = data_loader_id
        self.verbose_eval = verbose_eval
        self.use_gpus = use_gpus
        self.int_server_grpc_options = int_server_grpc_options
        self.model_file_name = model_file_name
        self.in_process = in_process
        self.metrics_writer_id = metrics_writer_id

    def get_adaptor(self, fl_ctx: FLContext):
        runner = XGBClientRunner(
            data_loader_id=self.data_loader_id,
            early_stopping_rounds=self.early_stopping_rounds,
            xgb_params=self.xgb_params,
            verbose_eval=self.verbose_eval,
            use_gpus=self.use_gpus,
            model_file_name=self.model_file_name,
            metrics_writer_id=self.metrics_writer_id,
        )
        runner.initialize(fl_ctx)
        adaptor = GrpcClientAdaptor(
            int_server_grpc_options=self.int_server_grpc_options,
            in_process=self.in_process,
        )
        adaptor.set_runner(runner)
        return adaptor
