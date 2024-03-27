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
from nvflare.fuel.utils.validation_utils import (
    check_non_negative_int,
    check_object_type,
    check_positive_number,
    check_str,
)


class FedXGBHistogramExecutor(XGBExecutor):
    def __init__(
        self,
        early_stopping_rounds,
        xgb_params: dict,
        data_loader_id: str,
        verbose_eval=False,
        use_gpus=False,
        per_msg_timeout=10.0,
        tx_timeout=100.0,
        model_file_name="model.json",
        metrics_writer_id: str = None,
        in_process: bool = True,
    ):
        """

        Args:
            early_stopping_rounds: early stopping rounds
            xgb_params: This dict is passed to `xgboost.train()` as the first argument `params`.
                It contains all the Booster parameters.
                Please refer to XGBoost documentation for details:
                https://xgboost.readthedocs.io/en/stable/parameter.html
            data_loader_id: the ID points to XGBDataLoader.
            verbose_eval: verbose_eval in xgboost.train
            use_gpus (bool): A convenient flag to enable gpu training, if gpu device is specified in
                the `xgb_params` then this flag can be ignored.
            metrics_writer_id: the ID points to a LogWriter, if provided, a MetricsCallback will be added.
                Users can then use the receivers from nvflare.app_opt.tracking.
            model_file_name (str): where to save the model.
            in_process (bool): Specifies whether to start the `XGBRunner` in the same process or not.
            per_msg_timeout: timeout for sending one message
            tx_timeout: transaction timeout
        """
        XGBExecutor.__init__(
            self,
            adaptor_component_id="",
        )

        if early_stopping_rounds is not None:
            check_non_negative_int("early_stopping_rounds", early_stopping_rounds)

        if xgb_params is not None:
            check_object_type("xgb_params", xgb_params, dict)

        check_str("data_loader_id", data_loader_id)
        check_positive_number("per_msg_timeout", per_msg_timeout)
        if tx_timeout:
            check_positive_number("tx_timeout", tx_timeout)

        check_str("model_file_name", model_file_name)

        if metrics_writer_id:
            check_str("metrics_writer_id", metrics_writer_id)

        self.early_stopping_rounds = early_stopping_rounds
        self.xgb_params = xgb_params
        self.data_loader_id = data_loader_id
        self.verbose_eval = verbose_eval
        self.use_gpus = use_gpus
        self.per_msg_timeout = per_msg_timeout
        self.tx_timeout = tx_timeout
        self.model_file_name = model_file_name
        self.in_process = in_process
        self.metrics_writer_id = metrics_writer_id

        # do not let user specify int_server_grpc_options in this version - always use default
        self.int_server_grpc_options = None

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
            per_msg_timeout=self.per_msg_timeout,
            tx_timeout=self.tx_timeout,
        )
        adaptor.set_runner(runner)
        return adaptor
