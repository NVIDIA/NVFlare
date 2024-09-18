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
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.grpc_server_adaptor import GrpcServerAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_server_runner import XGBServerRunner

from .controller import XGBController
from .defs import Constant
from .sec.server_handler import ServerSecurityHandler


class XGBFedController(XGBController):
    def __init__(
        self,
        num_rounds: int,
        data_split_mode: int,
        secure_training: bool,
        xgb_params: dict,
        xgb_options: Optional[dict] = None,
        disable_version_check=False,
        configure_task_name=Constant.CONFIG_TASK_NAME,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        start_task_name=Constant.START_TASK_NAME,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
        max_client_op_interval: float = Constant.MAX_CLIENT_OP_INTERVAL,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        client_ranks=None,
        in_process=True,
    ):
        XGBController.__init__(
            self,
            adaptor_component_id="",
            num_rounds=num_rounds,
            data_split_mode=data_split_mode,
            secure_training=secure_training,
            xgb_params=xgb_params,
            xgb_options=xgb_options,
            disable_version_check=disable_version_check,
            configure_task_name=configure_task_name,
            configure_task_timeout=configure_task_timeout,
            start_task_name=start_task_name,
            start_task_timeout=start_task_timeout,
            job_status_check_interval=job_status_check_interval,
            max_client_op_interval=max_client_op_interval,
            progress_timeout=progress_timeout,
            client_ranks=client_ranks,
        )
        # do not let user specify int_client_grpc_options in this version - always use default.
        self.int_client_grpc_options = None
        self.in_process = in_process

    def get_adaptor(self, fl_ctx: FLContext):

        engine = fl_ctx.get_engine()
        handler = ServerSecurityHandler()
        engine.add_component(str(uuid.uuid4()), handler)

        runner = XGBServerRunner()
        runner.initialize(fl_ctx)
        adaptor = GrpcServerAdaptor(
            int_client_grpc_options=self.int_client_grpc_options,
            in_process=self.in_process,
        )
        adaptor.set_runner(runner)
        return adaptor
