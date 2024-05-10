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
from nvflare.app_common.tie.cli_applet import CLIApplet
from nvflare.app_common.tie.controller import TieController
from nvflare.app_common.tie.defs import Constant as TieConstant
from nvflare.app_opt.flower.connectors.grpc_server_connector import GrpcServerConnector
from nvflare.fuel.utils.validation_utils import check_object_type, check_positive_number, check_str

from .defs import Constant


class FlowerController(TieController):
    def __init__(
        self,
        num_rounds: int,
        cli_cmd: str,
        cli_env=None,
        configure_task_name=TieConstant.CONFIG_TASK_NAME,
        configure_task_timeout=TieConstant.CONFIG_TASK_TIMEOUT,
        start_task_name=TieConstant.START_TASK_NAME,
        start_task_timeout=TieConstant.START_TASK_TIMEOUT,
        job_status_check_interval: float = TieConstant.JOB_STATUS_CHECK_INTERVAL,
        max_client_op_interval: float = TieConstant.MAX_CLIENT_OP_INTERVAL,
        progress_timeout: float = TieConstant.WORKFLOW_PROGRESS_TIMEOUT,
        int_client_grpc_options=None,
    ):
        TieController.__init__(
            self,
            configure_task_name=configure_task_name,
            configure_task_timeout=configure_task_timeout,
            start_task_name=start_task_name,
            start_task_timeout=start_task_timeout,
            job_status_check_interval=job_status_check_interval,
            max_client_op_interval=max_client_op_interval,
            progress_timeout=progress_timeout,
        )

        check_positive_number("num_rounds", num_rounds)
        check_str("cli_cmd", cli_cmd)
        if cli_env:
            check_object_type("cli_env", cli_env, dict)

        self.num_rounds = num_rounds
        self.cli_cmd = cli_cmd
        self.cli_env = cli_env
        self.int_client_grpc_options = int_client_grpc_options

    def get_connector(self, fl_ctx: FLContext):
        return GrpcServerConnector(
            int_client_grpc_options=self.int_client_grpc_options,
        )

    def get_applet(self, fl_ctx: FLContext):
        return CLIApplet()

    def get_client_config_params(self, fl_ctx: FLContext) -> dict:
        return {
            Constant.CONF_KEY_NUM_ROUNDS: self.num_rounds,
        }

    def get_connector_config_params(self, fl_ctx: FLContext) -> dict:
        config = {
            Constant.CONF_KEY_CLI_CMD: self.cli_cmd,
        }
        if self.cli_env:
            config[Constant.CONF_KEY_CLI_ENV] = self.cli_env
        return config
