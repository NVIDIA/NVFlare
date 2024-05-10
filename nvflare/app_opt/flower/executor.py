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
from nvflare.app_common.tie.executor import TieExecutor
from nvflare.app_opt.flower.connectors.grpc_client_connector import GrpcClientConnector
from nvflare.fuel.utils.validation_utils import check_object_type, check_str

from .defs import Constant


class FlowerExecutor(TieExecutor):
    def __init__(
        self,
        cli_cmd: str,
        cli_env=None,
        start_task_name=Constant.START_TASK_NAME,
        configure_task_name=Constant.CONFIG_TASK_NAME,
        per_msg_timeout=10.0,
        tx_timeout=100.0,
    ):
        TieExecutor.__init__(
            self,
            start_task_name=start_task_name,
            configure_task_name=configure_task_name,
        )

        check_str("cli_cmd", cli_cmd)
        if cli_env:
            check_object_type("cli_env", cli_env, dict)

        self.cli_cmd = cli_cmd
        self.cli_env = cli_env
        self.int_server_grpc_options = None
        self.per_msg_timeout = per_msg_timeout
        self.tx_timeout = tx_timeout
        self.num_rounds = None

    def get_connector(self, fl_ctx: FLContext):
        return GrpcClientConnector(
            int_server_grpc_options=self.int_server_grpc_options,
            per_msg_timeout=self.per_msg_timeout,
            tx_timeout=self.tx_timeout,
        )

    def get_applet(self, fl_ctx: FLContext):
        return CLIApplet()

    def configure(self, config: dict, fl_ctx: FLContext):
        self.num_rounds = config.get(Constant.CONF_KEY_NUM_ROUNDS)

    def get_connector_config(self, fl_ctx: FLContext) -> dict:
        config = {Constant.CONF_KEY_NUM_ROUNDS: self.num_rounds, Constant.CONF_KEY_CLI_CMD: self.cli_cmd}
        if self.cli_env:
            config[Constant.CONF_KEY_CLI_ENV] = self.cli_env
        return config
