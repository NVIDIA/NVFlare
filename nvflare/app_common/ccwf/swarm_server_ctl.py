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
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.ccwf.server_ctl import ServerSideController
from nvflare.fuel.utils.validation_utils import DefaultPolicy, validate_candidates


class SwarmServerController(ServerSideController):
    def __init__(
        self,
        num_rounds: int,
        start_round: int = 0,
        start_task_name=Constant.TASK_NAME_SWARM_START,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        configure_task_name=Constant.TASK_NAME_SWARM_CONFIGURE,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        task_check_period: float = Constant.TASK_CHECK_INTERVAL,
        job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
        participating_clients=None,
        result_clients=None,
        starting_client: str = "",
        max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        aggr_clients=None,
        train_clients=None,
    ):
        if not result_clients:
            result_clients = []

        super().__init__(
            num_rounds=num_rounds,
            start_round=start_round,
            start_task_name=start_task_name,
            start_task_timeout=start_task_timeout,
            configure_task_name=configure_task_name,
            configure_task_timeout=configure_task_timeout,
            task_check_period=task_check_period,
            job_status_check_interval=job_status_check_interval,
            participating_clients=participating_clients,
            result_clients=result_clients,
            result_clients_policy=DefaultPolicy.ALL,
            result_clients_allow_none=True,
            starting_client=starting_client,
            starting_client_policy=DefaultPolicy.ANY,
            starting_client_allow_none=False,
            max_status_report_interval=max_status_report_interval,
            progress_timeout=progress_timeout,
        )
        self.aggr_clients = aggr_clients
        self.train_clients = train_clients

    def start_controller(self, fl_ctx: FLContext):
        super().start_controller(fl_ctx)

        self.train_clients = validate_candidates(
            var_name="train_clients",
            candidates=self.train_clients,
            base=self.participating_clients,
            default_policy=DefaultPolicy.ALL,
            allow_none=False,
        )

        self.aggr_clients = validate_candidates(
            var_name="aggr_clients",
            candidates=self.aggr_clients,
            base=self.participating_clients,
            default_policy=DefaultPolicy.ALL,
            allow_none=False,
        )

        # make sure every participating client is either training or aggr client
        for c in self.participating_clients:
            if c not in self.train_clients and c not in self.aggr_clients:
                raise RuntimeError(f"Config Error:  client {c} is neither train client nor aggr client")

    def prepare_config(self):
        return {Constant.AGGR_CLIENTS: self.aggr_clients, Constant.TRAIN_CLIENTS: self.train_clients}
