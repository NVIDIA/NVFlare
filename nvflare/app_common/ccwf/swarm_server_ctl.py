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
from nvflare.fuel.utils.validation_utils import DefaultValuePolicy, normalize_config_arg, validate_candidates


class SwarmServerController(ServerSideController):
    def __init__(
        self,
        num_rounds: int,
        start_round: int = 0,
        task_name_prefix=Constant.TN_PREFIX_SWARM,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        task_check_period: float = Constant.TASK_CHECK_INTERVAL,
        job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
        participating_clients=None,
        result_clients=None,
        starting_client: str = "",
        max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        private_p2p: bool = True,
        aggr_clients=None,
        train_clients=None,
    ):
        result_clients = normalize_config_arg(result_clients)
        starting_client = normalize_config_arg(starting_client)
        if starting_client is None:
            raise ValueError("starting_client must be specified")

        super().__init__(
            num_rounds=num_rounds,
            start_round=start_round,
            task_name_prefix=task_name_prefix,
            start_task_timeout=start_task_timeout,
            configure_task_timeout=configure_task_timeout,
            task_check_period=task_check_period,
            job_status_check_interval=job_status_check_interval,
            participating_clients=participating_clients,
            result_clients=result_clients,
            result_clients_policy=DefaultValuePolicy.ALL,
            starting_client=starting_client,
            starting_client_policy=DefaultValuePolicy.ANY,
            max_status_report_interval=max_status_report_interval,
            progress_timeout=progress_timeout,
            private_p2p=private_p2p,
        )
        if not train_clients:
            train_clients = []

        if not aggr_clients:
            aggr_clients = []

        self.aggr_clients = aggr_clients
        self.train_clients = train_clients

    def start_controller(self, fl_ctx: FLContext):
        super().start_controller(fl_ctx)

        self.train_clients = validate_candidates(
            var_name="train_clients",
            candidates=self.train_clients,
            base=self.participating_clients,
            default_policy=DefaultValuePolicy.ALL,
            allow_none=False,
        )

        self.aggr_clients = validate_candidates(
            var_name="aggr_clients",
            candidates=self.aggr_clients,
            base=self.participating_clients,
            default_policy=DefaultValuePolicy.ALL,
            allow_none=False,
        )

        # make sure every participating client is either training or aggr client
        for c in self.participating_clients:
            if c not in self.train_clients and c not in self.aggr_clients:
                raise RuntimeError(f"Config Error:  client {c} is neither train client nor aggr client")

        # set train_clients as a sticky prop in fl_ctx
        # in case CSE (cross site eval) workflow follows, it will know that only training clients have local models
        fl_ctx.set_prop(key=Constant.PROP_KEY_TRAIN_CLIENTS, value=self.train_clients, private=True, sticky=True)

    def prepare_config(self):
        return {Constant.AGGR_CLIENTS: self.aggr_clients, Constant.TRAIN_CLIENTS: self.train_clients}
