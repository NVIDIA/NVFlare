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

from nvflare.app_common.ccwf.common import Constant, CyclicOrder
from nvflare.app_common.ccwf.server_ctl import ServerSideController
from nvflare.fuel.utils.validation_utils import DefaultValuePolicy, check_str, normalize_config_arg


class CyclicServerController(ServerSideController):
    def __init__(
        self,
        num_rounds: int,
        task_name_prefix=Constant.TN_PREFIX_CYCLIC,
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
        cyclic_order: str = CyclicOrder.FIXED,
    ):
        result_clients = normalize_config_arg(result_clients)
        starting_client = normalize_config_arg(starting_client)
        if starting_client is None:
            raise ValueError("starting_client must be specified")

        super().__init__(
            num_rounds=num_rounds,
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
        check_str("cyclic_order", cyclic_order)
        if cyclic_order not in [CyclicOrder.FIXED, CyclicOrder.RANDOM]:
            raise ValueError(
                f"invalid cyclic_order {cyclic_order}: must be in {[CyclicOrder.FIXED, CyclicOrder.RANDOM]}"
            )
        self.cyclic_order = cyclic_order

    def prepare_config(self):
        return {Constant.ORDER: self.cyclic_order}
