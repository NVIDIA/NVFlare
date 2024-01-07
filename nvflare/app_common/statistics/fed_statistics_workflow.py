# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import time
from typing import Dict, Optional

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.fed_stats_utils import (
    combine_all_statistics,
    get_client_stats,
    prepare_inputs,
    save_to_json,
)
from nvflare.app_common.statistics.numeric_stats import get_global_stats
from nvflare.app_common.statistics.statisitcs_objects_decomposer import fobs_registration
from nvflare.app_common.workflows.wf_comm.wf_comm_api_spec import (
    CMD,
    CMD_BROADCAST,
    CURRENT_ROUND,
    DATA,
    MIN_RESPONSES,
    NUM_ROUNDS,
    RESP_MAX_WAIT_TIME,
    START_ROUND,
)
from nvflare.app_common.workflows.wf_comm.wf_spec import WF


class FedStatistics(WF):
    def __init__(
        self,
        statistic_configs: Dict[str, dict],
        output_path: str,
        wait_time_after_1st_resp_received: float = 10,
        min_clients: Optional[int] = None,
        precision=4,
        streaming_enabled: bool = False,
        streaming_interval: float = 1,
    ):
        super().__init__()
        self.wait_time_after_1st_resp_received = wait_time_after_1st_resp_received
        self.min_clients = 0 if min_clients is None else min_clients
        self.precision = precision
        self.output_path = output_path
        self.streaming_enabled = streaming_enabled
        self.streaming_interval = streaming_interval

        self.statistic_configs: Dict[str, dict] = statistic_configs
        self.task_name = StC.FED_STATS_TASK
        self.round_tasks = [StC.STATS_1st_STATISTICS, StC.STATS_2nd_STATISTICS]

        self.logger = logging.getLogger(self.__class__.__name__)

        fobs_registration()

    def run(self):
        count = 1
        while True if self.streaming_enabled else (count <= 1):
            self.logger.info("start federated statistics run \n")
            global_statistics = {}
            client_statistics = {}
            client_features = {}
            for current_round, statistic_task in enumerate(self.round_tasks):

                self.logger.info(f"{current_round=}, {statistic_task} \n")

                global_statistics, client_statistics = self.statistics_task_flow(
                    current_round, global_statistics, client_statistics, client_features, statistic_task
                )

            self.logger.info("combine all clients' statistics")

            ds_stats = combine_all_statistics(
                self.statistic_configs, global_statistics, client_statistics, self.precision
            )

            save_to_json(ds_stats, self.output_path)
            self.logger.info(f"save statistics result to '{self.output_path}'\n ")

            count += 1
            if self.streaming_enabled:
                time.sleep(self.streaming_interval)

    def statistics_task_flow(
        self,
        current_round: int,
        global_statistics: dict,
        client_statistics: dict,
        client_features: dict,
        statistic_task: str,
    ):

        self.logger.info(f"start prepare inputs for task {statistic_task}")
        inputs = prepare_inputs(self.statistic_configs, statistic_task, global_statistics)

        self.logger.info(f"task: {self.task_name} statistics_flow for {statistic_task} started.")

        stats_config = FLModel(params_type=ParamsType.FULL, params=inputs)
        payload = {
            CMD: CMD_BROADCAST,
            DATA: stats_config,
            RESP_MAX_WAIT_TIME: self.wait_time_after_1st_resp_received,
            MIN_RESPONSES: self.min_clients,
            CURRENT_ROUND: current_round,
            NUM_ROUNDS: 2,
            START_ROUND: 0,
        }

        results: Dict[str, Dict[str, FLModel]] = self.flare_comm.broadcast_and_wait(payload)

        client_statistics, client_features = get_client_stats(results, client_statistics, client_features)

        global_statistics = get_global_stats(global_statistics, client_statistics, statistic_task)

        self.logger.info(f"statistics_flow task {statistic_task} flow completed.")

        return global_statistics, client_statistics
