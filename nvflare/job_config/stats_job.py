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
from typing import List

from nvflare import FedJob, FilterType
from nvflare.app_common.abstract.statistics_spec import Statistics
from nvflare.app_common.executors.statistics.statistics_executor import StatisticsExecutor
from nvflare.app_common.filters.statistics_privacy_filter import StatisticsPrivacyFilter
from nvflare.app_common.statistics.histogram_bins_cleanser import HistogramBinsCleanser
from nvflare.app_common.statistics.json_stats_file_persistor import JsonStatsFileWriter
from nvflare.app_common.statistics.min_count_cleanser import MinCountCleanser
from nvflare.app_common.statistics.min_max_cleanser import AddNoiseToMinMax
from nvflare.app_common.workflows.statistics_controller import StatisticsController


class StatsJob(FedJob):
    def __init__(
        self,
        job_name: str,
        statistic_configs: dict,
        stats_generator: Statistics,
        output_path: str,
        min_count: int = 10,
        min_noise_level=0.1,
        max_noise_level=0.3,
        max_bins_percent=10,
    ):
        super().__init__()
        self.writer_id = "stats_writer"
        self.stats_generator_id_prefix = "stats_generator"
        self.job_name = job_name
        self.stats_generator = stats_generator
        self.statistic_configs = statistic_configs
        self.output_path = output_path
        self.min_count = min_count
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.max_bins_percent = max_bins_percent

        self.setup_server()

    def setup_server(self):
        # define stats controller
        ctr = self.get_stats_controller()
        self.to(ctr, "server")
        # define stat writer to output Json file
        stats_writer = self.get_stats_output_writer()
        self.to(stats_writer, "server", id=self.writer_id)

    def setup_client(self, sites: List[str]):
        # Client side job config
        # Add client site
        for site_id in sites:
            stats_generator_id = self.to(self.stats_generator, site_id, id=self.stats_generator_id_prefix)
            executor = StatisticsExecutor(generator_id=stats_generator_id)
            self.to(executor, site_id, tasks=["fed_stats_pre_run", "fed_stats"])
            self.add_privacy_result_filters(site_id)

    def get_stats_controller(self) -> StatisticsController:
        return StatisticsController(
            statistic_configs=self.statistic_configs, writer_id=self.writer_id, enable_pre_run_task=False
        )

    def get_stats_output_writer(self):
        json_encoder_path = "nvflare.app_common.utils.json_utils.ObjectEncoder"
        return JsonStatsFileWriter(output_path=self.output_path, json_encoder_path=json_encoder_path)

    def add_privacy_result_filters(self, site_id: str):
        # add privacy filters
        min_count_cleanser = MinCountCleanser(min_count=self.min_count)
        min_max_cleanser = AddNoiseToMinMax(min_noise_level=self.min_noise_level, max_noise_level=self.max_noise_level)
        hist_bins_cleanser = HistogramBinsCleanser(max_bins_percent=self.max_bins_percent)
        result_cleanser_ids = [
            self.to(min_count_cleanser, site_id, id="min_count_cleanser"),
            self.to(min_max_cleanser, site_id, id="min_max_noise_cleanser"),
            self.to(hist_bins_cleanser, site_id, id="hist_bins_cleanser"),
        ]

        result_filter = StatisticsPrivacyFilter(result_cleanser_ids=result_cleanser_ids)
        self.to(result_filter, site_id, filter_type=FilterType.TASK_RESULT, tasks=["fed_stats"])
