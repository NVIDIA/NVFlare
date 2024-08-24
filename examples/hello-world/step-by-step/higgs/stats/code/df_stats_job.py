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
import argparse

from nvflare import FedJob, FilterType
from nvflare.app_common.executors.statistics.statistics_executor import StatisticsExecutor
from nvflare.app_common.filters.statistics_privacy_filter import StatisticsPrivacyFilter
from nvflare.app_common.statistics.histogram_bins_cleanser import HistogramBinsCleanser
from nvflare.app_common.statistics.json_stats_file_persistor import JsonStatsFileWriter
from nvflare.app_common.statistics.min_count_cleanser import MinCountCleanser
from nvflare.app_common.statistics.min_max_cleanser import AddNoiseToMinMax
from nvflare.app_common.workflows.statistics_controller import StatisticsController


def get_stats_controller(writer_id):
    statistic_configs = {
        "count": {},
        "mean": {},
        "sum": {},
        "stddev": {},
        "histogram": {"*": {"bins": 20}},
        "Age": {"bins": 20, "range": [0, 10]},
    }
    return StatisticsController(statistic_configs=statistic_configs, writer_id=writer_id, enable_pre_run_task=False)


def get_stats_output_writer(out_path):
    json_encoder_path = "nvflare.app_common.utils.json_utils.ObjectEncoder"
    return JsonStatsFileWriter(output_path=out_path, json_encoder_path=json_encoder_path)


def get_local_stats_generator(data_root_dir: str):
    from df_stats import DFStatistics

    return DFStatistics(data_root_dir=data_root_dir)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_clients", type=int, default=3)
    parser.add_argument("-d", "--data_root_dir", type=str, nargs="?", default="/tmp/nvflare/dataset/output")
    parser.add_argument("-o", "--stats_output_path", type=str, nargs="?", default="statistics/stats.json")
    parser.add_argument("-j", "--job_dir", type=str, nargs="?", default="/tmp/nvflare/jobs/stats_df")
    parser.add_argument("-w", "--work_dir", type=str, nargs="?", default="/tmp/nvflare/jobs/stats_df/work_dir")
    parser.add_argument("-co", "--export_config", action="store_true", help="config only mode, export config")

    return parser.parse_args()


def add_privacy_result_filters(job):
    # add privacy filters
    result_cleanser_ids = ["min_count_cleanser", "min_max_noise_cleanser", "hist_bins_cleanser"]
    result_filter = StatisticsPrivacyFilter(result_cleanser_ids=result_cleanser_ids)

    min_count_cleanser = MinCountCleanser(min_count=10)
    min_max_cleanser = AddNoiseToMinMax(min_noise_level=0.1, max_noise_level=0.3)
    hist_bins_cleanser = HistogramBinsCleanser(max_bins_percent=10)
    job.to(min_count_cleanser, site_id, id="min_count_cleanser")
    job.to(min_max_cleanser, site_id, id="min_max_noise_cleanser")
    job.to(hist_bins_cleanser, site_id, id="hist_bins_cleanser")
    job.to(result_filter, site_id, filter_type=FilterType.TASK_RESULT, tasks=["fed_stats"])


if __name__ == "__main__":

    args = define_parser()

    n_clients = args.n_clients
    data_root_dir = args.data_root_dir
    output_path = args.stats_output_path
    job_dir = args.job_dir
    work_dir = args.work_dir
    export_config = args.export_config

    job = FedJob(name="stats_df")

    # Server side Job Config

    # define stats controller
    ctr = get_stats_controller(writer_id="stats_writer")

    # define stat writer to output Json file
    stats_writer = get_stats_output_writer(out_path=output_path)

    job.to(ctr, "server")
    job.to(stats_writer, "server", id="stats_writer")

    # Client side job config

    # define local stats generator
    df_stats_generator = get_local_stats_generator(data_root_dir=data_root_dir)

    # Add client site
    for i in range(n_clients):
        site_id = f"site-{i + 1}"
        job.to(df_stats_generator, site_id, id="df_stats_generator")

        executor = StatisticsExecutor(generator_id="df_stats_generator")
        job.to(executor, site_id, tasks=["fed_stats_pre_run", "fed_stats"])
        add_privacy_result_filters(job)

    if export_config:
        job.export_job(job_dir)
    else:
        job.simulator_run(work_dir, gpu="0")
