# Copyright (c) 2022, NVIDIA CORPORATION.
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
from typing import List, Callable, Dict

from nvflare.apis.client import Client
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller, Task, ClientTask
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.metic_config import MetricConfig
from nvflare.app_common.statistics.numeric_stats import get_global_stats, get_global_dataset_stats, \
    get_client_dataset_stats, get_global_feature_data_types
from nvflare.app_common.statistics.stats_def import (
    BinRange,
    DatasetStatistics,
)
from nvflare.app_common.statistics.stats_file_persistor import StatsFileWriter, FileFormat


class GlobalStatistics(Controller):

    def __init__(self, metric_configs: Dict[str, dict], min_count, writer_id: str):
        super().__init__()
        self.metric_configs: Dict[str, dict] = metric_configs
        self.min_count = min_count
        self.writer_id = writer_id
        self.task_name = StC.FED_STATS_TASK
        self.client_metrics = {}
        self.global_metrics = {}
        self.client_features = {}

        self.result_callback_fns: Dict[str, Callable] = {
            StC.STATS_1st_METRICS: self.results_cb,
            StC.STATS_2nd_METRICS: self.results_cb
        }

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self.log_info(fl_ctx, f" {self.task_name} control flow started.")
        if abort_signal.triggered:
            return False

        clients = fl_ctx.get_engine().get_clients()
        self.metrics_task_flow(abort_signal, fl_ctx, clients, StC.STATS_1st_METRICS)
        self.metrics_task_flow(abort_signal, fl_ctx, clients, StC.STATS_2nd_METRICS)

        self.post_fn(self.task_name, fl_ctx)

        self.log_info(fl_ctx, f"task {self.task_name} control flow end.")

    def start_controller(self, fl_ctx: FLContext):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def process_result_of_unknown_task(self,
                                       client: Client,
                                       task_name: str,
                                       client_task_id: str,
                                       result: Shareable,
                                       fl_ctx: FLContext):
        pass

    def metrics_task_flow(self,
                          abort_signal: Signal,
                          fl_ctx: FLContext,
                          clients: List[Client],
                          metric_task: str):

        inputs = self._prepare_inputs(metric_task)
        results_cb_fn = self._get_result_cb(metric_task)

        self.log_info(fl_ctx, f"task: {self.task_name} metrics_flow for {metric_task} started.")

        if abort_signal.triggered:
            return False

        task = Task(name=self.task_name, data=inputs, result_received_cb=results_cb_fn)

        self.broadcast_and_wait(
            task=task,
            targets=None,
            min_responses=len(clients),
            fl_ctx=fl_ctx,
            wait_time_after_min_received=1,
            abort_signal=abort_signal,
        )

        self.log_info(fl_ctx, f"task {self.task_name} metrics_flow for {metric_task} flow end.")

    def results_cb(self, client_task: ClientTask, fl_ctx: FLContext):
        client_name = client_task.client.name
        task_name = client_task.task.name
        self.log_info(fl_ctx, f"Processing {task_name} result from client {client_name}")

        result = client_task.result
        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            metric_task = result.get_header(StC.METRIC_TASK_KEY)
            self.log_info(fl_ctx,
                          f"Received result entries from client:{client_name}, "
                          f"for task {task_name}, "
                          f"metrics task: {metric_task}")

            dxo = from_shareable(result)
            metrics = dxo.data[metric_task]
            for metric in metrics:
                self.client_metrics[metric] = {client_name: metrics[metric]}

            ds_features = result.get_header(StC.STATS_FEATURES)
            if ds_features:
                self.client_features = {client_name: ds_features}

        else:
            self.log_info(
                fl_ctx, f"Ignore the client {client_name} result. {task_name} tasked returned error code: {rc}"
            )

        # Cleanup task result
        client_task.result = None
        pass

    def post_fn(self, task_name: str, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"post processing for task {task_name}")
        self.global_metrics = get_global_stats(self.client_metrics)

        ds_stats: List[DatasetStatistics] = []
        feature_data_types = get_global_feature_data_types(self.client_features)
        global_dataset_prefix = StC.GLOBAL_DATASET_NAME_PREFIX

        global_ds_stats = get_global_dataset_stats(global_dataset_prefix, self.global_metrics, feature_data_types)
        ds_stats = global_ds_stats

        client_ds_stats = get_client_dataset_stats(self.client_metrics, self.client_features)
        for stat in client_ds_stats:
            ds_stats.append(stat)

        self.log_info(fl_ctx, f"save statistics result to persistence store")

        writer: StatsFileWriter = fl_ctx.get_engine().get_component(self.writer_id)
        writer.save("stats.json", ds_stats, file_format=FileFormat.JSON, overwrite_existing=True, fl_ctx=fl_ctx)

    def _get_target_metrics(self, ordered_metrics) -> List[MetricConfig]:
        # make sure the execution order of the metrics calculation
        targets = []
        if self.metric_configs:
            for metric in self.metric_configs:
                # if target metric has histogram, we are not in 2nd Metric task
                # we only need to calculate min/max if we have histogram metric,
                # and we only calculate in 1st metric task
                if metric == StC.STATS_HISTOGRAM and metric not in ordered_metrics:
                    targets.append(MetricConfig(StC.STATS_MIN, {}))
                    targets.append(MetricConfig(StC.STATS_MAX, {}))

                for rm in ordered_metrics:
                    if rm == metric:
                        targets.append(MetricConfig(metric, self.metric_configs[metric]))
        return targets

    def _prepare_inputs(self, metric_task: str) -> Shareable:
        inputs = Shareable()
        target_metrics: List[MetricConfig] = self._get_target_metrics(StC.ordered_metrics[metric_task])

        for tm in target_metrics:
            if tm.name == StC.STATS_HISTOGRAM:
                inputs[StC.STATS_MIN] = self.global_metrics[StC.STATS_MIN]
                inputs[StC.STATS_MAX] = self.global_metrics[StC.STATS_MAX]
                inputs[StC.STATS_BINS] = tm.config[StC.STATS_BINS]

            if tm.name == StC.STATS_VAR:
                inputs[StC.STATS_GLOBAL_COUNT] = self.global_metrics[StC.STATS_GLOBAL_COUNT]
                inputs[StC.STATS_GLOBAL_MEAN] = self.global_metrics[StC.STATS_GLOBAL_MEAN]

        inputs[StC.METRIC_TASK_KEY] = metric_task
        inputs[StC.STATS_TARGET_METRICS] = [mc.name for mc in target_metrics]

        return inputs

    def _get_result_cb(self, metric_task: str):
        return self.result_callback_fns[metric_task]
