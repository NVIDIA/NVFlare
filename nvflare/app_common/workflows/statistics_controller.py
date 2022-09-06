# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import time
from typing import Callable, Dict, List, Optional

from nvflare.apis.client import Client
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.statistics_spec import Bin, Histogram, MetricConfig
from nvflare.app_common.abstract.statistics_writer import StatisticsWriter
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.numeric_stats import get_global_stats
from nvflare.app_common.statistics.statisitcs_objects_decomposer import fobs_registration
from nvflare.fuel.utils import fobs


class StatisticsController(Controller):
    def __init__(
        self,
        metric_configs: Dict[str, dict],
        writer_id: str,
        result_wait_timeout: int = 10,
        precision=4,
        min_clients: Optional[int] = None,
    ):
        """
        Args:
            metric_configs: defines the input statistic metrics to be computed and each metric's configuration.
                            The key is one of metric names sum, count, mean, stddev, histogram
                            the value is the arguments needed.
                            all other metrics except histogram require no argument.
                             "metric_configs": {
                                  "count": {},
                                  "mean": {},
                                  "sum": {},
                                  "stddev": {},
                                  "histogram": { "*": {"bins": 20 },
                                                 "Age": {"bins": 10, "range":[0,120]}
                                               }
                              },

                            Histogram requires the following
                            arguments:
                               1) numbers of bins or buckets of the histogram
                               2) the histogram range values [min, max]
                               These arguments are different for each feature. Here are few examples:
                                "histogram": { "*": {"bins": 20 },
                                              "Age": {"bins": 10, "range":[0,120]
                                              }
                                The configuration specify that
                                    feature 'Age' will have 10 bins for histogram and the range is within [0, 120)
                                    for all other features, the default ("*") configuration is used, with bins = 20.
                                    but the range of histogram is not specified, thus requires the Statistics controller
                                    to dynamically estimate histogram range for each feature. Then this estimated global
                                    range (est global min, est. global max) will be used as histogram range.

                                    to dynamically estimated such histogram range, we need client to provide the local
                                    min and max values in order to calculate the global bin and max value. But to protect
                                    data privacy and avoid the data leak, the noise level is added to the local min/max
                                    value before send to the controller. Therefore the controller only get the 'estimated'
                                    values, the global min/max are estimated, or more accurately, noised global min/max
                                    values.

                                    Here is another example:

                                    "histogram": { "density": {"bins": 10, "range":[0,120] }

                                    in this example, there is no default histogram configuration for other features.

                                    This will work correctly if there is only one feature called "density"
                                    but will fail if there other features in the dataset

                                In the following configuration
                                 "metric_configs": {
                                      "count": {},
                                      "mean": {},
                                      "stddev": {}
                                }
                                only count, mean and stddev metrics are specified, then the statistics_controller
                                will only set tasks to calculate these three metrics


            writer_id:    ID for StatisticsWriter. The StatisticWriter will save the result to output specified by the
                          StatisticsWriter

            result_wait_timeout: numbers of seconds to wait until we received all results.
                                 Notice this is after the min_clients have arrived, and we wait for result process
                                 callback, this becomes important if the data size to be processed is large

            precision:  number of percision digits
            min_clients: if specified, min number of clients we have to wait before process.


        """
        super().__init__()
        self.metric_configs: Dict[str, dict] = metric_configs
        self.writer_id = writer_id
        self.task_name = StC.FED_STATS_TASK
        self.client_metrics = {}
        self.global_metrics = {}
        self.client_features = {}
        self.result_wait_timeout = result_wait_timeout
        self.precision = precision
        self.min_clients = min_clients
        self.result_cb_status = {}

        self.result_callback_fns: Dict[str, Callable] = {
            StC.STATS_1st_METRICS: self.results_cb,
            StC.STATS_2nd_METRICS: self.results_cb,
        }
        fobs_registration()
        self.fl_ctx = None
        self.abort_job_in_error = {
            ReturnCode.EXECUTION_EXCEPTION: True,
            ReturnCode.TASK_UNKNOWN: True,
            ReturnCode.EXECUTION_RESULT_ERROR: False,
            ReturnCode.TASK_DATA_FILTER_ERROR: False,
            ReturnCode.TASK_RESULT_FILTER_ERROR: False,
        }

    def start_controller(self, fl_ctx: FLContext):
        if self.metric_configs is None or len(self.metric_configs) == 0:
            self.system_panic(
                "At least one metric_config must be configured for task StatisticsController", fl_ctx=fl_ctx
            )
        self.fl_ctx = fl_ctx
        clients = fl_ctx.get_engine().get_clients()
        if not self.min_clients:
            self.min_clients = len(clients)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):

        self.log_info(fl_ctx, f" {self.task_name} control flow started.")
        if abort_signal.triggered:
            return False

        self.metrics_task_flow(abort_signal, fl_ctx, StC.STATS_1st_METRICS)
        self.metrics_task_flow(abort_signal, fl_ctx, StC.STATS_2nd_METRICS)

        if not self._wait_for_all_results(
            self.result_wait_timeout, self.min_clients, self.client_metrics, abort_signal
        ):
            return False

        self.log_info(fl_ctx, "start post processing")
        self.post_fn(self.task_name, fl_ctx)

        self.log_info(fl_ctx, f"task {self.task_name} control flow end.")

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass

    def metrics_task_flow(self, abort_signal: Signal, fl_ctx: FLContext, metric_task: str):

        self.log_info(fl_ctx, f"start prepare inputs for task {metric_task}")
        inputs = self._prepare_inputs(metric_task, fl_ctx)
        results_cb_fn = self._get_result_cb(metric_task)

        self.log_info(fl_ctx, f"task: {self.task_name} metrics_flow for {metric_task} started.")

        if abort_signal.triggered:
            return False

        task_props = {StC.METRIC_TASK_KEY: metric_task}
        task = Task(name=self.task_name, data=inputs, result_received_cb=results_cb_fn, props=task_props)

        self.broadcast_and_wait(
            task=task,
            targets=None,
            min_responses=self.min_clients,
            fl_ctx=fl_ctx,
            wait_time_after_min_received=1,
            abort_signal=abort_signal,
        )

        self.global_metrics = get_global_stats(self.global_metrics, self.client_metrics, metric_task)

        self.log_info(fl_ctx, f"task {self.task_name} metrics_flow for {metric_task} flow end.")

    def results_cb(self, client_task: ClientTask, fl_ctx: FLContext):
        client_name = client_task.client.name
        task_name = client_task.task.name
        self.log_info(fl_ctx, f"Processing {task_name} result from client {client_name}")

        result = client_task.result
        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"Received result entries from client:{client_name}, " f"for task {task_name}")
            dxo = from_shareable(result)
            client_result = dxo.data

            metric_task = client_result[StC.METRIC_TASK_KEY]
            self.log_info(fl_ctx, f"handle client {client_name} results for metrics task: {metric_task}")
            metrics = fobs.loads(client_result[metric_task])

            for metric in metrics:
                if metric not in self.client_metrics:
                    self.client_metrics[metric] = {client_name: metrics[metric]}
                else:
                    self.client_metrics[metric].update({client_name: metrics[metric]})

            ds_features = client_result.get(StC.STATS_FEATURES, None)
            if ds_features:
                self.client_features.update({client_name: fobs.loads(ds_features)})

        elif rc in self.abort_job_in_error.keys():
            abort = self.abort_job_in_error[rc]
            if abort:
                self.system_panic(
                    f"Failed in client-site statistics_executor for {client_name} during task {task_name}."
                    f"statistics controller is exiting.",
                    fl_ctx=fl_ctx,
                )
                self.log_info(fl_ctx, f"Execution failed for {client_name}")
            else:
                self.log_info(fl_ctx, f"Execution result is not received for {client_name}")

            self.result_cb_status[client_name] = {client_task.task.props[StC.METRIC_TASK_KEY]: False}
        else:
            self.result_cb_status[client_name] = {client_task.task.props[StC.METRIC_TASK_KEY]: True}

        self.result_cb_status[client_name] = {client_task.task.props[StC.METRIC_TASK_KEY]: True}
        # Cleanup task result
        client_task.result = None

    def _validate_min_clients(self, min_clients: int, client_metrics: dict) -> bool:
        self.logger.info("check if min_client result received")

        resulting_clients = {}
        for metric in client_metrics:
            resulting_clients.update({metric: len(client_metrics[metric])})

        for metric in resulting_clients:
            if resulting_clients[metric] < min_clients:
                return False
        return True

    def post_fn(self, task_name: str, fl_ctx: FLContext):

        if not self._validate_min_clients(self.min_clients, self.client_metrics):
            self.system_panic(f"not all required {self.min_clients} received, abort the job.", fl_ctx)

        self.log_info(fl_ctx, "save statistics result to persistence store")
        ds_stats = self._combine_all_metrics()

        writer: StatisticsWriter = fl_ctx.get_engine().get_component(self.writer_id)
        writer.save(ds_stats, overwrite_existing=True, fl_ctx=fl_ctx)

    def _combine_all_metrics(self):
        result = {}
        filtered_client_metrics = [metric for metric in self.client_metrics if metric in self.metric_configs]
        filtered_global_metrics = [metric for metric in self.global_metrics if metric in self.metric_configs]

        for metric in filtered_client_metrics:
            for client in self.client_metrics[metric]:
                for ds in self.client_metrics[metric][client]:
                    client_dataset = f"{client}-{ds}"
                    for feature_name in self.client_metrics[metric][client][ds]:
                        if feature_name not in result:
                            result[feature_name] = {}
                        if metric not in result[feature_name]:
                            result[feature_name][metric] = {}

                        if metric == StC.STATS_HISTOGRAM:
                            hist: Histogram = self.client_metrics[metric][client][ds][feature_name]
                            result[feature_name][metric][client_dataset] = hist.bins
                        else:
                            result[feature_name][metric][client_dataset] = self.client_metrics[metric][client][ds][
                                feature_name
                            ]

        precision = self.precision
        for metric in filtered_global_metrics:
            for ds in self.global_metrics[metric]:
                global_dataset = f"{StC.GLOBAL}-{ds}"
                for feature_name in self.global_metrics[metric][ds]:
                    if metric == StC.STATS_HISTOGRAM:
                        hist: Histogram = self.global_metrics[metric][ds][feature_name]
                        buckets = []
                        for bucket in hist.bins:
                            buckets.append(
                                Bin(
                                    round(bucket.low_value, precision),
                                    round(bucket.high_value, precision),
                                    bucket.sample_count,
                                )
                            )
                        result[feature_name][metric][global_dataset] = hist.bins
                    else:
                        result[feature_name][metric].update(
                            {global_dataset: round(self.global_metrics[metric][ds][feature_name], precision)}
                        )

        return result

    def _get_target_metrics(self, ordered_metrics) -> List[MetricConfig]:
        return StatisticsController._get_target_metrics2(self.metric_configs, ordered_metrics)

    @staticmethod
    def _get_target_metrics2(metric_configs: dict, ordered_metrics: list) -> List[MetricConfig]:
        # make sure the execution order of the metrics calculation
        targets = []
        if metric_configs:
            for metric in metric_configs:
                # if target metric has histogram, we are not in 2nd Metric task
                # we only need to estimate the global min/max if we have histogram metric,
                # If the user provided the global min/max for a specified feature, then we do nothing
                # if the user did not provide the global min/max for the feature, then we need to ask
                # client to provide the local estimated min/max for that feature.
                # then we used the local estimate min/max to estimate global min/max.
                # to do that, we calculate the local min/max in 1st metric task.
                # in all cases, we will still send the STATS_MIN/MAX tasks, but client executor may or may not
                # delegate to stats generator to calculate the local min/max depends on if the global bin ranges
                # are specified. to do this, we send over the histogram configuration when calculate the local min/max
                if metric == StC.STATS_HISTOGRAM and metric not in ordered_metrics:
                    targets.append(MetricConfig(StC.STATS_MIN, metric_configs[StC.STATS_HISTOGRAM]))
                    targets.append(MetricConfig(StC.STATS_MAX, metric_configs[StC.STATS_HISTOGRAM]))

                if metric == StC.STATS_STDDEV and metric in ordered_metrics:
                    targets.append(MetricConfig(StC.STATS_VAR, {}))

                for rm in ordered_metrics:
                    if rm == metric:
                        targets.append(MetricConfig(metric, metric_configs[metric]))
        return targets

    def _prepare_inputs(self, metric_task: str, fl_ctx: FLContext) -> Shareable:
        inputs = Shareable()
        target_metrics: List[MetricConfig] = self._get_target_metrics(StC.ordered_metrics[metric_task])

        for tm in target_metrics:
            if tm.name == StC.STATS_HISTOGRAM:
                if StC.STATS_MIN in self.global_metrics:
                    inputs[StC.STATS_MIN] = self.global_metrics[StC.STATS_MIN]
                if StC.STATS_MAX in self.global_metrics:
                    inputs[StC.STATS_MAX] = self.global_metrics[StC.STATS_MAX]

            if tm.name == StC.STATS_VAR:
                if StC.STATS_COUNT in self.global_metrics:
                    inputs[StC.STATS_GLOBAL_COUNT] = self.global_metrics[StC.STATS_COUNT]
                if StC.STATS_MEAN in self.global_metrics:
                    inputs[StC.STATS_GLOBAL_MEAN] = self.global_metrics[StC.STATS_MEAN]

        inputs[StC.METRIC_TASK_KEY] = metric_task

        inputs[StC.STATS_TARGET_METRICS] = fobs.dumps(target_metrics)

        return inputs

    def _wait_for_all_results(
        self,
        result_wait_timeout: float,
        requested_client_size: int,
        client_metrics: dict,
        sleep_time: float = 1,
        abort_signal=None,
    ) -> bool:
        """
            for each metric, we check if the number of requested clients (min_clients or all clients)
            is available, if not, we wait until result_wait_timeout.
            result_wait_timeout is reset for next metric. result_wait_timeout is per metrics, not overall
            timeout for all results.
        Args:
            result_wait_timeout: timeout we have to wait for each metric. reset for each metric
            requested_client_size: requested client size, usually min_clients or all clients
            client_metrics: client specific metrics received so far
            abort_signal:  abort signal

        Returns:

        """

        # record of each metric, number of clients processed
        metric_client_received = {}

        # current metrics obtained so far (across all clients)
        metric_names = client_metrics.keys()
        for m in metric_names:
            metric_client_received[m] = len(client_metrics[m].keys())

        timeout = result_wait_timeout
        for m in metric_client_received:
            if requested_client_size > metric_client_received[m]:
                t = 0
                while t < timeout and requested_client_size > metric_client_received[m]:
                    if abort_signal and abort_signal.triggered:
                        return False

                    msg = f"not all client received the metric {m}, need to wait for {sleep_time} seconds."
                    self.logger.info(msg)
                    time.sleep(sleep_time)
                    t += sleep_time
                    # check and update number of client processed for metric again
                    metric_client_received[m] = len(client_metrics[m].keys())

        return True

    def _get_result_cb(self, metric_task: str):
        return self.result_callback_fns[metric_task]
