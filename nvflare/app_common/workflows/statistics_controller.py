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

import time
from typing import Callable, Dict, List, Optional

from nvflare.apis.client import Client
from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.statistics_spec import Bin, Histogram, StatisticConfig
from nvflare.app_common.abstract.statistics_writer import StatisticsWriter
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.numeric_stats import get_global_stats
from nvflare.app_common.statistics.statisitcs_objects_decomposer import fobs_registration
from nvflare.fuel.utils import fobs


class StatisticsController(Controller):
    def __init__(
        self,
        statistic_configs: Dict[str, dict],
        writer_id: str,
        wait_time_after_min_received: int = 1,
        result_wait_timeout: int = 10,
        precision=4,
        min_clients: Optional[int] = None,
        enable_pre_run_task: bool = True,
    ):
        """Controller for Statistics.

        Args:
            statistic_configs: defines the input statistic to be computed and each statistic's configuration, see below for details.
            writer_id: ID for StatisticsWriter. The StatisticWriter will save the result to output specified by the
               StatisticsWriter
            wait_time_after_min_received: numbers of seconds to wait after minimum numer of clients specified has received.
            result_wait_timeout: numbers of seconds to wait until we received all results.
               Notice this is after the min_clients have arrived, and we wait for result process
               callback, this becomes important if the data size to be processed is large
            precision:  number of precision digits
            min_clients: if specified, min number of clients we have to wait before process.

        For statistic_configs, the key is one of statistics' names sum, count, mean, stddev, histogram, and
        the value is the arguments needed. All other statistics except histogram require no argument.

        .. code-block:: text

            "statistic_configs": {
                "count": {},
                "mean": {},
                "sum": {},
                "stddev": {},
                "histogram": {
                    "*": {"bins": 20},
                    "Age": {"bins": 10, "range": [0, 120]}
                }
            },

        Histogram requires the following arguments:
            1) numbers of bins or buckets of the histogram
            2) the histogram range values [min, max]

        These arguments are different for each feature. Here are few examples:

        .. code-block:: text

            "histogram": {
                            "*": {"bins": 20 },
                            "Age": {"bins": 10, "range":[0,120]}
                         }

        The configuration specifies that the
        feature 'Age' will have 10 bins for and the range is within [0, 120).
        For all other features, the default ("*") configuration is used, with bins = 20.
        The range of histogram is not specified, thus requires the Statistics controller
        to dynamically estimate histogram range for each feature. Then this estimated global
        range (est global min, est. global max) will be used as the histogram range.

        To dynamically estimate such a histogram range, we need the client to provide the local
        min and max values in order to calculate the global bin and max value. In order to protect
        data privacy and avoid data leakage, a noise level is added to the local min/max
        value before sending to the controller. Therefore the controller only gets the 'estimated'
        values, and the global min/max are estimated, or more accurately, they are noised global min/max
        values.

        Here is another example:

        .. code-block:: text

            "histogram": {
                            "density": {"bins": 10, "range":[0,120]}
                         }

        In this example, there is no default histogram configuration for other features.

        This will work correctly if there is only one feature called "density"
        but will fail if there are other features in the dataset.

        In the following configuration:

        .. code-block:: text

            "statistic_configs": {
                "count": {},
                "mean": {},
                "stddev": {}
            }

        Only count, mean and stddev statistics are specified, so the statistics_controller
        will only set tasks to calculate these three statistics.

        """
        super().__init__()
        self.statistic_configs: Dict[str, dict] = statistic_configs
        self.writer_id = writer_id
        self.task_name = StC.FED_STATS_TASK
        self.client_statistics = {}
        self.global_statistics = {}
        self.client_features = {}
        self.result_wait_timeout = result_wait_timeout
        self.wait_time_after_min_received = wait_time_after_min_received
        self.precision = precision
        self.min_clients = min_clients
        self.result_cb_status = {}
        self.client_handshake_ok = {}

        self.enable_pre_run_task = enable_pre_run_task

        self.result_callback_fns: Dict[str, Callable] = {
            StC.STATS_1st_STATISTICS: self.results_cb,
            StC.STATS_2nd_STATISTICS: self.results_cb,
        }
        fobs_registration()
        self.fl_ctx = None
        self.abort_job_in_error = {
            ReturnCode.EXECUTION_EXCEPTION: True,
            ReturnCode.TASK_UNKNOWN: True,
            ReturnCode.EXECUTION_RESULT_ERROR: False,
            ReturnCode.TASK_DATA_FILTER_ERROR: True,
            ReturnCode.TASK_RESULT_FILTER_ERROR: True,
        }

    def start_controller(self, fl_ctx: FLContext):
        if self.statistic_configs is None or len(self.statistic_configs) == 0:
            self.system_panic(
                "At least one statistic_config must be configured for task StatisticsController", fl_ctx=fl_ctx
            )
        self.fl_ctx = fl_ctx
        clients = fl_ctx.get_engine().get_clients()
        if not self.min_clients:
            self.min_clients = len(clients)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):

        self.log_info(fl_ctx, f"{self.task_name} control flow started.")

        if abort_signal.triggered:
            return False

        if self.enable_pre_run_task:
            self.pre_run_task_flow(abort_signal, fl_ctx)

        self.statistics_task_flow(abort_signal, fl_ctx, StC.STATS_1st_STATISTICS)
        self.statistics_task_flow(abort_signal, fl_ctx, StC.STATS_2nd_STATISTICS)

        if not StatisticsController._wait_for_all_results(
            self.logger, self.result_wait_timeout, self.min_clients, self.client_statistics, 1.0, abort_signal
        ):
            self.log_info(fl_ctx, f"task {self.task_name} timeout on wait for all results.")
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

    def _get_all_statistic_configs(self) -> List[StatisticConfig]:

        all_statistics = {
            StC.STATS_COUNT: StatisticConfig(StC.STATS_COUNT, {}),
            StC.STATS_FAILURE_COUNT: StatisticConfig(StC.STATS_FAILURE_COUNT, {}),
            StC.STATS_SUM: StatisticConfig(StC.STATS_SUM, {}),
            StC.STATS_MEAN: StatisticConfig(StC.STATS_MEAN, {}),
            StC.STATS_VAR: StatisticConfig(StC.STATS_VAR, {}),
            StC.STATS_STDDEV: StatisticConfig(StC.STATS_STDDEV, {}),
        }

        if StC.STATS_HISTOGRAM in self.statistic_configs:
            hist_config = self.statistic_configs[StC.STATS_HISTOGRAM]
            all_statistics[StC.STATS_MIN] = StatisticConfig(StC.STATS_MIN, hist_config)
            all_statistics[StC.STATS_MAX] = StatisticConfig(StC.STATS_MAX, hist_config)
            all_statistics[StC.STATS_HISTOGRAM] = StatisticConfig(StC.STATS_HISTOGRAM, hist_config)

        return [all_statistics[k] for k in all_statistics if k in self.statistic_configs]

    def pre_run_task_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        client_name = fl_ctx.get_identity_name()

        self.log_info(fl_ctx, f"start pre_run task for client {client_name}")
        inputs = Shareable()
        target_statistics: List[StatisticConfig] = self._get_all_statistic_configs()
        inputs[StC.STATS_TARGET_STATISTICS] = fobs.dumps(target_statistics)
        results_cb_fn = self.results_pre_run_cb

        if abort_signal.triggered:
            return False

        task = Task(name=StC.FED_STATS_PRE_RUN, data=inputs, result_received_cb=results_cb_fn)

        self.broadcast_and_wait(
            task=task,
            targets=None,
            min_responses=self.min_clients,
            fl_ctx=fl_ctx,
            wait_time_after_min_received=self.wait_time_after_min_received,
            abort_signal=abort_signal,
        )
        self.log_info(fl_ctx, f" client {client_name} pre_run task flow end.")

    def statistics_task_flow(self, abort_signal: Signal, fl_ctx: FLContext, statistic_task: str):

        self.log_info(fl_ctx, f"start prepare inputs for task {statistic_task}")
        inputs = self._prepare_inputs(statistic_task)
        results_cb_fn = self._get_result_cb(statistic_task)

        self.log_info(fl_ctx, f"task: {self.task_name} statistics_flow for {statistic_task} started.")

        if abort_signal.triggered:
            return False

        task_props = {StC.STATISTICS_TASK_KEY: statistic_task}
        task = Task(name=self.task_name, data=inputs, result_received_cb=results_cb_fn, props=task_props)

        self.broadcast_and_wait(
            task=task,
            targets=None,
            min_responses=self.min_clients,
            fl_ctx=fl_ctx,
            wait_time_after_min_received=self.wait_time_after_min_received,
            abort_signal=abort_signal,
        )

        self.global_statistics = get_global_stats(self.global_statistics, self.client_statistics, statistic_task)

        self.log_info(fl_ctx, f"task {self.task_name} statistics_flow for {statistic_task} flow end.")

    def handle_client_errors(self, rc: str, client_task: ClientTask, fl_ctx: FLContext):
        client_name = client_task.client.name
        task_name = client_task.task.name
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

    def results_pre_run_cb(self, client_task: ClientTask, fl_ctx: FLContext):
        client_name = client_task.client.name
        task_name = client_task.task.name
        self.log_info(fl_ctx, f"Processing {task_name} pre_run from client {client_name}")
        result = client_task.result

        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"Received pre-run handshake result from client:{client_name} for task {task_name}")
            self.client_handshake_ok = {client_name: True}
            fl_ctx.set_prop(StC.PRE_RUN_RESULT, {client_name: from_shareable(result)})
            self.fire_event(EventType.PRE_RUN_RESULT_AVAILABLE, fl_ctx)
        else:
            if rc in self.abort_job_in_error.keys():
                self.handle_client_errors(rc, client_task, fl_ctx)
            self.client_handshake_ok = {client_name: False}

        # Cleanup task result
        client_task.result = None

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

            statistics_task = client_result[StC.STATISTICS_TASK_KEY]
            self.log_info(fl_ctx, f"handle client {client_name} results for statistics task: {statistics_task}")
            statistics = fobs.loads(client_result[statistics_task])

            for statistic in statistics:
                if statistic not in self.client_statistics:
                    self.client_statistics[statistic] = {client_name: statistics[statistic]}
                else:
                    self.client_statistics[statistic].update({client_name: statistics[statistic]})

            ds_features = client_result.get(StC.STATS_FEATURES, None)
            if ds_features:
                self.client_features.update({client_name: fobs.loads(ds_features)})

        elif rc in self.abort_job_in_error.keys():
            self.handle_client_errors(rc, client_task, fl_ctx)

            self.result_cb_status[client_name] = {client_task.task.props[StC.STATISTICS_TASK_KEY]: False}
        else:
            self.result_cb_status[client_name] = {client_task.task.props[StC.STATISTICS_TASK_KEY]: True}

        self.result_cb_status[client_name] = {client_task.task.props[StC.STATISTICS_TASK_KEY]: True}
        # Cleanup task result
        client_task.result = None

    def _validate_min_clients(self, min_clients: int, client_statistics: dict) -> bool:
        self.logger.info("check if min_client result received for all features")

        resulting_clients = {}
        for statistic in client_statistics:
            clients = client_statistics[statistic].keys()
            if len(clients) < min_clients:
                return False
            for client in clients:
                ds_feature_statistics = client_statistics[statistic][client]
                for ds_name in ds_feature_statistics:
                    if ds_name not in resulting_clients:
                        resulting_clients[ds_name] = set()

                    if ds_feature_statistics[ds_name]:
                        resulting_clients[ds_name].update([client])

        for ds in resulting_clients:
            if len(resulting_clients[ds]) < min_clients:
                return False
        return True

    def post_fn(self, task_name: str, fl_ctx: FLContext):

        ok_to_proceed = self._validate_min_clients(self.min_clients, self.client_statistics)
        if not ok_to_proceed:
            self.system_panic(f"not all required {self.min_clients} received, abort the job.", fl_ctx)
        else:
            self.log_info(fl_ctx, "combine all clients' statistics")
            ds_stats = self._combine_all_statistics()
            self.log_info(fl_ctx, "save statistics result to persistence store")
            writer: StatisticsWriter = fl_ctx.get_engine().get_component(self.writer_id)
            writer.save(ds_stats, overwrite_existing=True, fl_ctx=fl_ctx)

    def _combine_all_statistics(self):
        result = {}
        filtered_client_statistics = [
            statistic for statistic in self.client_statistics if statistic in self.statistic_configs
        ]
        filtered_global_statistics = [
            statistic for statistic in self.global_statistics if statistic in self.statistic_configs
        ]

        for statistic in filtered_client_statistics:
            for client in self.client_statistics[statistic]:
                for ds in self.client_statistics[statistic][client]:
                    client_dataset = f"{client}-{ds}"
                    for feature_name in self.client_statistics[statistic][client][ds]:
                        if feature_name not in result:
                            result[feature_name] = {}
                        if statistic not in result[feature_name]:
                            result[feature_name][statistic] = {}

                        if statistic == StC.STATS_HISTOGRAM:
                            hist: Histogram = self.client_statistics[statistic][client][ds][feature_name]
                            buckets = StatisticsController._apply_histogram_precision(hist.bins, self.precision)
                            result[feature_name][statistic][client_dataset] = buckets
                        else:
                            result[feature_name][statistic][client_dataset] = round(
                                self.client_statistics[statistic][client][ds][feature_name], self.precision
                            )

        precision = self.precision
        for statistic in filtered_global_statistics:
            for ds in self.global_statistics[statistic]:
                global_dataset = f"{StC.GLOBAL}-{ds}"
                for feature_name in self.global_statistics[statistic][ds]:
                    if statistic == StC.STATS_HISTOGRAM:
                        hist: Histogram = self.global_statistics[statistic][ds][feature_name]
                        buckets = StatisticsController._apply_histogram_precision(hist.bins, self.precision)
                        result[feature_name][statistic][global_dataset] = buckets
                    else:
                        result[feature_name][statistic].update(
                            {global_dataset: round(self.global_statistics[statistic][ds][feature_name], precision)}
                        )

        return result

    @staticmethod
    def _apply_histogram_precision(bins: List[Bin], precision) -> List[Bin]:
        buckets = []
        for bucket in bins:
            buckets.append(
                Bin(
                    round(bucket.low_value, precision),
                    round(bucket.high_value, precision),
                    bucket.sample_count,
                )
            )
        return buckets

    @staticmethod
    def _get_target_statistics(statistic_configs: dict, ordered_statistics: list) -> List[StatisticConfig]:
        # make sure the execution order of the statistics calculation
        targets = []
        if statistic_configs:
            for statistic in statistic_configs:
                # if target statistic has histogram, we are not in 2nd statistic task
                # we only need to estimate the global min/max if we have histogram statistic,
                # If the user provided the global min/max for a specified feature, then we do nothing
                # if the user did not provide the global min/max for the feature, then we need to ask
                # client to provide the local estimated min/max for that feature.
                # then we used the local estimate min/max to estimate global min/max.
                # to do that, we calculate the local min/max in 1st statistic task.
                # in all cases, we will still send the STATS_MIN/MAX tasks, but client executor may or may not
                # delegate to stats generator to calculate the local min/max depends on if the global bin ranges
                # are specified. to do this, we send over the histogram configuration when calculate the local min/max
                if statistic == StC.STATS_HISTOGRAM and statistic not in ordered_statistics:
                    targets.append(StatisticConfig(StC.STATS_MIN, statistic_configs[StC.STATS_HISTOGRAM]))
                    targets.append(StatisticConfig(StC.STATS_MAX, statistic_configs[StC.STATS_HISTOGRAM]))

                if statistic == StC.STATS_STDDEV and statistic in ordered_statistics:
                    targets.append(StatisticConfig(StC.STATS_VAR, {}))

                for rm in ordered_statistics:
                    if rm == statistic:
                        targets.append(StatisticConfig(statistic, statistic_configs[statistic]))
        return targets

    def _prepare_inputs(self, statistic_task: str) -> Shareable:
        inputs = Shareable()
        target_statistics: List[StatisticConfig] = StatisticsController._get_target_statistics(
            self.statistic_configs, StC.ordered_statistics[statistic_task]
        )

        for tm in target_statistics:
            if tm.name == StC.STATS_HISTOGRAM:
                if StC.STATS_MIN in self.global_statistics:
                    inputs[StC.STATS_MIN] = self.global_statistics[StC.STATS_MIN]
                if StC.STATS_MAX in self.global_statistics:
                    inputs[StC.STATS_MAX] = self.global_statistics[StC.STATS_MAX]

            if tm.name == StC.STATS_VAR:
                if StC.STATS_COUNT in self.global_statistics:
                    inputs[StC.STATS_GLOBAL_COUNT] = self.global_statistics[StC.STATS_COUNT]
                if StC.STATS_MEAN in self.global_statistics:
                    inputs[StC.STATS_GLOBAL_MEAN] = self.global_statistics[StC.STATS_MEAN]

        inputs[StC.STATISTICS_TASK_KEY] = statistic_task

        inputs[StC.STATS_TARGET_STATISTICS] = fobs.dumps(target_statistics)

        return inputs

    @staticmethod
    def _wait_for_all_results(
        logger,
        result_wait_timeout: float,
        requested_client_size: int,
        client_statistics: dict,
        sleep_time: float = 1,
        abort_signal=None,
    ) -> bool:
        """Waits for all results.

        For each statistic, we check if the number of requested clients (min_clients or all clients)
        is available, if not, we wait until result_wait_timeout.
        result_wait_timeout is reset for next statistic. result_wait_timeout is per statistic, not overall
        timeout for all results.

        Args:
            result_wait_timeout: timeout we have to wait for each statistic. reset for each statistic
            requested_client_size: requested client size, usually min_clients or all clients
            client_statistics: client specific statistics received so far
            abort_signal:  abort signal

        Returns: False, when job is aborted else True

        """

        # record of each statistic, number of clients processed
        statistics_client_received = {}

        # current statistics obtained so far (across all clients)
        statistic_names = client_statistics.keys()
        for m in statistic_names:
            statistics_client_received[m] = len(client_statistics[m].keys())

        timeout = result_wait_timeout
        for m in statistics_client_received:
            if requested_client_size > statistics_client_received[m]:
                t = 0
                while t < timeout and requested_client_size > statistics_client_received[m]:
                    if abort_signal and abort_signal.triggered:
                        return False

                    msg = (
                        f"not all client received the statistic '{m}', need to wait for {sleep_time} seconds."
                        f"currently available clients are '{client_statistics[m].keys()}'."
                    )
                    logger.info(msg)
                    time.sleep(sleep_time)
                    t += sleep_time
                    # check and update number of client processed for statistics again
                    statistics_client_received[m] = len(client_statistics[m].keys())

        return True

    def _get_result_cb(self, statistics_task: str):
        return self.result_callback_fns[statistics_task]
