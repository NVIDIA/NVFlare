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
import random
from typing import Dict, List, Optional

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.statistics_spec import Feature, Histogram, HistogramType, MetricConfig, Statistics
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.executors.statistics.statistics_executor_exception import StatisticExecutorException
from nvflare.app_common.statistics.numeric_stats import filter_numeric_features
from nvflare.app_common.statistics.statisitcs_objects_decomposer import fobs_registration
from nvflare.fuel.utils import fobs

"""
    StatisticsExecutor is client-side executor that perform local statistics generation and communication to
    FL Server global statistics controller.
    The actual local statistics calculation would delegate to Statistics spec implementor.
"""


class StatisticsExecutor(Executor):
    def __init__(
        self,
        generator_id: str,
        min_count: int,
        min_random: float,
        max_random: float,
        max_bins_percent: float,
        precision=4,
    ):
        """

        Args:
            generator_id:  Id of the statistics component
            min_count:     minimum of data records (or tabular data rows) that required in order to perform statistics calculation
                           this is part the data privacy policy.
                           todo: This configuration will be moved to local/privacy.json files
            min_random:    minimum random noise -- used to protect min/max values before sending to server
            max_random:    maximum random noise -- used to protect min/max values before sending to server
                           min/max random is used to generate random noise between (min_random and max_random).
                           for example, the random noise is to be within (0.1 and 0.3), 10% to 30% level. These noise
                           will make local min values smaller than the true local min values, and max values larger than
                           the true local max values. As result, the estimate global max and min values (i.e. with noise)
                           are still bound the true global min/max values, in such that

                              est. global min value <
                                        true global min value <
                                                  client's min value <
                                                          client's max value <
                                                                  true global max <
                                                                           est. global max value
                       todo: the min/max noise level range (min_random, max_random) will be moved to
                       todo: local/privacy.json
            max_bins_percent:   max number of bins allowed in terms of percent of local data size. Set this number to
                                avoid number of bins equal or close equal to the data size, which can lead to data leak.
                                number of bins < max_bins_percent * local count
                                todo: this argument will be move to local/privacy.json

            precision: number of percision digts

        """

        super().__init__()
        self.generator_id = generator_id
        self.min_count = min_count
        self.stats_generator: Optional[Statistics] = None
        self.max_random = max_random
        self.min_random = min_random
        self.max_bins_percent = max_bins_percent
        self.precision = precision
        fobs_registration()

    def validate_inputs(self, fl_ctx: FLContext):
        try:
            if self.min_random < 0 or self.min_random > 1.0:
                raise ValueError(
                    f"minimum noise level provided by min_random {self.min_random} should be within (0, 1)"
                )
            if self.max_random < 0 or self.max_random > 1.0:
                raise ValueError(
                    f"maximum noise level provided by max_random {self.max_random} should be within (0, 1)"
                )

            if self.min_random > self.max_random:
                raise ValueError(
                    "minimum noise level {} should be less than maximum noise level {}".format(
                        self.min_random, self.max_random
                    )
                )

            if self.max_bins_percent < 0 or self.max_bins_percent > 1.0:
                raise ValueError(f"max_bins_percent should be within (0, 1), but {self.max_bins_percent} is provided")

        except ValueError as e:
            self.system_panic(f"input error {e}", fl_ctx)
            raise e

    def metric_functions(self) -> dict:
        return {
            StC.STATS_COUNT: self.get_count,
            StC.STATS_SUM: self.get_sum,
            StC.STATS_MEAN: self.get_mean,
            StC.STATS_STDDEV: self.get_stddev,
            StC.STATS_VAR: self.get_variance_with_mean,
            StC.STATS_HISTOGRAM: self.get_histogram,
            StC.STATS_MAX: self.get_max_value,
            StC.STATS_MIN: self.get_min_value,
        }

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        try:
            self.validate_inputs(fl_ctx)
            engine = fl_ctx.get_engine()
            self.stats_generator = engine.get_component(self.generator_id)
            if not isinstance(self.stats_generator, Statistics):
                raise TypeError(
                    f"{type(self.stats_generator).__name} must implement `Statistics` type."
                    f" Got: {type(self.stats_generator)}"
                )

            self.stats_generator.initialize(engine.get_all_components(), fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"statistics generator initialize exception: {e}")

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        client_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"Executing task '{task_name}' for client: '{client_name}'")
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        try:
            result = self._client_exec(task_name, shareable, fl_ctx, abort_signal)
            if result:
                dxo = DXO(data_kind=DataKind.STATISTICS, data=result)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

        except BaseException as e:
            self.log_exception(fl_ctx, f"Task {task_name} failed. Exception: {e.__str__()}")
            return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

    def _client_exec(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # client_name = fl_ctx.get_prop(ReservedKey.CLIENT_NAME)
        client_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"Executing task '{task_name}' for client: '{client_name}'")
        result = Shareable()
        metrics_result = {}
        if task_name == StC.FED_STATS_TASK:
            ds_features = self.get_numeric_features()
            self.validate(client_name, ds_features, shareable, fl_ctx)
            metric_task = shareable.get(StC.METRIC_TASK_KEY)
            target_metrics: List[MetricConfig] = fobs.loads(shareable.get(StC.STATS_TARGET_METRICS))
            for tm in target_metrics:
                fn = self.metric_functions()[tm.name]
                metrics_result[tm.name] = {}
                for ds_name in ds_features:
                    metrics_result[tm.name][ds_name] = {}
                    features: List[Feature] = ds_features[ds_name]
                    for feature in features:
                        metrics_result[tm.name][ds_name][feature.feature_name] = fn(
                            ds_name, feature.feature_name, tm, shareable, fl_ctx
                        )

            result[StC.METRIC_TASK_KEY] = metric_task
            if metric_task == StC.STATS_1st_METRICS:
                result[StC.STATS_FEATURES] = fobs.dumps(ds_features)
            result[metric_task] = fobs.dumps(metrics_result)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

        return result

    def get_numeric_features(self) -> Dict[str, List[Feature]]:
        ds_features: Dict[str, List[Feature]] = self.stats_generator.features()
        return filter_numeric_features(ds_features)

    def validate(
        self, client_name: str, ds_features: Dict[str, List[Feature]], shareable: Shareable, fl_ctx: FLContext
    ):
        count_config = MetricConfig(StC.STATS_COUNT, {})
        for ds_name in ds_features:
            features = ds_features[ds_name]
            for feature in features:
                feature_name = feature.feature_name
                count = self.get_count(ds_name, feature.feature_name, count_config, shareable, fl_ctx)
                if count < self.min_count:
                    raise StatisticExecutorException(
                        f" dataset {ds_name} feature '{feature_name}' item count is "
                        f"less than required minimum count {self.min_count} for client {client_name} "
                    )

    def get_count(
        self, dataset_name: str, feature_name: str, metric_config: MetricConfig, inputs: Shareable, fl_ctx: FLContext
    ) -> int:

        result = self.stats_generator.count(dataset_name, feature_name)
        return result

    def get_sum(
        self, dataset_name: str, feature_name: str, metric_config: MetricConfig, inputs: Shareable, fl_ctx: FLContext
    ) -> float:

        result = round(self.stats_generator.sum(dataset_name, feature_name), self.precision)
        return result

    def get_mean(
        self, dataset_name: str, feature_name: str, metric_config: MetricConfig, inputs: Shareable, fl_ctx: FLContext
    ) -> float:
        count = self.stats_generator.count(dataset_name, feature_name)
        sum_value = self.stats_generator.sum(dataset_name, feature_name)
        if count is not None and sum_value is not None:
            return round(sum_value / count, self.precision)
        else:
            # user did not implement count and/or sum, call means directly.
            mean = round(self.stats_generator.mean(dataset_name, feature_name), self.precision)
            # self._check_result(mean, metric_config.name)
            return mean

    def get_stddev(
        self, dataset_name: str, feature_name: str, metric_config: MetricConfig, inputs: Shareable, fl_ctx: FLContext
    ) -> float:

        result = round(self.stats_generator.stddev(dataset_name, feature_name), self.precision)
        return result

    def get_variance_with_mean(
        self, dataset_name: str, feature_name: str, metric_config: MetricConfig, inputs: Shareable, fl_ctx: FLContext
    ) -> float:
        if StC.STATS_GLOBAL_MEAN in inputs and StC.STATS_GLOBAL_COUNT in inputs:
            global_mean = inputs[StC.STATS_GLOBAL_MEAN][dataset_name][feature_name]
            global_count = inputs[StC.STATS_GLOBAL_COUNT][dataset_name][feature_name]
            result = self.stats_generator.variance_with_mean(dataset_name, feature_name, global_mean, global_count)
            result = round(result, self.precision)
            return result
        else:
            return None

    def get_histogram(
        self, dataset_name: str, feature_name: str, metric_config: MetricConfig, inputs: Shareable, fl_ctx: FLContext
    ) -> Histogram:
        if StC.STATS_MIN in inputs and StC.STATS_MAX in inputs:
            global_min_value = inputs[StC.STATS_MIN][dataset_name][feature_name]
            global_max_value = inputs[StC.STATS_MAX][dataset_name][feature_name]
            hist_config: dict = metric_config.config
            num_of_bins: int = self.get_number_of_bins(feature_name, hist_config)
            bin_range: List[int] = self.get_bin_range(feature_name, global_min_value, global_max_value, hist_config)
            item_count = self.stats_generator.count(dataset_name, feature_name)
            if num_of_bins >= item_count * self.max_bins_percent:
                raise ValueError(
                    f"number of bins: {num_of_bins} needs to be smaller than item count: {round(item_count * self.max_bins_percent)} "
                    f"for feature '{feature_name}' in dataset '{dataset_name}'"
                )

            result = self.stats_generator.histogram(dataset_name, feature_name, num_of_bins, bin_range[0], bin_range[1])
            return result
        else:
            return Histogram(HistogramType.STANDARD, list())

    def get_max_value(
        self, dataset_name: str, feature_name: str, metric_config: MetricConfig, inputs: Shareable, fl_ctx: FLContext
    ) -> float:
        """
        get randomized max value
        """
        hist_config: dict = metric_config.config
        user_bin_range = self.get_user_bin_range(feature_name, hist_config)
        if user_bin_range is None:
            client_max_value = self.stats_generator.max_value(dataset_name, feature_name)
            return self._get_max_value(client_max_value)
        else:
            return user_bin_range[1]

    def get_min_value(
        self, dataset_name: str, feature_name: str, metric_config: MetricConfig, inputs: Shareable, fl_ctx: FLContext
    ) -> float:
        """
        get randomized min value
        """
        hist_config: dict = metric_config.config
        user_bin_range = self.get_user_bin_range(feature_name, hist_config)
        if user_bin_range is None:
            client_min_value = self.stats_generator.min_value(dataset_name, feature_name)
            return self._get_min_value(client_min_value)
        else:
            return user_bin_range[0]

    def get_number_of_bins(self, feature_name: str, hist_config: dict) -> int:
        err_msg = (
            f"feature name = '{feature_name}': "
            f"missing required '{StC.STATS_BINS}' config in histogram config = {hist_config}"
        )
        try:
            num_of_bins = None
            if feature_name in hist_config:
                num_of_bins = hist_config[feature_name][StC.STATS_BINS]
            else:
                if "*" in hist_config:
                    default_config = hist_config["*"]
                    num_of_bins = default_config[StC.STATS_BINS]
            if num_of_bins:
                return num_of_bins
            else:
                raise Exception(err_msg)

        except KeyError as e:
            raise Exception(err_msg)

    def get_bin_range(
        self, feature_name: str, global_min_value: float, global_max_value: float, hist_config: dict
    ) -> List[int]:

        global_bin_range = [global_min_value, global_max_value]
        bin_range = self.get_user_bin_range(feature_name, hist_config)
        if bin_range is None:
            bin_range = global_bin_range

        return bin_range

    def get_user_bin_range(self, feature_name: str, hist_config: dict) -> Optional[List[int]]:
        bin_range = None
        if feature_name in hist_config:
            if StC.STATS_BIN_RANGE in hist_config[feature_name]:
                bin_range = hist_config[feature_name][StC.STATS_BIN_RANGE]
        elif "*" in hist_config:
            default_config = hist_config["*"]
            if StC.STATS_BIN_RANGE in default_config:
                bin_range = default_config[StC.STATS_BIN_RANGE]

        return bin_range

    def _get_max_value(self, local_max_value: float):
        r = random.uniform(self.min_random, self.max_random)
        if local_max_value == 0:
            max_value = (1 + r) * 1e-5
        else:
            if local_max_value > 0:
                max_value = local_max_value * (1 + r)
            else:
                max_value = local_max_value * (1 - r)

        return max_value

    def _get_min_value(self, local_min_value: float):
        r = random.uniform(self.min_random, self.max_random)
        if local_min_value == 0:
            min_value = -(1 - r) * 1e-5
        else:
            if local_min_value > 0:
                min_value = local_min_value * (1 - r)
            else:
                min_value = local_min_value * (1 + r)

        return min_value

    def finalize(self, fl_ctx: FLContext):
        try:
            if self.stats_generator:
                self.stats_generator.finalize()
        except Exception as e:
            self.log_exception(fl_ctx, f"Statistics generator finalize exception: {e}")
