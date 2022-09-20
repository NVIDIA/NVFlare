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
from nvflare.app_common.statistics.metrics_config_utils import get_feature_bin_range
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
        precision=4,
    ):
        """

        Args:
            generator_id:  Id of the statistics component

            precision: number of precision digits

        """

        super().__init__()
        self.init_status_ok = True
        self.init_failure = None
        self.generator_id = generator_id
        self.stats_generator: Optional[Statistics] = None
        self.precision = precision
        self.client_name = None
        fobs_registration()

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
            self.client_name = fl_ctx.get_identity_name()
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
            self.init_status_ok = False
            self.init_failure = e

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if not self.init_status_ok:
            self.logger.info(self.init_failure)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

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
        client_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"Executing task '{task_name}' for client: '{client_name}'")
        result = Shareable()
        metrics_result = {}
        if task_name == StC.FED_STATS_TASK:
            ds_features = self.get_numeric_features()
            metric_task = shareable.get(StC.METRIC_TASK_KEY)
            target_metrics: List[MetricConfig] = fobs.loads(shareable.get(StC.STATS_TARGET_METRICS))
            for tm in target_metrics:
                fn = self.metric_functions()[tm.name]
                metrics_result[tm.name] = {}
                StatisticsExecutor._populate_result_metrics(metrics_result, ds_features, tm, shareable, fl_ctx, fn)

            # always add count for data privacy needs
            if StC.STATS_COUNT not in metrics_result:
                tm = MetricConfig(StC.STATS_COUNT, {})
                fn = self.get_count
                metrics_result[tm.name] = {}
                StatisticsExecutor._populate_result_metrics(metrics_result, ds_features, tm, shareable, fl_ctx, fn)

            result[StC.METRIC_TASK_KEY] = metric_task
            if metric_task == StC.STATS_1st_METRICS:
                result[StC.STATS_FEATURES] = fobs.dumps(ds_features)
            result[metric_task] = fobs.dumps(metrics_result)

            target_metrics: List[MetricConfig]
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

        return result

    @staticmethod
    def _populate_result_metrics(metrics_result, ds_features, tm: MetricConfig, shareable, fl_ctx, fn):
        for ds_name in ds_features:
            metrics_result[tm.name][ds_name] = {}
            features: List[Feature] = ds_features[ds_name]
            for feature in features:
                metrics_result[tm.name][ds_name][feature.feature_name] = fn(
                    ds_name, feature.feature_name, tm, shareable, fl_ctx
                )

    def get_numeric_features(self) -> Dict[str, List[Feature]]:
        ds_features: Dict[str, List[Feature]] = self.stats_generator.features()
        return filter_numeric_features(ds_features)

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
        feature_bin_range = get_feature_bin_range(feature_name, hist_config)
        if feature_bin_range is None:
            client_max_value = self.stats_generator.max_value(dataset_name, feature_name)
            return client_max_value
        else:
            return feature_bin_range[1]

    def get_min_value(
        self, dataset_name: str, feature_name: str, metric_config: MetricConfig, inputs: Shareable, fl_ctx: FLContext
    ) -> float:
        """
        get randomized min value
        """
        hist_config: dict = metric_config.config
        feature_bin_range = get_feature_bin_range(feature_name, hist_config)
        if feature_bin_range is None:
            client_min_value = self.stats_generator.min_value(dataset_name, feature_name)
            return client_min_value
        else:
            return feature_bin_range[0]

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
        bin_range = get_feature_bin_range(feature_name, hist_config)
        if bin_range is None:
            bin_range = global_bin_range

        return bin_range

    def finalize(self, fl_ctx: FLContext):
        try:
            if self.stats_generator:
                self.stats_generator.finalize()
        except Exception as e:
            self.log_exception(fl_ctx, f"Statistics generator finalize exception: {e}")
