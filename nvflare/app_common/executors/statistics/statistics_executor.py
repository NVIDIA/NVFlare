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
from nvflare.app_common.abstract.statistics_spec import Feature, Histogram, HistogramType, StatisticConfig, Statistics
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.numeric_stats import filter_numeric_features
from nvflare.app_common.statistics.statisitcs_objects_decomposer import fobs_registration
from nvflare.app_common.statistics.statistics_config_utils import get_feature_bin_range
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
        self.init_failure = {"abort_job": None, "fail_client": None}
        self.generator_id = generator_id
        self.stats_generator: Optional[Statistics] = None
        self.precision = precision
        self.client_name = None
        fobs_registration()

    def statistic_functions(self) -> dict:
        return {
            StC.STATS_COUNT: self.get_count,
            StC.STATS_FAILURE_COUNT: self.get_failure_count,
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
            self._check_generator_type()
            self.stats_generator.initialize(engine.get_all_components(), fl_ctx)
        except TypeError as te:
            self.log_exception(
                fl_ctx,
                f"local statistics generator must implement `Statistics`,"
                f"but {type(self.stats_generator).__name__} is provided. {te}",
            )
            self.init_status_ok = False
            self.init_failure = {"abort_job": te}
        except Exception as e:
            self.init_status_ok = False
            self.init_failure = {"fail_client": e}

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        init_rc = self._check_init_status(fl_ctx)
        if init_rc:
            return make_reply(init_rc)

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
        statistics_result = {}
        if task_name == StC.FED_STATS_PRE_RUN:
            # initial handshake
            target_statistics: List[StatisticConfig] = fobs.loads(shareable.get(StC.STATS_TARGET_STATISTICS))
            return self.pre_run(target_statistics)

        elif task_name == StC.FED_STATS_TASK:
            ds_features = self.get_numeric_features()
            statistics_task = shareable.get(StC.STATISTICS_TASK_KEY)
            target_statistics: List[StatisticConfig] = fobs.loads(shareable.get(StC.STATS_TARGET_STATISTICS))
            if StC.STATS_FAILURE_COUNT not in target_statistics:
                target_statistics.append(StatisticConfig(StC.STATS_FAILURE_COUNT, {}))

            for tm in target_statistics:
                fn = self.statistic_functions()[tm.name]
                statistics_result[tm.name] = {}
                self._populate_result_statistics(statistics_result, ds_features, tm, shareable, fl_ctx, fn)

            # always add count for data privacy needs
            if StC.STATS_COUNT not in statistics_result:
                tm = StatisticConfig(StC.STATS_COUNT, {})
                fn = self.get_count
                statistics_result[tm.name] = {}
                self._populate_result_statistics(statistics_result, ds_features, tm, shareable, fl_ctx, fn)

            result[StC.STATISTICS_TASK_KEY] = statistics_task
            if statistics_task == StC.STATS_1st_STATISTICS:
                result[StC.STATS_FEATURES] = fobs.dumps(ds_features)
            result[statistics_task] = fobs.dumps(statistics_result)

            target_statistics: List[StatisticConfig]
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

        return result

    def _check_generator_type(self):
        if not isinstance(self.stats_generator, Statistics):
            raise TypeError(
                f"{type(self.stats_generator).__name} must implement `Statistics` type."
                f" Got: {type(self.stats_generator)}"
            )

    def _check_init_status(self, fl_ctx: FLContext):
        if not self.init_status_ok:
            for fail_key in self.init_failure:
                reason = self.init_failure[fail_key]
                if fail_key == "abort_job":
                    return ReturnCode.EXECUTION_EXCEPTION
                else:
                    self.system_panic(reason, fl_ctx)
                    # probably never reach here, but just for type consistency
                    return ReturnCode.EXECUTION_RESULT_ERROR
        return None

    def _populate_result_statistics(self, statistics_result, ds_features, tm: StatisticConfig, shareable, fl_ctx, fn):
        for ds_name in ds_features:
            statistics_result[tm.name][ds_name] = {}
            features: List[Feature] = ds_features[ds_name]
            for feature in features:
                try:
                    statistics_result[tm.name][ds_name][feature.feature_name] = fn(
                        ds_name, feature.feature_name, tm, shareable, fl_ctx
                    )
                except BaseException as e:
                    self.log_exception(
                        fl_ctx,
                        f"Failed to populate result  statistics of dataset {ds_name}"
                        f" and feature {feature.feature_name} with exception: {e}",
                    )

    def get_numeric_features(self) -> Dict[str, List[Feature]]:
        ds_features: Dict[str, List[Feature]] = self.stats_generator.features()
        return filter_numeric_features(ds_features)

    def pre_run(self, target_statistics: List[StatisticConfig]):
        feature_num_of_bins = None
        feature_bin_ranges = None
        target_statistic_keys = []
        for mc in target_statistics:
            target_statistic_keys.append(mc.name)
            if mc.name == StC.STATS_HISTOGRAM:
                hist_config = mc.config
                feature_num_of_bins = {}
                feature_bin_ranges = {}
                for feature_name in hist_config:
                    num_of_bins: int = self.get_number_of_bins(feature_name, hist_config)
                    feature_num_of_bins[feature_name] = num_of_bins
                    bin_range = get_feature_bin_range(feature_name, hist_config)
                    feature_bin_ranges[feature_name] = bin_range

        return self.stats_generator.pre_run(target_statistic_keys, feature_num_of_bins, feature_bin_ranges)

    def get_count(
        self,
        dataset_name: str,
        feature_name: str,
        statistic_configs: StatisticConfig,
        inputs: Shareable,
        fl_ctx: FLContext,
    ) -> int:

        result = self.stats_generator.count(dataset_name, feature_name)
        return result

    def get_failure_count(
        self,
        dataset_name: str,
        feature_name: str,
        statistic_configs: StatisticConfig,
        inputs: Shareable,
        fl_ctx: FLContext,
    ) -> int:

        result = self.stats_generator.failure_count(dataset_name, feature_name)
        return result

    def get_sum(
        self,
        dataset_name: str,
        feature_name: str,
        statistic_configs: StatisticConfig,
        inputs: Shareable,
        fl_ctx: FLContext,
    ) -> float:

        result = round(self.stats_generator.sum(dataset_name, feature_name), self.precision)
        return result

    def get_mean(
        self,
        dataset_name: str,
        feature_name: str,
        statistic_configs: StatisticConfig,
        inputs: Shareable,
        fl_ctx: FLContext,
    ) -> float:
        count = self.stats_generator.count(dataset_name, feature_name)
        sum_value = self.stats_generator.sum(dataset_name, feature_name)
        if count is not None and sum_value is not None:
            return round(sum_value / count, self.precision)
        else:
            # user did not implement count and/or sum, call means directly.
            mean = round(self.stats_generator.mean(dataset_name, feature_name), self.precision)
            # self._check_result(mean, statistic_configs.name)
            return mean

    def get_stddev(
        self,
        dataset_name: str,
        feature_name: str,
        statistic_configs: StatisticConfig,
        inputs: Shareable,
        fl_ctx: FLContext,
    ) -> float:

        result = round(self.stats_generator.stddev(dataset_name, feature_name), self.precision)
        return result

    def get_variance_with_mean(
        self,
        dataset_name: str,
        feature_name: str,
        statistic_configs: StatisticConfig,
        inputs: Shareable,
        fl_ctx: FLContext,
    ) -> float:
        result = None
        if StC.STATS_GLOBAL_MEAN in inputs and StC.STATS_GLOBAL_COUNT in inputs:
            global_mean = self._get_global_value_from_input(StC.STATS_GLOBAL_MEAN, dataset_name, feature_name, inputs)
            global_count = self._get_global_value_from_input(StC.STATS_GLOBAL_COUNT, dataset_name, feature_name, inputs)
            if global_mean is not None and global_count is not None:
                result = self.stats_generator.variance_with_mean(dataset_name, feature_name, global_mean, global_count)
                result = round(result, self.precision)

        return result

    def get_histogram(
        self,
        dataset_name: str,
        feature_name: str,
        statistic_configs: StatisticConfig,
        inputs: Shareable,
        fl_ctx: FLContext,
    ) -> Histogram:

        if StC.STATS_MIN in inputs and StC.STATS_MAX in inputs:

            global_min_value = self._get_global_value_from_input(StC.STATS_MIN, dataset_name, feature_name, inputs)
            global_max_value = self._get_global_value_from_input(StC.STATS_MAX, dataset_name, feature_name, inputs)
            if global_min_value is not None and global_max_value is not None:
                hist_config: dict = statistic_configs.config
                num_of_bins: int = self.get_number_of_bins(feature_name, hist_config)
                bin_range: List[int] = self.get_bin_range(feature_name, global_min_value, global_max_value, hist_config)
                result = self.stats_generator.histogram(
                    dataset_name, feature_name, num_of_bins, bin_range[0], bin_range[1]
                )
                return result
            else:
                return Histogram(HistogramType.STANDARD, list())
        else:
            return Histogram(HistogramType.STANDARD, list())

    def get_max_value(
        self,
        dataset_name: str,
        feature_name: str,
        statistic_configs: StatisticConfig,
        inputs: Shareable,
        fl_ctx: FLContext,
    ) -> float:
        """
        get randomized max value
        """
        hist_config: dict = statistic_configs.config
        feature_bin_range = get_feature_bin_range(feature_name, hist_config)
        if feature_bin_range is None:
            client_max_value = self.stats_generator.max_value(dataset_name, feature_name)
            return client_max_value
        else:
            return feature_bin_range[1]

    def get_min_value(
        self,
        dataset_name: str,
        feature_name: str,
        statistic_configs: StatisticConfig,
        inputs: Shareable,
        fl_ctx: FLContext,
    ) -> float:
        """
        get randomized min value
        """
        hist_config: dict = statistic_configs.config
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

    def _get_global_value_from_input(self, statistic_key: str, dataset_name: str, feature_name: str, inputs):
        global_value = None
        if dataset_name in inputs[statistic_key]:
            if feature_name in inputs[statistic_key][dataset_name]:
                global_value = inputs[statistic_key][dataset_name][feature_name]
            elif "*" in inputs[StC.STATS_MIN][dataset_name]:
                global_value = inputs[statistic_key][dataset_name][feature_name]

        return global_value
