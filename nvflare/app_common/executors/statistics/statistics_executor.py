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

from abc import ABC, abstractmethod

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors.statistics.base_statistics_executor import BaseStatsExecutor
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.stats_def import Histogram
from nvflare.app_common.validation_exception import ValidationException
from typing import Dict


class StatisticsExecutor(BaseStatsExecutor, ABC):
    def __init__(self,
                 data_path,
                 min_count
                 ):
        super().__init__()
        self.data_path = data_path
        self.min_count = min_count
        self.counts: Dict[str, int] = {}
        self.sums: Dict[str, int] = {}

    def metric_functions(self) -> dict:
        return \
            {StC.STATS_COUNT: self.get_counts,
             StC.STATS_SUM: self.get_sums,
             StC.STATS_MEAN: self.get_means,
             StC.STATS_STDDEV: self.get_stddevs,
             StC.STATS_VAR: self.get_variances_with_mean,
             StC.STATS_HISTOGRAM: self.get_histograms}

    @abstractmethod
    def get_counts(self, shareable: Shareable, fl_cxt: FLContext) -> Dict[str, int]:
        """
            get count based on input data (self.data) and other input information

        :param shareable: contains the input information, mostly from server
        :param accum_result: the results available so far based on previous metrics calculation
        :param fl_cxt:
        :return: dictionary of the feature name and count value
        """
        if len(self.counts) > 0:
            return self.counts
        else:
            raise NotImplemented

    @abstractmethod
    def get_sums(self, inputs: Shareable, fl_ctx: FLContext) -> Dict[str, float]:
        """
            get local sums based on input data (self.data) and other input information

        :param inputs: contains the input information, mostly from server
        :param accum_result: the results available so far based on previous metrics calculation
        :param fl_ctx: FLContext
        :return: dictionary of the feature name and sum value
        """
        if len(self.sums) > 0:
            return self.sums
        else:
            raise NotImplemented

    def get_means(self, inputs: Shareable, fl_ctx: FLContext) -> Dict[str, float]:
        """
            get local means based on input data (self.data) and other input information

        :param inputs: contains the input information, mostly from server
        :param fl_ctx: FLContext
        :return: dictionary of the feature name and mean value
        """
        counts = self.get_counts(inputs, fl_ctx)
        sums = self.get_sums(inputs, fl_ctx)
        results = {}
        for feature in counts:
            results[feature] = sums[feature] / counts[feature]

        return results

    @abstractmethod
    def get_stddevs(self, inputs: Shareable, fl_ctx: FLContext) -> Dict[str, float]:
        """
          get local stddev value based on input data (self.data) and other input information

        :param inputs: contains the input information, mostly from server
        :param fl_ctx: FLContext
        :return: dictionary of the feature name and stddev value
        """
        raise NotImplemented

    @abstractmethod
    def get_variances_with_mean(self, inputs: Shareable, fl_ctx: FLContext) -> Dict[str, float]:
        """
            calculate the variance with the given mean value from input sharable
            based on input data (self.data) and other input information.
            This is not local variance based on the local mean values.
            The calculation should be
            m = global mean
            N = global Count
            variance = (sum ( x - m)^2))/ (N-1)

        :param inputs: contains the input information, mostly from server
        :param fl_ctx: FLContext
        :return: dictionary of the feature name and variance value
        """
        raise NotImplemented

    @abstractmethod
    def get_histograms(self, inputs: Shareable, fl_ctx: FLContext) -> Dict[str, Histogram]:
        """
            get local histograms based on given numbers of bins, and range of the bins.
            we will get inputs:
                number of bins
                bin_range
            bins = inputs[StatsConstant.STATISTICS_NUMBER_BINS]
            bin_range = inputs[StatsConstant.STATISTICS_BIN_RANGE]

        :param inputs: contains the input information, mostly from server
        :param fl_ctx: FLContext
        :return: dictionary of the feature name and histogram value
        """
        raise NotImplemented

    @abstractmethod
    def client_data_validate(self, client_name: str, shareable: Shareable, fl_ctx: FLContext):
        """
            client_data_validate is called after the load_data(), at begining of the client_exec()
        :param client_name: client name
        :param shareable:  shareable
        :param fl_ctx:  FLContext
        :return: None
        """
        pass

    def client_exec(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        client_name = fl_ctx.get_prop(ReservedKey.CLIENT_NAME)
        self.log_info(fl_ctx, f"Executing task '{task_name}' for client: '{client_name}'")
        result = Shareable()
        self._validate(client_name, shareable, fl_ctx)

        metrics_result = {}
        if task_name == StC.FED_STATS_TASK:
            metric_task = shareable.get(StC.METRIC_TASK_KEY)
            metrics = shareable.get(StC.STATS_TARGET_METRICS)
            for metric in metrics:
                fn = self.metric_functions()[metric]
                metrics_result[metric] = fn(shareable, result, fl_ctx)

            result.set_header(StC.METRIC_TASK_KEY, metric_task)
            result[metric_task] = metrics_result

        return result

    def _validate(self, client_name: str, shareable: Shareable, fl_ctx: FLContext):
        self.client_data_validate(client_name, shareable, fl_ctx)
        self._validate_counts(client_name, shareable, fl_ctx)

    def _validate_counts(self, client_name: str, shareable: Shareable, fl_ctx: FLContext):
        self.counts = self.get_counts(shareable, fl_ctx)
        for feature in self.counts:
            count = self.counts[feature]
            if count < self.min_count:
                raise ValidationException(
                    f"feature {feature} item count is "
                    f"less than required minimum count {self.min_count} for client {client_name} ")
