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
from typing import Dict, List

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.statistics.stats_def import Histogram, Metric, DatasetMetrics, FeatureMetric, Feature


class Statistics(FLComponent):

    def initialize(self, parts: dict, fl_ctx: FLContext):
        """Initialize the Statistics generator object. This is called before the Statistics can perform calculation

        This is called only once.

        Args:
            parts: components to be used by the Statistics generator
            fl_ctx: FLContext of the running environment
        """
        pass

    def get_features(self, fl_ctx: FLContext) -> Dict[str, List[Feature]]:
        """
            return feature names for each dataset.
            For example, we have training and test datasets.

            the method will return
            { "train": features1, "test": features2}
            where features1,2 are the list of Features with contains feature name and DataType

        :param fl_ctx:
        :return:
        """
        pass

    def get_count(self,
                  dataset_name: str,
                  feature_name: str,
                  inputs: Shareable,
                  fl_cxt: FLContext) -> int:
        """
            return count for given dataset and feature

        :param dataset_name:
        :param feature_name:
        :param inputs:
        :param fl_cxt:
        :return: count
        """
        pass

    def get_sum(self,
                dataset_name: str,
                feature_name: str,
                inputs: Shareable,
                fl_ctx: FLContext) -> float:
        """
            get local sums for given dataset and feature
        :param dataset_name:
        :param feature_name:
        :param inputs:
        :param fl_ctx:
        :return: sum
        """
        pass

    def get_mean(self,
                 dataset_name: str,
                 feature_name: str,
                 inputs: Shareable,
                 fl_ctx: FLContext) -> float:
        """
            get local means for given dataset and feature
        :param dataset_name: data set
        :param feature_name:
        :param inputs:
        :param fl_ctx:
        """
        count: int = self.get_count(dataset_name, feature_name, inputs, fl_ctx)
        sum_value: float = self.get_sum(dataset_name, feature_name, inputs, fl_ctx)
        return sum_value / count

    def get_stddev(self,
                   dataset_name: str,
                   feature_name: str,
                   inputs: Shareable,
                   fl_ctx: FLContext) -> float:
        """
          get local stddev value for given dataset and feature
        :param dataset_name:
        :param feature_name:
        :param inputs: contains the input information, mostly from server
        :param fl_ctx: FLContext
        :return: dictionary of the feature name and stddev value
        """
        pass

    def get_variance_with_mean(self,
                               dataset_name: str,
                               feature_name: str,
                               inputs: Shareable,
                               fl_ctx: FLContext) -> float:
        """
            calculate the variance with the given mean value from input sharable
            based on input data (self.data) and other input information.
            This is not local variance based on the local mean values.
            The calculation should be
            m = global mean  = inputs[STATS_GLOBAL_COUNT][dataset_name][feature_name]
            N = global Count = inputs[STATS_GLOBAL_MEAN][dataset_name][feature_name]
            variance = (sum ( x - m)^2))/ (N-1)

        :param dataset_name:
        :param feature_name:
        :param inputs: contains the input information, mostly from server
        :param fl_ctx: FLContext
        :return: dictionary of the feature name and variance value
        """
        pass

    def get_histogram(self,
                      dataset_name: str,
                      feature_name: str,
                      inputs: Shareable,
                      fl_ctx: FLContext) -> Histogram:
        """
          get local histograms based on given numbers of bins, and global range of the bins.
          bins =  inputs[STATS_BINS]
          bin_range_min = inputs[STATS_MIN][dataset_name][feature_name]
          bin_range_max = inputs[STATS_MAX][dataset_name][feature_name]

        :param dataset_name:
        :param feature_name:
        :param inputs: contains the input information
        :param fl_ctx: FLContext
        :return: histogram
        """
        pass

    def get_max_value(self,
                      dataset_name: str,
                      feature_name: str,
                      inputs: Shareable,
                      fl_ctx: FLContext) -> float:
        """
            this method is needed to figure out the histogram global bucket/bin ranges.
            But the actual max_values are not returned to FL Server. Only random modified
            local max values are return to FL Server
            :param dataset_name:
            :param feature_name:
            :param inputs: contains the input information, mostly from server
            :param fl_ctx: FLContext
            :return: local max value
        """
        pass

    def get_min_value(self,
                      dataset_name: str,
                      feature_name: str,
                      inputs: Shareable,
                      fl_ctx: FLContext) -> float:
        """
            this method is needed to figure out the histogram global bucket/bin ranges.
            But the actual min_values are not returned to FL Server. Only random modified
            local min values are return to FL Server

            :param dataset_name:
            :param feature_name:
            :param inputs: contains the input information, mostly from server
            :param fl_ctx: FLContext
            :return: bool
        """
        pass

    def finalize(self, fl_ctx: FLContext):
        """Called to finalize the Statistic calculator (close/release resources gracefully).

        After this call, the Learner will be destroyed.

        Args:
            fl_ctx: FLContext of the running environment

        """
        pass
