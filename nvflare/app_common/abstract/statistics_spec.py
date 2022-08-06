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
from nvflare.app_common.statistics.stats_def import Feature, Histogram


class Statistics(FLComponent):
    def initialize(self, parts: dict, fl_ctx: FLContext):
        """Initialize the Statistics generator object. This is called before the Statistics can perform calculation

        This is called only once.

        Args:
            parts: components to be used by the Statistics generator
            fl_ctx: FLContext of the running environment
        """
        pass

    def features(self) -> Dict[str, List[Feature]]:
        """
           return feature names for each dataset.
           For example, we have training and test datasets.

           the method will return
           { "train": features1, "test": features2}
           where features1,2 are the list of Features with contains feature name and DataType

        :return: Dict[<dataset_name>, List[Feature]]
        """
        pass

    def count(self, dataset_name: str, feature_name: str) -> int:
        """
            return count for given dataset and feature

        :param dataset_name:
        :param feature_name:
        :return: count
        """
        pass

    def sum(self, dataset_name: str, feature_name: str) -> float:
        """
            get local sums for given dataset and feature
        :param dataset_name:
        :param feature_name:
        :return: sum
        """
        pass

    def mean(self, dataset_name: str, feature_name: str) -> float:
        """
            get local means for given dataset and feature
            if you already implemented the count and sum, there is no need to implement this method

        :param dataset_name: data set
        :param feature_name:
        """

        pass

    def stddev(self, dataset_name: str, feature_name: str) -> float:
        """
          get local stddev value for given dataset and feature
        :param dataset_name:
        :param feature_name:
        :return: dictionary of the feature name and stddev value
        """
        pass

    def variance_with_mean(
        self,
        dataset_name: str,
        feature_name: str,
        global_mean: float,
        global_count: float,
    ) -> float:
        """
            calculate the variance with the given mean and count values
            This is not local variance based on the local mean values.
            The calculation should be
            m = global mean
            N = global Count
            variance = (sum ( x - m)^2))/ (N-1)

        :param dataset_name:
        :param feature_name:
        :param global_mean:
        :param global_count:
        :return: dictionary of the feature name and variance value
        """
        pass

    def histogram(
        self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float, global_max_value: float
    ) -> Histogram:
        """
          get local histograms based on given numbers of bins, and global range of the bins.
          bin_range_min = global_min_value
          bin_range_max = global_max_value

        :param dataset_name:
        :param feature_name:
        :param num_of_bins:
        :param global_min_value:
        :param global_max_value:
        :return: histogram
        """
        pass

    def max_value(self, dataset_name: str, feature_name: str) -> float:
        """
        this method is needed to figure out the histogram global bucket/bin ranges.
        But the actual max_values are not returned to FL Server. Only random modified
        local max values are return to FL Server
        :param dataset_name:
        :param feature_name:
        :return: local max value
        """
        pass

    def min_value(self, dataset_name: str, feature_name: str) -> float:
        """
        this method is needed to figure out the histogram global bucket/bin ranges.
        But the actual min_values are not returned to FL Server. Only random modified
        local min values are return to FL Server

        :param dataset_name:
        :param feature_name:
        :return: bool
        """
        pass

    def finalize(self):
        """Called to finalize the Statistic calculator (close/release resources gracefully).

        After this call, the Learner will be destroyed.

        """
        pass
