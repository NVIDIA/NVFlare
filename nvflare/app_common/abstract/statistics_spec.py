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
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, List, NamedTuple, Optional

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.init_final_component import InitFinalComponent

"""
    Statistics defines methods that user need to implement in order to calculate the local statistics
    Only the metrics required by data privacy (such as count) or individual metrics of interested need to implement

"""


class DataType(IntEnum):
    INT = 0
    FLOAT = 1
    STRING = 2
    BYTES = 3
    STRUCT = 4
    DATETIME = 5


class BinRange(NamedTuple):
    # The minimum value of the bucket, inclusive.
    min_value: float
    # The max value of the bucket, exclusive (unless the highValue is positive infinity).
    max_value: float


class Bin(NamedTuple):
    # The low value of the bucket, inclusive.
    low_value: float

    # The high value of the bucket, exclusive (unless the highValue is positive infinity).
    high_value: float

    # quantile sample count could be fractional
    sample_count: float


class HistogramType(IntEnum):
    STANDARD = 0
    QUANTILES = 1


class Histogram(NamedTuple):
    # The type of the histogram. A standard histogram has equal-width buckets.
    # The quantiles type is used for when the histogram message is used to store
    # quantile information (by using equal-count buckets with variable widths).

    # The type of the histogram.
    hist_type: HistogramType

    # A list of buckets in the histogram, sorted from lowest bucket to highest bucket.
    bins: List[Bin]

    # An optional descriptive name of the histogram, to be used for labeling.
    hist_name: Optional[str] = None


class Feature(NamedTuple):
    feature_name: str
    data_type: DataType


class StatisticConfig(NamedTuple):
    # metric name
    name: str

    # metric configuration
    config: dict


class Statistics(InitFinalComponent, ABC):
    def initialize(self, fl_ctx: FLContext):
        """
        This is called when client is start Run. At this point
        the server hasn't communicated to the Statistics calculator yet.

        Args:
            fl_ctx: fl_ctx: FLContext of the running environment

        """

        pass

    def pre_run(
        self,
        statistics: List[str],
        num_of_bins: Optional[Dict[str, Optional[int]]],
        bin_ranges: Optional[Dict[str, Optional[List[float]]]],
    ):
        """This method is the initial hand-shake, where controller pass all the requested statistics configuration to client.

        This method invocation is optional and Configured via controller argument. If it is configured,
        this method will be called before all other statistic calculation methods

        Args:
            statistics: list of statistics to be calculated, count, sum, etc.
            num_of_bins: if histogram statistic is required, num_of_bins will be specified for each feature.
                         "*" implies default feature.
                         None value implies the feature's number of bins is not specified.
            bin_ranges: if histogram statistic is required, bin_ranges for the feature may be provided.
                        if bin_ranges is None. no bin_range is provided for any feature.
                        if bins_range is not None, but bins_ranges['feature_A'] is None, means that for specific feature
                        'feature_A', the bin_range is not provided by user.

        Returns: Dict

        """
        return {}

    @abstractmethod
    def features(self) -> Dict[str, List[Feature]]:
        """Return Features for each dataset.

        For example, if we have training and test datasets,
        the method will return
        { "train": features1, "test": features2}
        where features1,2 are the list of Features which contains feature name and DataType

        Returns: Dict[<dataset_name>, List[Feature]]

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def count(self, dataset_name: str, feature_name: str) -> int:
        """Returns record count for given dataset and feature.
           to perform data privacy min_count check, count is always required

        Args:
            dataset_name:
            feature_name:

        Returns: number of total records

        Raises:
            NotImplementedError

        """
        raise NotImplementedError

    def sum(self, dataset_name: str, feature_name: str) -> float:
        """Calculate local sums for given dataset and feature.

        Args:
            dataset_name:
            feature_name:

        Returns: sum of all records

        Raises:
            NotImplementedError will be raised when sum statistic is configured but not implemented. If the sum is not
            configured to be calculated, no need to implement this method and NotImplementedError will not be raised.

        """
        raise NotImplementedError

    def mean(self, dataset_name: str, feature_name: str) -> float:
        """

        Args:
            dataset_name: dataset name
            feature_name: feature name

        Returns: mean (average) value

        Raises:
            NotImplementedError will be raised when mean statistic is configured but not implemented. If the mean is not
            configured to be calculated, no need to implement this method and NotImplementedError will not be raised.

        """

        raise NotImplementedError

    def stddev(self, dataset_name: str, feature_name: str) -> float:
        """Get local stddev value for given dataset and feature.

        Args:
            dataset_name: dataset name
            feature_name: feature name

        Returns: local standard deviation

        Raises:
            NotImplementedError will be raised when stddev statistic is configured but not implemented. If the stddev is not
            configured to be calculated, no need to implement this method and NotImplementedError will not be raised.
        """
        raise NotImplementedError

    def variance_with_mean(
        self,
        dataset_name: str,
        feature_name: str,
        global_mean: float,
        global_count: float,
    ) -> float:
        """Calculate the variance with the given mean and count values.

        This is not local variance based on the local mean values.
        The calculation should be::

            m = global mean
            N = global Count
            variance = (sum ( x - m)^2))/ (N-1)

        This is used to calculate global standard deviation.
        Therefore, this method must be implemented if stddev statistic is requested

        Args:
            dataset_name: dataset name
            feature_name: feature name
            global_mean:  global mean value
            global_count: total count records across all sites

        Returns: variance result

        Raises:
            NotImplementedError will be raised when stddev statistic is configured but not implemented. If the stddev is not
            configured to be calculated, no need to implement this method and NotImplementedError will not be raised.
        """

        raise NotImplementedError

    def histogram(
        self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float, global_max_value: float
    ) -> Histogram:
        """
        Args:
            dataset_name: dataset name
            feature_name: feature name
            num_of_bins:  number of bins or buckets
            global_min_value: global min value for the histogram range
            global_max_value: global max value for the histogram range

        Returns: histogram

        Raises:
            NotImplementedError will be raised when histogram statistic is configured but not implemented. If the histogram
             is not configured to be calculated, no need to implement this method and NotImplementedError will not be raised.
        """

        raise NotImplementedError

    def max_value(self, dataset_name: str, feature_name: str) -> float:
        """Returns max value.

        This method is only needed when "histogram" statistic is configured and the histogram range is not specified.
        And the histogram range needs to dynamically estimated based on the client's local min/max values.
        this method returns local max value. The actual max value will not directly return to the FL server.
        the data privacy policy will add additional noise to the estimated value.

        Args:
            dataset_name: dataset name
            feature_name: feature name

        Returns: local max value

        Raises:
            NotImplementedError will be raised when histogram statistic is configured and histogram range for the
            given feature is not specified, and this method is not implemented. If the histogram
            is not configured to be calculated; or the given feature histogram range is already specified.
            no need to implement this method and NotImplementedError will not be raised.
        """

        raise NotImplementedError

    def min_value(self, dataset_name: str, feature_name: str) -> float:
        """Returns min value.

        This method is only needed when "histogram" statistic is configured and the histogram range is not specified.
        And the histogram range needs to dynamically estimated based on the client's local min/max values.
        this method returns local min value. The actual min value will not directly return to the FL server.
        the data privacy policy will add additional noise to the estimated value.

        Args:
            dataset_name: dataset name
            feature_name: feature name

        Returns: local min value

        Raises:
            NotImplementedError will be raised when histogram statistic is configured and histogram range for the
            given feature is not specified, and this method is not implemented. If the histogram
            is not configured to be calculated; or the given feature histogram range is already specified.
            no need to implement this method and NotImplementedError will not be raised.
        """

        raise NotImplementedError

    def failure_count(self, dataset_name: str, feature_name: str) -> int:
        """Return failed count for given dataset and feature.

        To perform data privacy min_count check, failure_count is always required.

        Args:
            dataset_name:
            feature_name:

        Returns: number of failure records, default to 0
        """
        return 0

    def finalize(self, fl_ctx: FLContext):
        """Called to finalize the Statistic calculator (close/release resources gracefully).

        After this call, the Learner will be destroyed.

        """
        pass
