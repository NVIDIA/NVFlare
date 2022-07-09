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

from enum import Enum
from typing import Dict, List, NamedTuple, Optional


class DataType(Enum):
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


class HistogramType(Enum):
    STANDARD = 0
    QUANTILES = 1


class Histogram(NamedTuple):
    # A list of buckets in the histogram, sorted from lowest bucket to highest bucket.
    bins: List[Bin]

    # The type of the histogram. A standard histogram has equal-width buckets.
    # The quantiles type is used for when the histogram message is used to store
    # quantile information (by using equal-count buckets with variable widths).

    # The type of the histogram.
    hist_type: HistogramType

    # An optional descriptive name of the histogram, to be used for labeling.
    hist_name: Optional[str] = None


class NumericStatistics(NamedTuple):
    mean: float
    sum: float
    stddev: float
    count: int
    histograms: List[Histogram]


class FeatureStatistics(NamedTuple):
    name: str
    data_type: DataType
    num_stats: Optional[NumericStatistics] = None


class DatasetStatistics(NamedTuple):
    name: str
    num_examples: int
    features: List[FeatureStatistics]
