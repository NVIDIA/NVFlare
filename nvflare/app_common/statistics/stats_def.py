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

from enum import IntEnum
from typing import List, NamedTuple, Optional, TypeVar

T = TypeVar("T")


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

    def to_json(self):
        bin_json = [b.to_json() for b in self.bins]
        return {"hist_name": self.hist_name, "hist_type": HistogramType(self.hist_type).name, "bins": bin_json}


class Feature(NamedTuple):
    feature_name: str
    data_type: DataType
