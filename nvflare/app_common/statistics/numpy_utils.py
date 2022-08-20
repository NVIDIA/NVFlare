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

import json
from typing import List, Optional

import numpy as np

from nvflare.app_common.abstract.statistics_spec import Bin, BinRange, DataType


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def dtype_to_data_type(dtype) -> DataType:
    if dtype.char in np.typecodes["AllFloat"]:
        return DataType.FLOAT
    elif dtype.char in np.typecodes["AllInteger"] or dtype == bool:
        return DataType.INT
    elif np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64):
        return DataType.DATETIME
    else:
        return DataType.STRING


def get_std_histogram_buckets(nums: np.ndarray, num_bins: int = 10, br: Optional[BinRange] = None):
    num_posinf = len(nums[np.isposinf(nums)])
    num_neginf = len(nums[np.isneginf(nums)])
    if br:
        counts, buckets = np.histogram(nums, bins=num_bins, range=(br.min_value, br.max_value))
    else:
        counts, buckets = np.histogram(nums, bins=num_bins)

    histogram_buckets: List[Bin] = []
    for bucket_count in range(len(counts)):
        # Add any negative or positive infinities to the first and last
        # buckets in the histogram.
        bucket_low_value = buckets[bucket_count]
        bucket_high_value = buckets[bucket_count + 1]
        bucket_sample_count = counts[bucket_count]
        if bucket_count == 0 and num_neginf > 0:
            bucket_low_value = float("-inf")
            bucket_sample_count += num_neginf
        elif bucket_count == len(counts) - 1 and num_posinf > 0:
            bucket_high_value = float("inf")
            bucket_sample_count += num_posinf

        histogram_buckets.append(
            Bin(low_value=bucket_low_value, high_value=bucket_high_value, sample_count=bucket_sample_count)
        )

    if buckets is not None and len(buckets) > 0:
        bucket = None
        if num_neginf:
            bucket = Bin(low_value=float("-inf"), high_value=float("-inf"), sample_count=num_neginf)
        if num_posinf:
            bucket = Bin(low_value=float("inf"), high_value=float("inf"), sample_count=num_posinf)

        if bucket:
            histogram_buckets.append(bucket)

    return histogram_buckets
