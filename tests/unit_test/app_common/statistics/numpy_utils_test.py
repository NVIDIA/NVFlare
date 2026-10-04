# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import pandas as pd
import pytest

from nvflare.app_common.abstract.statistics_spec import BinRange, DataType
from nvflare.app_common.statistics.numeric_stats import accumulate_hists
from nvflare.app_common.statistics.numpy_utils import dtype_to_data_type, get_std_histogram_buckets
from nvflare.app_opt.statistics.df.df_core_statistics import DFStatisticsCore


class TestDtypeToDataType:
    @pytest.mark.parametrize("dtype", [np.dtype("float32"), np.dtype("float64")])
    def test_float_dtypes(self, dtype):
        assert dtype_to_data_type(dtype) == DataType.FLOAT

    @pytest.mark.parametrize("dtype", [np.dtype("int32"), np.dtype("int64"), np.dtype("uint8")])
    def test_int_dtypes(self, dtype):
        assert dtype_to_data_type(dtype) == DataType.INT

    def test_bool_dtype(self):
        assert dtype_to_data_type(np.dtype("bool")) == DataType.INT

    def test_datetime_dtype(self):
        assert dtype_to_data_type(np.dtype("datetime64[ns]")) == DataType.DATETIME

    def test_timedelta_dtype(self):
        assert dtype_to_data_type(np.dtype("timedelta64[ns]")) == DataType.DATETIME

    def test_object_dtype(self):
        assert dtype_to_data_type(np.dtype("object")) == DataType.STRING

    def test_pandas_string_dtype(self):
        # pd.StringDtype has no .char attribute — this is the pandas 3.0 breaking case
        assert dtype_to_data_type(pd.StringDtype()) == DataType.STRING

    def test_pandas_string_dtype_inferred_from_series(self):
        # Simulate pandas 3.0 behavior where read_csv infers string columns as StringDtype
        s = pd.array(["a", "b", "c"], dtype=pd.StringDtype())
        assert dtype_to_data_type(s.dtype) == DataType.STRING

    def test_pandas_nullable_int_dtype(self):
        # pd.Int64Dtype is a nullable ExtensionDtype — must map to INT, not STRING
        assert dtype_to_data_type(pd.Int64Dtype()) == DataType.INT

    def test_pandas_nullable_float_dtype(self):
        # pd.Float64Dtype is a nullable ExtensionDtype — must map to FLOAT, not STRING
        assert dtype_to_data_type(pd.Float64Dtype()) == DataType.FLOAT

    def test_pandas_nullable_bool_dtype(self):
        # pd.BooleanDtype is a nullable ExtensionDtype — must map to INT, not STRING
        assert dtype_to_data_type(pd.BooleanDtype()) == DataType.INT


class TestHistogramInfinityCounts:
    @pytest.mark.parametrize(
        "values,num_bins",
        [
            ([-1.0, 0.0, 1.0], 3),
            ([-np.inf, -np.inf, 0.0], 3),
            ([0.0, np.inf, np.inf], 3),
            ([-np.inf, 0.0, np.inf], 3),
            ([-np.inf, 0.0, np.inf], 1),
            ([-np.inf, np.inf], 1),
            ([np.nan, -np.inf, 0.0, np.inf], 3),
            ([-2.0, -1.0, 0.0, 1.0, 2.0], 3),
            ([], 3),
        ],
    )
    def test_infinities_counted_once_in_edge_buckets(self, values, num_bins):
        # A read-only strided view verifies that input data is not modified.
        backing = np.repeat(np.asarray(values, dtype=np.float64), 2)
        nums = backing[::2]
        nums.setflags(write=False)
        counts, edges = np.histogram(nums[np.isfinite(nums)], bins=num_bins, range=(-1.0, 1.0))
        counts[0] += np.isneginf(nums).sum()
        counts[-1] += np.isposinf(nums).sum()
        expected_edges = edges.copy()
        if np.isneginf(nums).any():
            expected_edges[0] = -np.inf
        if np.isposinf(nums).any():
            expected_edges[-1] = np.inf

        buckets = get_std_histogram_buckets(nums, num_bins, BinRange(-1.0, 1.0))
        assert len(buckets) == num_bins
        np.testing.assert_array_equal([b.sample_count for b in buckets], counts)
        np.testing.assert_array_equal([b.low_value for b in buckets], expected_edges[:-1])
        np.testing.assert_array_equal([b.high_value for b in buckets], expected_edges[1:])
        assert sum(b.sample_count for b in buckets) == sum(counts)
        np.testing.assert_array_equal(nums, values)

    def test_dataframe_and_global_histogram_preserve_sample_count(self):
        global_histograms = {}
        expected_count = 0
        for values in ([-np.inf, 0.0, np.inf], [-np.inf, -0.5, 0.5, np.inf, np.nan]):
            statistics = DFStatisticsCore()
            statistics.data = {"train": pd.DataFrame({"value": values})}
            original = statistics.data["train"].copy(deep=True)
            histogram = statistics.histogram("train", "value", 3, -1.0, 1.0)
            expected_count += statistics.count("train", "value")
            global_histograms = accumulate_hists({"train": {"value": histogram}}, global_histograms)
            pd.testing.assert_frame_equal(statistics.data["train"], original)

        buckets = global_histograms["train"]["value"].bins
        assert len(buckets) == 3
        assert sum(b.sample_count for b in buckets) == expected_count == 7
        assert [b.sample_count for b in buckets] == [3, 1, 3]
