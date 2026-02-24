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

from nvflare.app_common.abstract.statistics_spec import DataType
from nvflare.app_common.statistics.numpy_utils import dtype_to_data_type


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
