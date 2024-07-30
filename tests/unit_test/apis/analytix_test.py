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

import pytest

from nvflare.apis.analytix import _DATA_TYPE_KEY, AnalyticsData, AnalyticsDataType, LogWriterName, TrackConst
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.utils.analytix_utils import create_analytic_dxo

FROM_DXO_TEST_CASES = [
    ("hello", 3.0, 1, AnalyticsDataType.SCALAR),
    ("world", "text", 2, AnalyticsDataType.TEXT),
    ("dict", {"key": 1.0}, 3, AnalyticsDataType.SCALARS),
]

TO_DXO_TEST_CASES = [
    AnalyticsData(key="hello", value=3.0, data_type=AnalyticsDataType.SCALAR),
    AnalyticsData(key="world", value="text", step=2, path="/tmp/", data_type=AnalyticsDataType.TEXT),
    AnalyticsData(
        key="dict",
        value={"key": 1.0},
        step=3,
        sender=LogWriterName.MLFLOW,
        kwargs={"experiment_name": "test"},
        data_type=AnalyticsDataType.SCALARS,
    ),
]

FROM_DXO_INVALID_TEST_CASES = [
    (dict(), TypeError, f"expect dxo to be an instance of DXO, but got {type(dict())}."),
    (
        DXO(data_kind=DataKind.WEIGHTS, data={"w": 1.0}),
        KeyError,
        "'track_key'",
    ),
]

INVALID_TEST_CASES = [
    (
        dict(),
        1.0,
        AnalyticsDataType.SCALAR,
        None,
        TypeError,
        f"expect tag to be an instance of str, but got {type(dict())}.",
    ),
    (
        "tag",
        1.0,
        "scalar",
        None,
        TypeError,
        f"expect data_type to be an instance of AnalyticsDataType, but got {type('')}.",
    ),
]


class TestAnalytix:
    @pytest.mark.parametrize("tag,value,data_type,kwargs,expected_error,expected_msg", INVALID_TEST_CASES)
    def test_invalid(self, tag, value, data_type, kwargs, expected_error, expected_msg):
        with pytest.raises(expected_error, match=expected_msg):
            if not kwargs:
                _ = AnalyticsData(key=tag, value=value, data_type=data_type)
            else:
                _ = AnalyticsData(key=tag, value=value, data_type=data_type, **kwargs)

    @pytest.mark.parametrize("tag,value,step, data_type", FROM_DXO_TEST_CASES)
    def test_from_dxo(self, tag, value, step, data_type):
        dxo = create_analytic_dxo(tag=tag, value=value, data_type=data_type, global_step=step)
        assert dxo.get_meta_prop(_DATA_TYPE_KEY) == data_type
        result = AnalyticsData.from_dxo(dxo)
        assert result.tag == tag
        assert result.value == value
        assert result.step == step
        assert result.sender == LogWriterName.TORCH_TB

    @pytest.mark.parametrize("data", TO_DXO_TEST_CASES)
    def test_to_dxo(self, data: AnalyticsData):
        result = data.to_dxo()
        assert result.data_kind == DataKind.ANALYTIC
        assert result.data[TrackConst.TRACK_KEY] == data.tag
        assert result.data[TrackConst.TRACK_VALUE] == data.value
        if data.step:
            assert result.data[TrackConst.GLOBAL_STEP_KEY] == data.step
        if data.path:
            assert result.data[TrackConst.PATH_KEY] == data.path
        if data.kwargs:
            assert result.data[TrackConst.KWARGS_KEY] == data.kwargs

        assert result.get_meta_prop(_DATA_TYPE_KEY) == data.data_type
        assert result.get_meta_prop(TrackConst.TRACKER_KEY) == data.sender

    @pytest.mark.parametrize("dxo,expected_error,expected_msg", FROM_DXO_INVALID_TEST_CASES)
    def test_from_dxo_invalid(self, dxo, expected_error, expected_msg):
        with pytest.raises(expected_error, match=expected_msg):
            _ = AnalyticsData.from_dxo(dxo)
