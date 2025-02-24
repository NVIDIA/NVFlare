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

from typing import Optional

import pytest

from nvflare.apis.analytix import AnalyticsDataType, LogWriterName, TrackConst
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.utils.analytix_utils import create_analytic_dxo, send_analytic_dxo

INVALID_TEST_CASES = [
    (list(), dict(), FLContext(), TypeError, f"expect comp to be an instance of FLComponent, but got {type(list())}"),
    (FLComponent(), dict(), FLContext(), TypeError, f"expect dxo to be an instance of DXO, but got {type(dict())}"),
    (
        FLComponent(),
        DXO(data={"k": "v"}, data_kind=DataKind.ANALYTIC),
        list(),
        TypeError,
        f"expect fl_ctx to be an instance of FLContext, but got {type(list())}",
    ),
]

INVALID_WRITE_TEST_CASES = [
    (
        list(),
        1.0,
        1,
        AnalyticsDataType.SCALAR,
        TypeError,
        f"expect tag to be an instance of str, but got {type(list())}",
    ),
    (
        "tag",
        list(),
        2,
        AnalyticsDataType.SCALAR,
        TypeError,
        f"expect 'tag' value to be an instance of float or int, but got '{type(list())}'",
    ),
    (
        list(),
        1.0,
        2,
        AnalyticsDataType.SCALARS,
        TypeError,
        f"expect tag to be an instance of str, but got {type(list())}",
    ),
    (
        "tag",
        1.0,
        3,
        AnalyticsDataType.SCALARS,
        TypeError,
        f"expect 'tag' value to be an instance of dict, but got '{type(1.0)}'",
    ),
    (list(), 1.0, 4, AnalyticsDataType.TEXT, TypeError, f"expect tag to be an instance of str, but got {type(list())}"),
    (
        "tag",
        1.0,
        5,
        AnalyticsDataType.TEXT,
        TypeError,
        f"expect 'tag' value to be an instance of str, but got '{type(1.0)}'",
    ),
    (
        list(),
        1.0,
        6,
        AnalyticsDataType.IMAGE,
        TypeError,
        f"expect tag to be an instance of str, but got {type(list())}",
    ),
]


class TestStreaming:
    @pytest.mark.parametrize("comp,dxo,fl_ctx,expected_error,expected_msg", INVALID_TEST_CASES)
    def test_invalid_send_analytic_dxo(self, comp, dxo, fl_ctx, expected_error, expected_msg):
        with pytest.raises(expected_error, match=expected_msg):
            send_analytic_dxo(comp=comp, dxo=dxo, fl_ctx=fl_ctx)

    @pytest.mark.parametrize("tag,value,step, data_type,expected_error,expected_msg", INVALID_WRITE_TEST_CASES)
    def test_invalid_write_func(self, tag, value, step, data_type, expected_error, expected_msg):
        with pytest.raises(expected_error, match=expected_msg):
            create_analytic_dxo(tag=tag, value=value, data_type=data_type, step=step, writer=LogWriterName.TORCH_TB)


def mock_add(tag: str, value, data_type: AnalyticsDataType, global_step: Optional[int] = None, **kwargs):
    # This mock_add tests writer behavior for MLflow and WandB too,
    # but to keep the signature of the func, we use writer=LogWriterName.TORCH_TB which shows up in expected_dxo_meta
    kwargs = kwargs if kwargs else {}
    if global_step is not None:
        if not isinstance(global_step, int):
            raise TypeError(f"Expect global step to be an instance of int, but got {type(global_step)}")
        kwargs[TrackConst.GLOBAL_STEP_KEY] = global_step
    dxo = create_analytic_dxo(tag=tag, value=value, data_type=data_type, writer=LogWriterName.TORCH_TB, **kwargs)
    return dxo


ANALYTICS_SENDER_TEST_CASES = [
    (
        "text",
        "textsample",
        AnalyticsDataType.TEXT,
        None,
        {},
        "ANALYTIC",
        {"track_key": "text", "track_value": "textsample"},
        {"analytics_data_type": AnalyticsDataType.TEXT, "tracker_key": LogWriterName.TORCH_TB},
    ),
    (
        "text",
        "textsample",
        AnalyticsDataType.TEXT,
        2,
        {},
        "ANALYTIC",
        {"track_key": "text", "track_value": "textsample", "global_step": 2, "analytics_kwargs": {"global_step": 2}},
        {"analytics_data_type": AnalyticsDataType.TEXT, "tracker_key": LogWriterName.TORCH_TB},
    ),
    (
        "text",
        "textsample",
        AnalyticsDataType.TEXT,
        3,
        {"extra_arg": 4},
        "ANALYTIC",
        {
            "track_key": "text",
            "track_value": "textsample",
            "global_step": 3,
            "analytics_kwargs": {"global_step": 3, "extra_arg": 4},
        },
        {"analytics_data_type": AnalyticsDataType.TEXT, "tracker_key": LogWriterName.TORCH_TB},
    ),
    (
        "set_tag_key_tag_name",
        "tagvalue",
        AnalyticsDataType.TAG,
        None,
        {},
        "ANALYTIC",
        {"track_key": "set_tag_key_tag_name", "track_value": "tagvalue"},
        {"analytics_data_type": AnalyticsDataType.TAG, "tracker_key": LogWriterName.TORCH_TB},
    ),
    (
        "log_metric_key_name",
        2.4,
        AnalyticsDataType.METRIC,
        20,
        {},
        "ANALYTIC",
        {
            "track_key": "log_metric_key_name",
            "track_value": 2.4,
            "global_step": 20,
            "analytics_kwargs": {"global_step": 20},
        },
        {"analytics_data_type": AnalyticsDataType.METRIC, "tracker_key": LogWriterName.TORCH_TB},
    ),
    (  # for WandBWriter
        "metrics",
        {"train_loss": 2.4},
        AnalyticsDataType.METRICS,
        20,
        {},
        "ANALYTIC",
        {
            "track_key": "metrics",
            "track_value": {"train_loss": 2.4},
            "global_step": 20,
            "analytics_kwargs": {"global_step": 20},
        },
        {"analytics_data_type": AnalyticsDataType.METRICS, "tracker_key": LogWriterName.TORCH_TB},
    ),
]

INVALID_SENDER_TEST_CASES = [
    (
        "text",
        "textsample",
        AnalyticsDataType.TEXT,
        None,
        {"global_step": 3, "extra_arg": 4},
        TypeError,
        "got multiple values for keyword argument 'global_step'",
    ),
]


class TestAnalyticsSender:
    @pytest.mark.parametrize(
        "tag,value,data_type,global_step,kwargs,expected_dxo_data_kind,expected_dxo_data,expected_dxo_meta",
        ANALYTICS_SENDER_TEST_CASES,
    )
    def test_add(
        self, tag, value, data_type, global_step, kwargs, expected_dxo_data_kind, expected_dxo_data, expected_dxo_meta
    ):
        dxo = mock_add(tag=tag, value=value, data_type=data_type, global_step=global_step, **kwargs)
        assert dxo.data_kind == expected_dxo_data_kind
        assert dxo.data == expected_dxo_data
        assert dxo.meta == expected_dxo_meta

    # Since global_step is already being set, it cannot also be in kwargs.
    @pytest.mark.parametrize(
        "tag,value,data_type,global_step,kwargs,expected_error,expected_msg",
        INVALID_SENDER_TEST_CASES,
    )
    def test_add_invalid(self, tag, value, data_type, global_step, kwargs, expected_error, expected_msg):
        with pytest.raises(expected_error, match=expected_msg):
            dxo = mock_add(tag=tag, value=value, data_type=data_type, global_step=global_step, **kwargs)
