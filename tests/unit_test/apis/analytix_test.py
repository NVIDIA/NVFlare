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

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType, LogWriterName, TrackConst
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
        assert dxo.get_meta_prop(TrackConst.DATA_TYPE_KEY) == data_type
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

        assert result.get_meta_prop(TrackConst.DATA_TYPE_KEY) == data.data_type
        assert result.get_meta_prop(TrackConst.TRACKER_KEY) == data.sender

    @pytest.mark.parametrize("dxo,expected_error,expected_msg", FROM_DXO_INVALID_TEST_CASES)
    def test_from_dxo_invalid(self, dxo, expected_error, expected_msg):
        with pytest.raises(expected_error, match=expected_msg):
            _ = AnalyticsData.from_dxo(dxo)

    @pytest.mark.parametrize(
        "sender_data_type,sender,receiver,expected",
        [
            # TORCH_TB → MLFLOW: SCALAR/SCALARS map to METRIC/METRICS
            (AnalyticsDataType.SCALAR, LogWriterName.TORCH_TB, LogWriterName.MLFLOW, AnalyticsDataType.METRIC),
            (AnalyticsDataType.SCALARS, LogWriterName.TORCH_TB, LogWriterName.MLFLOW, AnalyticsDataType.METRICS),
            (AnalyticsDataType.TEXT, LogWriterName.TORCH_TB, LogWriterName.MLFLOW, AnalyticsDataType.TEXT),
            # TORCH_TB → WANDB: same mapping as TORCH_TB → MLFLOW (regression test for the typo fix)
            (AnalyticsDataType.SCALAR, LogWriterName.TORCH_TB, LogWriterName.WANDB, AnalyticsDataType.METRIC),
            (AnalyticsDataType.SCALARS, LogWriterName.TORCH_TB, LogWriterName.WANDB, AnalyticsDataType.METRICS),
            (AnalyticsDataType.TEXT, LogWriterName.TORCH_TB, LogWriterName.WANDB, AnalyticsDataType.TEXT),
            # MLFLOW → TORCH_TB: METRIC/METRICS map back to SCALAR/SCALARS
            (AnalyticsDataType.METRIC, LogWriterName.MLFLOW, LogWriterName.TORCH_TB, AnalyticsDataType.SCALAR),
            (AnalyticsDataType.METRICS, LogWriterName.MLFLOW, LogWriterName.TORCH_TB, AnalyticsDataType.SCALARS),
            (AnalyticsDataType.TEXT, LogWriterName.MLFLOW, LogWriterName.TORCH_TB, AnalyticsDataType.TEXT),
            # WANDB → TORCH_TB: same mapping as MLFLOW → TORCH_TB
            (AnalyticsDataType.METRIC, LogWriterName.WANDB, LogWriterName.TORCH_TB, AnalyticsDataType.SCALAR),
            (AnalyticsDataType.METRICS, LogWriterName.WANDB, LogWriterName.TORCH_TB, AnalyticsDataType.SCALARS),
            (AnalyticsDataType.TEXT, LogWriterName.WANDB, LogWriterName.TORCH_TB, AnalyticsDataType.TEXT),
            # MLFLOW ↔ WANDB: pass-through (shared METRIC/METRICS naming)
            (AnalyticsDataType.METRIC, LogWriterName.MLFLOW, LogWriterName.WANDB, AnalyticsDataType.METRIC),
            (AnalyticsDataType.METRIC, LogWriterName.WANDB, LogWriterName.MLFLOW, AnalyticsDataType.METRIC),
            # Same sender == receiver: pass-through
            (AnalyticsDataType.SCALAR, LogWriterName.TORCH_TB, LogWriterName.TORCH_TB, AnalyticsDataType.SCALAR),
        ],
    )
    def test_convert_data_type(self, sender_data_type, sender, receiver, expected):
        result = AnalyticsData.convert_data_type(sender_data_type, sender, receiver)
        assert result == expected

    def test_convert_data_type_never_returns_none(self):
        """All combinations of sender/receiver/data_type must return a concrete AnalyticsDataType."""
        for sender in LogWriterName:
            for receiver in LogWriterName:
                for dt in AnalyticsDataType:
                    result = AnalyticsData.convert_data_type(dt, sender, receiver)
                    assert result is not None, f"convert_data_type returned None for {dt}, {sender}, {receiver}"
                    assert isinstance(result, AnalyticsDataType)

    def test_from_dxo_torch_tb_to_wandb_preserves_data(self):
        """from_dxo must not return None when a TORCH_TB DXO is received by a WANDB receiver."""
        dxo = create_analytic_dxo(tag="loss", value=0.5, data_type=AnalyticsDataType.SCALAR, global_step=1)
        result = AnalyticsData.from_dxo(dxo, receiver=LogWriterName.WANDB)
        assert result is not None
        assert result.tag == "loss"
        assert result.value == 0.5
        assert result.data_type == AnalyticsDataType.METRIC

    def test_scalar_like_value_is_normalized(self):
        class ScalarLike:
            shape = ()

            def item(self):
                return 1.25

        data = AnalyticsData(key="loss", value=ScalarLike(), data_type=AnalyticsDataType.SCALAR)

        assert data.value == 1.25
        assert isinstance(data.value, float)

    @pytest.mark.parametrize("shape", [(1,), (1, 1), (2,)])
    def test_non_scalar_shape_is_rejected(self, shape):
        class NonScalarLike:
            def __init__(self, value_shape):
                self.shape = value_shape

            def item(self):
                raise AssertionError("item() must not be called for a non-scalar shape")

        with pytest.raises(TypeError, match="expect 'loss' value to be a numeric scalar"):
            AnalyticsData(key="loss", value=NonScalarLike(shape), data_type=AnalyticsDataType.SCALAR)

    @pytest.mark.parametrize("path", [1, 0])
    def test_invalid_path_error_reports_path_type(self, path):
        with pytest.raises(TypeError, match="expect path to be an instance of str, but got <class 'int'>."):
            AnalyticsData(key="message", value="hello", data_type=AnalyticsDataType.TEXT, path=path)

    @pytest.mark.parametrize("data_type", [AnalyticsDataType.SCALARS, AnalyticsDataType.METRICS])
    def test_numeric_dict_values_are_normalized(self, data_type):
        class ScalarLike:
            shape = ()

            def item(self):
                return 1.25

        data = AnalyticsData(
            key="losses",
            value={"train": ScalarLike(), "valid": 2},
            data_type=data_type,
        )

        assert data.value == {"train": 1.25, "valid": 2}
        assert isinstance(data.value["train"], float)

    @pytest.mark.parametrize("data_type", [AnalyticsDataType.SCALARS, AnalyticsDataType.METRICS])
    def test_numeric_dict_values_reject_non_scalars(self, data_type):
        with pytest.raises(
            TypeError,
            match="expect all values in 'losses' dict to be numeric scalars, "
            "but got '<class 'str'>' for key 'train'.",
        ):
            AnalyticsData(
                key="losses",
                value={"train": "bad_string"},
                data_type=data_type,
            )

    @pytest.mark.parametrize("step", [0, 10])
    def test_numpy_integer_step_is_normalized(self, step):
        np = pytest.importorskip("numpy")

        data = AnalyticsData(
            key="loss",
            value=np.float32(1.25),
            data_type=AnalyticsDataType.SCALAR,
            global_step=np.int64(step),
        )
        dxo = data.to_dxo()

        assert data.step == step
        assert type(data.step) is int
        assert type(data.kwargs[TrackConst.GLOBAL_STEP_KEY]) is int
        assert type(dxo.data[TrackConst.GLOBAL_STEP_KEY]) is int
        assert type(dxo.data[TrackConst.KWARGS_KEY][TrackConst.GLOBAL_STEP_KEY]) is int

    def test_numpy_float_step_is_rejected(self):
        np = pytest.importorskip("numpy")

        with pytest.raises(TypeError, match="expect step to be an instance of int"):
            AnalyticsData(
                key="loss",
                value=1.25,
                data_type=AnalyticsDataType.SCALAR,
                global_step=np.float64(1.0),
            )

    def test_numpy_numeric_values_are_normalized(self):
        np = pytest.importorskip("numpy")

        data = AnalyticsData(key="loss", value=np.float32(1.25), data_type=AnalyticsDataType.SCALAR)
        dxo = create_analytic_dxo(
            tag="loss", value=np.asarray(1.25, dtype=np.float32), data_type=AnalyticsDataType.SCALAR
        )
        scalars = AnalyticsData(
            key="losses",
            value={"train": np.float32(1.25), "valid": np.asarray(2, dtype=np.int32)},
            data_type=AnalyticsDataType.SCALARS,
        )
        metrics = AnalyticsData(
            key="metrics",
            value={"train": np.float32(1.25), "valid": np.asarray(2, dtype=np.int32)},
            data_type=AnalyticsDataType.METRICS,
        )

        assert data.value == pytest.approx(1.25)
        assert isinstance(data.value, float)
        assert dxo.data[TrackConst.TRACK_VALUE] == pytest.approx(1.25)
        assert isinstance(dxo.data[TrackConst.TRACK_VALUE], float)
        assert scalars.value["train"] == pytest.approx(1.25)
        assert scalars.value["valid"] == 2
        assert isinstance(scalars.value["train"], float)
        assert isinstance(scalars.value["valid"], int)
        assert metrics.value["train"] == pytest.approx(1.25)
        assert metrics.value["valid"] == 2
        assert isinstance(metrics.value["train"], float)
        assert isinstance(metrics.value["valid"], int)
