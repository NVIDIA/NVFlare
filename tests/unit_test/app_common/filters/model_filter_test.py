# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from copy import deepcopy
from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.filters.model_filter import ModelFilter
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


class EditMetrics(ModelFilter):
    def process_model(self, model, fl_ctx):
        assert model.current_round == 3
        model.metrics = {"loss": 2.0}
        return model


@pytest.mark.parametrize("kind", ModelFilter.KINDS)
def test_model_filter_preserves_envelope_and_updates_metrics(kind):
    model = FLModel(params={"weight": 2.0}, metrics={"loss": 8.0}, start_round=1, current_round=3, total_rounds=10)
    if kind == DataKind.FL_MODEL:
        model.optimizer_params = {"step": 2}
        dxo = FLModelUtils.to_dxo(model)
        dxo.data["future_field"] = "preserve"
    elif kind == DataKind.METRICS:
        dxo = DXO(kind, {"loss": 8.0})
    else:
        model.params_type = ParamsType.DIFF if kind == DataKind.WEIGHT_DIFF else ParamsType.FULL
        dxo = from_shareable(FLModelUtils.to_shareable(model))
    dxo.meta.update(unknown={"x": 1}, NUM_STEPS_CURRENT_ROUND=13)
    shareable = dxo.to_shareable()
    shareable.set_header(AppConstants.CURRENT_ROUND, 3)
    shareable.set_header("unrelated", "keep")
    shareable.add_cookie("cookie", "keep")
    result = EditMetrics().process(shareable, FLContext())
    assert result is shareable
    assert result.get_header("unrelated") == "keep"
    assert result.get_cookie("cookie") == "keep"
    out = from_shareable(result)
    assert out.data_kind == kind
    assert out.meta["unknown"] == {"x": 1}
    assert out.meta["NUM_STEPS_CURRENT_ROUND"] == 13
    if kind == DataKind.FL_MODEL:
        decoded = FLModelUtils.from_dxo(out)
        assert decoded.start_round == 1
        assert decoded.optimizer_params == {"step": 2}
        assert out.data["future_field"] == "preserve"
    else:
        decoded = FLModelUtils.from_shareable(result)
    assert decoded.metrics == {"loss": 2.0}


def test_collection_leaves_are_not_decoded_as_parent():
    selected = DXO(DataKind.WEIGHTS, {"a": 1})
    sibling = DXO(DataKind.APP_DEFINED, {"b": 2})
    shareable = DXO(DataKind.COLLECTION, {"selected": selected, "sibling": sibling}).to_shareable()
    shareable.set_header(AppConstants.CURRENT_ROUND, 3)
    result = from_shareable(EditMetrics().process(shareable, FLContext()))
    assert result.data["selected"].meta[MetaKey.INITIAL_METRICS] == {"loss": 2.0}
    assert result.data["sibling"] is sibling


def test_conflicting_rounds_rejected():
    shareable = FLModelUtils.to_dxo(FLModel(params={"a": 1}, current_round=2)).to_shareable()
    shareable.set_header(AppConstants.CURRENT_ROUND, 3)
    with pytest.raises(ValueError, match="Conflicting"):
        EditMetrics().process(shareable, FLContext())


def test_legacy_aggregation_round_does_not_override_task_round():
    shareable = DXO(DataKind.WEIGHTS, {"a": 1}, {AppConstants.CURRENT_ROUND: 2}).to_shareable()
    shareable.set_header(AppConstants.CURRENT_ROUND, 3)
    result = EditMetrics().process(shareable, FLContext())
    assert FLModelUtils.from_shareable(result).current_round == 3
    assert from_shareable(result).meta[AppConstants.CURRENT_ROUND] == 2


def test_skip_and_failure_do_not_invoke_or_record_filter():
    filt = EditMetrics()
    filt.process_model = Mock(return_value=None)
    shareable = DXO(DataKind.WEIGHTS, {"a": 1}).to_shareable()
    assert filt.process(shareable, FLContext()) is shareable
    assert not from_shareable(shareable).get_filter_history()
    filt.process_model.reset_mock()
    shareable.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
    assert filt.process(shareable, FLContext()) is shareable
    filt.process_model.assert_not_called()


def test_legacy_metric_removal_and_native_start_round_round_trip():
    model = FLModel(params={"a": 1}, metrics=None, meta={MetaKey.INITIAL_METRICS: {"stale": 1}}, start_round=4)
    # Keep legacy callers that supply metrics solely via meta compatible.
    assert FLModelUtils.from_shareable(FLModelUtils.to_shareable(model)).metrics == {"stale": 1}
    filt = EditMetrics()

    def remove_metrics(model, fl_ctx):
        model.metrics = None
        return model

    filt.process_model = remove_metrics
    output = filt.process(FLModelUtils.to_shareable(model), FLContext())
    assert FLModelUtils.from_shareable(output).metrics is None
    assert FLModelUtils.from_dxo(FLModelUtils.to_dxo(model)).start_round == 4


@pytest.mark.parametrize("native", [False, True])
def test_none_preserves_input_with_borrowed_parameters_and_copied_metadata(native):
    model = FLModel(params={"w": [1.0]}, metrics={"loss": [2.0]}, meta={"custom": {"value": 3}})
    dxo = FLModelUtils.to_dxo(model) if native else from_shareable(FLModelUtils.to_shareable(model))
    shareable = dxo.to_shareable()
    before = deepcopy(shareable)
    filt = EditMetrics()

    def inspect_and_skip(decoded, fl_ctx):
        assert decoded.params is model.params
        assert decoded.metrics is not model.metrics
        assert decoded.meta["custom"] is not model.meta["custom"]
        decoded.metrics["loss"].append(4.0)
        decoded.meta["custom"]["value"] = 5
        return None

    filt.process_model = inspect_and_skip
    assert filt.process(shareable, FLContext()) is shareable
    assert shareable == before
    assert not from_shareable(shareable).get_filter_history()


@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("start_round", 2, "round identity"),
        ("current_round", 4, "round identity"),
        ("total_rounds", 11, "round identity"),
        ("params_type", ParamsType.DIFF, "parameter type"),
        ("optimizer_params", {"step": 1}, "optimizer parameters"),
        ("params", None, "data kind"),
    ],
)
def test_callback_cannot_change_legacy_identity_or_encoding(attribute, value, message):
    model = FLModel(params={"w": 1}, metrics={"loss": 2}, start_round=1, current_round=3, total_rounds=10)
    filt = EditMetrics()

    def invalid_change(decoded, fl_ctx):
        setattr(decoded, attribute, value)
        return decoded

    filt.process_model = invalid_change
    with pytest.raises(ValueError, match=message):
        filt.process(FLModelUtils.to_shareable(model), FLContext())


def test_invalid_callback_return_type_is_rejected():
    filt = EditMetrics()
    filt.process_model = Mock(return_value={"params": {"w": 1}})
    with pytest.raises(TypeError, match="FLModel or None"):
        filt.process(DXO(DataKind.WEIGHTS, {"w": 1}).to_shareable(), FLContext())


@pytest.mark.parametrize("header", [AppConstants.START_ROUND, AppConstants.CURRENT_ROUND, AppConstants.NUM_ROUNDS])
@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_invalid_round_header_never_reaches_callback(header, value):
    filt = EditMetrics()
    filt.process_model = Mock()
    shareable = DXO(DataKind.WEIGHTS, {"w": 1}).to_shareable()
    shareable.set_header(header, value)
    with pytest.raises(ValueError, match="Invalid"):
        filt.process(shareable, FLContext())
    filt.process_model.assert_not_called()


def test_nested_collection_preserves_encodings_and_unselected_leaf():
    weights = DXO(DataKind.WEIGHTS, {"w": 1})
    differences = DXO(DataKind.WEIGHT_DIFF, {"w": -1})
    excluded = DXO(DataKind.APP_DEFINED, {"application": "data"})
    nested = DXO(DataKind.COLLECTION, {"differences": differences, "excluded": excluded})
    shareable = DXO(DataKind.COLLECTION, {"weights": weights, "nested": nested}).to_shareable()
    shareable.set_header(AppConstants.CURRENT_ROUND, 3)
    result = from_shareable(EditMetrics().process(shareable, FLContext()))
    assert result.data_kind == DataKind.COLLECTION
    assert result.data["weights"].data_kind == DataKind.WEIGHTS
    assert result.data["weights"].data == {"w": 1}
    assert result.data["nested"].data_kind == DataKind.COLLECTION
    assert result.data["nested"].data["differences"].data_kind == DataKind.WEIGHT_DIFF
    assert result.data["nested"].data["differences"].data == {"w": -1}
    assert result.data["nested"].data["excluded"] is excluded
    for leaf in [result.data["weights"], result.data["nested"].data["differences"]]:
        assert leaf.meta[MetaKey.INITIAL_METRICS] == {"loss": 2.0}
        assert leaf.get_filter_history() == ["EditMetrics"]
    assert excluded.get_filter_history() is None


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("value", [True, 1.0])
def test_callback_cannot_change_round_type(native, value):
    model = FLModel(params={"w": 1}, current_round=1)
    shareable = FLModelUtils.to_dxo(model).to_shareable() if native else FLModelUtils.to_shareable(model)
    filt = EditMetrics()

    def invalid_change(decoded, fl_ctx):
        decoded.current_round = value
        return decoded

    filt.process_model = invalid_change
    with pytest.raises(ValueError, match="round identity"):
        filt.process(shareable, FLContext())


@pytest.mark.parametrize("header", [AppConstants.START_ROUND, AppConstants.CURRENT_ROUND, AppConstants.NUM_ROUNDS])
@pytest.mark.parametrize("value", [2, None, True, 1.0])
def test_callback_cannot_change_round_metadata_without_header(header, value):
    shareable = DXO(DataKind.WEIGHTS, {"w": 1}, {header: 1}).to_shareable()
    filt = EditMetrics()

    def change_metadata(decoded, fl_ctx):
        if value is None:
            decoded.meta.pop(header)
        else:
            decoded.meta[header] = value
        return decoded

    filt.process_model = change_metadata
    with pytest.raises(ValueError, match="round identity"):
        filt.process(shareable, FLContext())
    assert from_shareable(shareable).meta[header] == 1
