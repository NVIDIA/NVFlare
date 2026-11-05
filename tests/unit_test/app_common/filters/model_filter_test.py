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
        dxo = DXO(kind, {"loss": 8.0}, {MetaKey.INITIAL_METRICS: {"loss": 5.0}})
    else:
        model.params_type = ParamsType.DIFF if kind == DataKind.WEIGHT_DIFF else ParamsType.FULL
        dxo = from_shareable(FLModelUtils.to_shareable(model))
    dxo.meta.update(unknown={"x": 1}, NUM_STEPS_CURRENT_ROUND=13)
    dxo.add_filter_history("EarlierFilter")
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
    assert out.get_filter_history() == ["EarlierFilter", "EditMetrics"]
    if kind == DataKind.METRICS:
        assert out.meta[MetaKey.INITIAL_METRICS] == {"loss": 5.0}
    if kind == DataKind.FL_MODEL:
        decoded = FLModelUtils.from_dxo(out)
        assert decoded.start_round == 1
        assert decoded.optimizer_params == {"step": 2}
        assert out.data["future_field"] == "preserve"
    else:
        decoded = FLModelUtils.from_shareable(result)
    assert decoded.metrics == {"loss": 2.0}


@pytest.mark.parametrize("native", [False, True])
def test_fresh_model_preserves_transport_filter_history(native):
    class ReplaceModel(ModelFilter):
        def process_model(self, model, fl_ctx):
            if MetaKey.FILTER_HISTORY in model.meta:
                model.meta[MetaKey.FILTER_HISTORY].clear()
            return FLModel(params=model.params, start_round=model.start_round, meta={"replacement": True})

    model = FLModel(params={"w": 1})
    dxo = FLModelUtils.to_dxo(model) if native else from_shareable(FLModelUtils.to_shareable(model))
    dxo.add_filter_history("EarlierFilter")

    output = from_shareable(ReplaceModel().process(dxo.to_shareable(), FLContext()))

    assert output.get_filter_history() == ["EarlierFilter", "ReplaceModel"]
    decoded = FLModelUtils.from_dxo(output) if native else FLModelUtils.from_shareable(output.to_shareable())
    assert decoded.params == {"w": 1}
    assert decoded.meta["replacement"] is True


def test_model_metadata_headers_reach_collection_leaves():
    stale_meta = {MetaKey.VALIDATE_TYPE: "stale", MetaKey.JOB_ID: "old-job", MetaKey.SITE_NAME: "old-site"}
    legacy = DXO(DataKind.WEIGHTS, {"w": 1}, dict(stale_meta))
    native = FLModelUtils.to_dxo(FLModel(params={"w": 2}, meta=dict(stale_meta)))
    shareable = DXO(DataKind.COLLECTION, {"legacy": legacy, "native": native}).to_shareable()
    shareable.set_header(AppConstants.VALIDATE_TYPE, "before_train_validate")
    shareable.set_header(MetaKey.JOB_ID, "job-1")
    shareable.set_header(MetaKey.SITE_NAME, "site-1")
    shareable.set_header("unrelated", "keep")
    before = deepcopy(shareable)
    seen = []

    def inspect_and_skip(model, fl_ctx):
        assert model.meta[MetaKey.VALIDATE_TYPE] == "before_train_validate"
        assert model.meta[MetaKey.JOB_ID] == "job-1"
        assert model.meta[MetaKey.SITE_NAME] == "site-1"
        assert "unrelated" not in model.meta
        seen.append(model.params)
        model.meta[MetaKey.JOB_ID] = "callback-local"
        return None

    filt = EditMetrics()
    filt.process_model = inspect_and_skip
    assert filt.process(shareable, FLContext()) is shareable
    assert seen == [{"w": 1}, {"w": 2}]
    output = from_shareable(shareable)
    for name, leaf in output.data.items():
        assert leaf.to_dict() == from_shareable(before).data[name].to_dict()
    assert shareable.get_header(MetaKey.JOB_ID) == "job-1"


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


@pytest.mark.parametrize("metrics", [{"loss": 2.0}, {}])
def test_callback_metrics_override_stale_legacy_metadata(metrics):
    model = FLModel(
        params={"w": 1},
        metrics=metrics,
        meta={MetaKey.INITIAL_METRICS: {"loss": 8.0}, "custom": "preserve"},
    )
    filt = EditMetrics()

    def update_metrics(decoded, fl_ctx):
        decoded.metrics = metrics
        return decoded

    filt.process_model = update_metrics
    result = FLModelUtils.from_shareable(filt.process(FLModelUtils.to_shareable(model), FLContext()))
    assert result.metrics == metrics
    assert result.params_type == ParamsType.FULL
    assert result.meta["custom"] == "preserve"
    assert model.meta[MetaKey.INITIAL_METRICS] == {"loss": 8.0}


def test_filter_can_remove_legacy_metadata_metrics():
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


@pytest.mark.parametrize("native", [False, True])
def test_none_discards_top_level_edits_with_shared_nested_values(native):
    model = FLModel(params={"w": [1.0]}, metrics={"loss": [2.0]}, meta={"custom": {"value": 3}})
    dxo = FLModelUtils.to_dxo(model) if native else from_shareable(FLModelUtils.to_shareable(model))
    dxo.add_filter_history("EarlierFilter")
    shareable = dxo.to_shareable()
    before = deepcopy(shareable)
    filt = EditMetrics()

    def inspect_and_skip(decoded, fl_ctx):
        assert decoded.params is model.params
        assert decoded.metrics is not model.metrics
        assert decoded.metrics["loss"] is model.metrics["loss"]
        assert decoded.meta is not model.meta
        assert decoded.meta["custom"] is model.meta["custom"]
        decoded.metrics["loss"] = [4.0]
        decoded.meta["custom"] = {"value": 5}
        if not native:
            decoded.meta[MetaKey.FILTER_HISTORY].clear()
        return None

    filt.process_model = inspect_and_skip
    assert filt.process(shareable, FLContext()) is shareable
    assert shareable == before
    assert from_shareable(shareable).get_filter_history() == ["EarlierFilter"]


@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("current_round", 4, "round identity"),
        ("current_round", 3.0, "round identity"),
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


def test_invalid_round_header_never_reaches_callback():
    filt = EditMetrics()
    filt.process_model = Mock()
    shareable = DXO(DataKind.WEIGHTS, {"w": 1}).to_shareable()
    shareable.set_header(AppConstants.CURRENT_ROUND, True)
    with pytest.raises(ValueError, match="Invalid"):
        filt.process(shareable, FLContext())
    filt.process_model.assert_not_called()


def test_nested_collection_preserves_selected_and_unselected_leaves():
    selected = DXO(DataKind.WEIGHTS, {"w": 1})
    excluded = DXO(DataKind.APP_DEFINED, {"application": "data"})
    nested = DXO(DataKind.COLLECTION, {"selected": selected})
    shareable = DXO(DataKind.COLLECTION, {"nested": nested, "excluded": excluded}).to_shareable()
    shareable.set_header(AppConstants.CURRENT_ROUND, 3)

    result = from_shareable(EditMetrics().process(shareable, FLContext()))

    assert result.data_kind == DataKind.COLLECTION
    assert result.data["nested"].data_kind == DataKind.COLLECTION
    leaf = result.data["nested"].data["selected"]
    assert leaf.data_kind == DataKind.WEIGHTS
    assert leaf.data == {"w": 1}
    assert leaf.meta[MetaKey.INITIAL_METRICS] == {"loss": 2.0}
    assert leaf.get_filter_history() == ["EditMetrics"]
    assert result.data["excluded"] is excluded
    assert excluded.get_filter_history() is None


def test_callback_cannot_change_round_metadata_without_header():
    shareable = DXO(DataKind.WEIGHTS, {"w": 1}, {AppConstants.CURRENT_ROUND: 3}).to_shareable()
    filt = EditMetrics()

    def change_metadata(decoded, fl_ctx):
        decoded.meta[AppConstants.CURRENT_ROUND] = 4
        return decoded

    filt.process_model = change_metadata
    with pytest.raises(ValueError, match="round identity"):
        filt.process(shareable, FLContext())
    assert from_shareable(shareable).meta[AppConstants.CURRENT_ROUND] == 3


@pytest.mark.parametrize(
    "params_type,updated_params",
    [(ParamsType.FULL, None), (ParamsType.DIFF, None), (None, {"w": 2})],
)
def test_native_callback_rejects_inconsistent_parameters_without_changing_input(params_type, updated_params):
    model = FLModel(params_type=params_type, params={"w": 1} if params_type else None, metrics={"loss": 1.0})
    shareable = FLModelUtils.to_dxo(model).to_shareable()
    before = deepcopy(shareable)
    filt = EditMetrics()

    def change_parameters(decoded, fl_ctx):
        decoded.params = updated_params
        return decoded

    filt.process_model = change_parameters
    with pytest.raises(ValueError, match="params and params_type"):
        filt.process(shareable, FLContext())

    assert shareable == before
    output = from_shareable(shareable)
    assert not output.get_filter_history()
    assert FLModelUtils.from_dxo(output).params == model.params


@pytest.mark.parametrize("params_type,params", [(ParamsType.FULL, {}), (None, None)])
def test_native_callback_preserves_empty_parameters_and_metrics_only_models(params_type, params):
    model = FLModel(params_type=params_type, params=params, metrics={"loss": 1.0}, current_round=3)
    shareable = FLModelUtils.to_dxo(model).to_shareable()

    output = from_shareable(EditMetrics().process(shareable, FLContext()))
    decoded = FLModelUtils.from_dxo(output)

    assert output.data_kind == DataKind.FL_MODEL
    assert decoded.params_type == params_type
    assert decoded.params == params
    assert decoded.metrics == {"loss": 2.0}
    assert output.get_filter_history() == ["EditMetrics"]
