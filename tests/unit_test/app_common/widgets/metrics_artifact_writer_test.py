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

import json
import math
import os
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter

_METRICS_AGGREGATION_INFO = "metrics_aggregation_info"
_METRICS_DIR = "metrics"
_SUMMARY_FILE = "metrics_summary.json"
_ROUND_FILE = "round_metrics.jsonl"


class _UnsafeObject:
    def __repr__(self):
        raise AssertionError("unsafe metric object should not be stringified")

    def __str__(self):
        raise AssertionError("unsafe metric object should not be stringified")


class generic:
    __module__ = "numpy"


class _SpoofedNumpyScalar(generic):
    __module__ = "numpy"

    def item(self):
        raise AssertionError("spoofed metric object should not have item() called")


def _make_fl_ctx(run_dir):
    workspace = Mock()
    workspace.get_run_dir.return_value = str(run_dir)
    engine = Mock()
    engine.get_workspace.return_value = workspace

    fl_ctx = FLContext()
    fl_ctx.get_engine = Mock(return_value=engine)
    fl_ctx.get_job_id = Mock(return_value="job-1")
    return fl_ctx


def _artifact_paths(run_dir):
    metrics_dir = run_dir / _METRICS_DIR
    return metrics_dir, metrics_dir / _SUMMARY_FILE, metrics_dir / _ROUND_FILE


def _aggregation_info(sites=None, key_metric=None, site_weights=None, use_contribution_sites=True):
    info = {
        "metric_source": "client_reported_flmodel_metrics",
        "metric_split": "validation",
        "aggregation": {
            "method": "weighted_average",
            "weight_key": FLMetaKey.NUM_STEPS_CURRENT_ROUND,
            "metric_policy": "finite_numeric_metrics_only_per_key_denominator",
        },
    }
    if sites is not None:
        info["sites"] = sites
    if key_metric is not None:
        info["key_metric"] = key_metric
    if site_weights is not None:
        info["site_weights"] = site_weights
    if not use_contribution_sites:
        info["use_contribution_sites"] = False
    return info


def _site(name, metrics, weight=1.0):
    return {
        "name": name,
        "metrics": metrics,
        "weight": weight,
        "weight_key": FLMetaKey.NUM_STEPS_CURRENT_ROUND,
    }


def _record_round(
    writer,
    fl_ctx,
    round_num,
    metrics,
    sites=None,
    key_metric=None,
    site_weights=None,
    use_contribution_sites=True,
):
    aggr_result = FLModel(
        metrics=metrics,
        current_round=round_num,
        meta={
            AppConstants.CURRENT_ROUND: round_num,
            _METRICS_AGGREGATION_INFO: _aggregation_info(
                sites=sites,
                key_metric=key_metric,
                site_weights=site_weights,
                use_contribution_sites=use_contribution_sites,
            ),
        },
    )
    try:
        fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
        writer.handle_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
    finally:
        fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, None, private=True, sticky=False)


def _record_shareable_round(writer, fl_ctx, round_num, metrics):
    shareable = DXO(data_kind=DataKind.METRICS, data=metrics).to_shareable()
    shareable.set_header(AppConstants.CURRENT_ROUND, round_num)
    try:
        fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, shareable, private=True, sticky=False)
        writer.handle_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
    finally:
        fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, None, private=True, sticky=False)


def _record_contribution(writer, fl_ctx, round_num, site_name, metrics, weight=1):
    result = FLModel(
        metrics=metrics,
        current_round=round_num,
        meta={FLMetaKey.SITE_NAME: site_name, FLMetaKey.NUM_STEPS_CURRENT_ROUND: weight},
    )
    shareable = FLModelUtils.to_shareable(result)
    try:
        fl_ctx.set_prop(AppConstants.TRAINING_RESULT, shareable, private=True, sticky=False)
        writer.handle_event(AppEventType.AFTER_CONTRIBUTION_ACCEPT, fl_ctx)
    finally:
        fl_ctx.set_prop(AppConstants.TRAINING_RESULT, None, private=True, sticky=False)


def _finish_run(writer, fl_ctx):
    writer.handle_event(EventType.END_RUN, fl_ctx)


def _record_best_selection(writer, fl_ctx, selection):
    try:
        fl_ctx.set_prop(AppConstants.METRICS_SELECTION_INFO, selection, private=True, sticky=False)
        writer.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_ctx)
    finally:
        fl_ctx.set_prop(AppConstants.METRICS_SELECTION_INFO, None, private=True, sticky=False)


def _read_summary(run_dir):
    _, summary_path, _ = _artifact_paths(run_dir)
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def _read_rounds(run_dir):
    _, _, round_path = _artifact_paths(run_dir)
    with open(round_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _metrics_to_dict(metrics):
    assert isinstance(metrics, list)
    return {entry["name"]: entry["value"] for entry in metrics}


def _collect_object_keys(value):
    if isinstance(value, dict):
        keys = set(value.keys())
        for child in value.values():
            keys.update(_collect_object_keys(child))
        return keys
    if isinstance(value, list):
        keys = set()
        for child in value:
            keys.update(_collect_object_keys(child))
        return keys
    return set()


def _collect_metric_names(value):
    names = set()
    if isinstance(value, dict):
        if set(value.keys()) >= {"name", "value"}:
            names.add(value["name"])
        for child in value.values():
            names.update(_collect_metric_names(child))
    elif isinstance(value, list):
        for child in value:
            names.update(_collect_metric_names(child))
    return names


class TestMetricsArtifactWriterAggregationEvents:
    def test_writes_summary_and_jsonl_from_aggregation_events(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        key_metric = {"name": "accuracy", "mode": "max", "mode_source": "derived_from_existing_selection_logic"}
        _record_round(
            writer,
            fl_ctx,
            1,
            {"accuracy": 0.82, "loss": 0.40},
            sites=[
                _site("site-1", {"accuracy": 0.80, "loss": 0.38}, weight=10),
                _site("site-2", {"accuracy": 0.84, "loss": 0.42}, weight=10),
            ],
            key_metric=key_metric,
        )
        _record_round(
            writer,
            fl_ctx,
            2,
            {"accuracy": 0.81, "loss": 0.35},
            sites=[
                _site("site-1", {"accuracy": 0.79, "loss": 0.34}, weight=10),
                _site("site-2", {"accuracy": 0.83, "loss": 0.36}, weight=10),
            ],
            key_metric=key_metric,
        )
        _finish_run(writer, fl_ctx)

        metrics_dir, summary_path, round_path = _artifact_paths(run_dir)
        assert metrics_dir.is_dir()
        assert summary_path.is_file()
        assert round_path.is_file()

        summary = _read_summary(run_dir)
        assert summary["status"] == "metrics_reported"
        assert summary["final_round"] == 2
        assert _metrics_to_dict(summary["final_aggregated_metrics"]) == {"accuracy": 0.81, "loss": 0.35}
        assert summary["key_metric"] == key_metric
        assert summary["round_metrics_file"] == _ROUND_FILE
        assert "best_round" not in summary

        rounds = _read_rounds(run_dir)
        assert [record["round"] for record in rounds] == [1, 2]
        assert _metrics_to_dict(rounds[0]["aggregated_metrics"]) == {"accuracy": 0.82, "loss": 0.40}
        assert rounds[0]["sites"][0]["name"] == "site-1"
        assert _metrics_to_dict(rounds[0]["sites"][0]["metrics"]) == {"accuracy": 0.80, "loss": 0.38}
        assert rounds[0]["key_metric"] == key_metric
        assert rounds[0]["aggregation"]["method"] == "weighted_average"

    def test_key_metric_metadata_does_not_make_writer_select_best_round(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        key_metric = {"name": "score", "mode": "max", "mode_source": "workflow_metadata"}
        for index, value in enumerate([0.70, 0.75, 0.72], start=1):
            _record_round(writer, fl_ctx, index, {"score": value, "aux": index}, key_metric=key_metric)
        _finish_run(writer, fl_ctx)

        summary = _read_summary(run_dir)
        assert summary["key_metric"] == key_metric
        assert "best_round" not in summary
        assert "best_metrics" not in summary
        assert "best_aggregated_metrics" not in summary

    def test_records_shareable_aggregation_result_metrics(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_shareable_round(writer, fl_ctx, 0, {"accuracy": 0.81, "loss": 0.35})
        _finish_run(writer, fl_ctx)

        summary = _read_summary(run_dir)
        assert summary["final_round"] == 0
        assert _metrics_to_dict(summary["final_aggregated_metrics"]) == {"accuracy": 0.81, "loss": 0.35}

        rounds = _read_rounds(run_dir)
        assert rounds[0]["round"] == 0
        assert _metrics_to_dict(rounds[0]["aggregated_metrics"]) == {"accuracy": 0.81, "loss": 0.35}

    def test_records_official_best_selection_metadata(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_round(writer, fl_ctx, 1, {"score": 0.3})
        _record_best_selection(
            writer,
            fl_ctx,
            {
                "source": "IntimeModelSelector",
                "metric_source": "initial_metrics",
                "key_metric": {"name": "score", "mode": "min", "mode_source": "IntimeModelSelector"},
                "best_round": 1,
                "best_metrics": {"score": 0.3},
            },
        )
        _finish_run(writer, fl_ctx)

        summary = _read_summary(run_dir)
        assert summary["key_metric"] == {
            "name": "score",
            "mode": "min",
            "mode_source": "IntimeModelSelector",
        }
        assert summary["best_round"] == 1
        assert _metrics_to_dict(summary["best_metrics"]) == {"score": 0.3}
        assert summary["best_metric_source"] == "IntimeModelSelector"
        assert summary["best_metric_detail_source"] == "initial_metrics"

    def test_summary_key_metric_prefers_best_selection_metadata(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        selection_key_metric = {"name": "accuracy", "mode": "max", "mode_source": "selector"}
        aggregation_key_metric = {"name": "loss", "mode": "min", "mode_source": "stop_condition"}
        _record_best_selection(
            writer,
            fl_ctx,
            {
                "source": "IntimeModelSelector",
                "key_metric": selection_key_metric,
                "best_round": 0,
                "best_metrics": {"accuracy": 0.9},
            },
        )
        _record_round(writer, fl_ctx, 1, {"loss": 0.4, "accuracy": 0.8}, key_metric=aggregation_key_metric)
        _finish_run(writer, fl_ctx)

        summary = _read_summary(run_dir)
        assert summary["key_metric"] == selection_key_metric
        assert summary["best_round"] == 0
        assert _metrics_to_dict(summary["best_metrics"]) == {"accuracy": 0.9}

        rounds = _read_rounds(run_dir)
        assert rounds[0]["key_metric"] == aggregation_key_metric

    def test_contribution_with_none_meta_does_not_crash(self, tmp_path, monkeypatch):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        monkeypatch.setattr(
            FLModelUtils,
            "from_shareable",
            lambda _: FLModel(metrics={"loss": 0.3}, current_round=1, meta=None),
        )

        writer.handle_event(EventType.START_RUN, fl_ctx)
        fl_ctx.set_prop(AppConstants.TRAINING_RESULT, object(), private=True, sticky=False)
        writer.handle_event(AppEventType.AFTER_CONTRIBUTION_ACCEPT, fl_ctx)
        fl_ctx.set_prop(AppConstants.TRAINING_RESULT, None, private=True, sticky=False)
        _record_round(writer, fl_ctx, 1, {"loss": 0.3})
        _finish_run(writer, fl_ctx)

        rounds = _read_rounds(run_dir)
        assert rounds[0]["sites"][0]["name"] == AppConstants.CLIENT_UNKNOWN
        assert _metrics_to_dict(rounds[0]["sites"][0]["metrics"]) == {"loss": 0.3}

    def test_bool_site_metrics_are_recorded_without_synthetic_aggregation(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_round(
            writer,
            fl_ctx,
            1,
            metrics={},
            sites=[
                _site("site-1", {"accepted": True}, weight=3),
                _site("site-2", {"accepted": False}, weight=1),
            ],
        )
        _finish_run(writer, fl_ctx)

        rounds = _read_rounds(run_dir)
        assert _metrics_to_dict(rounds[0]["sites"][0]["metrics"]) == {"accepted": True}
        assert _metrics_to_dict(rounds[0]["sites"][1]["metrics"]) == {"accepted": False}
        assert rounds[0]["aggregated_metrics"] == []

        summary = _read_summary(run_dir)
        assert summary["final_aggregated_metrics"] == []

    def test_negative_weights_are_ignored_without_affecting_official_aggregation(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_contribution(writer, fl_ctx, 1, "site-1", {"score": -1.0}, weight=-100)
        _record_contribution(writer, fl_ctx, 1, "site-2", {"score": 1.0}, weight=1)
        _record_round(writer, fl_ctx, 1, metrics={"score": 0.5})
        _finish_run(writer, fl_ctx)

        rounds = _read_rounds(run_dir)
        assert _metrics_to_dict(rounds[0]["aggregated_metrics"]) == {"score": 0.5}
        assert _metrics_to_dict(rounds[0]["sites"][0]["metrics"]) == {"score": -1.0}
        assert "weight" not in rounds[0]["sites"][0]
        assert rounds[0]["sites"][1]["weight"] == 1

    def test_zero_weights_are_ignored_without_affecting_official_aggregation(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_contribution(writer, fl_ctx, 1, "site-1", {"score": 0.0}, weight=0)
        _record_contribution(writer, fl_ctx, 1, "site-2", {"score": 1.0}, weight=1)
        _record_round(writer, fl_ctx, 1, metrics={"score": 0.5})
        _finish_run(writer, fl_ctx)

        rounds = _read_rounds(run_dir)
        assert _metrics_to_dict(rounds[0]["aggregated_metrics"]) == {"score": 0.5}
        assert "weight" not in rounds[0]["sites"][0]
        assert rounds[0]["sites"][1]["weight"] == 1

    def test_numpy_scalar_metrics_are_normalized_to_json_scalars(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_contribution(
            writer,
            fl_ctx,
            1,
            "site-1",
            {"loss": np.float32(0.2), "count": np.int64(3), "accepted": np.bool_(True)},
            weight=np.int64(5),
        )
        _record_round(
            writer,
            fl_ctx,
            1,
            {"loss": np.float32(0.2), "count": np.int64(3), "accepted": np.bool_(True)},
        )
        _finish_run(writer, fl_ctx)

        rounds = _read_rounds(run_dir)
        assert rounds[0]["skipped_metrics"] == []
        site_metrics = _metrics_to_dict(rounds[0]["sites"][0]["metrics"])
        assert math.isclose(site_metrics["loss"], 0.2, rel_tol=1e-6)
        assert site_metrics["count"] == 3
        assert site_metrics["accepted"] is True
        assert rounds[0]["sites"][0]["weight"] == 5

        summary_metrics = _metrics_to_dict(_read_summary(run_dir)["final_aggregated_metrics"])
        assert math.isclose(summary_metrics["loss"], 0.2, rel_tol=1e-6)
        assert summary_metrics["count"] == 3
        assert summary_metrics["accepted"] is True

    def test_numpy_scalar_normalization_does_not_call_item_on_spoofed_objects(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_round(writer, fl_ctx, 1, {"loss": _SpoofedNumpyScalar()})
        _finish_run(writer, fl_ctx)

        rounds = _read_rounds(run_dir)
        assert rounds[0]["aggregated_metrics"] == []
        assert rounds[0]["sites"] == []
        assert rounds[0]["skipped_metrics"] == [{"name": "loss", "reason": "unsupported_type"}]

    def test_dynamic_and_dangerous_metric_names_are_array_values_not_json_keys_or_paths(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)
        dangerous_metrics = {
            "__proto__": 0.1,
            "constructor": 0.2,
            "../outside/loss": 0.3,
            "fold[0].score": 0.4,
            "line\nbreak\x00metric": 0.5,
        }

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_round(
            writer,
            fl_ctx,
            1,
            dangerous_metrics,
            sites=[_site("site/../1", dangerous_metrics, weight=1)],
            key_metric={"name": "__proto__", "mode": "max", "mode_source": "workflow_metadata"},
        )
        _finish_run(writer, fl_ctx)

        summary = _read_summary(run_dir)
        rounds = _read_rounds(run_dir)
        combined = {"summary": summary, "rounds": rounds}
        object_keys = _collect_object_keys(combined)
        for metric_name in dangerous_metrics:
            assert metric_name not in object_keys

        metric_names = _collect_metric_names(combined)
        assert {"__proto__", "constructor", "../outside/loss", "fold[0].score"} <= metric_names
        assert all("\n" not in name and "\x00" not in name for name in metric_names)

        written_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file())
        assert written_files == [f"{_METRICS_DIR}/{_SUMMARY_FILE}", f"{_METRICS_DIR}/{_ROUND_FILE}"]

    def test_invalid_metric_values_are_skipped_with_metadata_without_unsafe_stringification(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)
        unsafe_value = _UnsafeObject()

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_round(
            writer,
            fl_ctx,
            1,
            {
                "loss": 0.25,
                "nan_metric": float("nan"),
                "inf_metric": float("inf"),
                "debug_blob": unsafe_value,
            },
            sites=[
                _site(
                    "site-1",
                    {
                        "loss": 0.20,
                        "nan_metric": float("nan"),
                        "debug_blob": unsafe_value,
                        "raw_bytes": b"abc",
                    },
                    weight=1,
                )
            ],
        )
        _finish_run(writer, fl_ctx)

        _, summary_path, round_path = _artifact_paths(run_dir)
        assert "NaN" not in summary_path.read_text(encoding="utf-8")
        assert "Infinity" not in summary_path.read_text(encoding="utf-8")
        assert "NaN" not in round_path.read_text(encoding="utf-8")
        assert "Infinity" not in round_path.read_text(encoding="utf-8")

        summary = _read_summary(run_dir)
        rounds = _read_rounds(run_dir)
        assert _metrics_to_dict(summary["final_aggregated_metrics"]) == {"loss": 0.25}
        assert _metrics_to_dict(rounds[0]["aggregated_metrics"]) == {"loss": 0.25}
        assert _metrics_to_dict(rounds[0]["sites"][0]["metrics"]) == {"loss": 0.20}

        skipped = rounds[0]["skipped_metrics"]
        skipped_names = {entry["name"] for entry in skipped}
        assert {"nan_metric", "inf_metric", "debug_blob", "raw_bytes"} <= skipped_names
        assert all(entry.get("reason") for entry in skipped)

    def test_per_site_skipped_metrics_from_contribution_events_are_preserved(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)
        unsafe_value = _UnsafeObject()

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_contribution(
            writer,
            fl_ctx,
            1,
            "site-1",
            {"loss": 0.2, "bad_object": unsafe_value, "huge_int": 10**400, 2: 0.1},
            weight=3,
        )
        _record_round(writer, fl_ctx, 1, {"loss": 0.2})
        _finish_run(writer, fl_ctx)

        rounds = _read_rounds(run_dir)
        assert _metrics_to_dict(rounds[0]["sites"][0]["metrics"]) == {"loss": 0.2}
        skipped = {(entry.get("site"), entry["name"], entry["reason"]) for entry in rounds[0]["skipped_metrics"]}
        assert ("site-1", "bad_object", "unsupported_type") in skipped
        assert ("site-1", "huge_int", "number_too_large") in skipped
        assert ("site-1", "", "non_string_metric_name") in skipped

    def test_effective_site_weights_are_recorded_without_recomputing_aggregation(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_contribution(writer, fl_ctx, 1, "site-1", {"score": 0.0}, weight=100)
        _record_contribution(writer, fl_ctx, 1, "site-2", {"score": 1.0}, weight=100)
        _record_round(
            writer,
            fl_ctx,
            1,
            metrics={"score": 0.5},
            site_weights=[
                {"name": "site-1", "weight": 1.0, "weight_key": "effective_fedavg_metric_weight"},
                {"name": "site-2", "weight": 3.0, "weight_key": "effective_fedavg_metric_weight"},
            ],
        )
        _finish_run(writer, fl_ctx)

        rounds = _read_rounds(run_dir)
        assert _metrics_to_dict(rounds[0]["aggregated_metrics"]) == {"score": 0.5}
        assert [site["weight"] for site in rounds[0]["sites"]] == [1.0, 3.0]
        assert {site["weight_key"] for site in rounds[0]["sites"]} == {"effective_fedavg_metric_weight"}

    def test_custom_aggregator_metadata_can_disable_contribution_site_fallback(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_contribution(writer, fl_ctx, 1, "site-1", {"score": 0.1}, weight=1)
        _record_round(writer, fl_ctx, 1, {"score": 0.9}, use_contribution_sites=False)
        _finish_run(writer, fl_ctx)

        rounds = _read_rounds(run_dir)
        assert _metrics_to_dict(rounds[0]["aggregated_metrics"]) == {"score": 0.9}
        assert rounds[0]["sites"] == []

    def test_site_and_round_metric_limits_are_enforced(self, tmp_path):
        writer = MetricsArtifactWriter(limits={"max_sites_per_round": 1, "max_site_metric_records_per_round": 1})
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_round(
            writer,
            fl_ctx,
            1,
            metrics={},
            sites=[
                _site("site-1", {"m1": 1, "m2": 2}, weight=1),
                _site("site-2", {"m1": 3}, weight=1),
            ],
        )
        _finish_run(writer, fl_ctx)

        rounds = _read_rounds(run_dir)
        assert len(rounds[0]["sites"]) == 1
        assert _metrics_to_dict(rounds[0]["sites"][0]["metrics"]) == {"m1": 1}
        skipped_reasons = {entry["reason"] for entry in rounds[0]["skipped_metrics"]}
        assert {"too_many_metrics", "too_many_sites"} <= skipped_reasons

    def test_oversized_round_record_is_truncated_before_write(self, tmp_path):
        writer = MetricsArtifactWriter(
            limits={"max_round_record_bytes": 500, "max_string_value_length": 200, "max_metrics_per_site_per_round": 10}
        )
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)
        large_metrics = {f"metric_{i}": "x" * 100 for i in range(20)}

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_round(writer, fl_ctx, 1, {"score": 1.0}, sites=[_site("site-1", large_metrics, weight=1)])
        _finish_run(writer, fl_ctx)

        _, _, round_path = _artifact_paths(run_dir)
        assert round_path.stat().st_size <= 500
        rounds = _read_rounds(run_dir)
        assert rounds[0]["round"] == 1

    @pytest.mark.parametrize(
        "event_payload",
        [
            None,
            FLModel(metrics=None, current_round=1),
            FLModel(metrics={}, current_round=1),
        ],
    )
    def test_no_metrics_do_not_create_synthetic_artifacts(self, tmp_path, event_payload):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        writer.handle_event(AppEventType.VALIDATION_RESULT_RECEIVED, fl_ctx)
        if event_payload is not None:
            try:
                fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, event_payload, private=True, sticky=False)
                writer.handle_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
            finally:
                fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, None, private=True, sticky=False)
        _finish_run(writer, fl_ctx)

        _, summary_path, round_path = _artifact_paths(run_dir)
        assert not summary_path.exists()
        assert not round_path.exists()

    def test_best_selection_without_round_metrics_does_not_create_artifacts(self, tmp_path):
        writer = MetricsArtifactWriter()
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        writer.handle_event(EventType.START_RUN, fl_ctx)
        _record_best_selection(
            writer,
            fl_ctx,
            {
                "source": "IntimeModelSelector",
                "key_metric": {"name": "score", "mode": "max"},
                "best_round": 1,
                "best_metrics": {"score": 0.8},
            },
        )
        _finish_run(writer, fl_ctx)

        _, summary_path, round_path = _artifact_paths(run_dir)
        assert not summary_path.exists()
        assert not round_path.exists()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"results_dir": "../outside_metrics"},
            {"results_dir": "/tmp/outside_metrics"},
            {"summary_file_name": "../metrics_summary.json"},
            {"round_file_name": "/tmp/round_metrics.jsonl"},
        ],
    )
    def test_output_path_is_resolved_under_run_dir(self, tmp_path, kwargs):
        writer = MetricsArtifactWriter(**kwargs)
        run_dir = tmp_path / "run"
        fl_ctx = _make_fl_ctx(run_dir)

        with pytest.raises(ValueError, match="must (be relative|stay inside)"):
            writer.handle_event(EventType.START_RUN, fl_ctx)
            _record_round(writer, fl_ctx, 1, {"accuracy": 0.8})
            _finish_run(writer, fl_ctx)

        assert not os.path.exists(tmp_path / "outside_metrics")
