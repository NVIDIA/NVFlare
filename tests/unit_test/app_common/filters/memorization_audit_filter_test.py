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
from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.memorization_auditor import MemorizationAuditor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.filters.memorization_audit_filter import (
    AuditModelSnapshot,
    AuditSnapshotFilter,
    MemorizationAuditFilter,
)
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


class Auditor(MemorizationAuditor):
    def audit(self, model, baseline, fl_ctx):
        assert baseline.params == {"w": 1}
        assert model.params == {"w": 2}
        return {"model_fingerprint": "candidate", "metrics": {"score/auc": 0.5}}


def context(tmp_path):
    fl_ctx = FLContext()
    engine = Mock()
    engine.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    components = {"snapshot": AuditModelSnapshot(), "auditor": Auditor()}
    engine.get_component.side_effect = components.get
    fl_ctx.set_prop(ReservedKey.ENGINE, engine)
    fl_ctx.set_prop(FLContextKey.CURRENT_RUN, "test-job")
    fl_ctx.set_prop(FLContextKey.TASK_ID, "test-task")
    return fl_ctx, components


@pytest.mark.parametrize("round_number,status", [(0, "completed"), (1, "skipped"), (2, "completed")])
def test_schedule_snapshot_preservation_and_retry_report(tmp_path, round_number, status):
    fl_ctx, components = context(tmp_path)
    incoming = FLModelUtils.to_shareable(FLModel(params={"w": 1}, current_round=round_number))
    outgoing = FLModelUtils.to_shareable(
        FLModel(
            params={"w": 2}, metrics={"accuracy": 0.7}, meta={"NUM_STEPS_CURRENT_ROUND": 2}, current_round=round_number
        )
    )
    filt = MemorizationAuditFilter("auditor", "snapshot", frequency=2)
    for _ in range(2):
        AuditSnapshotFilter("snapshot").process(incoming, fl_ctx)
        result = filt.process(outgoing, fl_ctx)
        assert result is outgoing
        assert components["snapshot"].pop(FLModel(current_round=round_number), fl_ctx) is None
        model = FLModelUtils.from_shareable(result)
        assert model.params == {"w": 2}
        assert model.metrics == {"accuracy": 0.7}
        assert model.meta["NUM_STEPS_CURRENT_ROUND"] == 2
    reports = list(tmp_path.rglob("*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["status"] == status


def test_missing_baseline_and_runtime_failure_are_not_zero_scores(tmp_path):
    fl_ctx, components = context(tmp_path)
    outgoing = FLModelUtils.to_shareable(FLModel(params={"w": 2}, current_round=0))
    filt = MemorizationAuditFilter("auditor", "snapshot")
    filt.process(outgoing, fl_ctx)
    report = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert report["status"] == "unsupported"
    assert not report["metrics"]
    components["auditor"].audit = Mock(side_effect=TimeoutError())
    components["snapshot"].capture(FLModel(params={"w": 1}, current_round=0), fl_ctx)
    assert filt.process(outgoing, fl_ctx) is outgoing
    report = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert report["status"] == "failed"
    assert report["reason"] == "TimeoutError"


def test_wrong_task_baseline_is_not_reused(tmp_path):
    fl_ctx, components = context(tmp_path)
    model = FLModel(params={"w": 1}, current_round=0)
    components["snapshot"].capture(model, fl_ctx)
    fl_ctx.set_prop(FLContextKey.TASK_ID, "different-task")
    assert components["snapshot"].pop(model, fl_ctx) is None


@pytest.mark.parametrize(
    "kind,data,reason",
    [
        pytest.param(DataKind.WEIGHTS, {}, "missing_parameters", id="empty-weights"),
        pytest.param(DataKind.WEIGHT_DIFF, {}, "missing_parameters", id="empty-diff"),
        pytest.param(DataKind.FL_MODEL, {}, "missing_parameters", id="empty-native-dxo"),
        pytest.param(DataKind.FL_MODEL, {"params": {}}, "missing_parameters", id="empty-native-params"),
        pytest.param(DataKind.FL_MODEL, {"params": None}, "missing_parameters", id="missing-native-params"),
        pytest.param(DataKind.METRICS, {"accuracy": 0.7}, "unsupported_data_kind", id="metrics"),
        pytest.param(DataKind.APP_DEFINED, {"custom": 1}, "unsupported_data_kind", id="app-defined"),
        pytest.param(DataKind.COLLECTION, {}, "model_collection", id="empty-collection"),
        pytest.param(
            DataKind.COLLECTION, {"model": DXO(DataKind.WEIGHTS, {"w": 2})}, "model_collection", id="collection"
        ),
    ],
)
def test_unsupported_output_is_reported_and_consumes_snapshot(tmp_path, kind, data, reason):
    fl_ctx, components = context(tmp_path)
    baseline = FLModel(params={"w": 1}, current_round=0)
    components["snapshot"].capture(baseline, fl_ctx)
    components["auditor"].audit = Mock()
    outgoing = DXO(kind, data).to_shareable()
    outgoing.set_header(AppConstants.CURRENT_ROUND, 0)
    outgoing.set_header("custom_header", "preserve")
    outgoing.add_cookie("custom_cookie", "preserve")
    before = json.dumps(outgoing, default=lambda dxo: dxo.to_dict(), sort_keys=True)

    assert MemorizationAuditFilter("auditor", "snapshot").process(outgoing, fl_ctx) is outgoing

    reports = list(tmp_path.rglob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["status"] == "unsupported"
    assert report["reason"] == reason
    assert not report["metrics"]
    assert components["snapshot"].pop(baseline, fl_ctx) is None
    components["auditor"].audit.assert_not_called()
    assert json.dumps(outgoing, default=lambda dxo: dxo.to_dict(), sort_keys=True) == before


@pytest.mark.parametrize("case", ["non-dxo", "malformed-dxo", "failed-task", "invalid-round"])
def test_unauditable_result_reports_status_and_releases_baseline(tmp_path, case):
    fl_ctx, components = context(tmp_path)
    baseline = FLModel(params={"w": 1}, current_round=0)
    components["snapshot"].capture(baseline, fl_ctx)
    components["auditor"].audit = Mock()
    outgoing = FLModelUtils.to_shareable(FLModel(params={"w": 2}, current_round=0))
    status, reason = "unsupported", "non_model_result"
    if case == "non-dxo":
        outgoing = Shareable()
    elif case == "malformed-dxo":
        outgoing["DXO"] = "invalid"
    elif case == "failed-task":
        outgoing.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
        status, reason = "skipped", "task_result_not_ok"
    else:
        outgoing.set_header(AppConstants.CURRENT_ROUND, -1)
        status, reason = "failed", "ValueError"
    before = json.dumps(outgoing, sort_keys=True)

    assert MemorizationAuditFilter("auditor", "snapshot").process(outgoing, fl_ctx) is outgoing

    report = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert report["status"] == status
    assert report["reason"] == reason
    assert not report["metrics"]
    assert components["snapshot"].pop(baseline, fl_ctx) is None
    components["auditor"].audit.assert_not_called()
    assert json.dumps(outgoing, sort_keys=True) == before


def test_baseline_released_even_when_early_report_cannot_be_written(tmp_path):
    fl_ctx, components = context(tmp_path)
    baseline = FLModel(params={"w": 1}, current_round=0)
    components["snapshot"].capture(baseline, fl_ctx)
    fl_ctx.get_engine().get_workspace.side_effect = OSError("workspace unavailable")
    outgoing = DXO(DataKind.WEIGHTS, {}).to_shareable()

    assert MemorizationAuditFilter("auditor", "snapshot").process(outgoing, fl_ctx) is outgoing

    assert components["snapshot"].pop(baseline, fl_ctx) is None
    assert not list(tmp_path.rglob("*.json"))
