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

import importlib.util
import json
import sys
import weakref
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.memorization_auditor import MemorizationAuditor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.multi_process_executor import MultiProcessExecutor
from nvflare.app_common.filters import memorization_audit_filter as audit_filters
from nvflare.app_common.filters.memorization_audit_filter import (
    _BASELINE_ATTR,
    AuditSnapshotFilter,
    MemorizationAuditFilter,
)
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.common.multi_process_executor_constants import CommunicationMetaData
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.component_builder import ComponentBuilder
from nvflare.job_config.api import FedApp, FedJob
from nvflare.job_config.fed_app_config import ClientAppConfig
from nvflare.recipe.spec import Recipe


class Auditor(MemorizationAuditor):
    def audit(self, model, baseline, fl_ctx):
        assert baseline.params == {"w": 1}
        assert model.params == {"w": 2}
        return {"model_fingerprint": "candidate", "metrics": {"score/auc": 0.5}}


def context(tmp_path):
    engine = Mock()
    engine.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    fl_ctx = FLContextManager(engine=engine, identity_name="site-1", job_id="test-job").new_context()
    return fl_ctx, Auditor()


def capture(model, fl_ctx):
    AuditSnapshotFilter().process(FLModelUtils.to_shareable(model), fl_ctx)


@pytest.mark.parametrize("round_number,status", [(0, "skipped"), (1, "completed"), (2, "skipped"), (3, "completed")])
def test_schedule_snapshot_preservation_and_retry_report(tmp_path, monkeypatch, round_number, status):
    copy_model = Mock(wraps=deepcopy)
    monkeypatch.setattr(audit_filters, "deepcopy", copy_model)
    fl_ctx, auditor = context(tmp_path)
    incoming = FLModelUtils.to_shareable(FLModel(params={"w": 1}, current_round=round_number))
    outgoing = FLModelUtils.to_shareable(
        FLModel(
            params={"w": 2}, metrics={"accuracy": 0.7}, meta={"NUM_STEPS_CURRENT_ROUND": 2}, current_round=round_number
        )
    )
    incoming_before = json.dumps(incoming, sort_keys=True)
    outgoing_before = json.dumps(outgoing, sort_keys=True)
    filt = MemorizationAuditFilter(auditor, start_round=1, frequency=2)
    for _ in range(2):
        AuditSnapshotFilter(start_round=1, frequency=2).process(incoming, fl_ctx)
        result = filt.process(outgoing, fl_ctx)
        assert result is outgoing
        assert json.dumps(incoming, sort_keys=True) == incoming_before
        assert json.dumps(outgoing, sort_keys=True) == outgoing_before
        assert getattr(fl_ctx, _BASELINE_ATTR, None) is None
        model = FLModelUtils.from_shareable(result)
        assert model.params == {"w": 2}
        assert model.metrics == {"accuracy": 0.7}
        assert model.meta["NUM_STEPS_CURRENT_ROUND"] == 2
    reports = list(tmp_path.rglob("*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["status"] == status
    assert copy_model.call_count == (2 if status == "completed" else 0)


def test_missing_baseline_and_runtime_failure_are_not_zero_scores(tmp_path):
    fl_ctx, auditor = context(tmp_path)
    outgoing = FLModelUtils.to_shareable(FLModel(params={"w": 2}, current_round=0))
    filt = MemorizationAuditFilter(auditor)
    filt.process(outgoing, fl_ctx)
    report = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert report["status"] == "unsupported"
    assert report["reason"] == "missing_baseline"
    assert not report["metrics"]
    auditor.audit = Mock(side_effect=TimeoutError())
    capture(FLModel(params={"w": 1}, current_round=0), fl_ctx)
    assert filt.process(outgoing, fl_ctx) is outgoing
    report = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert report["status"] == "failed"
    assert report["reason"] == "TimeoutError"
    assert getattr(fl_ctx, _BASELINE_ATTR, None) is None


def test_round_mismatch_is_rejected_before_custom_auditor(tmp_path):
    fl_ctx, auditor = context(tmp_path)
    auditor.audit = Mock()
    capture(FLModel(params={"w": 1}, current_round=0), fl_ctx)
    outgoing = FLModelUtils.to_shareable(FLModel(params={"w": 2}, current_round=1))
    assert MemorizationAuditFilter(auditor).process(outgoing, fl_ctx) is outgoing
    auditor.audit.assert_not_called()
    report = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert (report["status"], report["reason"]) == ("unsupported", "baseline_round_mismatch")
    assert getattr(fl_ctx, _BASELINE_ATTR, None) is None


def test_snapshot_is_private_task_local_and_independent():
    manager = FLContextManager(identity_name="site-1", job_id="test-job")
    fl_ctx = manager.new_context()
    incoming = FLModel(params={"w": [1]}, current_round=0)
    filt = AuditSnapshotFilter()
    filt.process(FLModelUtils.to_shareable(incoming), fl_ctx)
    incoming.params["w"][0] = 2
    baseline = getattr(fl_ctx, _BASELINE_ATTR)
    assert baseline.params == {"w": [1]}
    assert _BASELINE_ATTR not in fl_ctx.props
    assert _BASELINE_ATTR not in fl_ctx.get_all_public_props()
    assert getattr(manager.new_context(), _BASELINE_ATTR, None) is None
    assert getattr(fl_ctx.clone(), _BASELINE_ATTR, None) is None
    baseline_ref = weakref.ref(baseline)
    # No output filter runs after an executor failure. Neither the manager nor
    # the long-lived input filter owns the baseline after its task is released.
    del baseline, fl_ctx
    assert baseline_ref() is None


def test_failed_capture_does_not_reuse_baseline(tmp_path):
    fl_ctx, _ = context(tmp_path)
    capture(FLModel(params={"w": 1}, current_round=0), fl_ctx)
    malformed = Shareable()
    assert AuditSnapshotFilter().process(malformed, fl_ctx) is malformed
    assert getattr(fl_ctx, _BASELINE_ATTR, None) is None


class RelayOnlyExecutor(MultiProcessExecutor):
    def get_multi_process_command(self):
        raise AssertionError("This serialization test must not launch workers")


def test_baseline_stays_local_through_worker_event_and_task_relays(tmp_path, isolated_fobs):
    fl_ctx, auditor = context(tmp_path)
    marker = "AUDIT_BASELINE_MUST_NOT_ENTER_WORKER_MESSAGES"
    capture(FLModel(params={"w": 1}, meta={"marker": marker}, current_round=0), fl_ctx)
    executor = RelayOnlyExecutor(num_of_processes=2)
    executor.engine = fl_ctx.get_engine()
    executor.targets = ["rank-0", "rank-1"]
    wires = []

    def send_message(**kwargs):
        # Use real message construction/FOBS with a fake Cell boundary. No
        # workers, threads or sockets are started by these relay methods.
        wires.append(fobs.dumps(kwargs["message"].payload))

    executor.engine.client.cell.fire_and_forget.side_effect = send_message
    executor._pass_event_to_rank_processes(EventType.BEFORE_TASK_EXECUTION, fl_ctx)
    outgoing = FLModelUtils.to_shareable(FLModel(params={"w": 2}, current_round=0))
    executor._execute_multi_process("train", outgoing, fl_ctx, Signal())
    assert len(wires) == 2
    for wire in wires:
        assert marker.encode() not in wire
        relayed = fobs.loads(wire)[CommunicationMetaData.FL_CTX]
        assert getattr(relayed, _BASELINE_ATTR, None) is None
        assert _BASELINE_ATTR not in relayed.props
    # Relays must neither consume nor replace the parent's baseline.
    assert MemorizationAuditFilter(auditor).process(outgoing, fl_ctx) is outgoing
    report = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert report["status"] == "completed"
    assert getattr(fl_ctx, _BASELINE_ATTR, None) is None


class Builder(ComponentBuilder):
    def get_module_scanner(self):
        return None  # Exported configs contain fully qualified paths.


def test_export_reload_pairs_separate_filters_via_context(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    source = Path(__file__).resolve().parents[4] / "research/memorization-reduction/model.py"
    spec = importlib.util.spec_from_file_location("model", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "model", module)
    spec.loader.exec_module(module)
    job = FedJob(name="paired-audit", min_clients=1)
    job._deploy_map["site-1"] = FedApp(ClientAppConfig())
    recipe = Recipe(job)
    recipe.add_client_input_filter(AuditSnapshotFilter(start_round=1, frequency=3), tasks=["train"])
    recipe.add_client_output_filter(
        MemorizationAuditFilter(auditor=module.SyntheticAuditor(max_seconds=30), start_round=1, frequency=3),
        tasks=["train"],
    )
    job.export_job(str(tmp_path / "jobs"))
    config_path = next((tmp_path / "jobs").rglob("config_fed_client.json"))
    config = json.loads(config_path.read_text())
    assert not config["components"]
    assert config["task_data_filters"][0]["filters"][0]["args"] == {"start_round": 1, "frequency": 3}
    output_config = config["task_result_filters"][0]["filters"][0]
    assert output_config["args"]["auditor"]["path"] == "model.SyntheticAuditor"
    assert output_config["args"]["auditor"]["args"]["max_seconds"] == 30
    # Rebuild the actual nested object from the exported custom source.
    monkeypatch.delitem(sys.modules, "model")
    monkeypatch.syspath_prepend(str(config_path.parent.parent / "custom"))
    incoming_filter = Builder().build_component(config["task_data_filters"][0]["filters"][0])
    outgoing_filter = Builder().build_component(output_config)
    assert isinstance(outgoing_filter.auditor, MemorizationAuditor)
    fl_ctx, _ = context(tmp_path / "reports")
    state = module.create_model().state_dict()
    incoming = FLModelUtils.to_shareable(FLModel(params=state, current_round=1))
    outgoing = FLModelUtils.to_shareable(FLModel(params={k: v + 0.01 for k, v in state.items()}, current_round=1))
    assert incoming_filter.process(incoming, fl_ctx) is incoming
    assert outgoing_filter.process(outgoing, fl_ctx) is outgoing
    report = json.loads(next((tmp_path / "reports").rglob("*.json")).read_text())
    assert report["status"] == "completed"
    assert report["baseline_fingerprint"] != report["model_fingerprint"]
    assert "likelihood/change_auc" in report["metrics"]
    assert getattr(fl_ctx, _BASELINE_ATTR, None) is None


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
    fl_ctx, auditor = context(tmp_path)
    baseline = FLModel(params={"w": 1}, current_round=0)
    capture(baseline, fl_ctx)
    auditor.audit = Mock()
    outgoing = DXO(kind, data).to_shareable()
    outgoing.set_header(AppConstants.CURRENT_ROUND, 0)
    outgoing.set_header("custom_header", "preserve")
    outgoing.add_cookie("custom_cookie", "preserve")
    before = json.dumps(outgoing, default=lambda dxo: dxo.to_dict(), sort_keys=True)

    assert MemorizationAuditFilter(auditor).process(outgoing, fl_ctx) is outgoing

    reports = list(tmp_path.rglob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["status"] == "unsupported"
    assert report["reason"] == reason
    assert not report["metrics"]
    assert getattr(fl_ctx, _BASELINE_ATTR, None) is None
    auditor.audit.assert_not_called()
    assert json.dumps(outgoing, default=lambda dxo: dxo.to_dict(), sort_keys=True) == before


@pytest.mark.parametrize("case", ["non-dxo", "malformed-dxo", "native-list", "failed-task", "invalid-round"])
def test_unauditable_result_reports_status_and_releases_baseline(tmp_path, case):
    fl_ctx, auditor = context(tmp_path)
    baseline = FLModel(params={"w": 1}, current_round=0)
    capture(baseline, fl_ctx)
    auditor.audit = Mock()
    outgoing = FLModelUtils.to_shareable(FLModel(params={"w": 2}, current_round=0))
    status, reason = "unsupported", "non_model_result"
    if case == "non-dxo":
        outgoing = Shareable()
    elif case == "malformed-dxo":
        outgoing["DXO"] = "invalid"
    elif case == "native-list":
        outgoing = DXO(DataKind.FL_MODEL, {"params": {"w": 2}}).to_shareable()
        outgoing["DXO"]["data"] = ["not", "a", "dictionary"]
    elif case == "failed-task":
        outgoing.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
        status, reason = "skipped", "task_result_not_ok"
    else:
        outgoing.set_header(AppConstants.CURRENT_ROUND, -1)
        status, reason = "failed", "ValueError"
    before = json.dumps(outgoing, sort_keys=True)

    assert MemorizationAuditFilter(auditor).process(outgoing, fl_ctx) is outgoing

    report = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert report["status"] == status
    assert report["reason"] == reason
    assert not report["metrics"]
    assert getattr(fl_ctx, _BASELINE_ATTR, None) is None
    auditor.audit.assert_not_called()
    assert json.dumps(outgoing, sort_keys=True) == before


def test_baseline_released_even_when_early_report_cannot_be_written(tmp_path):
    fl_ctx, auditor = context(tmp_path)
    baseline = FLModel(params={"w": 1}, current_round=0)
    capture(baseline, fl_ctx)
    fl_ctx.get_engine().get_workspace.side_effect = OSError("workspace unavailable")
    outgoing = DXO(DataKind.WEIGHTS, {}).to_shareable()

    assert MemorizationAuditFilter(auditor).process(outgoing, fl_ctx) is outgoing

    assert getattr(fl_ctx, _BASELINE_ATTR, None) is None
    assert not list(tmp_path.rglob("*.json"))


def test_tensor_dxo_decode_failure_reports_and_preserves_result(tmp_path):
    torch = pytest.importorskip("torch")
    fl_ctx, auditor = context(tmp_path)
    baseline = FLModel(params={"w": 1}, current_round=0)
    capture(baseline, fl_ctx)
    auditor.audit = Mock()
    outgoing = DXO(DataKind.WEIGHTS, {"w": 2}).to_shareable()
    malformed = torch.ones(2)
    outgoing["DXO"] = malformed

    assert MemorizationAuditFilter(auditor).process(outgoing, fl_ctx) is outgoing

    assert outgoing["DXO"] is malformed
    report = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert report["status"] == "failed" and report["reason"] == "RuntimeError"
    assert not report["metrics"]
    assert getattr(fl_ctx, _BASELINE_ATTR, None) is None
    auditor.audit.assert_not_called()
