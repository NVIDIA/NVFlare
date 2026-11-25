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
import os
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.dxo import DXO, DataKind, from_file
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.fuel.sec import task_audit
from nvflare.fuel.sec.audit import Auditor, AuditService


def _make_engine_and_ctx(run_dir):
    workspace = Mock()
    workspace.get_run_dir.return_value = str(run_dir)
    engine = Mock()
    engine.get_workspace.return_value = workspace

    fl_ctx = FLContext()
    fl_ctx.get_job_id = Mock(return_value="job-1")
    return engine, fl_ctx


class TestCrossSiteModelEvalPaths:
    @pytest.mark.parametrize("cross_val_dir", ["/tmp/outside_cse", "../outside_cse"])
    def test_start_controller_rejects_escaping_cross_val_dir(self, tmp_path, cross_val_dir):
        engine, fl_ctx = _make_engine_and_ctx(tmp_path / "run")
        controller = CrossSiteModelEval(cross_val_dir=cross_val_dir, participating_clients=["site-1"])
        controller._engine = engine

        with pytest.raises(ValueError, match="must (be relative|stay inside)"):
            controller.start_controller(fl_ctx)

    def test_start_controller_accepts_relative_cross_val_dir(self, tmp_path):
        engine, fl_ctx = _make_engine_and_ctx(tmp_path / "run")
        controller = CrossSiteModelEval(participating_clients=["site-1"])
        controller._engine = engine
        controller.fire_event = Mock()  # event plumbing is not under test

        controller.start_controller(fl_ctx)

        run_dir = os.path.realpath(str(tmp_path / "run"))
        assert os.path.isdir(os.path.join(run_dir, AppConstants.CROSS_VAL_DIR, AppConstants.CROSS_VAL_MODEL_DIR_NAME))
        assert os.path.isdir(os.path.join(run_dir, AppConstants.CROSS_VAL_DIR, AppConstants.CROSS_VAL_RESULTS_DIR_NAME))

    @pytest.mark.parametrize("key", ["normal", "/../../outside"])
    def test_collection_model_paths_stay_under_output_dir(self, tmp_path, key):
        models_dir = tmp_path / "models"
        # Traversal through the authenticated site prefix requires this existing directory.
        (models_dir / "site-1.").mkdir(parents=True)
        controller = CrossSiteModelEval(participating_clients=["site-1"])
        controller._cross_val_models_dir = str(models_dir)
        controller.fire_event = Mock()
        controller.broadcast = Mock()
        model = DXO(DataKind.WEIGHTS, {"w": 3.0})
        result = DXO(DataKind.COLLECTION, {key: model}).to_shareable()

        if key.startswith("/"):
            with pytest.raises(ValueError, match="must stay inside"):
                controller._accept_local_model("site-1", result, FLContext())
            assert not (tmp_path / "outside").exists()
        else:
            controller._accept_local_model("site-1", result, FLContext())
            assert from_file(controller._client_models["site-1.normal"]).data == model.data


_CLIENT = "private-client"
_MODEL = "private-model"
_METRIC = "private-metric"


@pytest.fixture
def evaluation_result_audit(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    writer = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    controller = CrossSiteModelEval()
    results_dir = tmp_path / "private-results"
    results_dir.mkdir()
    controller._cross_val_results_dir = str(results_dir)
    for method in ("fire_event", "log_error", "log_exception", "log_info"):
        monkeypatch.setattr(controller, method, Mock())
    _, ctx = _make_engine_and_ctx(tmp_path)
    ctx.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "original-scope", private=True, sticky=False)
    yield controller, ctx, path
    writer.close()


def _result_with_owner(result):
    result.add_cookie(AppConstants.MODEL_OWNER, _MODEL)
    return result


def _assert_result_audit(path, expected):
    raw = path.read_text()
    records = [json.loads(line) for line in raw.splitlines()]
    assert [(r["action"], r.get("message", "")) for r in records] == [
        (f"evaluation.result.{outcome}", detail) for outcome, detail in expected
    ]
    fields = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}
    assert all(r.keys() <= fields and all(isinstance(v, str) for v in r.values()) for r in records)
    assert all(r["user"] == "unknown" and r["job"] == "job-1" and r["scope"] == "original-scope" for r in records)
    for marker in (_CLIENT, _MODEL, _METRIC, "private-entry", "private-results", "private-transport"):
        assert marker not in raw


@pytest.mark.parametrize("routing", ["assigned", "unknown_task_hook"])
def test_legacy_evaluation_audits_partial_collection_and_keeps_valid_results(evaluation_result_audit, routing):
    controller, ctx, audit_path = evaluation_result_audit
    result = _result_with_owner(
        DXO(
            DataKind.COLLECTION,
            {
                "first": DXO(DataKind.METRICS, {_METRIC: 1}),
                "private-entry": "malformed",
                "last": DXO(DataKind.METRICS, {_METRIC: 2}),
            },
        ).to_shareable()
    )

    def change_scope(event, event_ctx):
        assert event == AppEventType.VALIDATION_RESULT_RECEIVED
        assert event_ctx.get_prop(AppConstants.VALIDATION_RESULT) is result
        event_ctx.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "changed-scope", private=True, sticky=False)

    controller.fire_event.side_effect = change_scope
    client = Client(_CLIENT, "private-token")
    if routing == "assigned":
        task = ClientTask(client, Task(name=AppConstants.TASK_VALIDATION, data=Shareable()))
        task.result = result
        controller._receive_val_result_cb(task, ctx)
        assert task.result is None
    else:
        controller.process_result_of_unknown_task(client, AppConstants.TASK_VALIDATION, "unknown-id", result, ctx)

    assert set(controller._val_results) == {f"{_CLIENT}.first", f"{_CLIENT}.last"}
    assert from_file(controller._val_results[f"{_CLIENT}.first"][_MODEL]).data == {_METRIC: 1}
    assert from_file(controller._val_results[f"{_CLIENT}.last"][_MODEL]).data == {_METRIC: 2}
    controller.fire_event.assert_called_once()
    _assert_result_audit(
        audit_path,
        [
            ("started", ""),
            ("stage_error", "reason=invalid_entries stage=decode_result"),
            ("stage_returned", "stage=store_result"),
            ("stage_returned", "stage=store_result"),
            ("returned", "status=processed"),
        ],
    )


@pytest.mark.parametrize("reason", ["non_ok_reply", "invalid_result"])
def test_legacy_evaluation_audits_rejection_without_storing_metrics(evaluation_result_audit, reason):
    controller, ctx, audit_path = evaluation_result_audit
    result = make_reply("private-transport-status") if reason == "non_ok_reply" else Shareable()
    controller._accept_val_result(_CLIENT, _result_with_owner(result), ctx)

    assert controller._val_results == ({_CLIENT: {_MODEL: {}}} if reason == "non_ok_reply" else {})
    assert list(audit_path.parent.joinpath("private-results").iterdir()) == []
    _assert_result_audit(audit_path, [("started", ""), ("rejected", f"reason={reason}")])


def test_legacy_evaluation_audits_storage_error_after_a_stored_result(evaluation_result_audit, monkeypatch):
    controller, ctx, audit_path = evaluation_result_audit
    first = DXO(DataKind.METRICS, {_METRIC: 3})
    last = DXO(DataKind.METRICS, {_METRIC: 4})
    result = _result_with_owner(DXO(DataKind.COLLECTION, {"first": first, "last": last}).to_shareable())
    original_store = controller._save_validation_result
    error = RuntimeError("private-transport-storage-error")

    def fail_last(client_name, model_name, dxo, fl_ctx):
        if dxo is last:
            raise error
        return original_store(client_name, model_name, dxo, fl_ctx)

    monkeypatch.setattr(controller, "_save_validation_result", fail_last)
    with pytest.raises(RuntimeError) as caught:
        controller._accept_val_result(_CLIENT, result, ctx)

    assert caught.value is error
    assert set(controller._val_results) == {f"{_CLIENT}.first"}
    assert from_file(controller._val_results[f"{_CLIENT}.first"][_MODEL]).data == first.data
    _assert_result_audit(
        audit_path,
        [
            ("started", ""),
            ("stage_returned", "stage=store_result"),
            ("stage_error", "stage=store_result"),
            ("error", ""),
        ],
    )


@pytest.mark.parametrize("fail_last", [False, True])
def test_legacy_evaluation_limits_storage_facts_without_losing_task_outcomes(
    evaluation_result_audit, monkeypatch, fail_last
):
    controller, ctx, audit_path = evaluation_result_audit
    entries = {f"private-entry-{i}": DXO(DataKind.METRICS, {_METRIC: i}) for i in range(1100)}
    result = _result_with_owner(DXO(DataKind.COLLECTION, entries).to_shareable())
    calls = []
    last_entry = next(reversed(entries))
    failure = RuntimeError("private-transport-storage-error")

    def store(client_name, model_name, dxo, fl_ctx):
        calls.append((client_name, model_name, dxo, fl_ctx))
        if fail_last and client_name == f"{_CLIENT}.{last_entry}":
            raise failure

    monkeypatch.setattr(controller, "_save_validation_result", store)
    with task_audit.collect():
        task_audit.record("task.result_manager.returned")
        if fail_last:
            with pytest.raises(RuntimeError) as caught:
                controller._accept_val_result(_CLIENT, result, ctx)
            assert caught.value is failure
            task_audit.record("task.result_callback.error")
        else:
            controller._accept_val_result(_CLIENT, result, ctx)
            task_audit.record("task.result_callback.returned")
        task_audit.record("task.result.handling_returned")
        assert audit_path.read_text() == ""

    assert calls == [(f"{_CLIENT}.{key}", _MODEL, dxo, ctx) for key, dxo in entries.items()]
    raw = audit_path.read_text()
    observations = [(r["action"], r.get("message", "")) for r in map(json.loads, raw.splitlines())]
    assert observations == [
        ("task.result_manager.returned", ""),
        ("evaluation.result.started", ""),
        *[("evaluation.result.stage_returned", "stage=store_result")] * 64,
        ("evaluation.result.observations_limited", "reason=record_limit stage=store_result"),
        *(
            [
                ("evaluation.result.stage_error", "stage=store_result"),
                ("evaluation.result.error", ""),
                ("task.result_callback.error", ""),
            ]
            if fail_last
            else [("evaluation.result.returned", "status=processed"), ("task.result_callback.returned", "")]
        ),
        ("task.result.handling_returned", ""),
    ]
    for marker in (_CLIENT, _MODEL, _METRIC, "private-entry", "private-transport"):
        assert marker not in raw
