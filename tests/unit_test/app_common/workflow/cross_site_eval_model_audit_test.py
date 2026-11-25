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
from unittest.mock import Mock, mock_open

import pytest

from nvflare.apis.fl_context import FLContextManager
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows import cross_site_eval
from nvflare.app_common.workflows.cross_site_eval import CrossSiteEval
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.fuel.utils import fobs


class _NamedPersistor(ModelPersistor):
    def __init__(self, model):
        super().__init__(filter_id="unused-post-get-filter")
        self.model = model
        self.calls = []

    def load_model(self, fl_ctx):
        pytest.fail("training-model initialization should not run")

    def save_model(self, model, fl_ctx):
        pytest.fail("model saving should not run")

    def get(self, model_file, fl_ctx):
        pytest.fail("cross-site evaluation must continue to bypass the post-get filter")

    def get_model(self, model_file, fl_ctx):
        self.calls.append((model_file, fl_ctx))
        return make_model_learnable(self.model.params, self.model.meta)


@pytest.fixture
def audit_path(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    writer = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    yield path
    writer.close()


@pytest.mark.parametrize("with_persistor, stage", [(True, "implementation"), (False, "file_read")])
def test_cross_site_eval_records_source_and_passes_model_to_validation(
    audit_path, tmp_path, monkeypatch, with_persistor, stage
):
    model = FLModel(params={"private-model-weight": 42}, meta={"tag": "private-model-metadata"})
    model_name = "private-model-alias"
    if with_persistor:
        persistor = _NamedPersistor(model)
    else:
        model_name = str(tmp_path / "private-model-file")
        fobs.dumpf(model, model_name)
        persistor = None

    controller = CrossSiteEval(server_models=[model_name], participating_clients=["site-1"])
    controller.persistor = persistor
    controller.fl_ctx = FLContextManager(identity_name="server", job_id="job-1").new_context()
    tasks = []
    monkeypatch.setattr(controller, "send_model", lambda **kwargs: tasks.append(kwargs))
    monkeypatch.setattr(controller, "get_num_standing_tasks", lambda: 0)
    monkeypatch.setattr(controller, "save_results", lambda: None)

    controller.run()

    assert [task["task_name"] for task in tasks] == [AppConstants.TASK_SUBMIT_MODEL, AppConstants.TASK_VALIDATION]
    result = tasks[1]["data"]
    assert result.params == model.params
    assert result.meta == {"tag": "private-model-metadata", AppConstants.MODEL_OWNER: model_name}
    if with_persistor:
        assert persistor.calls == [(model_name, controller.fl_ctx)]

    raw_audit = audit_path.read_text()
    records = [json.loads(line) for line in raw_audit.splitlines()]
    assert [(record["action"], record.get("message", "")) for record in records] == [
        ("model.retrieve.started", ""),
        ("model.retrieve.stage_returned", f"stage={stage}"),
        ("model.retrieve.returned", "status=value_returned"),
    ]
    assert all(record["job"] == "job-1" and record["user"] == "unknown" for record in records)
    for private in (model_name, str(tmp_path), "private-model-weight", "private-model-metadata"):
        assert private not in raw_audit


@pytest.mark.parametrize(
    "mode, value, stage_result, result",
    [
        ("direct_read", None, ("stage_error", "stage=file_read"), ("error", "stage=file_read")),
        ("conversion", None, ("stage_returned", "stage=implementation"), ("returned", "status=no_value")),
        ("conversion", {}, ("stage_returned", "stage=implementation"), ("returned", "status=value_returned")),
    ],
)
def test_cross_site_eval_distinguishes_read_and_conversion_errors(
    audit_path, tmp_path, monkeypatch, mode, value, stage_result, result
):
    model_name = str(tmp_path / "private-missing-model")
    controller = CrossSiteEval(server_models=[model_name], participating_clients=["site-1"])
    if mode == "conversion":
        controller.persistor = _NamedPersistor(None)

        def get_model(name, ctx):
            assert (name, ctx) == (model_name, controller.fl_ctx)
            return value

        monkeypatch.setattr(controller.persistor, "get_model", get_model)
    else:
        controller.persistor = None
    controller.fl_ctx = FLContextManager(identity_name="server", job_id="job-1").new_context()
    monkeypatch.setattr(controller, "send_model", lambda **kwargs: None)
    handler_error = RuntimeError("private-diagnostic-error")

    def stop_at_exception_handler(message):
        assert model_name in message
        raise handler_error

    monkeypatch.setattr(controller, "exception", stop_at_exception_handler)
    with pytest.raises(RuntimeError) as caught:
        controller.run()
    assert caught.value is handler_error

    raw_audit = audit_path.read_text()
    records = [json.loads(line) for line in raw_audit.splitlines()]
    assert [(record["action"], record.get("message", "")) for record in records] == [
        ("model.retrieve.started", ""),
        (f"model.retrieve.{stage_result[0]}", stage_result[1]),
        (f"model.retrieve.{result[0]}", result[1]),
    ]
    assert model_name not in raw_audit
    assert "private-diagnostic-error" not in raw_audit


@pytest.fixture
def evaluation_result(tmp_path, monkeypatch):
    controller = CrossSiteEval()
    controller.fl_ctx = FLContextManager(identity_name="server", job_id="evaluation-job").new_context()
    monkeypatch.setattr(controller, "info", Mock())
    monkeypatch.setattr(controller, "get_run_dir", lambda: str(tmp_path))
    model = FLModel(
        metrics={"private-metric": 0.25},
        meta={"client_name": "private-client", "props": {AppConstants.MODEL_OWNER: "private-owner"}},
    )
    artifacts = tmp_path / "private-artifacts"
    artifacts.mkdir()
    controller._cross_val_models_dir = str(artifacts)
    return controller, model, artifacts / "private-client_private-owner"


def _facts(audit_path):
    return [
        (record["action"], record.get("message", ""))
        for record in (json.loads(line) for line in audit_path.read_text().splitlines())
    ]


def test_cross_site_eval_records_tracking_and_result_write(audit_path, evaluation_result):
    controller, model, artifact = evaluation_result

    assert controller._receive_val_result_cb(model) is None

    assert fobs.loadf(str(artifact)).metrics == model.metrics
    assert controller._json_val_results == {"private-client": {"private-owner": model.metrics}}
    assert controller._val_results == {"private-client": {"private-owner": str(artifact)}}
    assert _facts(audit_path) == [
        ("evaluation.result.started", ""),
        ("evaluation.result.stage_returned", "stage=track_result"),
        ("evaluation.result.stage_returned", "stage=file_write"),
        ("evaluation.result.returned", "status=processed"),
    ]
    for record in (json.loads(line) for line in audit_path.read_text().splitlines()):
        assert record["user"] == "unknown"
        assert record["job"] == "evaluation-job"
    for private in ("private-client", "private-owner", "private-metric", str(artifact)):
        assert private not in audit_path.read_text()


@pytest.mark.parametrize(
    "failure_stage, stages, saved",
    [
        ("tracking", [("stage_error", "track_result")], False),
        ("write", [("stage_returned", "track_result"), ("stage_error", "file_write")], False),
        ("after_write", [("stage_returned", "track_result"), ("stage_returned", "file_write")], True),
    ],
)
def test_cross_site_eval_result_failure_keeps_completed_steps(
    audit_path, evaluation_result, monkeypatch, failure_stage, stages, saved
):
    controller, model, artifact = evaluation_result
    error = OSError("private-result-error")
    fail = Mock(side_effect=error)
    if failure_stage == "tracking":
        monkeypatch.setattr(controller, "track_results", fail)
    elif failure_stage == "write":
        monkeypatch.setattr(fobs, "dumpf", fail)
    else:
        monkeypatch.setattr(controller, "info", fail)

    with pytest.raises(OSError) as raised:
        controller._receive_val_result_cb(model)
    assert raised.value is error
    fail.assert_called_once()
    assert artifact.is_file() is saved
    assert controller._val_results == ({"private-client": {"private-owner": str(artifact)}} if saved else {})
    assert _facts(audit_path) == [
        ("evaluation.result.started", ""),
        *((f"evaluation.result.{outcome}", f"stage={stage}") for outcome, stage in stages),
        ("evaluation.result.error", ""),
    ]
    assert "private-" not in audit_path.read_text()


@pytest.mark.parametrize("serializable", [True, False])
def test_cross_site_eval_summary_keeps_serialization_separate_from_writing(
    audit_path, evaluation_result, tmp_path, serializable
):
    controller, _, _ = evaluation_result
    controller._json_val_results = {"private-metric": 0.25 if serializable else object()}
    summary = tmp_path / controller._results_dir / controller._json_file_name
    summary.parent.mkdir()
    summary.write_text("previous result")

    if serializable:
        assert controller.save_results() is None
        assert json.loads(summary.read_text()) == controller._json_val_results
        assert _facts(audit_path) == [
            ("evaluation.output.started", ""),
            ("evaluation.output.stage_returned", "stage=serialize_summary"),
            ("evaluation.output.stage_returned", "stage=file_write"),
            ("evaluation.output.returned", ""),
        ]
    else:
        with pytest.raises(TypeError, match="not JSON serializable"):
            controller.save_results()
        # The original code opens and truncates the destination before serialization.
        assert summary.read_text() == ""
        assert _facts(audit_path) == [
            ("evaluation.output.started", ""),
            ("evaluation.output.stage_error", "stage=serialize_summary"),
            ("evaluation.output.error", ""),
        ]
    assert "private-metric" not in audit_path.read_text()
    assert str(summary) not in audit_path.read_text()


@pytest.mark.parametrize("failure_stage", ["write", "close"])
def test_cross_site_eval_summary_write_and_close_can_fail_independently(
    audit_path, evaluation_result, monkeypatch, failure_stage
):
    controller, _, _ = evaluation_result
    error = OSError("private-output-error")
    opener = mock_open()
    file = opener.return_value
    failing_operation = file.write if failure_stage == "write" else file.__exit__
    failing_operation.side_effect = error
    monkeypatch.setattr(cross_site_eval, "open", opener, raising=False)

    with pytest.raises(OSError) as raised:
        controller.save_results()
    assert raised.value is error
    file.write.assert_called_once_with("{}")
    file.__exit__.assert_called_once()
    write_outcome = "stage_error" if failure_stage == "write" else "stage_returned"
    assert _facts(audit_path) == [
        ("evaluation.output.started", ""),
        ("evaluation.output.stage_returned", "stage=serialize_summary"),
        (f"evaluation.output.{write_outcome}", "stage=file_write"),
        ("evaluation.output.error", ""),
    ]
    assert "private-output-error" not in audit_path.read_text()
