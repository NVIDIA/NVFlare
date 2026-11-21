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

import pytest

from nvflare.apis.fl_context import FLContextManager
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
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
