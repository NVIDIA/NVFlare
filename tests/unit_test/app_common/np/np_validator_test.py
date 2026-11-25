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
from types import SimpleNamespace

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np import np_validator
from nvflare.app_common.np.constants import NPConstants
from nvflare.fuel.sec.audit import Auditor, AuditService


@pytest.fixture
def audit(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    auditor = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", auditor)
    engine = SimpleNamespace(fire_event=lambda event_type, fl_ctx: None)
    context = FLContextManager(engine=engine, identity_name="private-client-identity", job_id="job-1").new_context()
    context.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "scope-1", private=True, sticky=False)
    try:
        yield path, context
    finally:
        auditor.close()


def _facts(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        assert set(record) <= {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}
        assert all(isinstance(value, str) for value in record.values())
        assert record["user"] == "unknown"
        assert record["job"] == "job-1"
        assert record["scope"] == "scope-1"
    return [(record["action"], record.get("message", "")) for record in records]


@pytest.mark.parametrize(
    "mode, outcome, detail, expected_code, expected_calls",
    [
        ("reply", "returned", "status=reply_prepared", ReturnCode.OK, ["random", "sum"]),
        ("before_evaluation", "aborted", "stage=before_evaluation", ReturnCode.TASK_ABORTED, []),
        ("after_evaluation", "aborted", "stage=after_evaluation", ReturnCode.TASK_ABORTED, ["random", "sum"]),
        ("error", "error", "", ReturnCode.EXECUTION_EXCEPTION, ["random"]),
        ("reply_error", "error", "", ReturnCode.EXECUTION_EXCEPTION, ["random", "sum"]),
    ],
)
def test_validation_records_only_the_observed_outcome(
    audit, monkeypatch, mode, outcome, detail, expected_code, expected_calls
):
    path, context = audit
    abort = Signal()
    if mode == "before_evaluation":
        abort.trigger("private-abort-value")
    calls = []

    def random():
        calls.append("random")
        if mode == "error":
            raise RuntimeError("private-validation-error")
        if mode == "after_evaluation":
            abort.trigger("private-abort-value")
        return 0.25

    def sum_values(values):
        calls.append("sum")
        return np.sum(values)

    monkeypatch.setattr(
        np_validator, "np", SimpleNamespace(random=SimpleNamespace(random=random), sum=sum_values, max=np.max)
    )
    request = DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: np.array([197, 394])}).to_shareable()
    request.set_header(AppConstants.MODEL_OWNER, "private-model-owner")
    if mode == "reply_error":

        def fail_to_prepare_reply(self):
            raise RuntimeError("private-reply-error")

        monkeypatch.setattr(DXO, "to_shareable", fail_to_prepare_reply)
    reply = np_validator.NPValidator().execute(AppConstants.TASK_VALIDATION, request, context, abort)

    assert reply.get_return_code() == expected_code
    assert calls == expected_calls
    if mode == "reply":
        result = from_shareable(reply)
        assert result.data_kind == DataKind.METRICS
        assert result.data == {"accuracy": 1.75}
    assert _facts(path) == [("evaluation.execution.started", ""), (f"evaluation.execution.{outcome}", detail)]
    raw = path.read_text()
    assert "private-" not in raw
    assert "accuracy" not in raw


@pytest.mark.parametrize("input_kind", ["malformed", "wrong_kind", "missing_key"])
def test_invalid_model_is_rejected_before_evaluation(audit, monkeypatch, input_kind):
    path, context = audit
    if input_kind == "malformed":
        request = Shareable()
    else:
        kind = DataKind.METRICS if input_kind == "wrong_kind" else DataKind.WEIGHTS
        request = DXO(kind, {"private-unexpected-key": [1]}).to_shareable()

    def unexpected_evaluation():
        pytest.fail("an invalid model must not reach evaluation")

    monkeypatch.setattr(np_validator, "np", SimpleNamespace(random=SimpleNamespace(random=unexpected_evaluation)))
    reply = np_validator.NPValidator().execute(AppConstants.TASK_VALIDATION, request, context, Signal())

    assert reply.get_return_code() == ReturnCode.BAD_TASK_DATA
    assert _facts(path) == [
        ("evaluation.execution.started", ""),
        ("evaluation.execution.rejected", "reason=invalid_input"),
    ]
    assert "private-unexpected-key" not in path.read_text()


@pytest.mark.parametrize(
    "task_name, delay, expected_code",
    [
        (AppConstants.TASK_VALIDATION, 1, ReturnCode.TASK_ABORTED),
        (AppConstants.TASK_SUBMIT_MODEL, 1, ReturnCode.TASK_ABORTED),
        ("private-unsupported-task", 1, ReturnCode.TASK_ABORTED),
        (AppConstants.TASK_SUBMIT_MODEL, 0, ReturnCode.OK),
        ("private-unsupported-task", 0, ReturnCode.TASK_UNKNOWN),
    ],
)
def test_non_evaluation_paths_keep_the_existing_reply_and_emit_no_evaluation_record(
    audit, task_name, delay, expected_code
):
    path, context = audit
    abort = Signal()
    abort.trigger("private-abort-value")
    reply = np_validator.NPValidator(sleep_time=delay).execute(task_name, Shareable(), context, abort)

    assert reply.get_return_code() == expected_code
    assert path.read_text() == ""
