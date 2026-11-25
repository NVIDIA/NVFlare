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

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.app_common.executors.learner_executor import LearnerExecutor
from nvflare.app_common.executors.model_learner_executor import ModelLearnerExecutor
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.sec import task_audit
from nvflare.fuel.sec.audit import Auditor, AuditService
from tests.unit_test.fl_context_helper import make_fl_context


@pytest.fixture
def writer(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    auditor = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", auditor)
    yield path
    auditor.close()


@pytest.fixture
def executor():
    subject = ModelLearnerExecutor("private-learner-id")
    subject.learner = Mock()
    subject.learner_name = "private-learner-name"
    subject.is_initialized = True
    subject.log_info = Mock()
    subject.log_error = Mock()
    subject.log_exception = Mock()
    return subject


def _facts(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        assert record["job"] == "local-job"
        assert record["user"] == "unknown"
    assert "private-" not in path.read_text()
    return [(record["action"], record.get("message", "")) for record in records]


@pytest.mark.parametrize("field", ["params", "metrics"])
def test_model_learner_validation_keeps_reply_contents(writer, executor, field):
    fl_ctx = make_fl_context(run_num="local-job")
    request = FLModelUtils.to_shareable(FLModel(params={"private-parameter": 2}))
    expected = {"private-value": 3}
    executor.learner.validate.return_value = FLModel(**{field: expected})

    reply = executor.execute(AppConstants.TASK_VALIDATION, request, fl_ctx, Signal())

    executor.learner.validate.assert_called_once()
    model = executor.learner.validate.call_args.args[0]
    assert model.params == {"private-parameter": 2}
    assert getattr(FLModelUtils.from_shareable(reply), field) == expected
    assert _facts(writer) == [
        ("evaluation.execution.started", ""),
        ("evaluation.execution.stage_returned", "stage=learner"),
        ("evaluation.execution.stage_returned", "stage=encode_reply"),
        ("evaluation.execution.returned", "status=reply_prepared"),
    ]


@pytest.mark.parametrize(
    "failure, return_code, expected",
    [
        ("input", ReturnCode.BAD_REQUEST_DATA, [("rejected", "reason=invalid_input")]),
        (
            "empty",
            ReturnCode.EMPTY_RESULT,
            [("stage_returned", "stage=learner"), ("rejected", "reason=invalid_reply")],
        ),
        (
            "code",
            "private-application-code",
            [("stage_returned", "stage=learner"), ("rejected", "reason=learner_code")],
        ),
        (
            "learner",
            ReturnCode.EXECUTION_EXCEPTION,
            [("stage_error", "stage=learner"), ("error", "")],
        ),
    ],
)
def test_model_learner_validation_preserves_replies_for_failures(writer, executor, failure, return_code, expected):
    fl_ctx = make_fl_context(run_num="local-job")
    request = Shareable() if failure == "input" else FLModelUtils.to_shareable(FLModel(params={"model": 2}))
    executor.learner.validate.return_value = "private-application-code" if failure == "code" else None
    if failure == "learner":
        executor.learner.validate.side_effect = ValueError("private-learner-exception")

    reply = executor.execute(AppConstants.TASK_VALIDATION, request, fl_ctx, Signal())

    assert reply.get_return_code() == return_code
    assert executor.learner.validate.call_count == (0 if failure == "input" else 1)
    assert _facts(writer) == [("evaluation.execution.started", "")] + [
        (f"evaluation.execution.{outcome}", detail) for outcome, detail in expected
    ]


def test_model_learner_reply_encoding_failure_stays_distinct_from_learner_failure(writer, executor, monkeypatch):
    fl_ctx = make_fl_context(run_num="local-job")
    request = FLModelUtils.to_shareable(FLModel(params={"model": 2}))
    result = FLModel(metrics={"private-metric": 3})
    executor.learner.validate.return_value = result
    error = ValueError("private-reply-encoding-exception")

    def fail_encoding(value):
        assert value is result
        raise error

    monkeypatch.setattr(FLModelUtils, "to_shareable", fail_encoding)
    with pytest.raises(ValueError) as caught:
        executor.execute(AppConstants.TASK_VALIDATION, request, fl_ctx, Signal())

    assert caught.value is error
    executor.learner.validate.assert_called_once()
    assert _facts(writer) == [
        ("evaluation.execution.started", ""),
        ("evaluation.execution.stage_returned", "stage=learner"),
        ("evaluation.execution.stage_error", "stage=encode_reply"),
        ("evaluation.execution.error", ""),
    ]


def test_model_learner_lookup_failure_does_not_claim_learner_was_called(writer, executor):
    fl_ctx = make_fl_context(run_num="local-job")
    request = FLModelUtils.to_shareable(FLModel(params={"model": 2}))
    executor.learner = None

    reply = executor.validate(request, fl_ctx)

    assert reply.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert _facts(writer) == [("evaluation.execution.started", ""), ("evaluation.execution.error", "")]


@pytest.mark.parametrize("failure", [None, "empty", "learner"])
def test_legacy_validation_keeps_reply_and_exception_behavior(writer, failure):
    subject = LearnerExecutor("private-legacy-learner")
    error = ValueError("private-learner-exception")
    original_reply = make_reply("private-application-code")
    subject.learner = Mock(validate=Mock(return_value=original_reply if failure is None else None))
    subject.is_initialized = True
    subject.log_info = Mock()
    subject.log_debug = Mock()
    request = Shareable()
    fl_ctx = make_fl_context(run_num="local-job")
    abort = Signal()

    if failure == "learner":
        subject.learner.validate.side_effect = error
        with pytest.raises(ValueError) as caught:
            subject.execute(subject.validate_task, request, fl_ctx, abort)
        assert caught.value is error
        expected = [("evaluation.execution.stage_error", "stage=learner"), ("evaluation.execution.error", "")]
    else:
        actual = subject.execute(subject.validate_task, request, fl_ctx, abort)
        if failure == "empty":
            assert actual.get_return_code() == ReturnCode.EMPTY_RESULT
            terminal_fact = ("evaluation.execution.rejected", "reason=invalid_reply")
        else:
            assert actual is original_reply
            terminal_fact = ("evaluation.execution.returned", "status=reply_prepared")
        expected = [("evaluation.execution.stage_returned", "stage=learner"), terminal_fact]

    subject.learner.validate.assert_called_once_with(request, fl_ctx, abort)
    assert request.get_header(AppConstants.VALIDATE_TYPE) == ValidateType.MODEL_VALIDATE
    assert _facts(writer) == [("evaluation.execution.started", "")] + expected


def test_pretraining_validation_does_not_emit_evaluation_execution(writer, executor):
    request = FLModelUtils.to_shareable(FLModel(params={"model": 2}))
    fl_ctx = make_fl_context(run_num="local-job")
    executor.learner.validate.return_value = FLModel(metrics={"score": 3})
    executor.learner.train.return_value = FLModel(params={"model": 4})
    reply = executor.execute(AppConstants.TASK_TRAIN, request, fl_ctx, Signal())
    assert FLModelUtils.from_shareable(reply).params == {"model": 4}
    executor.learner.validate.assert_called_once()
    assert request.get_header(AppConstants.VALIDATE_TYPE) == ValidateType.BEFORE_TRAIN_VALIDATE

    legacy = LearnerExecutor("legacy")
    legacy.log_debug = Mock()
    legacy.learner = Mock(validate=Mock(return_value=None), train=Mock(return_value=reply))
    assert legacy.train(request, fl_ctx, Signal()) is reply
    legacy.learner.validate.assert_called_once()
    assert _facts(writer) == []


@pytest.mark.parametrize("write_fails", [False, True])
def test_model_learner_audit_io_waits_for_executor_lock_release(monkeypatch, executor, write_fails):
    fl_ctx = make_fl_context(run_num="local-job")
    request = FLModelUtils.to_shareable(FLModel(params={"model": 2}))
    lock = executor.learner_exe_lock
    learner_lock_state = []
    write_lock_states = []
    diagnostic_lock_states = []
    result = FLModel(metrics={"score": 3})

    def validate(_model):
        learner_lock_state.append(lock.locked())
        return result

    def write(**_fields):
        write_lock_states.append(lock.locked())
        if write_fails:
            raise OSError("private-audit-write-error")
        return "written"

    def diagnostic():
        diagnostic_lock_states.append(lock.locked())

    executor.learner.validate.side_effect = validate
    monkeypatch.setattr(AuditService, "the_auditor", Mock(add_event=write))
    monkeypatch.setattr(task_audit, "recording_failed", diagnostic)

    reply = executor.execute(AppConstants.TASK_VALIDATION, request, fl_ctx, Signal())

    assert FLModelUtils.from_shareable(reply).metrics == {"score": 3}
    assert learner_lock_state == [True]
    assert write_lock_states == [False] * 4
    assert diagnostic_lock_states == ([False] * 4 if write_fails else [])
