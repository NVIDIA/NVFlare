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

import pytest

from nvflare.fuel.sec import admin_audit, evaluation_audit, task_audit
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.fed.simulator.simulator_audit import SimulatorAuditor


def _context(job="job", scope="scope"):
    return SimpleNamespace(get_job_id=lambda: job, get_prop=lambda key, default: scope)


@pytest.fixture
def writer(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    auditor = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", auditor)
    monkeypatch.setattr(admin_audit, "_failed_attempts", 0)
    monkeypatch.setattr(admin_audit, "_last_warning", float("-inf"))
    yield auditor, path
    auditor.close()


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_evaluation_snapshots_local_context_and_approved_reference(writer):
    auditor, path = writer
    reference = auditor.add_event("unknown", "task.result.received", job_id="job", scope_name="scope")
    parent = task_audit.build_context(
        job_id="job", scope_name="scope", reference=reference, task_name="private-task", site="private-site"
    )
    context = _context()
    value = {"private-metric": "private-value"}

    def delegate():
        context.get_job_id = lambda: "changed-job"
        return value

    with task_audit.collect(), task_audit.use_context(parent):
        with evaluation_audit.operation("result", context) as audit:
            assert audit.call("store_result", delegate) is value
            audit.record("returned", status="processed")
        assert len(_records(path)) == 1
        with evaluation_audit.operation("result", context) as audit:
            audit.record("rejected", reason="invalid_result")
        with evaluation_audit.operation("output", _context(scope="other")) as audit:
            audit.record("returned")

    first, *records = _records(path)
    assert first["event_id"] == reference
    assert [(record["action"], record.get("message", "")) for record in records] == [
        ("evaluation.result.started", ""),
        ("evaluation.result.stage_returned", "stage=store_result"),
        ("evaluation.result.returned", "status=processed"),
        ("evaluation.result.started", ""),
        ("evaluation.result.rejected", "reason=invalid_result"),
        ("evaluation.output.started", ""),
        ("evaluation.output.returned", ""),
    ]
    for record in records[:3]:
        assert (record["job"], record["scope"], record["reference"]) == ("job", "scope", reference)
    for record in records[3:5]:
        assert (record["job"], record["scope"]) == ("changed-job", "scope")
    for record in records[5:]:
        assert (record["job"], record["scope"]) == ("job", "other")
    assert all("reference" not in record for record in records[3:])
    assert all(record["user"] == "unknown" and record["timestamp"].endswith("+00:00") for record in records)
    for record in records:
        assert set(record) <= {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}
        assert all(isinstance(v, str) for v in record.values())
    assert "private-" not in path.read_text()


def test_evaluation_preserves_the_original_exception_after_a_completed_stage(writer):
    _, path = writer
    failure = OSError("private post-write error")
    with pytest.raises(OSError) as caught:
        with evaluation_audit.operation("output", None) as audit:
            assert audit.call("file_write", lambda: 2) == 2
            raise failure
    assert caught.value is failure
    records = _records(path)
    assert [(r["action"], r.get("message", "")) for r in records] == [
        ("evaluation.output.started", ""),
        ("evaluation.output.stage_returned", "stage=file_write"),
        ("evaluation.output.error", ""),
    ]
    assert all(not ({"job", "scope", "reference"} & r.keys()) for r in records)
    assert "private" not in path.read_text()


@pytest.mark.parametrize("state", ["missing", "closed", "empty", "failing", "simulator", "invalid_context"])
def test_evaluation_recording_failure_cannot_change_or_repeat_operation(writer, monkeypatch, capfd, state):
    auditor, path = writer
    context = _context()
    failure = OSError("private failure")
    calls = []

    def fail_write(**kwargs):
        raise failure

    def delegate():
        calls.append(True)
        return "value"

    if state == "closed":
        auditor.close()
    elif state == "invalid_context":

        def fail_context():
            raise failure

        context.get_job_id = fail_context
    else:
        replacement = {
            "missing": None,
            "empty": SimpleNamespace(add_event=lambda **kwargs: ""),
            "failing": SimpleNamespace(add_event=fail_write),
            "simulator": SimulatorAuditor(),
        }[state]
        monkeypatch.setattr(AuditService, "the_auditor", replacement)

    with evaluation_audit.operation("execution", context) as audit:
        assert audit.call("learner", delegate) == "value"
        audit.record("returned", status="reply_prepared")

    assert calls == [True]
    expected = {"simulator": 0, "invalid_context": 1}.get(state, 3)
    assert admin_audit._failed_attempts == expected
    assert ("Audit recording failed" in capfd.readouterr().err) is bool(expected)
    assert _records(path) == []
