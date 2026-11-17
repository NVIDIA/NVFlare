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
import threading
from datetime import datetime, timezone
from uuid import UUID

import pytest

from nvflare.fuel.sec import audit as audit_module
from nvflare.fuel.sec.audit import Auditor, AuditService
from tests.timing_utils import WAIT_TIMEOUT, ObservedLock, ThreadGroup


def test_auditor_ignores_events_after_close(tmp_path):
    auditor = Auditor(str(tmp_path / "audit.log"))
    auditor.close()

    assert auditor.add_event("user", "action") == ""
    assert auditor.add_job_event("job-1") == ""


def test_close_waits_for_admitted_job_event():
    write_entered = threading.Event()
    release_write = threading.Event()
    close_finished = threading.Event()

    class BlockingFile:
        def __init__(self):
            self.closed = False

        def write(self, _line):
            write_entered.set()
            assert release_write.wait(WAIT_TIMEOUT)

        def flush(self):
            assert not self.closed

        def close(self):
            self.closed = True

    auditor = Auditor.__new__(Auditor)
    auditor.audit_file = BlockingFile()
    auditor._lock = ObservedLock()

    with ThreadGroup(release_write) as workers:
        workers.start(auditor.add_job_event, "job-1")
        assert write_entered.wait(WAIT_TIMEOUT)
        workers.start(lambda: (auditor.close(), close_finished.set()))
        assert auditor._lock.blocked.wait(WAIT_TIMEOUT), "close never contended with the admitted write"
        assert not close_finished.is_set()
    assert close_finished.is_set()
    assert auditor.audit_file is None


def test_audit_service_can_reinitialize_after_close(tmp_path):
    AuditService.close()
    try:
        first = AuditService.initialize(str(tmp_path / "first.log"))
        AuditService.close()

        assert AuditService.add_job_event("job-1") == ""
        second = AuditService.initialize(str(tmp_path / "second.log"))
        assert second is not first
    finally:
        AuditService.close()


@pytest.fixture(params=["direct", "service"])
def audit_writer(request, tmp_path):
    # Custom file names remain supported, including an already existing empty file.
    path = tmp_path / "custom.audit"
    path.write_bytes(b"")
    if request.param == "service":
        AuditService.close()
        try:
            AuditService.initialize(str(path))
            yield AuditService, path
        finally:
            AuditService.close()
    else:
        auditor = Auditor(str(path))
        try:
            yield auditor, path
        finally:
            auditor.close()


def read_records(path):
    lines = path.read_bytes().splitlines(keepends=True)
    assert all(line.endswith(b"\n") for line in lines)
    return [json.loads(line.decode("utf-8")) for line in lines]


@pytest.mark.parametrize(
    "microsecond, expected_timestamp",
    [
        (0, "2026-11-17T01:02:03.000000+00:00"),
        (123456, "2026-11-17T01:02:03.123456+00:00"),
    ],
)
def test_audit_records_have_exact_fields_and_explicit_utc(audit_writer, monkeypatch, microsecond, expected_timestamp):
    writer, path = audit_writer
    calls = []
    instant = datetime(2026, 11, 17, 1, 2, 3, microsecond, tzinfo=timezone.utc)

    class ControlledDatetime:
        @staticmethod
        def now(tz=None):
            assert tz is timezone.utc, "audit time must explicitly request UTC"
            calls.append(tz)
            return instant

    monkeypatch.setattr(audit_module, "datetime", ControlledDatetime)
    event_id = writer.add_event("alice", "submit", ref="request-1", msg="accepted")
    job_event_id = writer.add_job_event(
        "job-1", scope_name="site-1", task_name="train", task_id="task-1", ref=event_id, msg="started"
    )

    records = read_records(path)
    assert records == [
        {
            "event_id": event_id,
            "reference": "request-1",
            "timestamp": expected_timestamp,
            "user": "alice",
            "action": "submit",
            "message": "accepted",
        },
        {
            "event_id": job_event_id,
            "reference": event_id,
            "timestamp": expected_timestamp,
            "job": "job-1",
            "scope": "site-1",
            "action": "train#task-1",
            "message": "started",
        },
    ]
    assert len(calls) == 2
    assert event_id != job_event_id
    for record in records:
        assert all(isinstance(value, str) for value in record.values())
        assert str(UUID(record["event_id"])) == record["event_id"]


def test_event_omits_empty_optional_fields(audit_writer):
    writer, path = audit_writer
    event_id = writer.add_event("", "", ref="", msg="")

    record = read_records(path)[0]
    assert set(record) == {"event_id", "timestamp", "user", "action"}
    assert record["event_id"] == event_id
    assert record["user"] == ""
    assert record["action"] == ""


@pytest.mark.parametrize(
    "task_name, task_id, expected_action",
    [("", "", None), ("", "orphan-id", None), ("train", "", "train#"), ("train", "task-1", "train#task-1")],
)
def test_job_optional_fields_and_combined_action(audit_writer, task_name, task_id, expected_action):
    writer, path = audit_writer
    event_id = writer.add_job_event("", task_name=task_name, task_id=task_id)

    record = read_records(path)[0]
    expected_keys = {"event_id", "timestamp", "job"}
    if expected_action is not None:
        expected_keys.add("action")
        assert record["action"] == expected_action
    assert set(record) == expected_keys
    assert record["event_id"] == event_id
    assert record["job"] == ""


def test_audit_values_round_trip_without_extra_physical_lines(audit_writer):
    writer, path = audit_writer
    value = '雪 "quoted"\\path\nnext\rline\tend'
    event_id = writer.add_event(value, value, ref=value, msg=value)
    job_event_id = writer.add_job_event(value, scope_name=value, task_name=value, task_id=value, ref=value, msg=value)

    records = read_records(path)
    assert len(records) == 2
    assert records[0]["event_id"] == event_id
    assert records[1]["event_id"] == job_event_id
    for field in ("reference", "user", "action", "message"):
        assert records[0][field] == value
    for field in ("reference", "job", "scope", "message"):
        assert records[1][field] == value
    assert records[1]["action"] == f"{value}#{value}"


def test_reopening_audit_file_preserves_records_and_ids(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = Auditor(str(path))
    try:
        first_id = first.add_event("alice", "submit")
    finally:
        first.close()
    original_bytes = path.read_bytes()
    original_records = read_records(path)

    second = Auditor(str(path))
    try:
        second_id = second.add_job_event("job-1", ref=first_id)
    finally:
        second.close()

    assert path.read_bytes().startswith(original_bytes)
    records = read_records(path)
    assert len(records) == 2
    assert records[0] == original_records[0]
    assert records[0]["event_id"] == first_id
    assert records[1]["event_id"] == second_id
    assert records[1]["reference"] == first_id
    assert first_id != second_id


@pytest.mark.parametrize(
    "existing_content",
    [
        b"[E:legacy][T:2026-11-17 01:02:03][U:alice][A:submit]\n",
        b"{malformed}\n",
        b"[]\n",
        b'"not an object"\n',
        b'{"event_id":"previous"}',
        b'{}\n{"event_id":"previous"}',
    ],
)
def test_rejects_incompatible_existing_file_without_mutation(tmp_path, existing_content):
    path = tmp_path / "existing.audit"
    path.write_bytes(existing_content)

    # Close even an incorrectly accepted file before reporting the regression.
    auditor = None
    try:
        with pytest.raises(ValueError, match="(?i)choose.*new.*file"):
            auditor = Auditor(str(path))
    finally:
        if auditor is not None:
            auditor.close()
    assert path.read_bytes() == existing_content


@pytest.mark.parametrize("action", sorted(audit_module.EXCLUDED_ACTIONS))
def test_excluded_actions_do_not_write(audit_writer, action):
    writer, path = audit_writer
    before = path.read_bytes()

    assert writer.add_event("alice", action, ref="request-1", msg="ignored") == ""
    assert path.read_bytes() == before
