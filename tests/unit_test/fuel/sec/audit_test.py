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

import re
import threading
from datetime import datetime
from uuid import UUID

import pytest

from nvflare.fuel.sec.audit import Auditor, AuditService
from tests.timing_utils import WAIT_TIMEOUT, ObservedLock, ThreadGroup


@pytest.fixture
def auditor(tmp_path):
    instance = Auditor(str(tmp_path / "audit.log"))
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def isolated_audit_service(monkeypatch):
    # Restore any pre-existing singleton without closing a resource owned by another test.
    monkeypatch.setattr(AuditService, "the_auditor", None)
    try:
        yield AuditService
    finally:
        AuditService.close()


def _assert_event(line, event_id, tail, ref=""):
    assert UUID(event_id).version == 4
    prefix = f"[E:{event_id}]" + (f"[R:{ref}]" if ref else "")
    match = re.fullmatch(re.escape(prefix) + r"\[T:([^\]]+)\]" + re.escape(tail) + r"\n", line)
    assert match is not None, line
    datetime.fromisoformat(match.group(1))


def test_auditor_creates_file_before_first_event(tmp_path, auditor):
    assert (tmp_path / "audit.log").read_text() == ""


def test_auditor_appends_to_existing_file(tmp_path):
    path = tmp_path / "audit.log"
    path.write_text("previous event\n")
    instance = Auditor(str(path))
    try:
        event_id = instance.add_event("alice", "submit_job")
        # Read before close to verify that each event is flushed for other readers.
        lines = path.read_text().splitlines(keepends=True)
        assert lines[0] == "previous event\n"
        assert len(lines) == 2
        _assert_event(lines[1], event_id, "[U:alice][A:submit_job]")
    finally:
        instance.close()


@pytest.mark.parametrize("ref", ["", "origin-event"])
@pytest.mark.parametrize("msg", ["", "command accepted"])
def test_command_event_format_and_flush(tmp_path, auditor, ref, msg):
    event_id = auditor.add_event("alice", "submit_job", ref=ref, msg=msg)
    _assert_event((tmp_path / "audit.log").read_text(), event_id, f"[U:alice][A:submit_job]{msg}", ref)


@pytest.mark.parametrize("ref", ["", "server-event"])
def test_job_event_format_and_flush(tmp_path, auditor, ref):
    event_id = auditor.add_job_event("job-1", "scope-1", "train", "task-1", ref, "received task")
    _assert_event(
        (tmp_path / "audit.log").read_text(),
        event_id,
        "[S:scope-1][J:job-1][A:train#task-1]received task",
        ref,
    )


def test_job_event_omits_unspecified_optional_fields(tmp_path, auditor):
    event_id = auditor.add_job_event("job-1")
    _assert_event((tmp_path / "audit.log").read_text(), event_id, "[J:job-1]")


@pytest.mark.parametrize("action", ["scheduler.check_resource", "_check_session", "_commands", "__aux_command__"])
def test_housekeeping_actions_do_not_write_events(tmp_path, auditor, action):
    assert auditor.add_event("alice", action) == ""
    assert (tmp_path / "audit.log").read_text() == ""
    event_id = auditor.add_event("alice", "submit_job")
    _assert_event((tmp_path / "audit.log").read_text(), event_id, "[U:alice][A:submit_job]")


def test_each_event_has_a_distinct_id(tmp_path, auditor):
    first = auditor.add_event("alice", "submit_job")
    second = auditor.add_job_event("job-1")
    assert first != second
    lines = (tmp_path / "audit.log").read_text().splitlines(keepends=True)
    assert len(lines) == 2
    _assert_event(lines[0], first, "[U:alice][A:submit_job]")
    _assert_event(lines[1], second, "[J:job-1]")


def test_service_without_initialization_does_not_write(isolated_audit_service):
    assert AuditService.get_auditor() is None
    assert AuditService.add_event("alice", "submit_job") == ""
    assert AuditService.add_job_event("job-1") == ""
    AuditService.close()


def test_service_uses_first_initialized_destination(tmp_path, isolated_audit_service):
    path = tmp_path / "first.log"
    first = AuditService.initialize(str(path))
    assert AuditService.initialize(str(tmp_path / "second.log")) is first
    assert AuditService.get_auditor() is first
    assert not (tmp_path / "second.log").exists()
    command_id = AuditService.add_event("alice", "submit_job", "request-1", "accepted")
    job_id = AuditService.add_job_event("job-1", "scope-1", "train", "task-1", command_id, "received")
    lines = path.read_text().splitlines(keepends=True)
    assert len(lines) == 2
    _assert_event(lines[0], command_id, "[U:alice][A:submit_job]accepted", "request-1")
    _assert_event(lines[1], job_id, "[S:scope-1][J:job-1][A:train#task-1]received", command_id)


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


def test_audit_service_can_reinitialize_after_close(tmp_path, isolated_audit_service):
    AuditService.close()
    first = AuditService.initialize(str(tmp_path / "first.log"))
    AuditService.close()

    assert AuditService.add_job_event("job-1") == ""
    second = AuditService.initialize(str(tmp_path / "second.log"))
    assert second is not first
    AuditService.close()
