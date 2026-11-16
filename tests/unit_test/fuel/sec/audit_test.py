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

import threading

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
    first = AuditService.initialize(str(tmp_path / "first.log"))
    AuditService.close()

    assert AuditService.add_job_event("job-1") == ""
    second = AuditService.initialize(str(tmp_path / "second.log"))
    assert second is not first
    AuditService.close()
