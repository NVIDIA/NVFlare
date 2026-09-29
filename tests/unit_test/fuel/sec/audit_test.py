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
            assert release_write.wait(timeout=1.0)

        def flush(self):
            assert not self.closed

        def close(self):
            self.closed = True

    auditor = Auditor.__new__(Auditor)
    auditor.audit_file = BlockingFile()
    auditor._lock = threading.Lock()

    writer = threading.Thread(target=auditor.add_job_event, args=("job-1",))
    writer.start()
    assert write_entered.wait(timeout=1.0)

    closer = threading.Thread(target=lambda: (auditor.close(), close_finished.set()))
    closer.start()
    assert not close_finished.wait(timeout=0.05)

    release_write.set()
    writer.join(timeout=1.0)
    closer.join(timeout=1.0)
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
