# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import threading
import uuid
from datetime import datetime, timezone

EXCLUDED_ACTIONS = {
    "scheduler.check_resource",
    "_check_session",
    "_commands",
    "__aux_command__",
}


class Auditor(object):
    def __init__(self, audit_file_name: str):
        """Writes audit events as JSONL records with UTC timestamps.

        Args:
            audit_file_name (str): the location to save audit log file
        """
        assert isinstance(audit_file_name, str), "audit_file_name must be str"
        if os.path.exists(audit_file_name):
            assert os.path.isfile(audit_file_name), "audit_file_name is not a valid file"

        self.audit_file = open(audit_file_name, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def add_event(self, user: str, action: str, ref: str = "", msg: str = "") -> str:

        if action in EXCLUDED_ACTIONS:
            return ""

        with self._lock:
            # A late Cell callback can arrive while the process is shutting down.
            if self.audit_file is None:
                return ""

            event_id = str(uuid.uuid4())
            record = {"event_id": event_id}
            if ref:
                record["reference"] = str(ref)
            record.update(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                user=str(user),
                action=str(action),
            )
            if msg:
                record["message"] = str(msg)

            self.audit_file.write(json.dumps(record) + "\n")
            self.audit_file.flush()
            return event_id

    def add_job_event(
        self, job_id: str, scope_name: str = "", task_name: str = "", task_id: str = "", ref: str = "", msg: str = ""
    ) -> str:
        with self._lock:
            if self.audit_file is None:
                return ""

            event_id = str(uuid.uuid4())
            record = {"event_id": event_id}
            if ref:
                record["reference"] = str(ref)
            record["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            if scope_name:
                record["scope"] = str(scope_name)
            record["job"] = str(job_id)
            if task_name:
                record["action"] = f"{task_name}#{task_id}"
            if msg:
                record["message"] = str(msg)

            self.audit_file.write(json.dumps(record) + "\n")
            self.audit_file.flush()
            return event_id

    def close(self):
        with self._lock:
            if self.audit_file is not None:
                self.audit_file.close()
                self.audit_file = None


class AuditService(object):
    """Service for interacting with Auditor to add events to log."""

    the_auditor = None
    _lock = threading.Lock()

    @staticmethod
    def initialize(audit_file_name: str):
        with AuditService._lock:
            if not AuditService.the_auditor:
                AuditService.the_auditor = Auditor(audit_file_name)
            return AuditService.the_auditor

    @staticmethod
    def get_auditor():
        with AuditService._lock:
            return AuditService.the_auditor

    @staticmethod
    def add_event(user: str, action: str, ref: str = "", msg: str = "") -> str:
        with AuditService._lock:
            auditor = AuditService.the_auditor
        if not auditor:
            return ""
        return auditor.add_event(user, action, ref, msg)

    @staticmethod
    def add_job_event(
        job_id: str, scope_name: str = "", task_name: str = "", task_id: str = "", ref: str = "", msg: str = ""
    ) -> str:
        with AuditService._lock:
            auditor = AuditService.the_auditor
        if not auditor:
            return ""
        return auditor.add_job_event(
            scope_name=scope_name, job_id=job_id, task_name=task_name, task_id=task_id, ref=ref, msg=msg
        )

    @staticmethod
    def close():
        with AuditService._lock:
            auditor = AuditService.the_auditor
            AuditService.the_auditor = None
        if auditor:
            auditor.close()
