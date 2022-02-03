# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import uuid
from datetime import datetime


class Auditor(object):
    def __init__(self, audit_file_name: str):
        """Manages the audit file to log events.

        Args:
            audit_file_name (str): the location to save audit log file
        """
        assert isinstance(audit_file_name, str), "audit_file_name must be str"
        if os.path.exists(audit_file_name):
            assert os.path.isfile(audit_file_name), "audit_file_name is not a valid file"

        # create/open the file
        self.audit_file = open(audit_file_name, "a")

    def add_event(self, user: str, action: str, ref: str = "", msg: str = "") -> str:
        event_id = uuid.uuid4()
        event_id_str = "{}".format(event_id)

        if len(ref) > 0:
            ref = " [R:{}] ".format(ref)
        else:
            ref = " "

        if len(msg) > 0:
            msg = " [M:{}]".format(msg)

        line = "[E:{}]{}[T:{}] [U:{}] [A:{}]{}\n".format(event_id, ref, datetime.now(), user, action, msg)
        self.audit_file.write(line)
        self.audit_file.flush()
        return event_id_str

    def close(self):
        if self.audit_file is not None:
            self.audit_file.close()
            self.audit_file = None


class AuditService(object):
    """Service for interacting with Auditor to add events to log."""

    the_auditor = None

    @staticmethod
    def initialize(audit_file_name: str):
        if not AuditService.the_auditor:
            AuditService.the_auditor = Auditor(audit_file_name)
        return AuditService.the_auditor

    @staticmethod
    def get_auditor():
        return AuditService.the_auditor

    @staticmethod
    def add_event(user: str, action: str, ref: str = "", msg: str = "") -> str:
        if not AuditService.the_auditor:
            return ""
        return AuditService.the_auditor.add_event(user, action, ref, msg)

    @staticmethod
    def close():
        if AuditService.the_auditor:
            AuditService.the_auditor.close()
