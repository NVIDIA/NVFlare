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

"""Private payload markers for the existing bounded federation audit test."""

from pathlib import Path

from nvflare.apis.filter import Filter
from tests.peer_admin_test_utils import PeerAdminJobTrainer, _write_json

TASK_KEY = "audit_private_task_data"
TASK_MARKER = "private-audit-task-data-marker"
RESULT_KEY = "audit_private_result_data"
RESULT_MARKER = "private-audit-result-data-marker"


class TaskAuditDataFilter(Filter):
    def process(self, shareable, fl_ctx):
        shareable[TASK_KEY] = TASK_MARKER
        return shareable


class TaskAuditJobTrainer(PeerAdminJobTrainer):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if shareable.get(TASK_KEY) != TASK_MARKER:
            raise RuntimeError("audit regression task marker missing")
        result = super().execute(task_name, shareable, fl_ctx, abort_signal)
        result[RESULT_KEY] = RESULT_MARKER
        return result


class TaskAuditResultFilter(Filter):
    def __init__(self, directory):
        super().__init__()
        self.directory = Path(directory)

    def process(self, shareable, fl_ctx):
        if shareable.get(RESULT_KEY) != RESULT_MARKER:
            raise RuntimeError("audit regression result marker missing")
        self.directory.mkdir(parents=True, exist_ok=True)
        site = fl_ctx.get_peer_context().get_identity_name()
        _write_json(self.directory / f"result-{site}.json", {"private_result_seen": True})
        return shareable
