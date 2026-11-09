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

"""Controlled admin-session boundary for catalog CLI unit and process contract tests."""

import json
from pathlib import Path

from nvflare.app_common.fdn.catalog.common.payload import CATALOG_APPLY_META
from nvflare.app_common.fdn.catalog.server.commands import SEARCH, STATUS
from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.proto import ProtoKey


class CatalogSession:
    def __init__(self, status="FINISHED:COMPLETED", error=None, health="OK", api_status=None, reply_payload=None):
        """``error`` is raised by the call. ``api_status`` is returned as a failed API response with no payload,
        as the admin client does for a lost connection or an expired session. ``reply_payload`` replaces the
        catalog payload of a successful response."""
        self.status, self.error, self.health, self.api_status = status, error, health, api_status
        self.reply_payload = reply_payload
        self.monitor_timeout = None
        self.closed = False
        self.submitted = None
        self.monitored = False
        self.apply_status = "APPLIED"
        self.meta_reads = 0

    def submit_job(self, path):
        if self.error:
            raise self.error
        root = Path(path)
        self.submitted = {
            "path": root,
            "meta": json.loads((root / "meta.json").read_text()),
            "server": [json.loads(p.read_text()) for p in root.rglob("config_fed_server.json")],
        }
        return "job-1"

    def monitor_job_and_return_job_meta(self, job_id, timeout=0.0, cb=None):
        self.monitored = True
        self.monitor_timeout = timeout
        cb(self, job_id, {"status": "RUNNING"})
        if self.status == "TIMEOUT":
            return MonitorReturnCode.TIMEOUT, None
        meta = {"status": self.status}
        cb(self, job_id, meta)
        return MonitorReturnCode.JOB_FINISHED, meta

    def do_command(self, command, props):
        if self.error:
            raise self.error
        if self.api_status is not None:
            return {ProtoKey.STATUS: self.api_status, ProtoKey.DETAILS: "simulated"}
        if command == SEARCH:
            payload = {"entries": [], "next_cursor": None, "total": 0, "facets": {}}
        elif command == STATUS:
            payload = {
                "clients": [
                    {
                        "client": "site-a",
                        "health": self.health,
                        "reason_code": "BUDGET_EXCEEDED" if self.health == "DEGRADED" else None,
                        "failures": 0,
                        "last_attempt_at": "2026-06-01T00:00:00Z",
                        "last_success_at": None,
                        "stale": False,
                    }
                ]
            }
        else:
            raise AssertionError(f"unexpected command: {command}")
        if self.reply_payload is not None:
            payload = self.reply_payload
        return {
            ProtoKey.STATUS: APIStatus.SUCCESS,
            ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.DICT, ProtoKey.DATA: payload}],
        }

    def get_job_meta(self, job_id):
        if self.error:
            raise self.error
        self.meta_reads += 1
        return {"status": self.status, CATALOG_APPLY_META: self.apply_status}

    def close(self):
        self.closed = True
