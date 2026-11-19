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
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.sec.admin_audit import audit_context
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.fed.client import client_engine as subject
from nvflare.private.fed.client.client_engine import ClientEngine


@pytest.mark.parametrize(
    ("operation", "failure"),
    [(op, failure) for op in ("shutdown", "restart") for failure in ("none", "thread_start", "event")]
    + [("shutdown", "cleanup")],
)
def test_client_control_reports_scheduling_and_preserves_cleanup_failure(tmp_path, monkeypatch, operation, failure):
    path = tmp_path / "audit.jsonl"
    writer = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    engine = ClientEngine.__new__(ClientEngine)
    engine.client = object()
    engine.logger = MagicMock()
    engine.args = SimpleNamespace(workspace="/private-path")
    engine.new_context = MagicMock(return_value=object())
    engine.fire_event = MagicMock()
    engine.shutdown_streamer = MagicMock()
    thread = MagicMock()
    monkeypatch.setattr(subject.threading, "Thread", MagicMock(return_value=thread))
    if failure == "thread_start":
        thread.start.side_effect = RuntimeError("private-path")
    if failure == "event":
        engine.fire_event.side_effect = RuntimeError("private-path")
    if failure == "cleanup":
        engine.shutdown_streamer.side_effect = RuntimeError("private-path")
    try:
        with audit_context(
            user="admin@example.com", command=operation, request_id="receipt-event", identity_source="trusted_server"
        ):
            if failure == "none":
                getattr(engine, operation)()
            else:
                with pytest.raises(RuntimeError):
                    getattr(engine, operation)()
        records = [json.loads(line) for line in path.read_text().splitlines()]
        expected = (
            [f"client.{operation}.error"] if failure in ("thread_start", "event") else [f"client.{operation}.scheduled"]
        )
        if failure == "cleanup":
            expected.append("client.shutdown.cleanup_error")
        assert [item["action"] for item in records] == expected
        for item in records:
            assert item["user"] == "admin@example.com"
            assert item["reference"] == "receipt-event"
            assert "identity_source=trusted_server" in item["message"]
        assert "private-path" not in path.read_text()
    finally:
        writer.close()
