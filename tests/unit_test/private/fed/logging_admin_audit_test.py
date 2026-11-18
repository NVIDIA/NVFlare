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
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.sec.admin_audit import AdminContext, audit_context
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.admin_defs import Message, error_reply, ok_reply
from nvflare.private.defs import RequestHeader, SysCommandTopic, TrainingTopic
from nvflare.private.fed.client import admin_commands, sys_cmd, training_cmds
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.fed.server import server_commands
from nvflare.private.fed.server import sys_cmd as server_sys_cmd
from nvflare.private.fed.server.server_engine import ServerEngine


@pytest.fixture
def recording(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    writer = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    try:
        yield lambda: [json.loads(line) for line in path.read_text().splitlines()]
    finally:
        writer.close()


def _workspace():
    workspace = Mock()
    workspace.get_run_dir.return_value = "/private/workspace/job-1"
    workspace.get_root_dir.return_value = "/private/workspace"
    workspace.get_log_config_file_path.return_value = "/private/log-config.json"
    return workspace


def _check_record(event, action, *, job=None, user="administrator"):
    assert event["action"] == action
    assert event["user"] == user
    assert event.get("job") == job
    assert set(event) <= {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}
    assert all(isinstance(value, str) for value in event.values())
    assert "/private/" not in json.dumps(event)
    assert "SECRET" not in json.dumps(event)
    assert "DEBUG" not in json.dumps(event)


@pytest.mark.parametrize("module", [admin_commands, server_commands])
@pytest.mark.parametrize("failed", [False, True])
def test_actual_job_log_application_is_recorded_at_owner(recording, monkeypatch, module, failed):
    dynamic = Mock(side_effect=RuntimeError("SECRET_EXCEPTION /private/path") if failed else None)
    monkeypatch.setattr(module, "dynamic_log_config", dynamic)
    engine = Mock()
    engine.get_workspace.return_value = _workspace()
    fl_ctx = Mock()
    fl_ctx.get_engine.return_value = engine
    fl_ctx.get_job_id.return_value = "job-1"
    with audit_context(AdminContext(user="administrator", request_id="request")):
        result = module.ConfigureJobLogCommand().process("DEBUG", fl_ctx)
    dynamic.assert_called_once()
    assert bool(result) == failed
    events = recording()
    assert len(events) == 1
    _check_record(events[0], "logging.job.error" if failed else "logging.job.applied", job="job-1")
    assert events[0]["reference"] == "request"


@pytest.mark.parametrize("module", [admin_commands, server_commands])
def test_child_without_administrator_context_keeps_actor_unknown(recording, monkeypatch, module):
    monkeypatch.setattr(module, "dynamic_log_config", Mock())
    engine = Mock()
    engine.get_workspace.return_value = _workspace()
    fl_ctx = Mock()
    fl_ctx.get_engine.return_value = engine
    fl_ctx.get_job_id.return_value = "job-1"
    module.ConfigureJobLogCommand().process("DEBUG", fl_ctx)
    events = recording()
    assert len(events) == 1
    _check_record(events[0], "logging.job.applied", job="job-1", user="unknown")
    assert "reference" not in events[0]


@pytest.mark.parametrize("failed", [False, True])
def test_actual_client_site_log_application_and_error(recording, monkeypatch, failed):
    dynamic = Mock(side_effect=RuntimeError("SECRET_EXCEPTION /private/path") if failed else None)
    monkeypatch.setattr(sys_cmd, "dynamic_log_config", dynamic)
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, _workspace())
    engine = Mock()
    engine.new_context.return_value = fl_ctx
    request = Message(SysCommandTopic.CONFIGURE_SITE_LOG, "DEBUG")
    with audit_context(AdminContext(user="administrator", request_id="receipt", identity_source="trusted_server")):
        sys_cmd.ConfigureSiteLogProcessor().process(request, engine)
    events = recording()
    assert len(events) == 1
    _check_record(events[0], "logging.site.error" if failed else "logging.site.applied")
    assert events[0]["reference"] == "receipt"
    assert "identity_source=trusted_server" in events[0]["message"]


@pytest.mark.parametrize("failed", [False, True])
def test_actual_server_site_log_application_and_error(recording, monkeypatch, failed):
    dynamic = Mock(side_effect=RuntimeError("SECRET_EXCEPTION /private/path") if failed else None)
    monkeypatch.setattr(server_sys_cmd, "dynamic_log_config", dynamic)
    engine = Mock(spec=ServerEngine)
    engine.get_workspace.return_value = _workspace()
    conn = Mock()
    conn.app_ctx = engine
    with audit_context(AdminContext(user="administrator", request_id="request")):
        server_sys_cmd.SystemCommandModule().configure_site_log(conn, ["configure_site_log", "server", "DEBUG"])
    events = recording()
    assert len(events) == 1
    _check_record(events[0], "logging.site.error" if failed else "logging.site.applied")
    assert events[0]["reference"] == "request"


def test_remote_site_results_are_acknowledgements_not_applied_facts(recording, monkeypatch):
    module = server_sys_cmd.SystemCommandModule()
    replies = [
        SimpleNamespace(client_name="site-1", reply=ok_reply(body="arbitrary-text")),
        SimpleNamespace(client_name="site-2", reply=error_reply("SECRET_REMOTE_ERROR")),
        SimpleNamespace(client_name="site-3", reply=None),
    ]
    monkeypatch.setattr(module, "send_request_to_clients", lambda *args: replies)
    monkeypatch.setattr(module, "process_replies_to_table", Mock())
    conn = Mock()
    with audit_context(AdminContext(user="administrator", request_id="request")):
        module.configure_site_log(conn, ["configure_site_log", "client", "DEBUG"])
    events = recording()
    assert len(events) == 3
    for event in events:
        _check_record(event, "logging.site.remote_reported")
        assert event["reference"] == "request"
    assert "status=acknowledged" in events[0]["message"]
    assert "site=site-1" in events[0]["message"]
    assert "status=error" in events[1]["message"]
    assert "site=site-2" in events[1]["message"]
    assert "status=no_reply" in events[2]["message"]
    assert "site=site-3" in events[2]["message"]


@pytest.mark.parametrize("failed", [False, True])
def test_client_reports_job_child_reply_without_claiming_local_application(recording, failed):
    engine = Mock(spec=ClientEngineInternalSpec)
    engine.configure_job_log = Mock(return_value="SECRET_REMOTE_ERROR" if failed else None)
    request = Message(TrainingTopic.CONFIGURE_JOB_LOG, "DEBUG")
    request.set_header(RequestHeader.JOB_ID, "job-1")
    with audit_context(AdminContext(user="administrator", request_id="receipt", identity_source="trusted_server")):
        training_cmds.ConfigureJobLogProcessor().process(request, engine)
    events = recording()
    assert len(events) == 1
    _check_record(events[0], "logging.job.remote_reported", job="job-1")
    assert events[0]["reference"] == "receipt"
    assert "status=" + ("error" if failed else "acknowledged") in events[0]["message"]


def test_job_log_audit_context_preparation_failure_does_not_change_success(recording, monkeypatch):
    dynamic = Mock()
    monkeypatch.setattr(admin_commands, "dynamic_log_config", dynamic)
    engine = Mock()
    engine.get_workspace.return_value = _workspace()
    fl_ctx = Mock()
    fl_ctx.get_engine.return_value = engine
    # The first access is required by the operation; the second is audit-only.
    fl_ctx.get_job_id.side_effect = ["job-1", RuntimeError("SECRET_EXCEPTION")]
    assert admin_commands.ConfigureJobLogCommand().process("DEBUG", fl_ctx) is None
    dynamic.assert_called_once()
    assert len(recording()) == 1
    assert "job" not in recording()[0]
