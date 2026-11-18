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
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import ProtoKey
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.admin_audit import AdminContext, audit_context
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode, error_reply, ok_reply
from nvflare.private.defs import RequestHeader, SysCommandTopic, TrainingTopic
from nvflare.private.fed.client import admin_commands, sys_cmd, training_cmds
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.fed.server import server_commands
from nvflare.private.fed.server import sys_cmd as server_sys_cmd
from nvflare.private.fed.server.admin import FedAdminServer
from nvflare.private.fed.server.message_send import ClientReply
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
        SimpleNamespace(client_token="token-1", client_name="site-1", reply=ok_reply(body="arbitrary-text")),
        SimpleNamespace(client_token="token-2", client_name="site-2", reply=error_reply("SECRET_REMOTE_ERROR")),
        SimpleNamespace(client_token="token-3", client_name="site-3", reply=None),
    ]
    monkeypatch.setattr(module, "send_request_to_clients", lambda *args: replies)
    monkeypatch.setattr(module, "process_replies_to_table", Mock())
    conn = Connection(
        props={
            module.TARGET_CLIENT_TOKENS: ["token-1", "token-2", "token-3"],
            module.TARGET_CLIENTS: {"token-1": "site-1", "token-2": "site-2", "token-3": "site-3"},
        }
    )
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


@pytest.mark.parametrize(
    "child_result, status", [(None, "unknown_outcome"), ("", "unknown_outcome"), ("SECRET_REMOTE_ERROR", "error")]
)
def test_client_reports_job_child_reply_without_claiming_local_application(recording, child_result, status):
    engine = Mock(spec=ClientEngineInternalSpec)
    engine.configure_job_log = Mock(return_value=child_result)
    request = Message(TrainingTopic.CONFIGURE_JOB_LOG, "DEBUG")
    request.set_header(RequestHeader.JOB_ID, "job-1")
    with audit_context(AdminContext(user="administrator", request_id="receipt", identity_source="trusted_server")):
        reply = training_cmds.ConfigureJobLogProcessor().process(request, engine)
    events = recording()
    assert len(events) == 1
    _check_record(events[0], "logging.job.remote_reported", job="job-1")
    assert events[0]["reference"] == "receipt"
    assert "status=" + status in events[0]["message"]
    assert reply.get_header(MsgHeader.RETURN_CODE) == (ReturnCode.ERROR if child_result else ReturnCode.OK)
    engine.configure_job_log.assert_called_once_with("job-1", "DEBUG")


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


@pytest.mark.parametrize(
    "code, expected",
    [
        (ReturnCode.OK, "acknowledged"),
        (ReturnCode.ERROR, "error"),
        (None, "unknown_outcome"),
        ("", "unknown_outcome"),
        ("SECRET_CUSTOM_CODE", "unknown_outcome"),
    ],
)
def test_site_logging_requires_explicit_success_code(recording, monkeypatch, code, expected):
    module = server_sys_cmd.SystemCommandModule()
    reply = Message("reply", "SECRET_REMOTE_BODY")
    if code is not None:
        reply.set_header(MsgHeader.RETURN_CODE, code)
    result = ClientReply("SECRET_TOKEN", "site-1", None, reply)
    send = Mock(return_value=[result])
    monkeypatch.setattr(module, "send_request_to_clients", send)
    handled = Mock()
    monkeypatch.setattr(module, "process_replies_to_table", handled)
    conn = Connection(
        props={module.TARGET_CLIENT_TOKENS: ["SECRET_TOKEN"], module.TARGET_CLIENTS: {"SECRET_TOKEN": "site-1"}}
    )
    with audit_context(AdminContext(user="administrator", request_id="request")):
        module.configure_site_log(conn, ["configure_site_log", "client", "DEBUG"])
    send.assert_called_once()
    handled.assert_called_once_with(conn, [result])
    events = recording()
    assert len(events) == 1
    _check_record(events[0], "logging.site.remote_reported")
    assert events[0]["reference"] == "request"
    assert "status=" + expected in events[0]["message"]
    assert "site=site-1" in events[0]["message"]


def test_site_logging_retains_disconnected_selected_target_through_real_send(recording):
    module = server_sys_cmd.SystemCommandModule()
    selected = {"SECRET_ALIVE": "site-alive", "SECRET_GONE": "site-gone", "SECRET_EMPTY": "site-empty"}
    tokens = list(selected)
    clients = {token: SimpleNamespace(token=token, name=name, fqcn=name) for token, name in selected.items()}
    admin = FedAdminServer.__new__(FedAdminServer)
    admin.clients = clients
    admin.timeout = 2.0
    admin.sai = SimpleNamespace(new_context=FLContext, fire_event=Mock())
    broadcast = Mock(
        return_value={
            "site-alive": CellMessage(payload=ok_reply(body="reply-body")),
            "site-empty": CellMessage(payload=None),
        }
    )
    admin.cell = SimpleNamespace(broadcast_multi_requests=broadcast)
    original_send = admin.send_requests
    conn = Connection(
        props={module.TARGET_CLIENT_TOKENS: tokens, module.TARGET_CLIENTS: selected, ConnProps.ADMIN_SERVER: admin}
    )

    def disconnect_before_send(requests, fl_ctx, **kwargs):
        clients.pop("SECRET_GONE")
        selected.pop("SECRET_GONE")
        return original_send(requests, fl_ctx, **kwargs)

    admin.send_requests = Mock(side_effect=disconnect_before_send)
    with audit_context(AdminContext(user="administrator", request_id="request")):
        module.configure_site_log(conn, ["configure_site_log", "client", "DEBUG"])
    admin.send_requests.assert_called_once()
    broadcast.assert_called_once()
    assert set(broadcast.call_args.args[0]) == {"site-alive", "site-empty"}
    events = recording()
    assert len(events) == 3
    for event, site, status in zip(
        events, ["site-alive", "site-gone", "site-empty"], ["acknowledged", "no_reply", "no_reply"]
    ):
        _check_record(event, "logging.site.remote_reported")
        assert event["reference"] == "request"
        assert "site=" + site in event["message"]
        assert "status=" + status in event["message"]
    assert conn.buffer.data == [
        {
            ProtoKey.TYPE: ProtoKey.TABLE,
            ProtoKey.ROWS: [["CLIENT", "RESPONSE"], ["site-alive", "reply-body"], ["site-empty", ""]],
        }
    ]


def test_site_logging_records_each_selected_token_once_and_ignores_unselected_names(recording):
    replies = [
        ClientReply("SECRET_SELECTED", "SECRET_UNTRUSTED_NAME", None, ok_reply()),
        ClientReply("SECRET_SELECTED", "SECRET_DUPLICATE", None, error_reply("SECRET_ERROR")),
        ClientReply("SECRET_UNSELECTED", "SECRET_UNSELECTED_NAME", None, ok_reply()),
    ]
    with audit_context(AdminContext(user="administrator", request_id="request")):
        server_sys_cmd.SystemCommandModule._record_logging_replies(
            replies, {"SECRET_SELECTED": "site-1", "SECRET_MISSING": "site-2"}
        )
    events = recording()
    assert len(events) == 2
    assert "status=acknowledged" in events[0]["message"] and "site=site-1" in events[0]["message"]
    assert "status=no_reply" in events[1]["message"] and "site=site-2" in events[1]["message"]
    for event in events:
        _check_record(event, "logging.site.remote_reported")


@pytest.mark.parametrize("failure", ["getter", "iteration"])
def test_site_logging_target_snapshot_failure_preserves_send_and_reply(recording, monkeypatch, failure):
    module = server_sys_cmd.SystemCommandModule()

    class BrokenMapping:
        def __iter__(self):
            raise RuntimeError("SECRET_MAPPING_FAILURE")

    class AuditConnection(Connection):
        def get_prop(self, key, default=None):
            if key == module.TARGET_CLIENTS and failure == "getter":
                raise RuntimeError("SECRET_GETTER_FAILURE")
            return super().get_prop(key, default)

    conn = AuditConnection(
        props={module.TARGET_CLIENT_TOKENS: ["SECRET_TOKEN"], module.TARGET_CLIENTS: BrokenMapping()}
    )
    result = ClientReply("SECRET_TOKEN", "site-1", None, ok_reply(body="response"))
    send = Mock(return_value=[result])
    monkeypatch.setattr(module, "send_request_to_clients", send)
    with audit_context(AdminContext(user="administrator", request_id="request")):
        module.configure_site_log(conn, ["configure_site_log", "client", "DEBUG"])
    send.assert_called_once()
    assert conn.buffer.data == [
        {ProtoKey.TYPE: ProtoKey.TABLE, ProtoKey.ROWS: [["CLIENT", "RESPONSE"], ["site-1", "response"]]}
    ]
    assert recording() == []
