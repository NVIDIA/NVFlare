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
from unittest.mock import MagicMock, Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.admin_audit import AdminContext, audit_context, snapshot
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode, ok_reply
from nvflare.private.defs import RequestHeader, SysCommandTopic, TrainingTopic, new_cell_message
from nvflare.private.fed.client import admin
from nvflare.private.fed.client.admin import FedAdminAgent, RequestProcessor


@pytest.fixture
def audited_agent(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    writer = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    cell = Mock()
    cell.get_fqcn.return_value = "relay.site-2"
    engine = MagicMock()
    fl_ctx = FLContext()
    engine.new_context.return_value.__enter__.return_value = fl_ctx
    agent = FedAdminAgent("site-2", cell, engine)
    processor = Mock(spec=RequestProcessor)
    processor.get_topics.return_value = [
        SysCommandTopic.SHELL,
        TrainingTopic.NOTIFY_JOB_STATUS,
        TrainingTopic.CHECK_STATUS,
        TrainingTopic.CHECK_RESOURCE,
    ]
    processor.process.return_value = ok_reply(body="processed")
    agent.register_processor(processor)

    def records():
        return [json.loads(line) for line in path.read_text().splitlines()]

    try:
        yield agent, processor, fl_ctx, records
    finally:
        writer.close()


def _request(topic=SysCommandTopic.SHELL, *, origin="server", endpoint="relay", command=None):
    payload = Message(topic, "SECRET_RAW_ARGUMENTS /private/path")
    payload.set_header(RequestHeader.USER_NAME, "verified-admin@example.org")
    payload.set_header(ConnProps.EVENT_ID, "server-request")
    payload.set_header(RequestHeader.JOB_ID, "job-1")
    if command is not None:
        payload.set_header(RequestHeader.ADMIN_COMMAND, command)
    message = new_cell_message({MessageHeaderKey.ORIGIN: origin}, payload)
    if endpoint:
        message.set_prop(MessagePropKey.ENDPOINT, Endpoint(endpoint))
    return message


def _assert_safe(records):
    permitted = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}
    for record in records:
        assert set(record) <= permitted
        assert all(isinstance(value, str) for value in record.values())
    output = json.dumps(records)
    for excluded in ["SECRET_RAW_ARGUMENTS", "/private/path", "SECRET_EXCEPTION", "SECRET_POLICY"]:
        assert excluded not in output


@pytest.mark.parametrize("origin,endpoint", [("peer", "relay"), ("server", "peer"), ("server", None)])
def test_rejected_origin_cannot_supply_identity_or_reference(audited_agent, origin, endpoint):
    agent, processor, _, records = audited_agent
    reply = agent._dispatch_request(_request(origin=origin, endpoint=endpoint))
    assert reply.payload.get_header(MsgHeader.RETURN_CODE) == ReturnCode.ERROR
    processor.process.assert_not_called()
    events = records()
    assert len(events) == 1
    assert events[0]["action"] == "client.origin.rejected"
    assert events[0]["user"] == "unknown"
    assert "reference" not in events[0]
    assert "job" not in events[0]
    assert "verified-admin" not in json.dumps(events)
    _assert_safe(events)


@pytest.mark.parametrize("endpoint", ["server", "relay"])
def test_sensitive_receipt_and_return_reference_chain(audited_agent, endpoint):
    agent, processor, _, records = audited_agent
    request = _request(endpoint=endpoint)
    observed = []

    def handle(req, engine):
        observed.append(snapshot())
        return ok_reply(body="processed")

    processor.process.side_effect = handle
    with audit_context(AdminContext(user="outer")):
        assert agent._dispatch_request(request).payload.body == "processed"
        assert snapshot().user == "outer"
    events = records()
    assert [event["action"] for event in events] == ["client.command.received", "client.command.handler_returned"]
    receipt, outcome = events
    assert receipt["reference"] == "server-request"
    assert outcome["reference"] == receipt["event_id"]
    assert all(event["user"] == "verified-admin@example.org" for event in events)
    assert all("identity_source=trusted_server" in event["message"] for event in events)
    assert all(event["job"] == "job-1" for event in events)
    assert observed[0].request_id == receipt["event_id"]
    assert request.payload.get_header(ConnProps.EVENT_ID) == "server-request"
    _assert_safe(events)


def test_child_status_does_not_use_administrator_headers_or_add_records(audited_agent):
    agent, processor, _, records = audited_agent
    request = _request(TrainingTopic.NOTIFY_JOB_STATUS, origin="relay.site-2.job-1", endpoint="relay.site-2.job-1")
    observed = []
    processor.process.side_effect = lambda *args: observed.append(snapshot()) or ok_reply()
    with audit_context(AdminContext(user="outer-admin", request_id="outer-request")):
        agent._dispatch_request(request)
        assert snapshot().user == "outer-admin"
    assert records() == []
    assert observed == [AdminContext()]


@pytest.mark.parametrize(
    "topic,actions", [(TrainingTopic.CHECK_STATUS, [TrainingTopic.CHECK_STATUS]), (TrainingTopic.CHECK_RESOURCE, [])]
)
def test_routine_request_has_no_extra_handling_records(audited_agent, topic, actions):
    agent, _, _, records = audited_agent
    agent._dispatch_request(_request(topic))
    assert [event["action"] for event in records()] == actions


@pytest.mark.parametrize(
    "kind", ["site_denied", "site_error", "policy_denied", "policy_error", "policy_exception", "missing_command"]
)
def test_permission_decisions_are_recorded_once_without_secrets(audited_agent, monkeypatch, kind):
    agent, processor, _, records = audited_agent
    site_check = Mock(return_value=(True, ""))
    monkeypatch.setattr(admin.SiteSecurity, "authorization_check", site_check)
    policy_check = Mock(return_value=(True, ""))
    monkeypatch.setattr(admin.AuthorizationService, "authorize", policy_check)
    request = _request(command=None if kind == "missing_command" else "cat")
    request.payload.set_header(RequestHeader.REQUIRE_AUTHZ, "true")
    if kind == "site_denied":
        site_check.return_value = False, "SECRET_POLICY"
    elif kind == "site_error":
        site_check.side_effect = RuntimeError("SECRET_EXCEPTION")
    elif kind == "policy_denied":
        policy_check.return_value = False, ""
    elif kind == "policy_error":
        policy_check.return_value = False, "SECRET_POLICY"
    elif kind == "policy_exception":
        policy_check.side_effect = RuntimeError("SECRET_EXCEPTION")

    reply = agent._dispatch_request(request)
    assert reply.payload.get_header(MsgHeader.RETURN_CODE) == ReturnCode.ERROR
    processor.process.assert_not_called()
    events = records()
    decisions = [event for event in events if event["action"].startswith("client.permission.")]
    assert len(decisions) == 1
    assert decisions[0]["reference"] == events[0]["event_id"]
    assert all(event["user"] == "verified-admin@example.org" for event in events)
    assert "client.command.handler_returned" not in [event["action"] for event in events]
    _assert_safe(events)


def test_site_policy_event_processing_error_is_observed_without_changing_decision(audited_agent, monkeypatch):
    agent, processor, fl_ctx, records = audited_agent
    fl_ctx.set_prop(FLContextKey.EXCEPTIONS, {"policy": RuntimeError("SECRET_POLICY")})
    monkeypatch.setattr(admin.SiteSecurity, "authorization_check", lambda *args: (True, ""))
    reply = agent._dispatch_request(_request(command="cat"))
    assert reply.payload.body == "processed"
    processor.process.assert_called_once()
    decisions = [event for event in records() if event["action"] == "client.permission.error"]
    assert len(decisions) == 1
    assert "reason=site_policy_error" in decisions[0]["message"]
    _assert_safe(records())


@pytest.mark.parametrize(
    "failure,expected_stage,returned",
    [
        ("setup", "pre_handler", False),
        ("handler", "handler", False),
        ("reply", "post_handler", True),
        ("exit", "post_handler", True),
    ],
)
def test_handling_errors_describe_stage_and_preserve_context(audited_agent, failure, expected_stage, returned):
    agent, processor, _, records = audited_agent
    if failure == "setup":
        agent.app_ctx.new_context.side_effect = RuntimeError("SECRET_EXCEPTION")
    elif failure == "handler":
        processor.process.side_effect = RuntimeError("SECRET_EXCEPTION")
    elif failure == "reply":
        processor.process.return_value = "invalid-reply"
    elif failure == "exit":
        agent.app_ctx.new_context.return_value.__exit__.side_effect = RuntimeError("SECRET_EXCEPTION")
    if failure in ("setup", "exit"):
        with pytest.raises(RuntimeError):
            agent._dispatch_request(_request())
    else:
        reply = agent._dispatch_request(_request())
        assert reply.payload.get_header(MsgHeader.RETURN_CODE) == ReturnCode.ERROR
    assert snapshot() == AdminContext()
    events = records()
    errors = [event for event in events if event["action"] == "client.command.error"]
    assert len(errors) == 1
    assert "stage=" + expected_stage in errors[0]["message"]
    assert errors[0]["reference"] == events[0]["event_id"]
    assert ("client.command.handler_returned" in [event["action"] for event in events]) == returned
    _assert_safe(events)


def test_recording_failure_cannot_block_or_invent_receipt_reference(audited_agent, monkeypatch):
    agent, processor, _, records = audited_agent
    original = agent.auditor.add_event
    attempts = 0

    def fail_receipt(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("SECRET_EXCEPTION")
        return original(**kwargs)

    monkeypatch.setattr(agent.auditor, "add_event", fail_receipt)
    assert agent._dispatch_request(_request()).payload.body == "processed"
    processor.process.assert_called_once()
    events = records()
    assert len(events) == 1
    assert events[0]["action"] == "client.command.handler_returned"
    assert "reference" not in events[0]
    _assert_safe(events)
