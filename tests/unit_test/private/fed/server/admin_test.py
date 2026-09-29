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

from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.f3.cellnet.defs import CellChannel
from nvflare.fuel.hci.proto import StreamChannel
from nvflare.fuel.hci.server.hci import AdminServer
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.fed.server.admin import FedAdminServer, check_client_replies
from nvflare.private.fed.server.message_send import ClientReply


def _make_client_reply(client_name: str, return_code=ReturnCode.OK, body="ok"):
    req = Message(topic="req", body="")
    reply = Message(topic="reply", body=body)
    reply.set_header(MsgHeader.RETURN_CODE, return_code)
    return ClientReply(client_token=f"token-{client_name}", client_name=client_name, req=req, reply=reply)


def _make_timeout_reply(client_name: str):
    """Simulate a client that did not respond (reply=None)."""
    return ClientReply(
        client_token=f"token-{client_name}", client_name=client_name, req=Message(topic="req", body=""), reply=None
    )


# ---------------------------------------------------------------------------
# Legacy (non-strict) mode
# ---------------------------------------------------------------------------


def test_check_client_replies_legacy_allows_timeout_reply():
    """Non-strict mode silently accepts a timeout reply."""
    replies = [_make_timeout_reply("C1")]

    result = check_client_replies(replies=replies, client_sites=["C1"], command="start", strict=False)

    assert result == []


def test_check_client_replies_legacy_uses_dict_lookup_not_zip():
    """Non-strict mode uses name-keyed lookup; reply order does not matter."""
    # Replies in reverse order of client_sites — old zip() would give wrong names.
    replies = [_make_client_reply("C2"), _make_client_reply("C1")]

    result = check_client_replies(replies=replies, client_sites=["C1", "C2"], command="start", strict=False)

    assert result == []


# ---------------------------------------------------------------------------
# Strict mode — timeouts
# ---------------------------------------------------------------------------


def test_check_client_replies_strict_returns_timed_out_clients():
    """In strict mode a timeout reply is returned as a timed-out client, NOT raised."""
    replies = [_make_timeout_reply("C1")]

    timed_out = check_client_replies(replies=replies, client_sites=["C1"], command="start", strict=True)

    assert timed_out == ["C1"]


def test_check_client_replies_strict_returns_only_timed_out_clients():
    """Mixed: one OK, one timeout — only the timed-out client is returned."""
    replies = [_make_client_reply("C1"), _make_timeout_reply("C2")]

    timed_out = check_client_replies(replies=replies, client_sites=["C1", "C2"], command="start", strict=True)

    assert timed_out == ["C2"]


def test_check_client_replies_strict_no_timeouts_returns_empty():
    """All clients responded successfully — returns empty list."""
    replies = [_make_client_reply("C1"), _make_client_reply("C2")]

    result = check_client_replies(replies=replies, client_sites=["C1", "C2"], command="start", strict=True)

    assert result == []


# ---------------------------------------------------------------------------
# Strict mode — explicit errors always raise
# ---------------------------------------------------------------------------


def test_check_client_replies_strict_raises_for_non_ok_return_code():
    replies = [_make_client_reply("C1", return_code=ReturnCode.ERROR, body="start failed")]

    with pytest.raises(RuntimeError, match="start failed"):
        check_client_replies(replies=replies, client_sites=["C1"], command="start", strict=True)


def test_check_client_replies_strict_raises_for_missing_client_reply():
    """Structurally missing entry (client not in replies dict at all) always raises."""
    replies = [_make_client_reply("C1"), _make_client_reply("CX")]

    with pytest.raises(RuntimeError, match=r"missing replies from \["):
        check_client_replies(replies=replies, client_sites=["C1", "C2"], command="start", strict=True)


def test_check_client_replies_strict_raises_but_not_for_timeout_when_mixed():
    """If one client has explicit error and another times out, explicit error raises."""
    replies = [_make_client_reply("C1", return_code=ReturnCode.ERROR, body="err"), _make_timeout_reply("C2")]

    with pytest.raises(RuntimeError, match="err"):
        check_client_replies(replies=replies, client_sites=["C1", "C2"], command="start", strict=True)


# ---------------------------------------------------------------------------
# Strict mode — reply ordering
# ---------------------------------------------------------------------------


def test_check_client_replies_strict_allows_reordered_success_replies():
    replies = [_make_client_reply("C2"), _make_client_reply("C1")]

    check_client_replies(replies=replies, client_sites=["C1", "C2"], command="start", strict=True)


# ---------------------------------------------------------------------------
# Non-strict mode — ERROR_MSG_PREFIX detection
# ---------------------------------------------------------------------------


def test_check_client_replies_legacy_raises_when_body_starts_with_error_prefix():
    """Non-strict mode raises when reply body starts with ERROR_MSG_PREFIX."""
    from nvflare.private.defs import ERROR_MSG_PREFIX

    replies = [_make_client_reply("C1", body=f"{ERROR_MSG_PREFIX}: something went wrong")]

    with pytest.raises(RuntimeError, match="something went wrong"):
        check_client_replies(replies=replies, client_sites=["C1"], command="start", strict=False)


def test_check_client_replies_legacy_does_not_raise_when_prefix_not_at_start():
    """Non-strict mode uses startswith — a body containing the prefix mid-string is NOT an error."""
    from nvflare.private.defs import ERROR_MSG_PREFIX

    replies = [_make_client_reply("C1", body=f"info: see {ERROR_MSG_PREFIX} for details")]

    result = check_client_replies(replies=replies, client_sites=["C1"], command="start", strict=False)

    assert result == []


def test_admin_server_registers_hci_command_and_upload_handlers_by_default():
    cell = MagicMock()
    cmd_reg = MagicMock()
    engine = MagicMock()
    fl_ctx = MagicMock()
    engine.new_context.return_value = fl_ctx

    with patch("nvflare.fuel.hci.server.hci.FileStreamer.register_stream_processing") as register_stream:
        server = AdminServer(cell=cell, cmd_reg=cmd_reg, engine=engine)

    cmd_reg.finalize.assert_called_once_with()
    cell.register_request_cb.assert_called_once_with(
        channel=CellChannel.HCI,
        topic="*",
        cb=server._process_admin_request,
    )
    register_stream.assert_called_once_with(
        fl_ctx=fl_ctx,
        channel=StreamChannel.UPLOAD,
        topic="*",
        stream_done_cb=server._process_upload,
    )


def test_admin_server_disabled_registers_no_hci_command_or_upload_handlers():
    cell = MagicMock()
    engine = MagicMock()

    with patch("nvflare.fuel.hci.server.hci.FileStreamer.register_stream_processing") as register_stream:
        server = AdminServer(cell=cell, cmd_reg=None, engine=engine, enable_hci=False)

    assert server.cmd_reg is None
    assert server.cred_keeper is None
    cell.register_request_cb.assert_not_called()
    engine.new_context.assert_not_called()
    register_stream.assert_not_called()

    server.start()
    server.stop()


def test_fed_admin_server_disabled_preserves_outbound_helpers_without_admin_components():
    cell = MagicMock()
    fed_admin_interface = MagicMock()

    with (
        patch("nvflare.private.fed.server.admin.new_command_register_with_builtin_module") as new_cmd_reg,
        patch("nvflare.private.fed.server.admin.SessionManager") as session_manager,
        patch("nvflare.private.fed.server.admin.LoginModule") as login_module,
        patch("nvflare.private.fed.server.admin.AuthzFilter") as authz_filter,
        patch("nvflare.private.fed.server.admin.CommandAudit") as command_audit,
        patch("nvflare.private.fed.server.admin.AuditService.get_auditor") as get_auditor,
        patch("nvflare.private.fed.server.admin.NetAgent") as net_agent,
        patch("nvflare.private.fed.server.admin.NetManager") as net_manager,
        patch("nvflare.private.fed.server.admin.mpm.add_cleanup_cb") as add_cleanup_cb,
    ):
        server = FedAdminServer(
            cell=cell,
            fed_admin_interface=fed_admin_interface,
            cmd_modules=[],
            file_upload_dir="upload",
            file_download_dir="download",
            enable_hci=False,
        )

    new_cmd_reg.assert_not_called()
    session_manager.assert_not_called()
    login_module.assert_not_called()
    authz_filter.assert_not_called()
    command_audit.assert_not_called()
    get_auditor.assert_not_called()
    net_agent.assert_not_called()
    net_manager.assert_not_called()
    add_cleanup_cb.assert_not_called()
    assert server.sess_mgr is None
    assert server.net_agent is None
    assert server.net_mgr is None

    server.client_heartbeat("token", "site-1", "site-1")
    assert server.get_client_tokens() == ["token"]

    request = Message(topic="request", body="")
    fl_ctx = MagicMock()
    with (
        patch("nvflare.private.fed.server.admin.gen_new_peer_ctx", return_value=MagicMock()),
        patch("nvflare.private.fed.server.admin.send_requests", return_value=[]) as send_client_requests,
    ):
        replies = server.send_requests({"token": request}, fl_ctx)

    assert replies == []
    fed_admin_interface.fire_event.assert_called_once()
    send_client_requests.assert_called_once_with(
        cell=cell,
        command="admin",
        requests={"token": request},
        clients=server.clients,
        timeout_secs=2.0,
        optional=False,
    )
