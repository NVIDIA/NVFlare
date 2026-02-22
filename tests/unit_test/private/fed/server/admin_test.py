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

import pytest

from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.fed.server.admin import check_client_replies
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

    with pytest.raises(RuntimeError, match="missing replies from C2"):
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
