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


def test_check_client_replies_legacy_allows_timeout_reply():
    replies = [ClientReply(client_token="t1", client_name="C1", req=Message(topic="req", body=""), reply=None)]

    check_client_replies(replies=replies, client_sites=["C1"], command="start", strict=False)


def test_check_client_replies_strict_raises_for_timeout_reply():
    replies = [ClientReply(client_token="t1", client_name="C1", req=Message(topic="req", body=""), reply=None)]

    with pytest.raises(RuntimeError, match=r"no reply \(timeout\)"):
        check_client_replies(replies=replies, client_sites=["C1"], command="start", strict=True)


def test_check_client_replies_strict_raises_for_non_ok_return_code():
    replies = [_make_client_reply("C1", return_code=ReturnCode.ERROR, body="start failed")]

    with pytest.raises(RuntimeError, match="start failed"):
        check_client_replies(replies=replies, client_sites=["C1"], command="start", strict=True)


def test_check_client_replies_strict_raises_for_missing_client_reply():
    replies = [_make_client_reply("C1"), _make_client_reply("CX")]

    with pytest.raises(RuntimeError, match="missing replies from C2"):
        check_client_replies(replies=replies, client_sites=["C1", "C2"], command="start", strict=True)


def test_check_client_replies_strict_allows_reordered_success_replies():
    replies = [_make_client_reply("C2"), _make_client_reply("C1")]

    check_client_replies(replies=replies, client_sites=["C1", "C2"], command="start", strict=True)
