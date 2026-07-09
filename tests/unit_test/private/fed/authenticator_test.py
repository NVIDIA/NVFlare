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

import logging

import pytest

from nvflare.apis.fl_constant import ServerCommandNames
from nvflare.fuel.f3.cellnet.defs import CellChannel, CellChannelTopic, MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.stream_const import STREAM_CHANNEL, STREAM_DATA_TOPIC
from nvflare.private.defs import CellMessageHeaderKeys
from nvflare.private.fed.authenticator import MISSING_CLIENT_FQCN, validate_auth_headers


class _TokenVerifier:
    def verify(self, _client_name, _token, _signature):
        return True


def _make_message(origin, channel=CellChannel.SERVER_COMMAND, topic=ServerCommandNames.GET_TASK):
    return Message(
        headers={
            MessageHeaderKey.CHANNEL: channel,
            MessageHeaderKey.TOPIC: topic,
            MessageHeaderKey.ORIGIN: origin,
            CellMessageHeaderKeys.CLIENT_NAME: "site-a",
            CellMessageHeaderKeys.TOKEN: "token-a",
            CellMessageHeaderKeys.TOKEN_SIGNATURE: "sig-a",
        }
    )


def _validate(origin, client_fqcn_resolver, channel=CellChannel.SERVER_COMMAND, topic=ServerCommandNames.GET_TASK):
    return validate_auth_headers(
        message=_make_message(origin, channel=channel, topic=topic),
        token_verifier=_TokenVerifier(),
        logger=logging.getLogger(__name__),
        client_fqcn_resolver=client_fqcn_resolver,
    )


@pytest.mark.parametrize("origin", ["site-a", "site-a.job-1", "site-a.site-a-child.job-1"])
def test_validate_auth_headers_accepts_token_from_registered_origin(origin):
    # Job and hierarchical child cells under a registered site are still part of that site's trust boundary.
    assert _validate(origin, lambda _client_name, _token: "site-a") is None


@pytest.mark.parametrize(
    "origin",
    [
        "site-a_8065f1c4-fd35-47ef-b945-800f4d0d5176_active",
        "site-a_8065f1c4-fd35-47ef-b945-800f4d0d5176_passive",
    ],
)
def test_validate_auth_headers_accepts_direct_cell_pipe_stream_alias(origin):
    assert (
        _validate(
            origin,
            lambda _client_name, _token: "site-a",
            channel=STREAM_CHANNEL,
            topic=STREAM_DATA_TOPIC,
        )
        is None
    )


def test_validate_auth_headers_rejects_token_from_different_origin():
    reply = _validate("site-b", lambda _client_name, _token: "site-a")

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_validate_auth_headers_rejects_cell_pipe_alias_for_different_client():
    reply = _validate(
        "site-b_8065f1c4-fd35-47ef-b945-800f4d0d5176_passive",
        lambda _client_name, _token: "site-a",
        channel=STREAM_CHANNEL,
        topic=STREAM_DATA_TOPIC,
    )

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_validate_auth_headers_rejects_cell_pipe_alias_on_non_stream_channel():
    reply = _validate(
        "site-a_8065f1c4-fd35-47ef-b945-800f4d0d5176_passive",
        lambda _client_name, _token: "site-a",
    )

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_validate_auth_headers_rejects_cell_pipe_alias_with_hierarchical_runtime_id():
    reply = _validate(
        "site-a_8065f1c4-fd35-47ef-b945-800f4d0d5176.child_passive",
        lambda _client_name, _token: "site-a",
        channel=STREAM_CHANNEL,
        topic=STREAM_DATA_TOPIC,
    )

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_validate_auth_headers_rejects_cell_pipe_alias_with_underscore_runtime_id():
    reply = _validate(
        "site-a_simulate_job_passive",
        lambda _client_name, _token: "site-a",
        channel=STREAM_CHANNEL,
        topic=STREAM_DATA_TOPIC,
    )

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_validate_auth_headers_rejects_underscore_prefixed_client_ambiguity():
    reply = _validate(
        "site-a_x_8065f1c4-fd35-47ef-b945-800f4d0d5176_passive",
        lambda _client_name, _token: "site-a",
        channel=STREAM_CHANNEL,
        topic=STREAM_DATA_TOPIC,
    )

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_validate_auth_headers_rejects_registered_token_with_missing_origin_binding():
    reply = _validate("site-a", lambda _client_name, _token: MISSING_CLIENT_FQCN)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_validate_auth_headers_keeps_compatibility_when_origin_cannot_be_resolved():
    assert _validate("site-b", lambda _client_name, _token: None) is None


def _make_bye_message(destination, to_cell=None):
    # Cellnet Cell.stop() broadcasts an empty Message() carrying no FL auth headers;
    # the transport stamps DESTINATION/TO_CELL. A crafted bye can set them to anything.
    headers = {
        MessageHeaderKey.CHANNEL: CellChannel.CELLNET,
        MessageHeaderKey.TOPIC: CellChannelTopic.Bye,
        MessageHeaderKey.ORIGIN: "attacker",
        MessageHeaderKey.DESTINATION: destination,
    }
    if to_cell is not None:
        headers[MessageHeaderKey.TO_CELL] = to_cell
    return Message(headers=headers)


def _validate_bye(destination, local_cell_fqcn, to_cell=None):
    return validate_auth_headers(
        message=_make_bye_message(destination, to_cell=to_cell),
        token_verifier=_TokenVerifier(),
        logger=logging.getLogger(__name__),
        local_cell_fqcn=local_cell_fqcn,
    )


def test_validate_auth_headers_bypasses_bye_terminating_at_this_cell():
    # A legitimate bye is a direct-neighbor message: DESTINATION == the receiving cell's own FQCN.
    assert _validate_bye(destination="server", local_cell_fqcn="server") is None


def test_validate_auth_headers_rejects_forwarded_bye_for_other_cell():
    # A crafted bye that names another cell as DESTINATION (so this cell would forward it and evict
    # that cell's upstream agent) must NOT bypass auth — DESTINATION != this cell's FQCN.
    reply = _validate_bye(destination="site-1", local_cell_fqcn="server")

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_validate_auth_headers_rejects_bye_forged_to_cell_matching_destination():
    # The old guard compared two sender-controlled headers (TO_CELL == DESTINATION); an attacker can
    # satisfy that while DESTINATION still points at another cell. The FQCN-based guard rejects it.
    reply = _validate_bye(destination="site-1", local_cell_fqcn="server", to_cell="site-1")

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_validate_auth_headers_does_not_bypass_bye_when_local_fqcn_unknown():
    # Without a known local FQCN the bypass must not trigger (fail closed).
    reply = _validate_bye(destination="server", local_cell_fqcn=None)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED
