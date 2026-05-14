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
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.private.defs import CellMessageHeaderKeys
from nvflare.private.fed.authenticator import MISSING_CLIENT_FQCN, validate_auth_headers


class _TokenVerifier:
    def verify(self, _client_name, _token, _signature):
        return True


def _make_message(origin):
    return Message(
        headers={
            MessageHeaderKey.CHANNEL: CellChannel.SERVER_COMMAND,
            MessageHeaderKey.TOPIC: ServerCommandNames.GET_TASK,
            MessageHeaderKey.ORIGIN: origin,
            CellMessageHeaderKeys.CLIENT_NAME: "site-a",
            CellMessageHeaderKeys.TOKEN: "token-a",
            CellMessageHeaderKeys.TOKEN_SIGNATURE: "sig-a",
        }
    )


def _validate(origin, client_fqcn_resolver):
    return validate_auth_headers(
        message=_make_message(origin),
        token_verifier=_TokenVerifier(),
        logger=logging.getLogger(__name__),
        client_fqcn_resolver=client_fqcn_resolver,
    )


@pytest.mark.parametrize("origin", ["site-a", "site-a.job-1", "site-a.site-a-child.job-1"])
def test_validate_auth_headers_accepts_token_from_registered_origin(origin):
    # Job and hierarchical child cells under a registered site are still part of that site's trust boundary.
    assert _validate(origin, lambda _client_name, _token: "site-a") is None


def test_validate_auth_headers_rejects_token_from_different_origin():
    reply = _validate("site-b", lambda _client_name, _token: "site-a")

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_validate_auth_headers_rejects_registered_token_with_missing_origin_binding():
    reply = _validate("site-a", lambda _client_name, _token: MISSING_CLIENT_FQCN)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_validate_auth_headers_keeps_compatibility_when_origin_cannot_be_resolved():
    assert _validate("site-b", lambda _client_name, _token: None) is None
