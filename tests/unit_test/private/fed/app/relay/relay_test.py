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
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, sign_content
from nvflare.private.defs import CellMessageHeaderKeys
from nvflare.private.fed.app.relay.relay import _RelayAuth, _validate_auth_headers
from nvflare.private.fed.authenticator import Authenticator
from nvflare.private.fed.utils.identity_utils import TokenVerifier


def _relay_core_cell(fqcn="relay-1"):
    core_cell = CoreCell.__new__(CoreCell)
    core_cell.my_info = FqcnInfo(fqcn)
    return core_cell


def _server_transit_message(incoming_endpoint):
    message = Message(
        headers={
            MessageHeaderKey.SERVER_TRANSIT_REQUIRED: True,
            MessageHeaderKey.ROUTE: [("site-1", 0.0), ("server", 1.0)],
        }
    )
    message.set_prop(MessagePropKey.ENDPOINT, Endpoint(incoming_endpoint))
    return message


@pytest.mark.parametrize(("relay_fqcn", "upstream_fqcn"), [("relay-1", "server"), ("relay-1.relay-2", "relay-1")])
def test_relay_accepts_sanitized_server_transit_from_configured_upstream(relay_fqcn, upstream_fqcn):
    core_cell = _relay_core_cell(relay_fqcn)
    message = _server_transit_message(upstream_fqcn)

    with patch("nvflare.private.fed.app.relay.relay.validate_auth_headers") as validate:
        result = _validate_auth_headers(
            message,
            token_verifier=MagicMock(),
            logger=MagicMock(),
            local_cell_fqcn=relay_fqcn,
            core_cell=core_cell,
        )

    assert result is None
    validate.assert_not_called()


def test_relay_does_not_trust_forged_server_route_from_child():
    core_cell = _relay_core_cell()
    message = _server_transit_message("relay-1.site-1")
    expected = Message(payload="rejected")

    with patch("nvflare.private.fed.app.relay.relay.validate_auth_headers", return_value=expected) as validate:
        result = _validate_auth_headers(
            message,
            token_verifier=MagicMock(),
            logger=MagicMock(),
            local_cell_fqcn="relay-1",
            core_cell=core_cell,
        )

    assert result is expected
    validate.assert_called_once()


@pytest.fixture
def relay_auth():
    key, public_key = generate_keys()
    cert = generate_cert(Identity("server", "org"), Identity("server", "org"), key, public_key)
    verifier = TokenVerifier(cert)
    signature = sign_content("relay-1token", key)
    authenticator = MagicMock(client_name="relay-1")
    authenticator.challenge_server.return_value = ("nonce", verifier)
    return _RelayAuth(
        authenticator,
        FLContext(),
        ("token", signature, "ssid", verifier),
        threading.Event(),
        logging.getLogger(__name__),
    )


def _signed_message(auth_result, name="relay-1"):
    return Message(
        headers={
            MessageHeaderKey.ORIGIN: name,
            CellMessageHeaderKeys.CLIENT_NAME: name,
            CellMessageHeaderKeys.TOKEN: auth_result[0],
            CellMessageHeaderKeys.TOKEN_SIGNATURE: auth_result[1],
        }
    )


def test_relay_new_server_key_recovery_and_repeated_replacement(relay_auth):
    for _ in range(2):
        old = relay_auth.auth_result
        key, public_key = generate_keys()
        cert = generate_cert(Identity("server", "org"), Identity("server", "org"), key, public_key)
        new = ("new-token", sign_content("relay-1new-token", key), "new-ssid", TokenVerifier(cert))
        message = _signed_message(new)
        assert relay_auth.validate(message).get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED
        relay_auth.authenticator.challenge_server.return_value = ("nonce", new[3])
        relay_auth.authenticator.authenticate.return_value = new
        relay_auth.refresh()
        assert relay_auth.auth_result is new
        assert relay_auth.validate(message) is None
        assert (
            relay_auth.validate(_signed_message(old)).get_header(MessageHeaderKey.RETURN_CODE)
            == ReturnCode.UNAUTHENTICATED
        )
        outgoing = Message()
        relay_auth.add_request_headers(outgoing)
        assert outgoing.get_header(CellMessageHeaderKeys.TOKEN_SIGNATURE) == new[1]
        assert outgoing.get_header(CellMessageHeaderKeys.SSID) == new[2]
        outgoing.set_header(MessageHeaderKey.ORIGIN, "server")
        relay_auth.add_reply_headers(outgoing)
        assert outgoing.get_header(CellMessageHeaderKeys.TOKEN_SIGNATURE) == new[1]


def test_relay_same_key_and_invalid_tokens_do_not_replace_session(relay_auth):
    old = relay_auth.auth_result
    for _ in range(3):
        assert relay_auth.validate(_signed_message(("forged", "bad-signature"))) is not None
    relay_auth.refresh()
    relay_auth.refresh()
    relay_auth.authenticator.challenge_server.assert_called_once()
    relay_auth.authenticator.authenticate.assert_not_called()
    assert relay_auth.auth_result is old
    assert relay_auth.validate(_signed_message(old)) is None


@pytest.mark.parametrize(
    "failure", ["untrusted-challenge", "unreachable", "registration-timeout", "cancel-challenge", "cancel-registration"]
)
def test_relay_failed_or_cancelled_recovery_preserves_authenticated_state(relay_auth, failure):
    old = relay_auth.auth_result
    verifier = MagicMock()
    verifier.verify.return_value = False
    relay_auth.authenticator.challenge_server.return_value = ("nonce", verifier)
    if failure == "untrusted-challenge":
        relay_auth.authenticator.challenge_server.side_effect = ValueError("untrusted challenge")
    elif failure == "unreachable":
        relay_auth.authenticator.challenge_server.return_value = (None, None)
    elif failure == "registration-timeout":
        relay_auth.authenticator.authenticate.side_effect = TimeoutError()
    elif failure == "cancel-challenge":

        def cancel():
            relay_auth.stop_event.set()
            return "nonce", verifier

        relay_auth.authenticator.challenge_server.side_effect = cancel
    else:

        def cancel(*args):
            relay_auth.stop_event.set()
            return ("new-token", "new-signature", "new-ssid", verifier)

        relay_auth.authenticator.authenticate.side_effect = cancel
    relay_auth.refresh_needed.set()
    relay_auth.refresh()
    relay_auth.refresh()
    relay_auth.authenticator.challenge_server.assert_called_once()
    assert relay_auth.auth_result is old
    assert relay_auth.validate(_signed_message(old)) is None
    assert relay_auth.validate(_signed_message(("forged", "bad-signature"))) is not None


def test_relay_no_recovery_without_rejection_or_after_stop(relay_auth):
    relay_auth.refresh()
    assert relay_auth.validate(_signed_message(relay_auth.auth_result)) is None
    relay_auth.refresh()
    relay_auth.stop_event.set()
    relay_auth.refresh_needed.set()
    relay_auth.refresh()
    relay_auth.authenticator.challenge_server.assert_not_called()


def test_relay_registration_recovery_is_bounded_and_retries_after_later_rejection(relay_auth, monkeypatch):
    old = relay_auth.auth_result
    auth = Authenticator(MagicMock(), "project", "relay-1", "relay", "server", True, None, None, None, 5, 2)
    relay_auth = _RelayAuth(auth, FLContext(), old, threading.Event(), logging.getLogger(__name__))
    verifier = MagicMock()
    verifier.verify.return_value = False
    # First detect a changed key, then the server disappears during registration.
    auth.challenge_server = MagicMock(side_effect=[("nonce", verifier), (None, None)])
    clock = iter([0, 6])
    monkeypatch.setattr("nvflare.private.fed.authenticator.time", SimpleNamespace(time=lambda: next(clock)))
    assert relay_auth.validate(_signed_message(("forged", "bad-signature"))) is not None
    relay_auth.refresh()
    assert relay_auth.auth_result is old
    assert auth.challenge_server.call_count == 2
    auth.challenge_server.side_effect = None
    auth.challenge_server.return_value = ("nonce", old[3])
    assert relay_auth.validate(_signed_message(("forged", "bad-signature"))) is not None
    relay_auth.refresh()
    assert auth.challenge_server.call_count == 3
    assert relay_auth.auth_result is old
