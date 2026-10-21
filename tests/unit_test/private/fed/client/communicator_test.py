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

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from nvflare.apis.fl_constant import FLContextKey, SystemComponents, TaskResultReceipt
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.f3.cellnet.core_cell import MessageHeaderKey, ReturnCode
from nvflare.private.defs import ClientRegMsgKey, new_cell_message
from nvflare.private.fed.client.communicator import Communicator


def test_get_site_config_for_registration_from_loaded_client_config():
    site_config = {"labels": {"region": "us-east"}}
    communicator = Communicator(client_config={"client_name": "site-1", ClientRegMsgKey.SITE_CONFIG: site_config})

    assert communicator._get_site_config_for_registration(FLContext()) == site_config


def test_get_site_config_for_registration_ignores_non_dict_config():
    communicator = Communicator(client_config={"client_name": "site-1", ClientRegMsgKey.SITE_CONFIG: ["bad"]})

    assert communicator._get_site_config_for_registration(FLContext()) is None


@pytest.mark.parametrize(
    "acknowledgement",
    [TaskResultReceipt.RECEIVED, TaskResultReceipt.TASK_CLOSED, TaskResultReceipt.RETRY, None, True, "accepted"],
)
@pytest.mark.parametrize("mismatch", [None, "task", "attempt", "workflow", "payload", "headers"])
def test_fenced_submit_requires_explicit_assignment_bound_receipt(monkeypatch, acknowledgement, mismatch):
    communicator = Communicator(client_config={"client_name": "site-1"})
    fl_ctx = FLContextManager(identity_name="site-1", job_id="job-1").new_context()
    fl_ctx.set_prop(FLContextKey.SSID, "session", private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.RECEIVED, private=True, sticky=False)
    response = Shareable()
    for key, value, field in [
        (ReservedHeaderKey.TASK_ID, "task-1", "task"),
        (ReservedHeaderKey.TASK_ATTEMPT_ID, "attempt-1", "attempt"),
        (ReservedHeaderKey.WORKFLOW, "workflow", "workflow"),
    ]:
        response.set_header(key, "other" if mismatch == field else value)
    response.set_header(ReservedHeaderKey.TASK_RESULT_RECEIPT, acknowledgement)
    if mismatch == "headers":
        response[ReservedHeaderKey.HEADERS] = "malformed"
    communicator.cell = Mock()

    def send_request(**kwargs):
        kwargs["request"].set_header(MessageHeaderKey.PAYLOAD_LEN, 0)
        return new_cell_message(
            {MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, None if mismatch == "payload" else response
        )

    communicator.cell.send_request.side_effect = send_request
    monkeypatch.setattr("nvflare.private.fed.client.communicator.determine_parent_fqcn", lambda *_args: "server")
    submitted = Shareable()
    submitted.add_cookie(FLContextKey.TASK_ID, "task-1")
    submitted.add_cookie(FLContextKey.TASK_ATTEMPT_ID, "attempt-1")
    submitted.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow")
    rc = communicator.submit_update("project", "token", "session", fl_ctx, "site-1", submitted, "train")
    valid = mismatch is None and acknowledgement in (
        TaskResultReceipt.RECEIVED,
        TaskResultReceipt.TASK_CLOSED,
        TaskResultReceipt.RETRY,
    )
    expected_rc = ReturnCode.OK if valid and acknowledgement != TaskResultReceipt.RETRY else ReturnCode.COMM_ERROR
    assert rc == expected_rc
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == (acknowledgement if valid else None)


def test_unfenced_old_server_preserves_legacy_transport_success_with_unknown_receipt(monkeypatch):
    communicator = Communicator(client_config={"client_name": "site-1"})
    fl_ctx = FLContextManager(identity_name="site-1", job_id="job-1").new_context()
    fl_ctx.set_prop(FLContextKey.SSID, "session", private=True, sticky=False)
    communicator.cell = Mock()

    def send_request(**kwargs):
        kwargs["request"].set_header(MessageHeaderKey.PAYLOAD_LEN, 0)
        return new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, Shareable())

    communicator.cell.send_request.side_effect = send_request
    monkeypatch.setattr("nvflare.private.fed.client.communicator.determine_parent_fqcn", lambda *_args: "server")
    assert (
        communicator.submit_update("project", "token", "session", fl_ctx, "site-1", Shareable(), "train")
        == ReturnCode.OK
    )
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) is None


@pytest.mark.parametrize("token_valid", [True, False])
def test_rejected_heartbeat_requires_registration_only_when_old_token_no_longer_verifies(token_valid):
    communicator = Communicator(client_config={"client_name": "site-1"}, secure_train=True)
    communicator.token = "old-token"
    communicator.token_signature = "old-signature"
    communicator._authenticator = MagicMock()
    verifier = MagicMock()
    verifier.verify.return_value = token_valid
    communicator._authenticator.challenge_server.return_value = ("nonce", verifier)
    communicator.client_registration = MagicMock(return_value=("new-token", "new-signature", "new-session"))
    client = SimpleNamespace(token="old-token", token_signature="old-signature", ssid="old-session")
    ctx = FLContext()
    ctx.set_prop(SystemComponents.FED_CLIENT, client)
    communicator._check_server_session("site-1", "project", ctx)

    verifier.verify.assert_called_once_with("site-1", "old-token", "old-signature", log_error=False)
    if token_valid:
        communicator.client_registration.assert_not_called()
        assert client.token == "old-token"
    else:
        communicator.client_registration.assert_called_once_with(
            "site-1", "project", ctx, timeout=communicator.maint_msg_timeout
        )
        assert (client.token, client.token_signature, client.ssid) == ("new-token", "new-signature", "new-session")
        assert ctx.get_prop(FLContextKey.CLIENT_TOKEN) == "new-token"


@pytest.mark.parametrize("first_challenge", ["unreachable", "untrusted", "new-key"])
@pytest.mark.parametrize("failure_code", [ReturnCode.UNAUTHENTICATED, ReturnCode.TIMEOUT])
def test_heartbeat_recovery_retries_without_a_direct_connection_event(monkeypatch, first_challenge, failure_code):
    communicator = Communicator(client_config={"client_name": "site-1"}, secure_train=True)
    communicator.token, communicator.token_signature = "old-token", "old-signature"
    client = SimpleNamespace(token="old-token", token_signature="old-signature", ssid="old-session")
    ctx = FLContext()
    ctx.set_prop(SystemComponents.FED_CLIENT, client)
    engine = Mock()
    engine.new_context.return_value = ctx
    engine.get_all_job_ids.return_value = []
    communicator._authenticator = Mock()
    verifier = Mock()
    verifier.verify.return_value = False
    challenge = ("nonce", verifier)
    first = {
        "unreachable": (None, None),
        "untrusted": FLCommunicationError("Invalid server identity"),
        "new-key": challenge,
    }[first_challenge]
    communicator._authenticator.challenge_server.side_effect = [first, challenge]
    communicator.client_registration = Mock(return_value=("new-token", "new-signature", "new-session"))

    def heartbeat(**kwargs):
        if communicator.cell.send_request.call_count == 3:
            communicator.heartbeat_done = True
            return new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, Shareable())
        return new_cell_message({MessageHeaderKey.RETURN_CODE: failure_code}, Shareable())

    communicator.cell = Mock()
    communicator.cell.send_request.side_effect = heartbeat
    monkeypatch.setattr("nvflare.private.fed.client.communicator.time.sleep", lambda _: None)
    communicator.send_heartbeat([], "project", "old-token", "old-session", "site-1", engine, 2)

    assert communicator._authenticator.challenge_server.call_count == 2
    assert communicator.client_registration.call_count == (2 if first_challenge == "new-key" else 1)
    assert client.token == "new-token"


def test_cancelled_registration_preserves_existing_authentication():
    communicator = Communicator(client_config={"client_name": "site-1"}, cell=Mock())
    communicator.token, communicator.token_signature, communicator.ssid = "old-token", "old-signature", "old-session"
    with patch("nvflare.private.fed.client.communicator.Authenticator") as factory:
        factory.return_value.authenticate.return_value = (None, None, None, None)
        with pytest.raises(FLCommunicationError, match="did not complete"):
            communicator.client_registration("site-1", "project", FLContext(), timeout=5)
        assert factory.call_args.kwargs["timeout"] == 5

    assert (communicator.token, communicator.token_signature, communicator.ssid) == (
        "old-token",
        "old-signature",
        "old-session",
    )
