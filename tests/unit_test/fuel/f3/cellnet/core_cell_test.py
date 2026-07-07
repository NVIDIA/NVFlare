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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.f3.cellnet.core_cell import (
    CellAgent,
    CertificateExchanger,
    CoreCell,
    TargetMessage,
    _is_failed_cert_exchange,
    _validate_url,
)
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message


def _cell(fqcn="site-1"):
    cell = CoreCell.__new__(CoreCell)
    cell.my_info = FqcnInfo(fqcn)
    cell.logger = logging.getLogger(__name__)
    cell.fobs_ctx = {"base": 1}
    cell.root_url = "tcp://server:8002"
    cell.running = True
    cell.agents = {}
    cell.ext_listeners = {}
    cell.ALL_CELLS = {}
    return cell


def test_target_message_round_trip_adds_routing_headers():
    message = Message(headers={"existing": "value"}, payload={"data": 1})

    restored = TargetMessage.from_dict(TargetMessage("site-1", "channel", "topic", message).to_dict())

    assert restored.target == "site-1"
    assert restored.channel == "channel"
    assert restored.topic == "topic"
    assert restored.message.payload == {"data": 1}
    assert restored.message.get_header(MessageHeaderKey.DESTINATION) == "site-1"
    assert restored.message.get_header(MessageHeaderKey.CHANNEL) == "channel"
    assert restored.message.get_header(MessageHeaderKey.TOPIC) == "topic"


def test_cell_agent_validates_fqcn():
    agent = CellAgent("SITE-1", Endpoint("site-1"))

    assert agent.get_fqcn() == "SITE-1"
    with pytest.raises(ValueError, match="Invalid FQCN"):
        CellAgent("bad name", Endpoint("bad"))


@pytest.mark.parametrize(
    "url, expected",
    [
        ("tcp://localhost:8002", True),
        ("http://example.test", True),
        ("localhost:8002", False),
        ("", False),
        (None, False),
    ],
)
def test_validate_url(url, expected):
    assert _validate_url(url) is expected


def test_failed_certificate_exchange_requires_matching_channel_topic_and_code():
    failed = Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.PROCESS_EXCEPTION})
    ok = Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK})

    assert _is_failed_cert_exchange("credential_manager", "key_exchange", failed)
    assert not _is_failed_cert_exchange("credential_manager", "key_exchange", ok)
    assert not _is_failed_cert_exchange("other", "key_exchange", failed)
    assert not _is_failed_cert_exchange("credential_manager", "key_exchange", None)


def test_certificate_exchanger_uses_cache_then_remote_exchange():
    core_cell = MagicMock()
    manager = MagicMock()
    manager.get_certificate.side_effect = [b"cached", None]
    manager.create_request.return_value = b"request"
    manager.process_response.return_value = b"remote"
    core_cell.send_request.return_value = Message(payload=b"response")
    exchanger = CertificateExchanger(core_cell, manager)

    assert exchanger.get_certificate("site-1") == b"cached"
    assert exchanger.get_certificate("site-2") == b"remote"
    manager.create_request.assert_called_once()
    manager.process_response.assert_called_once_with(core_cell.send_request.return_value)


def test_certificate_exchanger_reports_empty_response_and_handles_requests():
    core_cell = MagicMock()
    manager = MagicMock()
    core_cell.send_request.return_value = Message(
        headers={MessageHeaderKey.RETURN_CODE: ReturnCode.PROCESS_EXCEPTION}, payload=None
    )
    exchanger = CertificateExchanger(core_cell, manager)

    with pytest.raises(RuntimeError, match="Cert exchanged to site-1 failed"):
        exchanger.exchange_certificate("site-1")

    manager.process_request.return_value = b"reply"
    assert exchanger._handle_cert_request(Message(payload=b"request")).payload == b"reply"


def test_fobs_context_is_validated_and_copied():
    cell = _cell()

    cell.update_fobs_context({"added": 2})
    context = cell.get_fobs_context({"local": 3})
    context["base"] = 99

    assert cell.fobs_ctx == {"base": 1, "added": 2}
    assert context == {"base": 99, "added": 2, "local": 3}
    with pytest.raises(ValueError, match="props must be dict"):
        cell.update_fobs_context([])


@pytest.mark.parametrize(
    "fqcn, agents, local_cells, listeners, expected",
    [
        ("server", [], [], ["tcp://server:8002"], True),
        ("server", [], [], [], False),
        ("site-1", ["server"], [], [], True),
        ("site-1", [], ["server"], [], True),
        ("site-1.job", ["site-1"], [], [], True),
        ("site-1.job", [], [], [], False),
    ],
)
def test_backbone_readiness(fqcn, agents, local_cells, listeners, expected):
    cell = _cell(fqcn)
    cell.agents = {name: object() for name in agents}
    cell.ALL_CELLS = {name: object() for name in local_cells}
    cell.ext_listeners = {name: object() for name in listeners}

    assert cell.is_backbone_ready() is expected
    cell.running = False
    assert not cell.is_backbone_ready()


def test_connection_queries_handle_local_connected_and_routed_cells():
    cell = _cell()
    cell.ALL_CELLS = {"local": object()}
    cell.agents = {"connected": object()}
    cell._find_endpoint = MagicMock(return_value=("", Endpoint("route")))

    assert cell.is_cell_connected("local")
    assert cell.is_cell_connected("connected")
    assert not cell.is_cell_connected("missing")
    assert cell.is_cell_reachable("local")
    assert cell.is_cell_reachable("routed")


def test_listener_accessors_and_callbacks():
    cell = _cell()
    cell.int_listener = None

    assert cell.get_internal_listener_url() is None
    assert cell.get_internal_listener_params() is None

    cell.int_listener = SimpleNamespace(
        get_connection_url=lambda: "tcp://parent:9000", get_connection_params=lambda: {"secure": True}
    )
    assert cell.get_internal_listener_url() == "tcp://parent:9000"
    assert cell.get_internal_listener_params() == {"secure": True}

    def callback():
        return None

    cell.set_cell_connected_cb(callback, 1, key=2)
    cell.set_cell_disconnected_cb(callback, 3, key=4)
    cell.set_message_interceptor(callback, 5, key=6)
    assert cell.cell_connected_cb_args == (1,)
    assert cell.cell_disconnected_cb_kwargs == {"key": 4}
    assert cell.message_interceptor_args == (5,)

    for setter in (cell.set_cell_connected_cb, cell.set_cell_disconnected_cb, cell.set_message_interceptor):
        with pytest.raises(ValueError, match="not callable"):
            setter(None)


def test_encrypt_and_decrypt_secure_payload():
    cell = _cell()
    cell.cert_ex = MagicMock()
    cell.cert_ex.get_certificate.side_effect = [b"target-cert", b"origin-cert"]
    cell.credential_manager = MagicMock()
    cell.credential_manager.encrypt.return_value = b"encrypted"
    cell.credential_manager.decrypt.return_value = b"clear"
    message = Message(
        headers={
            MessageHeaderKey.SECURE: True,
            MessageHeaderKey.DESTINATION: "site-2",
            MessageHeaderKey.ORIGIN: "site-2",
        },
        payload=bytearray(b"clear"),
    )

    cell.encrypt_payload(message)
    assert message.payload == b"encrypted"
    assert message.get_header(MessageHeaderKey.CLEAR_PAYLOAD_LEN) == 5
    assert message.get_header(MessageHeaderKey.ENCRYPTED)

    cell.decrypt_payload(message)
    assert message.payload == b"clear"
    assert not message.get_header(MessageHeaderKey.ENCRYPTED, False)


@pytest.mark.parametrize("payload", ["text", 1, {}])
def test_encrypt_rejects_unsupported_payload(payload):
    cell = _cell()
    message = Message(headers={MessageHeaderKey.SECURE: True, MessageHeaderKey.DESTINATION: "site-2"}, payload=payload)

    with pytest.raises(RuntimeError, match="Payload type"):
        cell.encrypt_payload(message)
