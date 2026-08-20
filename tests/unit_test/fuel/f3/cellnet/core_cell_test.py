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

from nvflare.fuel.f3.cellnet.connector_manager import ConnectorManager
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
from nvflare.fuel.f3.cellnet.registry import Registry
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.stream_const import (
    STREAM_ACK_TOPIC,
    STREAM_CHANNEL,
    STREAM_DATA_TOPIC,
    StreamDataType,
    StreamHeaderKey,
)


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


def test_make_internal_listener_forwards_listener_override():
    cell = _cell("site-1.job-1")
    cell.int_listener = None
    cell.connector_manager = MagicMock()
    cell.connector_manager.get_internal_listener.return_value = SimpleNamespace(
        get_connection_url=lambda: "tcp://localhost:49152"
    )
    resources = {
        DriverParams.HOST.value: "localhost",
        DriverParams.CONNECTION_SECURITY.value: "clear",
    }

    cell.make_internal_listener("tcp", resources)

    cell.connector_manager.get_internal_listener.assert_called_once_with("tcp", resources)
    assert cell.get_internal_listener_url() == "tcp://localhost:49152"


def test_internal_listener_host_overrides_default_resources():
    comm_configurator = MagicMock()
    comm_configurator.get_config.return_value = None

    manager = ConnectorManager(
        communicator=MagicMock(),
        secure=False,
        comm_configurator=comm_configurator,
        internal_listener_host="127.0.0.1",
    )

    assert manager.int_resources[DriverParams.HOST.value] == "127.0.0.1"
    assert manager.int_resources[DriverParams.LISTEN_HOST.value] == "127.0.0.1"


def test_configured_internal_host_overrides_internal_listener_host():
    comm_configurator = MagicMock()
    comm_configurator.get_config.return_value = {
        "internal": {
            "scheme": "tcp",
            "resources": {
                DriverParams.HOST.value: "0.0.0.0",
                DriverParams.PORT.value: 19100,
            },
        }
    }

    manager = ConnectorManager(
        communicator=MagicMock(),
        secure=False,
        comm_configurator=comm_configurator,
        internal_listener_host="127.0.0.1",
    )

    assert manager.int_resources[DriverParams.HOST.value] == "0.0.0.0"
    assert DriverParams.LISTEN_HOST.value not in manager.int_resources
    assert manager.int_resources[DriverParams.PORT.value] == 19100


def test_internal_listener_override_does_not_mutate_configured_resources():
    configured_resources = {
        DriverParams.HOST.value: "localhost",
        DriverParams.LISTEN_HOST.value: "localhost",
        DriverParams.PORTS.value: "8102-8102",
        DriverParams.CONNECTION_SECURITY.value: "mtls",
    }
    comm_configurator = MagicMock()
    comm_configurator.get_internal_connection_scheme.return_value = "stcp"
    comm_configurator.get_config.return_value = {
        "internal": {
            "scheme": "stcp",
            "resources": configured_resources,
        }
    }
    communicator = MagicMock()
    communicator.start_listener.return_value = ("handle", "tcp://localhost:49152", {})
    manager = ConnectorManager(
        communicator=communicator,
        secure=True,
        comm_configurator=comm_configurator,
    )
    listener_resources = {
        DriverParams.HOST.value: "localhost",
        DriverParams.CONNECTION_SECURITY.value: "clear",
    }

    listener = manager.get_internal_listener("tcp", listener_resources)

    assert listener.get_connection_url() == "tcp://localhost:49152"
    communicator.start_listener.assert_called_once_with("tcp", {"secure": False, **listener_resources})
    assert manager.int_scheme == "stcp"
    assert manager.int_resources == configured_resources


def test_public_send_reply_applies_filters_before_server_transit_routing():
    cell = _cell("site-2.job")
    cell.out_reply_filter_reg = Registry()
    cell.received_msg_counter_pool = MagicMock()
    cell._stats_category = MagicMock(return_value="reply:test")
    cell._send_to_endpoint = MagicMock(return_value="")
    cell._find_endpoint = MagicMock(
        side_effect=[
            ("", Endpoint("site-1.job")),
            ("", Endpoint("server")),
        ]
    )

    def require_server_transit(message):
        message.set_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED, True)

    cell.add_outgoing_reply_filter("*", "*", require_server_transit)

    rc = cell.send_reply(Message(), "site-1.job", ["req-1"])

    assert rc == ""
    assert cell._find_endpoint.call_count == 2
    assert cell._send_to_endpoint.call_args.args[0].name == "server"


def test_send_reply_reports_comm_error_when_server_transit_path_is_unreachable():
    cell = _cell("site-2.job")
    cell.out_reply_filter_reg = Registry()
    cell.received_msg_counter_pool = MagicMock()
    cell._find_endpoint = MagicMock(return_value=(ReturnCode.TARGET_UNREACHABLE, None))
    cell._send_to_endpoint = MagicMock()
    reply = Message(
        headers={
            MessageHeaderKey.DESTINATION: "site-1.job",
            MessageHeaderKey.SERVER_TRANSIT_REQUIRED: True,
        }
    )

    rc = cell._send_reply(reply, Endpoint("site-1.job"))

    assert rc == ReturnCode.COMM_ERROR
    cell._send_to_endpoint.assert_not_called()


def test_server_transit_request_does_not_advertise_ad_hoc_connector():
    cell = _cell("site-1")
    cell.out_req_filter_reg = Registry()
    cell.sent_msg_counter_pool = MagicMock()
    cell.connector_manager = MagicMock()
    cell.connector_manager.is_adhoc_allowed.return_value = True
    cell._create_external_listener = MagicMock()
    cell._send_to_endpoint = MagicMock(return_value="")
    cell._find_endpoint = MagicMock(
        side_effect=[
            ("", Endpoint("site-2")),
            ("", Endpoint("server")),
        ]
    )

    def require_server_transit(message):
        message.set_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED, True)

    cell.add_outgoing_request_filter("*", "*", require_server_transit)
    target_message = TargetMessage("site-2", "peer", "ping", Message(payload="hello"))

    result = cell._send_target_messages({"site-2": target_message})

    assert result == {"site-2": ""}
    cell._create_external_listener.assert_not_called()
    sent_message = cell._send_to_endpoint.call_args.args[1]
    assert sent_message.get_header(MessageHeaderKey.CONN_URL) is None


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


@pytest.mark.parametrize("cell_cipher, expected", [(None, False), (MagicMock(), True)])
def test_supports_secure_messages_requires_cell_cipher(cell_cipher, expected):
    cell = _cell()
    cell.credential_manager = SimpleNamespace(cell_cipher=cell_cipher)

    assert cell.supports_secure_messages() is expected


@pytest.mark.parametrize("payload", ["text", 1, {}])
def test_encrypt_rejects_unsupported_payload(payload):
    cell = _cell()
    message = Message(headers={MessageHeaderKey.SECURE: True, MessageHeaderKey.DESTINATION: "site-2"}, payload=payload)

    with pytest.raises(RuntimeError, match="Payload type"):
        cell.encrypt_payload(message)


def test_stream_filter_rejection_sends_error_ack_to_stream_sender():
    cell = _cell("server")
    cell.fire_and_forget = MagicMock(return_value={})
    request = Message(
        headers={
            MessageHeaderKey.CHANNEL: STREAM_CHANNEL,
            MessageHeaderKey.TOPIC: STREAM_DATA_TOPIC,
            MessageHeaderKey.ORIGIN: "site-1.job.trainer",
            StreamHeaderKey.STREAM_ID: 42,
        }
    )
    rejection = Message(
        headers={
            MessageHeaderKey.RETURN_CODE: ReturnCode.UNAUTHENTICATED,
            MessageHeaderKey.ERROR: "missing client name",
        }
    )

    cell._send_stream_filter_rejection(request, rejection)

    kwargs = cell.fire_and_forget.call_args.kwargs
    assert kwargs["channel"] == STREAM_CHANNEL
    assert kwargs["topic"] == STREAM_ACK_TOPIC
    assert kwargs["targets"] == "site-1.job.trainer"
    assert kwargs["message"].get_header(StreamHeaderKey.STREAM_ID) == 42
    assert kwargs["message"].get_header(StreamHeaderKey.DATA_TYPE) == StreamDataType.ERROR
    assert "missing client name" in kwargs["message"].get_header(StreamHeaderKey.ERROR_MSG)


def test_non_stream_filter_rejection_does_not_send_stream_error():
    cell = _cell("server")
    cell.fire_and_forget = MagicMock(return_value={})

    cell._send_stream_filter_rejection(
        Message(
            headers={
                MessageHeaderKey.CHANNEL: "app",
                MessageHeaderKey.TOPIC: "task",
                MessageHeaderKey.ORIGIN: "site-1",
            }
        ),
        Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.UNAUTHENTICATED}),
    )

    cell.fire_and_forget.assert_not_called()


def test_stream_filter_rejection_without_stream_id_does_not_send_stream_error():
    cell = _cell("server")
    cell.fire_and_forget = MagicMock(return_value={})

    cell._send_stream_filter_rejection(
        Message(
            headers={
                MessageHeaderKey.CHANNEL: STREAM_CHANNEL,
                MessageHeaderKey.TOPIC: STREAM_DATA_TOPIC,
                MessageHeaderKey.ORIGIN: "site-1.job.trainer",
            }
        ),
        Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.UNAUTHENTICATED}),
    )

    cell.fire_and_forget.assert_not_called()


def test_stream_filter_rejection_with_ok_reply_uses_filter_error_detail():
    cell = _cell("server")
    cell.fire_and_forget = MagicMock(return_value={})

    cell._send_stream_filter_rejection(
        Message(
            headers={
                MessageHeaderKey.CHANNEL: STREAM_CHANNEL,
                MessageHeaderKey.TOPIC: STREAM_DATA_TOPIC,
                MessageHeaderKey.ORIGIN: "site-1.job.trainer",
                StreamHeaderKey.STREAM_ID: 42,
            }
        ),
        Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}),
    )

    error = cell.fire_and_forget.call_args.kwargs["message"].get_header(StreamHeaderKey.ERROR_MSG)
    assert ReturnCode.FILTER_ERROR in error
    assert "rejected" in error
