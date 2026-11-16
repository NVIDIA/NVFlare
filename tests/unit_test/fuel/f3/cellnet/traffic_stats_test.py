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

from unittest.mock import MagicMock

import pytest

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessageType, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.cellnet.registry import Callback
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.traffic_stats import PeerTrafficCounters


def _cell():
    cell = CoreCell.__new__(CoreCell)
    cell.my_info = FqcnInfo("server.job")
    cell.logger = MagicMock()
    cell.fobs_ctx = {}
    cell.max_msg_size = 1024 * 1024
    cell.ALL_CELLS = {}
    cell.communicator = MagicMock()
    cell.encrypt_payload = MagicMock()
    cell.log_error = MagicMock()
    cell.sent_msg_size_pool = MagicMock()
    cell.sent_msg_counter_pool = MagicMock()
    cell.received_msg_size_pool = MagicMock()
    cell.received_msg_counter_pool = MagicMock()
    cell.in_filter_reg = MagicMock()
    cell.in_filter_reg.find.return_value = []
    cell.message_interceptor = None
    cell._add_to_route = MagicMock()
    cell._process_reply = MagicMock()
    cell._forward = MagicMock()
    cell.traffic_stats = PeerTrafficCounters()
    return cell


def _message(payload=b"data", destination="server.job"):
    return Message(
        {
            MessageHeaderKey.ORIGIN: "claimed-origin.job",
            MessageHeaderKey.FROM_CELL: "claimed-origin.job",
            MessageHeaderKey.DESTINATION: destination,
            MessageHeaderKey.MSG_TYPE: MessageType.REPLY,
            MessageHeaderKey.CHANNEL: "control",
            MessageHeaderKey.TOPIC: "ack",
            # Received length headers must not determine the observed size.
            MessageHeaderKey.PAYLOAD_LEN: 999999,
        },
        payload,
    )


def _receive(cell, message, peer="site-1.job", connection=None):
    if connection is None:
        connection = MagicMock()
        connection.get_conn_properties.return_value = {}
    cell._process_received_msg(Endpoint(peer), connection, message)


def test_snapshot_contains_peer_totals_and_does_not_reset_counters():
    cell = _cell()
    cell.traffic_stats.record_sent("site-1.job", 4)

    snapshot = cell.get_traffic_stats()
    snapshot["site-1.job"]["sent"] = 99

    assert cell.get_traffic_stats() == {
        "site-1.job": {"sent": 4, "received": 0},
    }


def test_high_level_cell_exposes_the_core_snapshot_without_a_new_wrapper():
    cell = Cell.__new__(Cell)
    cell.core_cell = _cell()
    cell.logger = MagicMock()
    cell.core_cell.traffic_stats.record_sent("site-1.job", 4)

    assert cell.get_traffic_stats() == cell.core_cell.get_traffic_stats()


def test_high_level_cell_can_remove_completed_peer_without_changing_another_job():
    cell = Cell.__new__(Cell)
    cell.core_cell = _cell()
    cell.logger = MagicMock()
    cell.core_cell.traffic_stats.record_sent("site-1.job", 4)
    cell.core_cell.traffic_stats.record_received("site-1.other-job", 7)

    assert cell.pop_traffic_stats("site-1.job") == {"sent": 4, "received": 0}
    assert cell.pop_traffic_stats("site-1.job") is None
    assert cell.get_traffic_stats() == {"site-1.other-job": {"sent": 0, "received": 7}}


def test_cells_do_not_share_counters():
    first, second = _cell(), _cell()
    first.traffic_stats.record_sent("site-1.job", 4)

    assert second.get_traffic_stats() == {}


def test_send_counts_actual_adjacent_peer_and_includes_retries_and_control_messages():
    cell = _cell()
    message = _message(destination="final-destination.job")
    endpoint = Endpoint("relay")

    assert cell._send_to_endpoint(endpoint, message) == ""
    assert cell._send_to_endpoint(endpoint, message) == ""

    assert cell.get_traffic_stats() == {"relay": {"sent": 8, "received": 0}}
    assert cell.communicator.send.call_count == 2


def test_send_counts_encrypted_payload_after_transport_returns():
    cell = _cell()
    message = _message()
    cell.encrypt_payload.side_effect = lambda msg: setattr(msg, "payload", b"encrypted-payload")

    def send(*_args):
        assert cell.get_traffic_stats() == {}

    cell.communicator.send.side_effect = send

    assert cell._send_to_endpoint(Endpoint("site-1.job"), message) == ""

    assert cell.get_traffic_stats()["site-1.job"]["sent"] == len(b"encrypted-payload")


def test_send_counts_buffer_bytes_not_buffer_list_entries():
    cell = _cell()
    message = _message()
    cell.encrypt_payload.side_effect = lambda msg: setattr(msg, "payload", [b"abc", b"defgh"])

    assert cell._send_to_endpoint(Endpoint("site-1.job"), message) == ""

    assert cell.get_traffic_stats()["site-1.job"]["sent"] == 8


@pytest.mark.parametrize("payload", [None, b"", bytearray(b"123"), memoryview(b"12345")])
def test_byte_buffers_and_empty_payloads_have_their_actual_size(payload):
    cell = _cell()

    assert cell._send_to_endpoint(Endpoint("site-1.job"), _message(payload=payload)) == ""
    _receive(cell, _message(payload=payload))

    size = 0 if payload is None else len(payload)
    assert cell.get_traffic_stats() == {"site-1.job": {"sent": size, "received": size}}


@pytest.mark.parametrize("failure", ["encoding", "encryption", "transport", "oversized"])
def test_failed_sends_do_not_increment_bytes(monkeypatch, failure):
    cell = _cell()
    if failure == "encoding":
        monkeypatch.setattr(
            "nvflare.fuel.f3.cellnet.core_cell.encode_payload", MagicMock(side_effect=RuntimeError("encode"))
        )
    elif failure == "encryption":
        cell.encrypt_payload.side_effect = RuntimeError("encrypt")
    elif failure == "transport":
        cell.communicator.send.side_effect = RuntimeError("send")
    else:
        cell.max_msg_size = 1

    expected = ReturnCode.MSG_TOO_BIG if failure == "oversized" else ReturnCode.COMM_ERROR
    assert cell._send_to_endpoint(Endpoint("site-1.job"), _message()) == expected
    assert cell.get_traffic_stats() == {}


def test_direct_in_process_send_is_delivered_without_remote_bytes():
    cell = _cell()
    receiver = MagicMock()
    cell.ALL_CELLS = {"site-1.job": receiver}

    assert cell._send_to_endpoint(Endpoint("site-1.job"), _message()) == ""

    receiver.process_message.assert_called_once()
    cell.communicator.send.assert_not_called()
    assert cell.get_traffic_stats() == {}


def test_self_loopback_send_is_not_remote_traffic():
    cell = _cell()

    assert cell._send_to_endpoint(Endpoint("server.job"), _message()) == ""

    cell.communicator.send.assert_called_once()
    assert cell.get_traffic_stats() == {}


def test_receive_counts_actual_peer_and_actual_payload_before_dispatch():
    cell = _cell()
    message = _message(payload=[b"abc", b"defgh"])
    observed_at_dispatch = []

    def dispatch(*_args):
        observed_at_dispatch.append(cell.get_traffic_stats()["site-1.job"]["received"])
        message.payload = b""

    cell._process_reply.side_effect = dispatch
    _receive(cell, message)

    assert observed_at_dispatch == [8]
    assert cell.get_traffic_stats() == {"site-1.job": {"sent": 0, "received": 8}}


def test_forwarded_receive_counts_incoming_hop_only():
    cell = _cell()
    message = _message(destination="final-destination.job")

    _receive(cell, message, peer="relay")

    cell._forward.assert_called_once()
    assert cell.get_traffic_stats() == {"relay": {"sent": 0, "received": 4}}


def test_receive_is_counted_even_when_rejected_by_an_incoming_filter():
    cell = _cell()

    def reject(_message):
        return Message(payload=b"rejected")

    cell.in_filter_reg.find.return_value = [Callback(reject, (), {})]
    _receive(cell, _message())

    cell._process_reply.assert_not_called()
    assert cell.get_traffic_stats()["site-1.job"]["received"] == 4


@pytest.mark.parametrize("peer", ["site-1.job", "server.job"])
def test_direct_or_self_loopback_receive_does_not_count_remote_bytes(peer):
    cell = _cell()
    connection = None if peer == "site-1.job" else MagicMock()
    if connection is not None:
        connection.get_conn_properties.return_value = {}

    cell._process_received_msg(Endpoint(peer), connection, _message())

    cell._process_reply.assert_called_once()
    assert cell.get_traffic_stats() == {}


@pytest.mark.parametrize("direction", ["sent", "received"])
def test_counter_failure_does_not_change_delivery(direction):
    cell = _cell()
    setattr(cell.traffic_stats, f"record_{direction}", MagicMock(side_effect=RuntimeError("counter")))

    if direction == "sent":
        assert cell._send_to_endpoint(Endpoint("site-1.job"), _message()) == ""
        cell.communicator.send.assert_called_once()
    else:
        _receive(cell, _message())
        cell._process_reply.assert_called_once()


@pytest.mark.parametrize("direction", ["sent", "received"])
def test_byte_sampling_failure_does_not_change_delivery(monkeypatch, direction):
    cell = _cell()
    monkeypatch.setattr("nvflare.fuel.f3.cellnet.core_cell.buffer_len", MagicMock(side_effect=RuntimeError("sampling")))

    if direction == "sent":
        assert cell._send_to_endpoint(Endpoint("site-1.job"), _message()) == ""
        cell.communicator.send.assert_called_once()
    else:
        _receive(cell, _message())
        cell._process_reply.assert_called_once()
    assert cell.get_traffic_stats() == {}


def test_old_minimal_corecell_fixture_without_counters_still_delivers():
    cell = _cell()
    del cell.traffic_stats

    assert cell._send_to_endpoint(Endpoint("site-1.job"), _message()) == ""
    _receive(cell, _message())

    cell.communicator.send.assert_called_once()
    cell._process_reply.assert_called_once()
