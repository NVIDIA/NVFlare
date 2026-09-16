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

import copy
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from nvflare.app_common.decomposers.numpy_decomposers import register
from nvflare.fuel.f3.cellnet.cell import Adapter, Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.cellnet.utils import decode_payload
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming import cacheable, download_service, obj_downloader
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.utils.fobs import FOBSContextKey
from tests.unit_test.fuel.f3.streaming.download_test_utils import make_service_no_monitor, run_monitor_once


def make_cell(name, context=None):
    cell = Cell.__new__(Cell)
    cell.logger = logging.getLogger(name)
    cell.get_fqcn = lambda: name
    shared = {FOBSContextKey.CELL: cell, **(context or {})}
    cell.get_fobs_context = lambda props=None: {**shared, **(props or {})}
    cell.register_request_cb = MagicMock()
    cell.send_blob = MagicMock(return_value=object())
    return cell


@pytest.fixture
def service(monkeypatch):
    isolated = make_service_no_monitor()
    monkeypatch.setattr(download_service, "DownloadService", isolated)
    monkeypatch.setattr(obj_downloader, "DownloadService", isolated)
    monkeypatch.setattr(cacheable, "DownloadService", isolated)
    register()
    yield isolated
    isolated.shutdown()


def decode_at(message, name, service):
    """Use the real FOBS/NumPy downloader; substitute only request transport."""
    receiver = make_cell(name)

    def route(message):
        message.set_header(MessageHeaderKey.ORIGIN, name)
        return service._handle_download(message)

    receiver.send_request = lambda **kwargs: route(kwargs["request"])
    receiver.fire_and_forget = lambda **kwargs: route(kwargs["message"])
    received = Message(copy.copy(message.headers), message.payload)
    decode_payload(received, StreamHeaderKey.PAYLOAD_ENCODING, fobs_ctx=receiver.get_fobs_context())
    return received.payload


@pytest.mark.parametrize("finish_second", [True, False])
def test_fanout_retains_source_until_second_receiver_settles(service, finish_second):
    source = make_cell("source")
    array = np.arange(1024 * 1024, dtype=np.float64)
    message = Message(payload={"array": array})
    source.fire_and_forget(channel="weights", topic="snapshot", targets=["a", "b"], message=message)
    tx = next(iter(service._tx_table.values()))
    waiter = service.get_transfer_waiter(tx.tid)
    ref = tx.snapshot_refs()[0]

    np.testing.assert_array_equal(decode_at(message, "a", service)["array"], array)
    assert waiter.wait(timeout=0.05) is None
    assert service.get_transaction_id(ref.rid) == tx.tid
    assert ref.obj.base_obj is not None

    if finish_second:
        np.testing.assert_array_equal(decode_at(message, "b", service)["array"], array)
    else:
        # A missing consumer must not count as success. Drive the existing
        # inactivity timeout, without changing deadlines or waiting minutes.
        run_monitor_once(service, now=tx.last_active_time + tx.timeout + 1)
    outcome = waiter.wait(timeout=2)
    assert outcome is not None
    assert outcome.completed is finish_second
    assert service.get_transaction_id(ref.rid) is None
    if finish_second:
        assert dict(outcome.refs[0].receiver_statuses) == {"a": "success", "b": "success"}


@pytest.mark.parametrize(
    "targets,pass_through,consumers",
    [
        ("a", False, ["a"]),
        (["a", "a", "b"], False, ["a", "b"]),
        (["source", "a"], False, ["source", "a"]),
        (["a", "b"], False, ["a", "b"]),
        (["a", "b"], True, ["a.trainer", "b.trainer"]),
    ],
)
def test_fanout_accounts_for_actual_consumers(service, targets, pass_through, consumers):
    context = {FOBSContextKey.RECEIVER_IDS: ["stale"], FOBSContextKey.NUM_RECEIVERS: 99}
    source = make_cell("source", context)
    array = np.arange(16, dtype=np.float64)
    message = Message(headers={MessageHeaderKey.PASS_THROUGH: pass_through}, payload={"array": array})
    result = source.fire_and_forget(channel="weights", topic="snapshot", targets=targets, message=message)
    tx = next(iter(service._tx_table.values()))
    waiter = service.get_transfer_waiter(tx.tid)
    assert tx.num_receivers == len(consumers)
    assert tx.receiver_ids is None
    for consumer in consumers:
        np.testing.assert_array_equal(decode_at(message, consumer, service)["array"], array)
    assert waiter.wait(timeout=2).completed
    sent = [call.kwargs["target"] for call in source.send_blob.call_args_list]
    assert sent == ([targets] if isinstance(targets, str) else targets)
    assert set(result) == set(sent)
    assert set(message.get_prop(MessagePropKey.FUTURES)) == set(sent)
    assert source.get_fobs_context()[FOBSContextKey.RECEIVER_IDS] == ["stale"]
    assert source.get_fobs_context()[FOBSContextKey.NUM_RECEIVERS] == 99


def test_empty_fanout_does_not_allocate_download_source(service):
    source = make_cell("source")
    message = Message(payload={"array": np.arange(16)})
    assert source.fire_and_forget(channel="weights", topic="snapshot", targets=[], message=message) == {}
    assert not service._tx_table
    source.send_blob.assert_not_called()
    assert message.get_prop(MessagePropKey.FUTURES) == {}


@pytest.mark.parametrize("configure_topic", [False, True])
def test_receiver_configured_forwarding_completes_source_receipt(service, configure_topic):
    source = make_cell("source")
    array = np.arange(16, dtype=np.float64)
    message = Message(payload={"array": array})
    source.fire_and_forget(channel="weights", topic="snapshot", targets=["relay"], message=message)
    tx = next(iter(service._tx_table.values()))
    waiter = service.get_transfer_waiter(tx.tid)
    relay = make_cell("relay")
    relay.decode_pass_through_channels = set() if configure_topic else {"weights"}
    relay.decode_pass_through_topics = {("weights", "snapshot")} if configure_topic else set()
    forwarded = []

    def forward(request):
        outgoing = Message(payload=request.payload)
        relay.fire_and_forget(channel="weights", topic="snapshot", targets=["trainer"], message=outgoing)
        forwarded.append(outgoing)

    future = SimpleNamespace(
        headers={**message.headers, StreamHeaderKey.CHANNEL: "weights", StreamHeaderKey.TOPIC: "snapshot"},
        result=lambda: message.payload,
    )
    # Exercise Adapter's actual receiver-side configuration, with no sender header.
    Adapter(forward, SimpleNamespace(fqcn="relay"), relay).call(future)
    np.testing.assert_array_equal(decode_at(forwarded[0], "trainer", service)["array"], array)
    outcome = waiter.wait(timeout=0.1)
    assert outcome is not None and outcome.completed
    assert dict(outcome.refs[0].receiver_statuses) == {"trainer": "success"}
