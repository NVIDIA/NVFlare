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

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.cellnet import cell as cell_module
from nvflare.fuel.f3.cellnet.cell import Adapter
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.send_accounting import SourceContributionAccumulator, attach_logical_send_context
from nvflare.fuel.f3.streaming.download_service import TransactionDoneStatus
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import StreamFuture
from nvflare.private.fed.resource_stats.f3_counter import F3Counter, F3TrafficClass
from tests.unit_test.fuel.f3.streaming.download_test_utils import (
    MockDownloadable,
    make_service_no_monitor,
    pull_request,
)


def _source(chunks=(b"tensor-bytes",), recipients=("server.job",)):
    source = SourceContributionAccumulator("trainer", recipients)
    service = make_service_no_monitor()
    tx_id = service.new_transaction(
        cell=Mock(), timeout=10, num_receivers=len(recipients), receiver_ids=recipients, send_accounting_context=source
    )
    ref_id = service.add_object(tx_id, MockDownloadable(list(chunks)))
    return source, service, tx_id, ref_id


def _data(service, ref_id, recipient="server.job", state=None):
    return service._handle_download(pull_request(ref_id, recipient, state=state))


def test_source_records_accepted_tensor_bytes_not_production_or_encoded_reply_size():
    source, service, tx_id, ref_id = _source()
    reply = _data(service, ref_id)
    assert service._tx_table[tx_id].total_bytes == len(b"tensor-bytes")
    attempt = reply.get_logical_send_context().try_begin("trainer", "server.job")
    assert attempt is not None
    attempt.accepted(99999)  # DownloadService framing is not tensor payload.
    source.settle_oob_transaction(tx_id, ("server.job",))

    assert source.snapshot() == {
        "version": 1,
        "groups": [
            {
                "source_ref": ref_id,
                "recipient_fqcn": "server.job",
                "accepted_payload_bytes": str(len(b"tensor-bytes")),
                "complete": True,
            }
        ],
    }


@pytest.mark.parametrize("actual_consumer", ["server.job", "unexpected.job"])
def test_source_uses_trusted_accounting_recipient_without_changing_count_based_lifetime(actual_consumer):
    source = SourceContributionAccumulator("trainer", ("server.job",))
    service = make_service_no_monitor()
    tx_id = service.new_transaction(cell=Mock(), timeout=10, num_receivers=1, send_accounting_context=source)
    ref_id = service.add_object(tx_id, MockDownloadable([b"tensor-bytes"]))
    tx = service._tx_table[tx_id]
    assert tx.receiver_ids is None

    reply = _data(service, ref_id, actual_consumer)
    context = reply.get_logical_send_context()
    if actual_consumer == "server.job":
        context.try_begin("trainer", actual_consumer).accepted(99999)
    else:
        assert context is None
    terminal = _data(service, ref_id, actual_consumer, reply.payload["state"])
    assert terminal.payload["status"] == "eof"
    assert tx.is_finished()
    assert tx.receiver_ids is None  # Accounting never pins source lifetime.
    tx.transaction_done(TransactionDoneStatus.FINISHED)

    assert source.snapshot()["groups"] == [
        {
            "source_ref": ref_id,
            "recipient_fqcn": "server.job",
            "accepted_payload_bytes": str(len(b"tensor-bytes")) if actual_consumer == "server.job" else "0",
            "complete": actual_consumer == "server.job",
        }
    ]


@pytest.mark.parametrize("declared_receivers", [(), ("unexpected.job",)])
def test_source_rejects_conflicting_explicit_lifetime_receivers(declared_receivers):
    source = SourceContributionAccumulator("trainer", ("server.job",))
    source.register_oob_transaction("tx", declared_receivers)
    source.register_source_reference("tx", "ref")
    assert source.snapshot()["groups"] == []


def test_source_deduplicates_retry_and_can_recover_an_abandoned_chunk():
    source, service, tx_id, ref_id = _source()
    first = _data(service, ref_id).get_logical_send_context().try_begin("trainer", "server.job")
    first.abandon()
    retry = _data(service, ref_id).get_logical_send_context().try_begin("trainer", "server.job")
    retry.accepted(100)
    assert _data(service, ref_id).get_logical_send_context().try_begin("trainer", "server.job") is None
    source.settle_oob_transaction(tx_id, ("server.job",))

    group = source.snapshot()["groups"][0]
    assert group["complete"] is True
    assert group["accepted_payload_bytes"] == str(len(b"tensor-bytes"))


@pytest.mark.parametrize("terminal", ["abandoned", "inflight", "failed"])
def test_unaccepted_or_failed_source_is_not_a_complete_zero(terminal):
    source, service, tx_id, ref_id = _source()
    attempt = _data(service, ref_id).get_logical_send_context().try_begin("trainer", "server.job")
    if terminal == "abandoned":
        attempt.abandon()
    if terminal == "failed":
        attempt.accepted(100)
    source.settle_oob_transaction(tx_id, () if terminal == "failed" else ("server.job",))

    group = source.snapshot()["groups"][0]
    assert group["complete"] is False
    assert group["accepted_payload_bytes"] == (str(len(b"tensor-bytes")) if terminal == "failed" else "0")


def test_zero_source_and_multiple_references_and_recipients_are_explicit():
    source, service, tx_id, first_ref = _source(chunks=(), recipients=("server.job", "peer.job"))
    second_ref = service.add_object(tx_id, MockDownloadable([]))
    source.settle_oob_transaction(tx_id, ("server.job", "peer.job"))

    groups = source.snapshot()["groups"]
    assert {(group["source_ref"], group["recipient_fqcn"]) for group in groups} == {
        (first_ref, "server.job"),
        (first_ref, "peer.job"),
        (second_ref, "server.job"),
        (second_ref, "peer.job"),
    }
    assert all(group["complete"] and group["accepted_payload_bytes"] == "0" for group in groups)


def test_source_snapshot_is_frozen_and_does_not_expose_mutable_internal_state():
    source, service, tx_id, ref_id = _source()
    attempt = _data(service, ref_id).get_logical_send_context().try_begin("trainer", "server.job")
    source.settle_oob_transaction(tx_id, ("server.job",))
    before = source.snapshot()
    attempt.accepted(100)
    before["groups"][0]["accepted_payload_bytes"] = "999"

    after = source.snapshot()["groups"][0]
    assert after["accepted_payload_bytes"] == "0"
    assert after["complete"] is False


def test_conflicting_retry_size_and_group_overflow_fail_closed():
    source = SourceContributionAccumulator("trainer", ("server.job",), max_groups=1)
    source.register_oob_transaction("tx", ("server.job",))
    source.register_source_reference("tx", "ref")
    first = source.make_oob_contribution_context("tx", "server.job", ("tx", "ref", "server.job", 0), 3)
    first.try_begin("trainer", "server.job").accepted(100)
    conflict = source.make_oob_contribution_context("tx", "server.job", ("tx", "ref", "server.job", 0), 4)
    assert conflict.try_begin("trainer", "server.job") is None
    source.register_source_reference("tx", "overflow-ref")
    source.settle_oob_transaction("tx", ("server.job",))

    assert source.snapshot()["groups"] == [
        {"source_ref": "ref", "recipient_fqcn": "server.job", "accepted_payload_bytes": "3", "complete": False}
    ]


@pytest.mark.parametrize("actual_consumer,resolver", [("client.job", None), ("client.job.trainer", True)])
def test_logical_destination_is_separate_from_count_based_download_lifetime(actual_consumer, resolver):
    service = make_service_no_monitor()
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    assert context.prepare_destination("server.job", "client.job")
    if resolver:
        context.set_consumer_resolver(lambda consumer, request: "client.job" if consumer == actual_consumer else None)
    tx_id = service.new_transaction(cell=Mock(), timeout=10, num_receivers=1, send_accounting_context=context)
    ref_id = service.add_object(tx_id, MockDownloadable([b"tensor"]))
    tx = service._tx_table[tx_id]
    assert tx.receiver_ids is None
    main = context.try_begin("server.job", "client.job")
    main.accepted(7)
    reply = _data(service, ref_id, actual_consumer)
    reply.get_logical_send_context().try_begin("server.job", actual_consumer).accepted(1000)
    terminal = _data(service, ref_id, actual_consumer, reply.payload["state"])
    assert terminal.payload["status"] == "eof"
    assert tx.is_finished()
    tx.transaction_done(TransactionDoneStatus.FINISHED)

    assert counter.freeze() == {"sent_to": [{"participant_name": "site-1", "payload_bytes": "13", "messages": "1"}]}


def test_unknown_consumer_marks_partial_without_preventing_source_completion():
    service = make_service_no_monitor()
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    context.prepare_destination("server.job", "client.job")
    tx_id = service.new_transaction(cell=Mock(), timeout=10, num_receivers=1, send_accounting_context=context)
    ref_id = service.add_object(tx_id, MockDownloadable([b"tensor"]))
    main = context.try_begin("server.job", "client.job")
    main.accepted(7)

    reply = _data(service, ref_id, "unverified.trainer")
    assert reply.payload["data"] == b"tensor"
    assert reply.get_logical_send_context() is None
    _data(service, ref_id, "unverified.trainer", reply.payload["state"])
    assert service._tx_table[tx_id].is_finished()
    assert counter.freeze()["status"] == "partial"


@pytest.mark.parametrize("summary_before_envelope", [True, False])
def test_external_source_summary_and_envelope_complete_one_message(summary_before_envelope):
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="server")
    key = ("session", "task", "source-ref", "server.job")
    assert context.register_external_contribution(key, "server.job")
    if summary_before_envelope:
        context.settle_external_contribution(key, "server.job", 100, True)
    main = context.try_begin("client.job", "server.job")
    main.accepted(7)
    if not summary_before_envelope:
        assert counter.pending_count == 1
        context.settle_external_contribution(key, "server.job", 100, True)
    context.settle_external_contribution(key, "server.job", 100, True)  # Idempotent replay.

    assert counter.freeze() == {"sent_to": [{"participant_name": "server", "payload_bytes": "107", "messages": "1"}]}


def test_conflicting_and_missing_external_summaries_are_partial():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="server")
    assert context.register_external_contribution("source", "server.job")
    main = context.try_begin("client.job", "server.job")
    main.accepted(7)
    context.settle_external_contribution("source", "server.job", 100, True)
    context.settle_external_contribution("source", "server.job", 101, True)
    assert counter.freeze()["status"] == "partial"

    other_counter = F3Counter()
    other = attach_logical_send_context(Message(), other_counter, F3TrafficClass.TASK_RESULT, recipient_name="server")
    assert other.register_external_contribution("missing", "server.job")
    other.try_begin("client.job", "server.job").accepted(7)
    frozen = other_counter.freeze()
    other.settle_external_contribution("missing", "server.job", 100, True)
    assert frozen["status"] == "partial"
    assert other_counter.freeze() == frozen


def test_simultaneous_duplicate_external_summaries_do_not_double_count():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="server")
    assert context.register_external_contribution("source", "server.job")
    context.try_begin("client.job", "server.job").accepted(7)
    start = threading.Barrier(3)

    def settle():
        start.wait()
        context.settle_external_contribution("source", "server.job", 100, True)

    workers = [threading.Thread(target=settle) for _ in range(2)]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert counter.freeze()["sent_to"][0]["payload_bytes"] == "107"


def test_adapter_restores_only_local_receive_properties(monkeypatch):
    future = StreamFuture(
        1,
        {
            MessageHeaderKey.ORIGIN: "site-1.job.trainer",
            MessagePropKey.COMMON_NAME: "forged-wire-site",
            StreamHeaderKey.CHANNEL: "download",
            StreamHeaderKey.TOPIC: "pull",
        },
    )
    endpoint = SimpleNamespace(name="site-1.job.trainer")
    future.receive_properties = {
        MessagePropKey.ENDPOINT: endpoint,
        MessagePropKey.COMMON_NAME: "site-1",
        "arbitrary-property": "not-copied",
    }
    future.set_result(b"payload")
    received = []
    cell = SimpleNamespace(
        decode_pass_through_channels=set(),
        decode_pass_through_topics=set(),
        get_fobs_context=lambda props: props,
    )
    adapter = Adapter(lambda request: received.append(request), SimpleNamespace(fqcn="server.job"), cell)
    monkeypatch.setattr(cell_module, "decode_payload", lambda *_args, **_kwargs: None)
    adapter.call(future)

    assert received[0].get_prop(MessagePropKey.ENDPOINT) is endpoint
    assert received[0].get_prop(MessagePropKey.COMMON_NAME) == "site-1"
    assert received[0].get_prop("arbitrary-property") is None
    assert received[0].get_header(MessagePropKey.COMMON_NAME) == "forged-wire-site"
