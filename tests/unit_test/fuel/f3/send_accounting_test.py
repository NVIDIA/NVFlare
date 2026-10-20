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
from unittest.mock import MagicMock, call

import pytest

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.send_accounting import attach_logical_send_context
from nvflare.fuel.f3.streaming.blob_streamer import BlobStream
from nvflare.fuel.f3.streaming.byte_streamer import STREAM_TYPE_BLOB, ByteStreamer, TxTask
from nvflare.fuel.f3.streaming.download_service import TransactionDoneStatus, _Transaction
from nvflare.private.fed.resource_stats.f3_counter import F3Counter, F3TrafficClass
from tests.unit_test.fuel.f3.streaming.download_test_utils import (
    MockDownloadable,
    make_isolated_download_service,
    make_service_no_monitor,
    pull_request,
)


def _total_value(snapshot):
    return {
        "payload_bytes": str(sum(int(group["payload_bytes"]) for group in snapshot["sent_to"])),
        "messages": str(sum(int(group["messages"]) for group in snapshot["sent_to"])),
    }


def _counter_value(counter: F3Counter):
    return _total_value(counter.freeze())


def _core_cell(fqcn="origin"):
    cell = CoreCell.__new__(CoreCell)
    cell.my_info = FqcnInfo(fqcn)
    cell.logger = logging.getLogger(__name__)
    cell.fobs_ctx = {}
    cell.max_msg_size = 1024 * 1024
    cell.ALL_CELLS = {}
    cell.communicator = MagicMock()
    cell.sent_msg_size_pool = MagicMock()
    cell._stats_category = MagicMock(return_value="test")
    cell.log_error = MagicMock()
    return cell


def test_message_clone_preserves_context_without_serializing_it():
    message = Message(headers={"wire": "value"}, payload=b"payload")
    counter = F3Counter()
    context = attach_logical_send_context(message, counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")

    clone = message.clone(deep_copy_headers=True)

    assert clone.get_logical_send_context() is context
    assert clone.headers == {"wire": "value"}
    assert set(clone.__dict__) == {"headers", "payload", "_logical_send_context"}


def test_one_context_rejects_a_second_destination_to_protect_recipient_attribution():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")

    first = context.try_begin("origin", "site-1")
    second = context.try_begin("origin", "site-2")
    assert first is not None
    assert second is None
    first.accepted(10)

    assert context.try_begin("origin", "site-1") is None
    assert context.try_begin("relay", "site-3") is None
    snapshot = counter.freeze()
    assert snapshot["status"] == "partial"
    assert snapshot["issues"] == ["counter_gap"]
    assert snapshot["sent_to"] == [{"participant_name": "site-1", "payload_bytes": "10", "messages": "1"}]


def test_pre_admission_is_reused_once_by_transport():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")

    assert context.pre_admit("server/job", "client/job")
    assert counter.pending_count == 1

    attempt = context.try_begin("server/job", "client/job")
    assert attempt is not None
    assert counter.pending_count == 1
    assert context.try_begin("server/job", "client/job") is None
    attempt.accepted(17)

    assert counter.freeze() == {
        "sent_to": [{"participant_name": "site-1", "payload_bytes": "17", "messages": "1"}],
    }


def test_excluded_delivery_releases_only_the_matching_pre_admission():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    assert context.pre_admit("server/job", "client/job")

    assert not context.abandon_excluded_delivery("relay", "client/job")
    assert not context.abandon_excluded_delivery("server/job", "other/job")
    assert counter.pending_count == 1
    assert context.abandon_excluded_delivery("server/job", "client/job")
    assert not context.abandon_excluded_delivery("server/job", "client/job")
    assert counter.pending_count == 0
    assert counter.freeze() == {"sent_to": []}


def test_excluded_delivery_cannot_abandon_transport_that_has_already_started():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    assert context.pre_admit("server/job", "client/job")
    attempt = context.try_begin("server/job", "client/job")

    assert not context.abandon_excluded_delivery("server/job", "client/job")
    assert counter.pending_count == 1
    attempt.accepted(17)
    assert counter.freeze() == {
        "sent_to": [{"participant_name": "site-1", "payload_bytes": "17", "messages": "1"}],
    }


def test_concurrent_excluded_deliveries_release_the_pre_admission_once():
    counter = F3Counter()
    counter.abandon = MagicMock(wraps=counter.abandon)
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    assert context.pre_admit("server/job", "client/job")
    start = threading.Barrier(5)
    results = []

    def release():
        start.wait(timeout=2.0)
        results.append(context.abandon_excluded_delivery("server/job", "client/job"))

    threads = [threading.Thread(target=release) for _ in range(4)]
    for thread in threads:
        thread.start()
    start.wait(timeout=2.0)
    for thread in threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    assert sorted(results) == [False, False, False, True]
    counter.abandon.assert_called_once()
    assert counter.pending_count == 0
    assert counter.freeze() == {"sent_to": []}


@pytest.mark.parametrize("counter_state", ["collecting", "closed", "frozen"])
def test_excluded_delivery_without_pre_admission_never_changes_counter(counter_state):
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    if counter_state != "collecting":
        counter.close()
    if counter_state == "frozen":
        counter.freeze()

    assert not context.abandon_excluded_delivery("server/job", "client/job")
    assert counter.pending_count == 0
    assert counter.freeze() == {"sent_to": []}


def test_pre_admission_allows_oob_registration_before_transport():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")

    assert context.pre_admit("server/job", "client/job")
    context.register_oob_transaction("tx-1", ("client/job",))
    main = context.try_begin("server/job", "client/job")
    assert main is not None
    main.accepted(7)
    contribution = context.make_oob_contribution_context("tx-1", "client/job", "chunk-1", 11)
    contribution.try_begin("server/job", "client/job").accepted(999)
    context.settle_oob_transaction("tx-1", {"client/job"})

    assert _counter_value(counter) == {"payload_bytes": "18", "messages": "1"}


def test_failed_pre_admission_marks_closed_owner_partial():
    counter = F3Counter()
    counter.close()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")

    assert not context.pre_admit("server/job", "client/job")

    assert counter.freeze() == {
        "status": "partial",
        "issues": ["counter_gap"],
        "sent_to": [],
    }


def test_cached_consumer_mapping_rechecks_each_request():
    context = attach_logical_send_context(Message(), F3Counter(), F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    context.prepare_destination("server.job", "client.job")
    context.register_oob_transaction("tx", None)
    resolver = MagicMock(return_value="client.job")
    context.set_consumer_resolver(resolver)
    first = Message(headers={"proof": "first"})
    second = Message(headers={"proof": "second"})

    assert context.resolve_oob_consumer("tx", "client.job.trainer", first) == "client.job"
    assert context.resolve_oob_consumer("tx", "client.job.trainer", second) == "client.job"

    assert resolver.call_args_list == [
        call("client.job.trainer", first),
        call("client.job.trainer", second),
    ]


def test_direct_logical_consumer_keeps_no_resolver_compatibility():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    context.prepare_destination("server.job", "client.job")
    context.register_oob_transaction("tx", None)
    first = Message()
    second = Message()

    assert context.resolve_oob_consumer("tx", "client.job", first) == "client.job"
    assert context.resolve_oob_consumer("tx", "client.job", second) == "client.job"
    assert "status" not in counter.freeze()


def test_unknown_transaction_never_asks_resolver_to_authorize_consumer():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    context.prepare_destination("server.job", "client.job")
    resolver = MagicMock(return_value="client.job")
    context.set_consumer_resolver(resolver)

    assert context.resolve_oob_consumer("unknown", "client.job.trainer", Message()) is None
    resolver.assert_not_called()
    assert counter.freeze()["status"] == "partial"


def test_consumer_cannot_switch_between_known_logical_receivers():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    context.register_oob_transaction("tx", ("site-1", "site-2"))
    context.set_consumer_resolver(MagicMock(side_effect=["site-1", "site-2"]))

    assert context.resolve_oob_consumer("tx", "trainer", Message()) == "site-1"
    assert context.resolve_oob_consumer("tx", "trainer", Message()) is None
    assert context.make_oob_contribution_context("tx", "trainer", "chunk", 1) is None
    assert counter.freeze()["status"] == "partial"


@pytest.mark.parametrize("rejected", [None, "other.job", {"invalid": "type"}, RuntimeError("private-proof")])
def test_rejected_cached_consumer_proof_is_evicted_and_stays_incomplete(rejected, caplog):
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    context.prepare_destination("server.job", "client.job")
    context.register_oob_transaction("tx", None)
    context.set_consumer_resolver(MagicMock(side_effect=["client.job", rejected, "client.job"]))
    main = context.try_begin("server.job", "client.job")
    main.accepted(7)
    assert context.resolve_oob_consumer("tx", "client.job.trainer", Message()) == "client.job"
    stale = context.make_oob_contribution_context("tx", "client.job.trainer", "first", 5)
    inflight = stale.try_begin("server.job", "client.job.trainer")

    assert context.resolve_oob_consumer("tx", "client.job.trainer", Message()) is None
    assert context.make_oob_contribution_context("tx", "client.job.trainer", "rejected", 5) is None
    # Accounting stays stopped after proof loss; delivery and source lifetime
    # are independent of whether a later pull supplies valid proof.
    assert context.resolve_oob_consumer("tx", "client.job.trainer", Message()) is None
    inflight.accepted(100)
    assert stale.try_begin("server.job", "client.job.trainer") is None
    context.settle_oob_transaction("tx", ("client.job.trainer",))

    assert counter.pending_count == 0
    assert counter.freeze() == {"status": "partial", "issues": ["counter_gap"], "sent_to": []}
    assert "private-proof" not in caplog.text


def test_oob_bytes_complete_the_same_message_only_after_settlement():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    context.register_oob_transaction("tx-1", ("site-1",))

    main = context.try_begin("origin", "site-1")
    assert main is not None
    main.accepted(11)
    assert counter.pending_count == 1

    contribution = context.make_oob_contribution_context("tx-1", "site-1", "chunk-1", 101)
    assert contribution is not None
    chunk = contribution.try_begin("origin", "site-1")
    assert chunk is not None
    chunk.accepted(9999)  # encoded DownloadService reply size is deliberately ignored

    duplicate = contribution.try_begin("origin", "site-1")
    assert duplicate is None
    assert counter.pending_count == 1

    context.settle_oob_transaction("tx-1", {"site-1"})

    assert counter.pending_count == 0
    assert _counter_value(counter) == {"payload_bytes": "112", "messages": "1"}


def test_oob_settlement_waits_for_an_inflight_last_contribution():
    class BlockingAccounting:
        def __init__(self):
            self.counter = F3Counter()
            self.add_started = threading.Event()
            self.release_add = threading.Event()

        def __getattr__(self, name):
            return getattr(self.counter, name)

        def add_accepted_payload_bytes(self, admission, payload_bytes):
            self.add_started.set()
            assert self.release_add.wait(timeout=1.0)
            return self.counter.add_accepted_payload_bytes(admission, payload_bytes)

    accounting = BlockingAccounting()
    counter = accounting.counter
    context = attach_logical_send_context(Message(), accounting, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    context.register_oob_transaction("tx-1", ("site-1",))
    main = context.try_begin("origin", "site-1")
    main.accepted(7)

    contribution = context.make_oob_contribution_context("tx-1", "site-1", "last", 5)
    chunk = contribution.try_begin("origin", "site-1")
    send_thread = threading.Thread(target=chunk.accepted, args=(123,))
    send_thread.start()
    assert accounting.add_started.wait(timeout=1.0)

    context.settle_oob_transaction("tx-1", {"site-1"})
    assert counter.pending_count == 1

    accounting.release_add.set()
    send_thread.join(timeout=1.0)
    assert not send_thread.is_alive()

    assert counter.pending_count == 0
    assert _counter_value(counter) == {"payload_bytes": "12", "messages": "1"}


def test_failed_oob_settlement_marks_the_observation_partial():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    context.register_oob_transaction("tx-1", ("site-1",))
    main = context.try_begin("origin", "site-1")
    main.accepted(7)

    context.settle_oob_transaction("tx-1", set())

    snapshot = counter.freeze()
    assert snapshot["status"] == "partial"
    assert snapshot["issues"] == ["counter_gap"]
    assert _total_value(snapshot) == {"payload_bytes": "0", "messages": "0"}


def test_oob_rejects_untrusted_second_destination_and_retains_first_subtotal():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    context.register_oob_transaction("tx-1", ("site-1", "site-2"))

    site_1 = context.try_begin("origin", "site-1")
    site_2 = context.try_begin("origin", "site-2")
    site_1.accepted(7)
    assert site_2 is None
    site_1_chunk = context.make_oob_contribution_context("tx-1", "site-1", "site-1-chunk", 11)
    site_1_chunk.try_begin("origin", "site-1").accepted(1000)

    context.settle_oob_transaction("tx-1", {"site-1"})

    snapshot = counter.freeze()
    assert snapshot["status"] == "partial"
    assert snapshot["issues"] == ["counter_gap"]
    assert _total_value(snapshot) == {"payload_bytes": "18", "messages": "1"}


def test_unresolved_oob_consumers_mark_the_main_send_partial_at_cutoff():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    context.register_oob_transaction("tx-unknown", None)

    main = context.try_begin("origin", "site-1")
    assert main is not None
    main.accepted(7)

    snapshot = counter.freeze()
    assert snapshot["status"] == "partial"
    assert snapshot["issues"] == ["counter_gap"]
    assert _total_value(snapshot) == {"payload_bytes": "0", "messages": "0"}


def test_late_oob_registration_marks_a_completed_send_partial():
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    main = context.try_begin("origin", "site-1")
    main.accepted(7)

    context.register_oob_transaction("late-tx", ("site-1",))

    snapshot = counter.freeze()
    assert snapshot["status"] == "partial"
    assert snapshot["issues"] == ["counter_gap"]
    assert _total_value(snapshot) == {"payload_bytes": "7", "messages": "1"}


def test_core_cell_counts_post_fobs_pre_encryption_size(monkeypatch):
    import nvflare.fuel.f3.cellnet.core_cell as core_cell_module

    cell = _core_cell()
    counter = F3Counter()
    message = Message(headers={MessageHeaderKey.DESTINATION: "site-1"}, payload={"large": "object"})
    attach_logical_send_context(message, counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")

    def encode(msg, fobs_ctx):
        msg.payload = b"encoded"
        return 7

    monkeypatch.setattr(core_cell_module, "encode_payload", encode)
    cell.encrypt_payload = lambda msg: setattr(msg, "payload", b"encrypted-payload-is-larger")

    assert cell._send_to_endpoint(Endpoint("site-1"), message) == ""
    assert cell.communicator.send.call_count == 1
    assert _counter_value(counter) == {"payload_bytes": "7", "messages": "1"}


def test_core_cell_reuses_pre_admission(monkeypatch):
    import nvflare.fuel.f3.cellnet.core_cell as core_cell_module

    cell = _core_cell("server/job")
    counter = F3Counter()
    message = Message(headers={MessageHeaderKey.DESTINATION: "client/job"}, payload=b"payload")
    context = attach_logical_send_context(message, counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    assert context.pre_admit("server/job", "client/job")
    monkeypatch.setattr(core_cell_module, "encode_payload", lambda _message, fobs_ctx: 7)
    cell.encrypt_payload = MagicMock()

    assert cell._send_to_endpoint(Endpoint("client/job"), message) == ""
    assert counter.pending_count == 0
    assert _counter_value(counter) == {"payload_bytes": "7", "messages": "1"}


def test_core_cell_excludes_direct_in_process_delivery(monkeypatch):
    import nvflare.fuel.f3.cellnet.core_cell as core_cell_module

    cell = _core_cell()
    cell.ALL_CELLS["site-1"] = object()
    cell._send_direct_message = MagicMock()
    counter = F3Counter()
    message = Message(headers={MessageHeaderKey.DESTINATION: "site-1"}, payload=b"payload")
    attach_logical_send_context(message, counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    monkeypatch.setattr(core_cell_module, "encode_payload", lambda _message, fobs_ctx: 7)
    cell.encrypt_payload = MagicMock()

    assert cell._send_to_endpoint(Endpoint("site-1"), message) == ""
    cell._send_direct_message.assert_called_once()
    assert _counter_value(counter) == {"payload_bytes": "0", "messages": "0"}


def test_core_cell_abandons_pre_admission_for_direct_final_delivery(monkeypatch):
    import nvflare.fuel.f3.cellnet.core_cell as core_cell_module

    cell = _core_cell("server/job")
    cell.ALL_CELLS["client/job"] = object()
    cell._send_direct_message = MagicMock()
    counter = F3Counter()
    message = Message(headers={MessageHeaderKey.DESTINATION: "client/job"}, payload=b"payload")
    context = attach_logical_send_context(message, counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    assert context.pre_admit("server/job", "client/job")
    monkeypatch.setattr(core_cell_module, "encode_payload", lambda _message, fobs_ctx: 7)
    cell.encrypt_payload = MagicMock()

    assert cell._send_to_endpoint(Endpoint("client/job"), message) == ""
    assert counter.pending_count == 0
    assert _counter_value(counter) == {"payload_bytes": "0", "messages": "0"}


def test_encode_failure_releases_pre_admission_without_waiting(monkeypatch):
    import nvflare.fuel.f3.cellnet.core_cell as core_cell_module

    cell = _core_cell("server/job")
    counter = F3Counter()
    message = Message(headers={MessageHeaderKey.DESTINATION: "client/job"}, payload=b"payload")
    context = attach_logical_send_context(message, counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    assert context.pre_admit("server/job", "client/job")

    def fail_encode(_message, fobs_ctx):
        raise RuntimeError("encode failed")

    monkeypatch.setattr(core_cell_module, "encode_payload", fail_encode)
    cell.encrypt_payload = MagicMock()

    assert cell._send_to_endpoint(Endpoint("client/job"), message)
    assert counter.pending_count == 0
    assert counter.close_and_drain(0.0)
    assert counter.freeze() == {"sent_to": []}


def test_core_cell_counts_a_local_first_hop_to_a_remote_destination(monkeypatch):
    import nvflare.fuel.f3.cellnet.core_cell as core_cell_module

    cell = _core_cell()
    cell.ALL_CELLS["relay"] = object()
    cell._send_direct_message = MagicMock()
    counter = F3Counter()
    message = Message(headers={MessageHeaderKey.DESTINATION: "site-1"}, payload=b"payload")
    attach_logical_send_context(message, counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    monkeypatch.setattr(core_cell_module, "encode_payload", lambda _message, fobs_ctx: 7)
    cell.encrypt_payload = MagicMock()

    assert cell._send_to_endpoint(Endpoint("relay"), message) == ""
    cell._send_direct_message.assert_called_once()
    assert counter.freeze() == {
        "sent_to": [{"participant_name": "site-1", "payload_bytes": "7", "messages": "1"}],
    }


def test_accounting_exception_never_changes_core_transport(monkeypatch):
    import nvflare.fuel.f3.cellnet.core_cell as core_cell_module

    class BrokenAccounting:
        def __getattr__(self, _name):
            raise RuntimeError("accounting failed")

    cell = _core_cell()
    message = Message(headers={MessageHeaderKey.DESTINATION: "site-1"}, payload=b"payload")
    attach_logical_send_context(message, BrokenAccounting(), object(), recipient_name="site-1")
    monkeypatch.setattr(core_cell_module, "encode_payload", lambda _message, fobs_ctx: 7)
    cell.encrypt_payload = MagicMock()

    assert cell._send_to_endpoint(Endpoint("site-1"), message) == ""
    cell.communicator.send.assert_called_once()


def test_context_exception_never_changes_core_transport(monkeypatch):
    import nvflare.fuel.f3.cellnet.core_cell as core_cell_module

    class BrokenContext:
        def is_origin(self, _origin):
            raise RuntimeError("context failed")

        def mark_counter_gap(self):
            raise RuntimeError("gap reporting also failed")

    cell = _core_cell()
    message = Message(headers={MessageHeaderKey.DESTINATION: "site-1"}, payload=b"payload")
    message.set_logical_send_context(BrokenContext())
    monkeypatch.setattr(core_cell_module, "encode_payload", lambda _message, fobs_ctx: 7)
    cell.encrypt_payload = MagicMock()

    assert cell._send_to_endpoint(Endpoint("site-1"), message) == ""
    cell.communicator.send.assert_called_once()


def test_terminal_accounting_exception_never_changes_core_transport(monkeypatch):
    import nvflare.fuel.f3.cellnet.core_cell as core_cell_module

    class BrokenAttempt:
        def accepted(self, _payload_bytes):
            raise RuntimeError("completion failed")

    context = MagicMock()
    context.is_origin.return_value = True
    context.try_begin.return_value = BrokenAttempt()
    cell = _core_cell()
    message = Message(headers={MessageHeaderKey.DESTINATION: "site-1"}, payload=b"payload")
    message.set_logical_send_context(context)
    monkeypatch.setattr(core_cell_module, "encode_payload", lambda _message, fobs_ctx: 7)
    cell.encrypt_payload = MagicMock()

    assert cell._send_to_endpoint(Endpoint("site-1"), message) == ""
    cell.communicator.send.assert_called_once()
    context.mark_counter_gap.assert_called_once()


def test_byte_streamer_accounts_one_whole_stream_at_terminal_success(monkeypatch):
    cell = SimpleNamespace(my_info=FqcnInfo("origin"))
    streamer = ByteStreamer.__new__(ByteStreamer)
    streamer.cell = cell
    streamer.chunk_size = 4
    counter = F3Counter()
    message = Message(payload=b"abcdef")
    context = attach_logical_send_context(message, counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    stream = BlobStream(message.payload, message.headers)

    def admit_task(task, _handler):
        task.task_future = object()

    monkeypatch.setattr(TxTask, "start_task_thread", admit_task)
    monkeypatch.setattr(ByteStreamer.sent_stream_counter_pool, "increment", lambda **_kwargs: None)
    monkeypatch.setattr(ByteStreamer.sent_stream_size_pool, "record_value", lambda **_kwargs: None)

    future = streamer.send(
        "channel",
        "topic",
        "site-1",
        message.headers,
        stream,
        STREAM_TYPE_BLOB,
        accounting_context=context,
    )
    with ByteStreamer.map_lock:
        ByteStreamer.tx_task_map.pop(future.stream_id, None)

    assert counter.pending_count == 1
    assert _total_value(counter.snapshot()) == {"payload_bytes": "0", "messages": "0"}
    future.set_result(6)

    assert _counter_value(counter) == {"payload_bytes": "6", "messages": "1"}


@pytest.mark.parametrize("declared_size,payload", [(6, b"abcdef"), (0, b"abcdef"), (0, b""), (99, b"abcdef")])
def test_byte_streamer_counts_actual_streamed_bytes_instead_of_declared_size(monkeypatch, declared_size, payload):
    cell = SimpleNamespace(my_info=FqcnInfo("origin"), ALL_CELLS={}, fire_and_forget=MagicMock(return_value={}))
    streamer = ByteStreamer.__new__(ByteStreamer)
    streamer.cell = cell
    streamer.chunk_size = 4
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    stream = MagicMock()
    stream.get_size.return_value = declared_size
    stream.read.side_effect = [payload[:4], payload[4:], b""] if payload else [b""]
    tasks = []

    def admit_task(task, _handler):
        tasks.append(task)
        task.task_future = MagicMock()

    monkeypatch.setattr(TxTask, "start_task_thread", admit_task)
    monkeypatch.setattr(ByteStreamer.sent_stream_counter_pool, "increment", lambda **_kwargs: None)
    monkeypatch.setattr(ByteStreamer.sent_stream_size_pool, "record_value", lambda **_kwargs: None)

    future = streamer.send("channel", "topic", "site-1", {}, stream, reliable=False, accounting_context=context)
    try:
        assert counter.pending_count == 1
        tasks[0].send_loop()

        assert future.result(timeout=0) == len(payload)
        assert counter.pending_count == 0
        assert _counter_value(counter) == {"payload_bytes": str(len(payload)), "messages": "1"}
        sent_bytes = sum(len(sent.args[3].payload) for sent in cell.fire_and_forget.call_args_list)
        assert sent_bytes == len(payload)
    finally:
        with ByteStreamer.map_lock:
            ByteStreamer.tx_task_map.pop(future.stream_id, None)
            ByteStreamer.error_context_map.pop(future.stream_id, None)


@pytest.mark.parametrize("terminal", ("error", "cancel"))
@pytest.mark.parametrize("declared_size", (0, 6))
def test_byte_streamer_abandons_accounting_on_async_failure_or_cancel(monkeypatch, terminal, declared_size):
    from nvflare.fuel.f3.streaming.stream_types import StreamError

    cell = SimpleNamespace(my_info=FqcnInfo("origin"), ALL_CELLS={})
    streamer = ByteStreamer.__new__(ByteStreamer)
    streamer.cell = cell
    streamer.chunk_size = 4
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    stream = BlobStream(b"abcdef", {})
    stream.size = declared_size

    def admit_task(task, _handler):
        task.task_future = MagicMock()

    monkeypatch.setattr(TxTask, "start_task_thread", admit_task)
    monkeypatch.setattr(ByteStreamer.sent_stream_counter_pool, "increment", lambda **_kwargs: None)
    monkeypatch.setattr(ByteStreamer.sent_stream_size_pool, "record_value", lambda **_kwargs: None)

    future = streamer.send("channel", "topic", "site-1", {}, stream, accounting_context=context)
    with ByteStreamer.map_lock:
        ByteStreamer.tx_task_map.pop(future.stream_id, None)
    assert counter.pending_count == 1

    if terminal == "error":
        future.set_exception(StreamError("async send failed"))
    else:
        assert future.cancel()

    assert counter.pending_count == 0
    assert _counter_value(counter) == {"payload_bytes": "0", "messages": "0"}


def test_byte_streamer_excludes_direct_final_but_counts_remote_target_with_local_relay(monkeypatch):
    cell = SimpleNamespace(my_info=FqcnInfo("origin"), ALL_CELLS={"site-1": object(), "relay": object()})
    streamer = ByteStreamer.__new__(ByteStreamer)
    streamer.cell = cell
    streamer.chunk_size = 4
    direct_counter = F3Counter()
    direct_context = attach_logical_send_context(
        Message(), direct_counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1"
    )
    remote_counter = F3Counter()
    remote_context = attach_logical_send_context(
        Message(), remote_counter, F3TrafficClass.TASK_RESULT, recipient_name="site-2"
    )

    def admit_task(task, _handler):
        task.task_future = object()

    monkeypatch.setattr(TxTask, "start_task_thread", admit_task)
    monkeypatch.setattr(ByteStreamer.sent_stream_counter_pool, "increment", lambda **_kwargs: None)
    monkeypatch.setattr(ByteStreamer.sent_stream_size_pool, "record_value", lambda **_kwargs: None)

    direct = streamer.send(
        "channel", "topic", "site-1", {}, BlobStream(b"local", {}), accounting_context=direct_context
    )
    remote = streamer.send(
        "channel", "topic", "site-2", {}, BlobStream(b"remote", {}), accounting_context=remote_context
    )
    with ByteStreamer.map_lock:
        ByteStreamer.tx_task_map.pop(direct.stream_id, None)
        ByteStreamer.tx_task_map.pop(remote.stream_id, None)
    direct.set_result(5)
    remote.set_result(6)

    assert _counter_value(direct_counter) == {"payload_bytes": "0", "messages": "0"}
    assert _counter_value(remote_counter) == {"payload_bytes": "6", "messages": "1"}


@pytest.mark.parametrize("pre_admitted", [False, True])
def test_byte_streamer_direct_delivery_releases_pre_admission_without_changing_stream(monkeypatch, pre_admitted):
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    if pre_admitted:
        assert context.pre_admit("server/job", "client/job")
    counter.close()
    cell = SimpleNamespace(my_info=FqcnInfo("server/job"), ALL_CELLS={"client/job": object()})
    streamer = ByteStreamer.__new__(ByteStreamer)
    streamer.cell = cell
    streamer.chunk_size = 4
    started = []

    def admit_task(task, handler):
        started.append((task.target, handler))
        task.task_future = object()

    monkeypatch.setattr(TxTask, "start_task_thread", admit_task)
    monkeypatch.setattr(ByteStreamer.sent_stream_counter_pool, "increment", lambda **_kwargs: None)
    monkeypatch.setattr(ByteStreamer.sent_stream_size_pool, "record_value", lambda **_kwargs: None)

    future = streamer.send("channel", "topic", "client/job", {}, BlobStream(b"local", {}), accounting_context=context)
    with ByteStreamer.map_lock:
        ByteStreamer.tx_task_map.pop(future.stream_id, None)
    assert started == [("client/job", streamer._transmit_task)]
    assert counter.pending_count == 0
    assert counter.close_and_drain(0.0)
    future.set_result(5)
    assert future.result() == 5
    assert counter.freeze() == {"sent_to": []}


def test_excluded_delivery_accounting_exception_never_changes_the_stream(monkeypatch):
    context = MagicMock()
    context.abandon_excluded_delivery.side_effect = RuntimeError("excluded accounting failed")
    cell = SimpleNamespace(my_info=FqcnInfo("server/job"), ALL_CELLS={"client/job": object()})
    streamer = ByteStreamer.__new__(ByteStreamer)
    streamer.cell = cell
    streamer.chunk_size = 4
    started = []

    def admit_task(task, handler):
        started.append((task.target, handler))
        task.task_future = object()

    monkeypatch.setattr(TxTask, "start_task_thread", admit_task)
    monkeypatch.setattr(ByteStreamer.sent_stream_counter_pool, "increment", lambda **_kwargs: None)
    monkeypatch.setattr(ByteStreamer.sent_stream_size_pool, "record_value", lambda **_kwargs: None)

    future = streamer.send("channel", "topic", "client/job", {}, BlobStream(b"local", {}), accounting_context=context)
    with ByteStreamer.map_lock:
        ByteStreamer.tx_task_map.pop(future.stream_id, None)
    future.set_result(5)

    assert started == [("client/job", streamer._transmit_task)]
    assert future.result() == 5
    context.try_begin.assert_not_called()
    context.mark_counter_gap.assert_called_once()


def test_context_exception_never_changes_byte_streamer_admission(monkeypatch):
    class BrokenContext:
        def try_begin(self, _origin, _destination):
            raise RuntimeError("context failed")

        def mark_counter_gap(self):
            raise RuntimeError("gap reporting also failed")

    cell = SimpleNamespace(my_info=FqcnInfo("origin"))
    streamer = ByteStreamer.__new__(ByteStreamer)
    streamer.cell = cell
    streamer.chunk_size = 4
    stream = BlobStream(b"abcdef", {})

    def admit_task(task, _handler):
        task.task_future = object()

    monkeypatch.setattr(TxTask, "start_task_thread", admit_task)
    monkeypatch.setattr(ByteStreamer.sent_stream_counter_pool, "increment", lambda **_kwargs: None)
    monkeypatch.setattr(ByteStreamer.sent_stream_size_pool, "record_value", lambda **_kwargs: None)

    future = streamer.send("channel", "topic", "site-1", {}, stream, accounting_context=BrokenContext())
    with ByteStreamer.map_lock:
        ByteStreamer.tx_task_map.pop(future.stream_id, None)

    assert future is not None


def test_terminal_accounting_exception_never_changes_byte_streamer_admission(monkeypatch):
    class BrokenAttempt:
        def accepted(self, _payload_bytes):
            raise RuntimeError("completion failed")

    context = MagicMock()
    context.try_begin.return_value = BrokenAttempt()
    cell = SimpleNamespace(my_info=FqcnInfo("origin"))
    streamer = ByteStreamer.__new__(ByteStreamer)
    streamer.cell = cell
    streamer.chunk_size = 4
    stream = BlobStream(b"abcdef", {})

    def admit_task(task, _handler):
        task.task_future = object()

    monkeypatch.setattr(TxTask, "start_task_thread", admit_task)
    monkeypatch.setattr(ByteStreamer.sent_stream_counter_pool, "increment", lambda **_kwargs: None)
    monkeypatch.setattr(ByteStreamer.sent_stream_size_pool, "record_value", lambda **_kwargs: None)

    future = streamer.send("channel", "topic", "site-1", {}, stream, accounting_context=context)
    with ByteStreamer.map_lock:
        ByteStreamer.tx_task_map.pop(future.stream_id, None)
    future.set_result(6)

    assert future is not None
    context.mark_counter_gap.assert_called_once()


def test_download_service_contribution_is_deduped_and_settled_after_transport_acceptance():
    service = make_isolated_download_service()
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    tx_id = service.new_transaction(
        cell=MagicMock(),
        timeout=10.0,
        num_receivers=1,
        receiver_ids=("site-1",),
        send_accounting_context=context,
    )
    ref_id = service.add_object(tx_id, MockDownloadable([b"large-data"]))

    main = context.try_begin("origin", "site-1")
    assert main is not None
    main.accepted(4)
    first_reply = service._handle_download(pull_request(ref_id, "site-1"))
    retry_reply = service._handle_download(pull_request(ref_id, "site-1"))
    first_contribution = first_reply.get_logical_send_context()
    retry_contribution = retry_reply.get_logical_send_context()
    first_attempt = first_contribution.try_begin("origin", "site-1")
    assert retry_contribution.try_begin("origin", "site-1") is None

    eof_reply = service._handle_download(pull_request(ref_id, "site-1", state=first_reply.payload["state"]))
    assert eof_reply.payload["status"] == "eof"
    tx = service._tx_table[tx_id]
    tx.transaction_done(TransactionDoneStatus.FINISHED)
    assert counter.pending_count == 1

    first_attempt.accepted(9999)

    assert counter.pending_count == 0
    assert _counter_value(counter) == {"payload_bytes": "14", "messages": "1"}


def _trainer_download(chunks):
    service = make_service_no_monitor()
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")
    context.prepare_destination("server.job", "client.job")
    tx_id = service.new_transaction(cell=MagicMock(), timeout=10, num_receivers=1, send_accounting_context=context)
    source = MockDownloadable(chunks)
    ref_id = service.add_object(tx_id, source)
    main = context.try_begin("server.job", "client.job")
    main.accepted(7)
    return service, counter, context, service._tx_table[tx_id], source, ref_id


def _trainer_pull(ref_id, proof, state=None):
    request = pull_request(ref_id, "client.job.trainer", state=state)
    request.set_header("proof", proof)
    return request


def test_revalidated_trainer_download_retries_count_once_and_keep_count_based_lifetime():
    service, counter, context, tx, source, ref_id = _trainer_download([b"first", b"second"])
    resolver = MagicMock(
        side_effect=lambda consumer, request: "client.job" if request.get_header("proof") == "valid" else None
    )
    context.set_consumer_resolver(resolver)

    first = service._handle_download(_trainer_pull(ref_id, "valid"))
    first.get_logical_send_context().try_begin("server.job", "client.job.trainer").accepted(1000)
    retry = service._handle_download(_trainer_pull(ref_id, "valid"))
    assert retry.payload["data"] == b"first"
    assert retry.get_logical_send_context().try_begin("server.job", "client.job.trainer") is None
    second = service._handle_download(_trainer_pull(ref_id, "valid", first.payload["state"]))
    second.get_logical_send_context().try_begin("server.job", "client.job.trainer").accepted(1000)
    eof = service._handle_download(_trainer_pull(ref_id, "valid", second.payload["state"]))

    assert resolver.call_count == 4  # Includes duplicate chunks and the EOF pull.
    assert eof.payload["status"] == "eof"
    assert tx.receiver_ids is None
    assert tx.num_receivers == 1
    assert tx.is_finished()
    outcome = tx.transaction_done(TransactionDoneStatus.FINISHED)

    assert outcome.receiver_ids is None
    assert source.released
    assert source.transaction_done_calls == [(tx.tid, TransactionDoneStatus.FINISHED)]
    assert counter.pending_count == 0
    assert counter.freeze() == {"sent_to": [{"participant_name": "site-1", "payload_bytes": "18", "messages": "1"}]}


@pytest.mark.parametrize("reject_on", ["data", "eof"])
@pytest.mark.parametrize("rejected", [None, "changed", RuntimeError("private-proof")])
def test_proof_loss_after_first_chunk_never_changes_download_or_source_release(reject_on, rejected):
    service, counter, context, tx, source, ref_id = _trainer_download([b"first", b"second"])

    def resolve(_consumer, request):
        proof = request.get_header("proof")
        if isinstance(proof, Exception):
            raise proof
        return "client.job" if proof == "valid" else None

    resolver = MagicMock(side_effect=resolve)
    context.set_consumer_resolver(resolver)
    first = service._handle_download(_trainer_pull(ref_id, "valid"))
    first.get_logical_send_context().try_begin("server.job", "client.job.trainer").accepted(1000)
    second = service._handle_download(
        _trainer_pull(ref_id, rejected if reject_on == "data" else "valid", first.payload["state"])
    )
    assert second.payload["data"] == b"second"
    if reject_on == "data":
        assert second.get_logical_send_context() is None
    else:
        second.get_logical_send_context().try_begin("server.job", "client.job.trainer").accepted(1000)
    eof = service._handle_download(
        _trainer_pull(ref_id, rejected if reject_on == "eof" else "valid", second.payload["state"])
    )

    assert resolver.call_count == (2 if reject_on == "data" else 3)
    assert eof.payload["status"] == "eof"
    assert tx.receiver_ids is None
    assert tx.num_receivers == 1
    assert tx.is_finished()
    outcome = tx.transaction_done(TransactionDoneStatus.FINISHED)

    assert outcome.receiver_ids is None
    assert source.downloaded_to_one_calls == [("client.job.trainer", "success")]
    assert source.released
    assert source.transaction_done_calls == [(tx.tid, TransactionDoneStatus.FINISHED)]
    assert counter.pending_count == 0
    assert counter.freeze() == {"status": "partial", "issues": ["counter_gap"], "sent_to": []}


def test_first_unresolved_pull_then_valid_proof_continues_delivery_but_stays_partial():
    service, counter, context, tx, source, ref_id = _trainer_download([b"first", b"second"])
    context.set_consumer_resolver(
        lambda _consumer, request: "client.job" if request.get_header("proof") == "valid" else None
    )
    first = service._handle_download(_trainer_pull(ref_id, None))
    assert first.payload["data"] == b"first"
    assert first.get_logical_send_context() is None
    second = service._handle_download(_trainer_pull(ref_id, "valid", first.payload["state"]))
    assert second.payload["data"] == b"second"
    assert second.get_logical_send_context() is None
    eof = service._handle_download(_trainer_pull(ref_id, "valid", second.payload["state"]))

    assert eof.payload["status"] == "eof"
    assert tx.is_finished()
    assert tx.receiver_ids is None
    tx.transaction_done(TransactionDoneStatus.FINISHED)
    assert source.released
    assert counter.freeze() == {"status": "partial", "issues": ["counter_gap"], "sent_to": []}


def test_download_chunk_identity_uses_produced_chunk_not_extra_request_fields():
    tx = _Transaction(timeout=10.0, num_receivers=1)
    ref = tx.add_object(MockDownloadable([b"data"]))
    first_key = ref.accounting_chunk_key({"extra": object()}, {"chunk_idx": 1})
    retry_key = ref.accounting_chunk_key({"extra": b"x" * 100_000}, {"chunk_idx": 1})
    assert first_key == retry_key == (0, 1)
    assert ref.accounting_chunk_key(None, {"chunk_idx": 2}) == (1, 2)


def test_unaccounted_download_does_not_compute_a_chunk_identity(monkeypatch):
    hook = MagicMock(side_effect=AssertionError("unaccounted download requested an accounting key"))
    monkeypatch.setattr(MockDownloadable, "accounting_chunk_key", hook)
    service = make_isolated_download_service()
    tx_id = service.new_transaction(cell=MagicMock(), timeout=10.0, num_receivers=1)
    ref_id = service.add_object(tx_id, MockDownloadable([b"data"]))

    reply = service._handle_download(pull_request(ref_id, "site-1"))

    assert reply.payload["data"] == b"data"
    hook.assert_not_called()


def test_unsupported_download_producer_marks_f3_partial_without_changing_served_data():
    service = make_isolated_download_service()
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, recipient_name="site-1")
    tx_id = service.new_transaction(
        cell=MagicMock(),
        timeout=10.0,
        num_receivers=1,
        receiver_ids=("site-1",),
        send_accounting_context=context,
    )
    obj = MockDownloadable([b"data"])
    obj.accounting_chunk_key = lambda current_state, next_state: None
    ref_id = service.add_object(tx_id, obj)
    main = context.try_begin("origin", "site-1")
    main.accepted(4)

    reply = service._handle_download(pull_request(ref_id, "site-1", state={"custom": object()}))

    assert reply.payload["data"] == b"data"
    assert reply.get_logical_send_context() is None
    snapshot = counter.freeze()
    assert snapshot["status"] == "partial"
    assert snapshot["issues"] == ["counter_gap"]
    assert _total_value(snapshot) == {"payload_bytes": "0", "messages": "0"}


def test_download_service_accounting_callback_failures_do_not_change_serving_or_settlement():
    class BrokenContext:
        def __init__(self):
            self.gaps = 0

        def register_oob_transaction(self, _transaction_id, _receiver_ids):
            raise RuntimeError("registration failed")

        def make_oob_contribution_context(self, **_kwargs):
            raise RuntimeError("contribution failed")

        def mark_oob_incomplete(self, _destination):
            raise RuntimeError("incomplete failed")

        def settle_oob_transaction(self, _transaction_id, _successful_receivers):
            raise RuntimeError("settlement failed")

        def mark_counter_gap(self):
            self.gaps += 1

    service = make_isolated_download_service()
    context = BrokenContext()
    tx_id = service.new_transaction(
        cell=MagicMock(),
        timeout=10.0,
        num_receivers=1,
        receiver_ids=("site-1",),
        send_accounting_context=context,
    )
    ref_id = service.add_object(tx_id, MockDownloadable([b"data"]))

    reply = service._handle_download(pull_request(ref_id, "site-1"))
    assert reply.payload["data"] == b"data"
    assert reply.get_logical_send_context() is None
    tx = service._tx_table[tx_id]
    outcome = tx.transaction_done(TransactionDoneStatus.DELETED)

    assert outcome is not None
    assert tx._settlement_complete
    assert context.gaps >= 2
