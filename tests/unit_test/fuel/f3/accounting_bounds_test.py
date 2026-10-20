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

from unittest.mock import Mock

import pytest

from nvflare.fuel.f3 import send_accounting
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.send_accounting import SourceContributionAccumulator, attach_logical_send_context
from nvflare.fuel.f3.streaming.cacheable import CacheableObject
from nvflare.fuel.f3.streaming.download_service import Downloadable, ProduceRC, _Ref, _Transaction
from nvflare.fuel.f3.streaming.file_downloader import FileDownloadable
from nvflare.private.fed.resource_stats.f3_counter import F3Counter, F3TrafficClass
from tests.unit_test.fuel.f3.streaming.download_test_utils import make_service_no_monitor, pull_request


def _logical(*, admit=True):
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, "site-1")
    context.prepare_destination("origin", "site-1")
    context.register_oob_transaction("tx", ("site-1",))
    if admit:
        context.try_begin("origin", "site-1").accepted(7)
    return counter, context


def _chunk(context, index, size=1):
    return context.make_oob_contribution_context("tx", "site-1", ("tx", "ref", "site-1", (index,)), size)


@pytest.mark.parametrize("terminal", ["inflight", "accepted", "queued"])
def test_logical_send_has_one_cumulative_metadata_budget(monkeypatch, terminal):
    monkeypatch.setattr(send_accounting, "MAX_ACCOUNTING_ENTRIES", 3)
    counter, context = _logical(admit=terminal != "queued")
    attempts = []
    for index in range(3):
        attempt = _chunk(context, index).try_begin("origin", "site-1")
        assert attempt is not None
        attempts.append(attempt)
        if terminal != "inflight":
            attempt.accepted(999)

    assert _chunk(context, 3).try_begin("origin", "site-1") is None
    for index in range(4, 100):
        assert _chunk(context, index).try_begin("origin", "site-1") is None
    state = context._destinations["site-1"]
    assert len(state.contribution_lengths) == 3
    assert len(state.inflight_contributions) <= 3
    assert len(state.accepted_contributions) <= 3
    assert len(state.queued_contributions) <= 3
    assert context._metadata_count == 3
    if terminal == "queued":
        assert context.try_begin("origin", "site-1") is None
    for attempt in attempts:
        attempt.accepted(999)
    context.settle_oob_transaction("tx", ("site-1",))
    assert counter.pending_count == 0
    assert counter.freeze() == {"status": "partial", "issues": ["counter_gap"], "sent_to": []}


@pytest.mark.parametrize("terminal", ["inflight", "accepted", "completed"])
def test_conflicting_chunk_size_never_looks_complete(terminal):
    counter, context = _logical()
    first = _chunk(context, 0, 3).try_begin("origin", "site-1")
    if terminal != "inflight":
        first.accepted(99)
    if terminal == "completed":
        context.settle_oob_transaction("tx", ("site-1",))
    assert _chunk(context, 0, 4).try_begin("origin", "site-1") is None
    first.accepted(99)
    context.settle_oob_transaction("tx", ("site-1",))
    assert counter.pending_count == 0
    snapshot = counter.freeze()
    assert snapshot["status"] == "partial"
    assert snapshot["issues"] == ["counter_gap"]


def test_delegated_consumers_share_the_chunk_budget_and_stop_after_saturation(monkeypatch):
    monkeypatch.setattr(send_accounting, "MAX_ACCOUNTING_ENTRIES", 3)
    counter, context = _logical()
    context.set_consumer_resolver(lambda consumer, request: "site-1")
    assert context.resolve_oob_consumer("tx", "site-1", Message()) == "site-1"
    assert context._consumer_destinations == {}  # Direct consumers need no cached mapping.
    for index in range(2):
        assert context.resolve_oob_consumer("tx", f"site-1.trainer{index}", Message()) == "site-1"
    _chunk(context, 0).try_begin("origin", "site-1").accepted(99)
    assert context.resolve_oob_consumer("tx", "site-1.overflow", Message()) is None
    for index in range(100):
        assert context.resolve_oob_consumer("tx", f"site-1.later{index}", Message()) is None
    assert len(context._consumer_destinations) == 2
    assert context._metadata_count == 3
    assert counter.pending_count == 0
    assert counter.freeze()["issues"] == ["counter_gap"]


def test_queued_external_contributions_use_the_same_budget(monkeypatch):
    monkeypatch.setattr(send_accounting, "MAX_ACCOUNTING_ENTRIES", 3)
    counter, context = _logical(admit=False)
    for index in range(3):
        assert context.register_external_contribution(index, "site-1")
        context.settle_external_contribution(index, "site-1", 1, True)
    assert context.register_external_contribution(3, "site-1")
    context.settle_external_contribution(3, "site-1", 1, True)
    assert not context.register_external_contribution(4, "site-1")
    state = context._destinations["site-1"]
    assert len(state.contribution_lengths) == len(state.queued_contributions) == 3
    assert len(context._external_results) == 3
    assert context.try_begin("origin", "site-1") is None
    assert counter.pending_count == 0
    assert counter.freeze()["issues"] == ["counter_gap"]


def test_delegated_consumer_names_are_bounded_without_rejecting_downloads():
    counter, context = _logical()
    context.set_consumer_resolver(lambda consumer, request: "site-1")
    assert context.resolve_oob_consumer("tx", "x" * 513, Message()) is None
    assert context._consumer_destinations == {}
    assert counter.freeze()["issues"] == ["counter_gap"]


def test_source_result_budget_covers_all_references_and_stops_retention(monkeypatch):
    monkeypatch.setattr(send_accounting, "MAX_ACCOUNTING_ENTRIES", 3)
    source = SourceContributionAccumulator("trainer", ("server.job",))
    source.register_oob_transaction("tx", ("server.job",))
    for ref in ("ref1", "ref2"):
        source.register_source_reference("tx", ref)
    attempts = []
    for index, ref in enumerate(("ref1", "ref2", "ref1")):
        context = source.make_oob_contribution_context("tx", "server.job", ("tx", ref, "server.job", (index,)), 1)
        attempt = context.try_begin("trainer", "server.job")
        assert attempt is not None
        attempts.append(attempt)
    for index in range(3, 100):
        context = source.make_oob_contribution_context("tx", "server.job", ("tx", "ref2", "server.job", (index,)), 1)
        assert context.try_begin("trainer", "server.job") is None
    for attempt in attempts:
        attempt.accepted(99)
    source.settle_oob_transaction("tx", ("server.job",))
    assert source._chunk_count == 3
    assert sum(len(group.seen) for group in source._groups.values()) == 3
    assert all(not group["complete"] for group in source.snapshot()["groups"])


def test_repeated_file_download_ignores_arbitrary_peer_state(tmp_path):
    path = tmp_path / "one-byte"
    path.write_bytes(b"x")
    service = make_service_no_monitor()
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, "site-1")
    tx_id = service.new_transaction(
        cell=Mock(), timeout=10, num_receivers=1, receiver_ids=("site-1",), send_accounting_context=context
    )
    ref_id = service.add_object(tx_id, FileDownloadable(str(path), chunk_size=1))
    context.try_begin("origin", "site-1").accepted(7)
    for index in range(64):
        reply = service._handle_download(
            pull_request(ref_id, "site-1", state={"received_bytes": 0, "extra": str(index) + "x" * 65_536})
        )
        assert reply.payload["data"] == b"x"
        attempt = reply.get_logical_send_context().try_begin("origin", "site-1")
        if index == 0:
            assert attempt is not None
            attempt.accepted(99)
        else:
            assert attempt is None
    lengths = context._destinations["site-1"].contribution_lengths
    assert list(lengths) == [(tx_id, ref_id, "site-1", (0, 1))]
    assert context._metadata_count == 1


@pytest.mark.parametrize("source_only", [False, True])
def test_budget_exhaustion_preserves_real_file_delivery_and_completion(monkeypatch, tmp_path, source_only):
    monkeypatch.setattr(send_accounting, "MAX_ACCOUNTING_ENTRIES", 3)
    path = tmp_path / "five-bytes"
    path.write_bytes(b"abcde")
    counter = F3Counter()
    context = (
        SourceContributionAccumulator("origin", ("site-1",))
        if source_only
        else attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, "site-1")
    )
    service = make_service_no_monitor()
    tx_id = service.new_transaction(
        cell=Mock(), timeout=10, num_receivers=1, receiver_ids=("site-1",), send_accounting_context=context
    )
    ref_id = service.add_object(tx_id, FileDownloadable(str(path), chunk_size=1))
    if not source_only:
        context.try_begin("origin", "site-1").accepted(7)
    state = None
    received = b""
    for _ in range(5):
        reply = service._handle_download(pull_request(ref_id, "site-1", state=state))
        received += reply.payload["data"]
        state = reply.payload["state"]
        contribution = reply.get_logical_send_context()
        if contribution is not None:
            attempt = contribution.try_begin("origin", "site-1")
            if attempt is not None:
                attempt.accepted(999)
    terminal = service._handle_download(pull_request(ref_id, "site-1", state=state))
    assert received == b"abcde"
    assert terminal.payload["status"] == ProduceRC.EOF
    tx = service._tx_table[tx_id]
    assert tx.is_finished()
    context.settle_oob_transaction(tx_id, ("site-1",))
    if source_only:
        assert context._chunk_count == 3
        assert context.snapshot()["groups"][0]["complete"] is False
    else:
        assert context._metadata_count == 3
        assert counter.pending_count == 0
        assert counter.freeze()["issues"] == ["counter_gap"]


class _UnidentifiedProducer(Downloadable):
    def produce(self, state, requester):
        return ProduceRC.OK, b"data", {}  # No stable progress/chunk identity.


@pytest.mark.parametrize("bad_key", [None, (), (True,), (-1,), (1 << 64,), (0, 1, 2, 3, 4), [0], ("0",)])
def test_invalid_optional_chunk_identity_does_not_change_delivery(bad_key):
    service = make_service_no_monitor()
    counter = F3Counter()
    context = attach_logical_send_context(Message(), counter, F3TrafficClass.TASK_RESULT, "site-1")
    tx_id = service.new_transaction(
        cell=Mock(), timeout=10, num_receivers=1, receiver_ids=("site-1",), send_accounting_context=context
    )
    obj = _UnidentifiedProducer(None)
    if bad_key is not None:
        obj.accounting_chunk_key = lambda current_state, next_state: bad_key
    ref_id = service.add_object(tx_id, obj)
    context.try_begin("origin", "site-1").accepted(7)
    for _ in range(2):
        reply = service._handle_download(pull_request(ref_id, "site-1"))
        assert reply.payload["data"] == b"data"
        assert reply.get_logical_send_context() is None
    assert context._metadata_count == 0
    assert counter.pending_count == 0
    assert counter.freeze() == {"status": "partial", "issues": ["counter_gap"], "sent_to": []}


def test_optional_identity_hook_failure_does_not_change_delivery():
    obj = _UnidentifiedProducer(None)
    obj.accounting_chunk_key = Mock(side_effect=ValueError("bad custom key"))
    ref = _Transaction(timeout=10, num_receivers=1).add_object(obj)
    assert ref.accounting_chunk_key(None, {}) is None
    assert obj.produce(None, "site-1")[1] == b"data"


def test_cacheable_identity_is_actual_produced_item_range():
    class Items(CacheableObject):
        def get_item_count(self):
            return 3

        def produce_item(self, index):
            return b"x"

    obj = Items(None, max_chunk_size=2)
    ref = _Ref(Mock(), obj)
    rc, data, state = obj.produce({"start": 0, "count": 1, "ignored": object()}, "site-1")
    assert rc == ProduceRC.OK
    assert data == [b"x"]
    assert ref.accounting_chunk_key({"start": 0, "count": 1}, state) == (1, 1)
    assert ref.accounting_chunk_key({"start": 1, "count": 0}, state) == (1, 1)
