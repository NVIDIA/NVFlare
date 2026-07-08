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

"""Tests for receiver-confirmed completion + retry-aware accounting (plan: F3-2).

Producer side: a confirm-capable receiver's served EOF/ERROR is PROVISIONAL; the receiver's
confirmation finalizes it (receiver truth wins, first confirm is final, retries overwrite
provisional state). Legacy receivers keep today's producer-served semantics -- both version
skews degrade to the current behavior, and a runtime kill-switch disables the wire behavior
entirely without a code revert.

Receiver side: capability is advertised per request, confirmations are sent fire-and-forget
only toward producers that advertised they consume them, and the receiver truth is decided by
Consumer.download_completed() (finalization), not by the served EOF.
"""

import time
from unittest.mock import Mock, patch

import pytest

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.streaming import download_service as ds_module
from nvflare.fuel.f3.streaming.download_service import Consumer, DownloadStatus, ProduceRC, _PropKey, download_object
from tests.unit_test.fuel.f3.streaming.download_test_utils import (
    MockDownloadable,
    make_isolated_download_service,
    run_monitor_once,
)


@pytest.fixture(autouse=True)
def confirm_switch_on():
    """Pin the kill-switch ON for tests (individual tests patch it OFF explicitly)."""
    with patch.object(ds_module, "_receiver_confirm_cached", True):
        yield


def _make_service():
    service = make_isolated_download_service()
    service._tx_monitor = Mock()  # suppress the real monitor thread
    return service


def _new_tx(service, num_receivers=1, timeout=10.0, chunks=1):
    tx_id = service.new_transaction(cell=Mock(), timeout=timeout, num_receivers=num_receivers)
    obj = MockDownloadable([b"chunk"] * chunks)
    rid = service.add_object(tx_id, obj)
    return tx_id, rid, obj


def _pull_request(rid, requester, confirm_capable=False, state=None):
    payload = {_PropKey.REF_ID: rid}
    if confirm_capable:
        payload[_PropKey.CONFIRM_CAPABLE] = True
    if state is not None:
        payload[_PropKey.STATE] = state
    return new_cell_message(headers={MessageHeaderKey.ORIGIN: requester}, payload=payload)


def _confirm_request(rid, requester, status):
    return new_cell_message(
        headers={MessageHeaderKey.ORIGIN: requester}, payload={_PropKey.REF_ID: rid, _PropKey.CONFIRM: status}
    )


def _pull_to_terminal(service, rid, requester, confirm_capable=False):
    """Drives the pull loop for one receiver until the producer serves a terminal status."""
    state = None
    for _ in range(50):
        reply = service._handle_download(_pull_request(rid, requester, confirm_capable=confirm_capable, state=state))
        body = reply.payload
        status = body.get(_PropKey.STATUS)
        if status in (ProduceRC.EOF, ProduceRC.ERROR):
            return reply
        state = body.get(_PropKey.STATE)
    raise AssertionError("pull loop never reached a terminal status")


def _ref(service, rid):
    return service._ref_table[rid]


class TestProducerSide:
    def test_legacy_receiver_finalizes_at_serve(self):
        service = _make_service()
        tx_id, rid, _ = _new_tx(service)

        reply = _pull_to_terminal(service, rid, "r1", confirm_capable=False)

        assert _PropKey.CONFIRM_EXPECTED not in reply.payload
        ref = _ref(service, rid)
        assert ref.snapshot_receiver_statuses() == {"r1": DownloadStatus.SUCCESS}
        assert ref.snapshot_pending_confirms() == {}
        assert service._tx_table[tx_id].is_finished()

    def test_confirm_capable_receiver_is_provisional_at_serve(self):
        service = _make_service()
        tx_id, rid, _ = _new_tx(service)

        reply = _pull_to_terminal(service, rid, "r1", confirm_capable=True)

        assert reply.payload.get(_PropKey.CONFIRM_EXPECTED) is True
        ref = _ref(service, rid)
        # served EOF is NOT the receiver's truth yet
        assert ref.snapshot_receiver_statuses() == {}
        assert ref.snapshot_pending_confirms() == {"r1": DownloadStatus.SUCCESS}
        assert not service._tx_table[tx_id].is_finished()

    def test_confirmation_finalizes_and_finishes(self):
        service = _make_service()
        tx_id, rid, _ = _new_tx(service)
        _pull_to_terminal(service, rid, "r1", confirm_capable=True)

        reply = service._handle_download(_confirm_request(rid, "r1", DownloadStatus.SUCCESS))

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        ref = _ref(service, rid)
        assert ref.snapshot_receiver_statuses() == {"r1": DownloadStatus.SUCCESS}
        assert ref.snapshot_pending_confirms() == {}
        assert service._tx_table[tx_id].is_finished()

    def test_receiver_truth_wins_failed_confirm_after_served_eof(self):
        # the motivating case: producer served EOF, but the receiver's finalization failed
        service = _make_service()
        tx_id, rid, _ = _new_tx(service)
        _pull_to_terminal(service, rid, "r1", confirm_capable=True)

        service._handle_download(_confirm_request(rid, "r1", DownloadStatus.FAILED))

        ref = _ref(service, rid)
        assert ref.snapshot_receiver_statuses() == {"r1": DownloadStatus.FAILED}
        # the transaction reaches its receiver count, but the aggregate outcome must fail
        assert service._tx_table[tx_id].is_finished()
        run_monitor_once(service, now=time.time())
        outcome = service.get_transaction_outcome(tx_id)
        assert not outcome.completed

    def test_retry_overwrites_provisional_and_confirm_wins(self):
        # retry-aware accounting: a served ERROR is provisional; a healed retry (EOF) plus a
        # SUCCESS confirmation must finalize SUCCESS, not stick at first failure
        service = _make_service()
        tx_id, rid, obj = _new_tx(service)
        ref = _ref(service, rid)

        # simulate a produce-time failure serve, provisionally recorded
        ref.obj_served("r1", DownloadStatus.FAILED, expect_confirm=True)
        assert ref.snapshot_pending_confirms() == {"r1": DownloadStatus.FAILED}

        # the receiver retries; this time the pull succeeds end-to-end
        _pull_to_terminal(service, rid, "r1", confirm_capable=True)
        assert ref.snapshot_pending_confirms() == {"r1": DownloadStatus.SUCCESS}

        service._handle_download(_confirm_request(rid, "r1", DownloadStatus.SUCCESS))
        assert ref.snapshot_receiver_statuses() == {"r1": DownloadStatus.SUCCESS}
        run_monitor_once(service, now=time.time())
        assert service.get_transaction_outcome(tx_id).completed

    def test_first_confirm_is_final(self):
        service = _make_service()
        tx_id, rid, _ = _new_tx(service)
        _pull_to_terminal(service, rid, "r1", confirm_capable=True)

        service._handle_download(_confirm_request(rid, "r1", DownloadStatus.SUCCESS))
        service._handle_download(_confirm_request(rid, "r1", DownloadStatus.FAILED))

        assert _ref(service, rid).snapshot_receiver_statuses() == {"r1": DownloadStatus.SUCCESS}

    def test_late_duplicate_serve_cannot_resurrect_provisional(self):
        service = _make_service()
        tx_id, rid, _ = _new_tx(service)
        _pull_to_terminal(service, rid, "r1", confirm_capable=True)
        service._handle_download(_confirm_request(rid, "r1", DownloadStatus.SUCCESS))

        ref = _ref(service, rid)
        ref.obj_served("r1", DownloadStatus.FAILED, expect_confirm=True)

        assert ref.snapshot_pending_confirms() == {}
        assert ref.snapshot_receiver_statuses() == {"r1": DownloadStatus.SUCCESS}

    def test_invalid_confirm_status_is_ignored(self):
        service = _make_service()
        tx_id, rid, _ = _new_tx(service)
        _pull_to_terminal(service, rid, "r1", confirm_capable=True)

        service._handle_download(_confirm_request(rid, "r1", "bogus"))

        ref = _ref(service, rid)
        assert ref.snapshot_receiver_statuses() == {}
        assert ref.snapshot_pending_confirms() == {"r1": DownloadStatus.SUCCESS}

    def test_unsolicited_confirm_cannot_certify(self):
        # fail-open hole guard: a CONFIRM for a receiver that was never served a terminal
        # reply on THIS incarnation of the ref must be dropped -- otherwise a stale confirm
        # delayed across a ref_id reuse could certify (or pre-poison) a transfer that never
        # delivered a byte
        service = _make_service()
        tx_id, rid, _ = _new_tx(service)

        service._handle_download(_confirm_request(rid, "r1", DownloadStatus.SUCCESS))

        ref = _ref(service, rid)
        assert ref.snapshot_receiver_statuses() == {}
        assert not service._tx_table[tx_id].is_finished()
        run_monitor_once(service, now=time.time())
        assert service.get_transaction_outcome(tx_id) is None  # still live, nothing certified

        # the FAILED mirror cannot pre-poison either
        service._handle_download(_confirm_request(rid, "r1", DownloadStatus.FAILED))
        assert ref.snapshot_receiver_statuses() == {}

    def test_tombstoned_ref_replay_does_not_solicit_confirm(self):
        # after cleanup, a confirm-capable receiver retrying a finished ref gets the EOF
        # replay WITHOUT CONFIRM_EXPECTED (no live ref to confirm against), and a stray
        # confirm for the tombstoned rid is dropped OK without disturbing the outcome
        service = _make_service()
        tx_id, rid, _ = _new_tx(service)
        _pull_to_terminal(service, rid, "r1", confirm_capable=True)
        service._handle_download(_confirm_request(rid, "r1", DownloadStatus.SUCCESS))
        run_monitor_once(service, now=time.time())  # FINISHED -> tombstoned
        assert rid not in service._ref_table

        replay = service._handle_download(_pull_request(rid, "r1", confirm_capable=True))
        assert replay.payload.get(_PropKey.STATUS) == ProduceRC.EOF
        assert _PropKey.CONFIRM_EXPECTED not in replay.payload

        reply = service._handle_download(_confirm_request(rid, "r1", DownloadStatus.FAILED))
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        assert service.get_transaction_outcome(tx_id).completed  # unchanged

    def test_kill_switch_reads_application_conf(self):
        # the switch resolves through the standard application-config source (env var
        # NVFLARE_STREAMING_RECEIVER_CONFIRM_ENABLED / job config), like neighboring vars
        with patch.object(ds_module, "_receiver_confirm_cached", None):
            with patch.object(ds_module.ConfigService, "get_bool_var", return_value=False) as gv:
                assert ds_module._receiver_confirm_enabled() is False
                assert gv.call_args.kwargs.get("conf") == ds_module.SystemConfigs.APPLICATION_CONF

    def test_late_confirm_for_unknown_ref_is_dropped_ok(self):
        service = _make_service()
        reply = service._handle_download(_confirm_request("R-GONE", "r1", DownloadStatus.SUCCESS))
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK

    def test_unconfirmed_receiver_fails_closed_at_timeout(self):
        # a lost fire-and-forget confirmation must not certify success: at the transaction
        # timeout the unconfirmed receiver has no final status and the outcome fails
        service = _make_service()
        tx_id, rid, _ = _new_tx(service, timeout=10.0)
        _pull_to_terminal(service, rid, "r1", confirm_capable=True)

        tx = service._tx_table[tx_id]
        tx.last_active_time = time.time() - 11.0
        run_monitor_once(service, now=time.time())

        outcome = service.get_transaction_outcome(tx_id)
        assert outcome is not None
        assert not outcome.completed

    def test_kill_switch_off_restores_legacy_semantics(self):
        service = _make_service()
        with patch.object(ds_module, "_receiver_confirm_cached", False):
            tx_id, rid, _ = _new_tx(service)
            reply = _pull_to_terminal(service, rid, "r1", confirm_capable=True)

            # producer ignores the advertised capability entirely
            assert _PropKey.CONFIRM_EXPECTED not in reply.payload
            ref = _ref(service, rid)
            assert ref.snapshot_receiver_statuses() == {"r1": DownloadStatus.SUCCESS}
            assert ref.snapshot_pending_confirms() == {}
            assert service._tx_table[tx_id].is_finished()

    def test_mixed_fleet_finishes_on_mixed_semantics(self):
        # one legacy receiver (final at serve) + one confirm-capable receiver (final at confirm)
        service = _make_service()
        tx_id, rid, _ = _new_tx(service, num_receivers=2)
        tx = service._tx_table[tx_id]

        _pull_to_terminal(service, rid, "legacy", confirm_capable=False)
        _pull_to_terminal(service, rid, "modern", confirm_capable=True)
        assert not tx.is_finished()  # modern receiver not confirmed yet

        service._handle_download(_confirm_request(rid, "modern", DownloadStatus.SUCCESS))
        assert tx.is_finished()
        run_monitor_once(service, now=time.time())
        assert service.get_transaction_outcome(tx_id).completed


class _ScriptedCell:
    """A cell whose send_request returns scripted replies and which records fire_and_forget."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.requests = []
        self.confirms = []

    def send_request(self, channel, target, topic, request, timeout, secure, optional, abort_signal):
        self.requests.append(request.payload)
        if not self._replies:
            raise AssertionError("scripted cell ran out of replies")
        return self._replies.pop(0)

    def fire_and_forget(self, channel, topic, targets, message, secure=False, optional=False):
        self.confirms.append(message.payload)


class _RecordingConsumer(Consumer):
    def __init__(self, fail_on_complete=False):
        super().__init__()
        self.consumed = []
        self.completed = False
        self.failed_reason = None
        self._fail_on_complete = fail_on_complete

    def consume(self, ref_id, state, data):
        self.consumed.append(data)
        return state or {}

    def download_completed(self, ref_id):
        if self._fail_on_complete:
            raise RuntimeError("finalization failed")
        self.completed = True

    def download_failed(self, ref_id, reason):
        self.failed_reason = reason


def _ok_reply(body):
    return make_reply(ReturnCode.OK, body=body)


def _chunk_reply(confirm_expected=True):
    body = {_PropKey.STATUS: ProduceRC.OK, _PropKey.STATE: {"seq": 1}, _PropKey.DATA: b"chunk"}
    if confirm_expected:
        body[_PropKey.CONFIRM_EXPECTED] = True
    return _ok_reply(body)


def _terminal_reply(status, confirm_expected=True):
    body = {_PropKey.STATUS: status}
    if confirm_expected:
        body[_PropKey.CONFIRM_EXPECTED] = True
    return _ok_reply(body)


class TestReceiverSide:
    def test_advertises_capability_and_confirms_success_after_finalization(self):
        cell = _ScriptedCell([_chunk_reply(), _terminal_reply(ProduceRC.EOF)])
        consumer = _RecordingConsumer()

        download_object(from_fqcn="site-1", ref_id="R1", per_request_timeout=5.0, cell=cell, consumer=consumer)

        assert all(req.get(_PropKey.CONFIRM_CAPABLE) is True for req in cell.requests)
        assert consumer.completed
        assert cell.confirms == [{_PropKey.REF_ID: "R1", _PropKey.CONFIRM: DownloadStatus.SUCCESS}]

    def test_confirms_failed_when_finalization_raises(self):
        # served EOF but download_completed raises: the producer must learn receiver truth
        cell = _ScriptedCell([_terminal_reply(ProduceRC.EOF)])
        consumer = _RecordingConsumer(fail_on_complete=True)

        with pytest.raises(RuntimeError, match="finalization failed"):
            download_object(from_fqcn="site-1", ref_id="R1", per_request_timeout=5.0, cell=cell, consumer=consumer)

        assert cell.confirms == [{_PropKey.REF_ID: "R1", _PropKey.CONFIRM: DownloadStatus.FAILED}]

    def test_confirms_failed_on_producer_error(self):
        cell = _ScriptedCell([_chunk_reply(), _terminal_reply(ProduceRC.ERROR)])
        consumer = _RecordingConsumer()

        download_object(from_fqcn="site-1", ref_id="R1", per_request_timeout=5.0, cell=cell, consumer=consumer)

        assert consumer.failed_reason is not None
        assert cell.confirms == [{_PropKey.REF_ID: "R1", _PropKey.CONFIRM: DownloadStatus.FAILED}]

    def test_no_confirm_toward_legacy_producer(self):
        # the producer never advertised CONFIRM_EXPECTED: a new receiver must send nothing extra
        cell = _ScriptedCell(
            [_chunk_reply(confirm_expected=False), _terminal_reply(ProduceRC.EOF, confirm_expected=False)]
        )
        consumer = _RecordingConsumer()

        download_object(from_fqcn="site-1", ref_id="R1", per_request_timeout=5.0, cell=cell, consumer=consumer)

        assert consumer.completed
        assert cell.confirms == []

    def test_kill_switch_off_receiver_is_fully_legacy(self):
        with patch.object(ds_module, "_receiver_confirm_cached", False):
            cell = _ScriptedCell([_terminal_reply(ProduceRC.EOF, confirm_expected=True)])
            consumer = _RecordingConsumer()

            download_object(from_fqcn="site-1", ref_id="R1", per_request_timeout=5.0, cell=cell, consumer=consumer)

            assert all(_PropKey.CONFIRM_CAPABLE not in req for req in cell.requests)
            assert cell.confirms == []
            assert consumer.completed
