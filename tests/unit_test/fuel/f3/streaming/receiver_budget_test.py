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

"""Tests for per-(transfer, receiver) acquire/idle budgets and the quorum surface.

A receiver that exhausts its acquire budget (never issued its first pull) or idle budget
(stopped making requests) is finalized FAILED for the aggregate outcome on a monitor pass --
without waiting for the whole-transaction TTL, and without a live receiver masking a stalled
one behind the tx-wide activity timestamp. Budgets are per-transaction opt-in (config-var
defaults); disabled budgets preserve today's TTL-only behavior exactly.
"""

import time
from unittest.mock import Mock, patch

import pytest

from nvflare.fuel.f3.streaming import download_service as ds_module
from nvflare.fuel.f3.streaming.download_service import DownloadStatus
from nvflare.fuel.f3.streaming.transfer_outcome import TransferOutcomeReason, compute_transfer_outcome
from tests.unit_test.fuel.f3.streaming.download_test_utils import MockDownloadable, confirm_request
from tests.unit_test.fuel.f3.streaming.download_test_utils import make_confirm_test_service as _make_service
from tests.unit_test.fuel.f3.streaming.download_test_utils import pull_request
from tests.unit_test.fuel.f3.streaming.download_test_utils import pull_to_terminal as _pull_to_terminal
from tests.unit_test.fuel.f3.streaming.download_test_utils import run_monitor_once, serve_nonce


def _new_tx(service, chunks=1, **tx_kwargs):
    tx_kwargs.setdefault("timeout", 1000.0)  # the whole-tx TTL is deliberately huge:
    tx_kwargs.setdefault("num_receivers", 1)  # budgets, not the TTL, must resolve these tests
    tx_id = service.new_transaction(cell=Mock(), **tx_kwargs)
    obj = MockDownloadable([b"chunk"] * chunks)
    rid = service.add_object(tx_id, obj)
    return tx_id, rid


class TestIdleBudget:
    def test_stalled_receiver_fails_without_waiting_tx_ttl(self):
        service = _make_service()
        tx_id, rid = _new_tx(service, chunks=3, num_receivers=2, receiver_idle_timeout=5.0)

        _pull_to_terminal(service, rid, "healthy")  # legacy receiver, final at serve
        service._handle_download(pull_request(rid, "stalled"))  # one pull, then silence

        # a monitor pass past the idle budget (but far inside the 1000s TTL)
        run_monitor_once(service, now=time.time() + 30.0)

        outcome = service.get_transaction_outcome(tx_id)
        assert outcome is not None, "budget failure must resolve the tx on this pass, not at TTL"
        assert not outcome.completed
        assert outcome.reason == TransferOutcomeReason.RECEIVER_FAILED
        statuses = outcome.refs[0].receiver_statuses
        assert statuses["healthy"] == DownloadStatus.SUCCESS
        assert statuses["stalled"] == DownloadStatus.FAILED

    def test_lost_confirmation_is_bounded_by_idle_budget(self):
        # fire-and-forget confirms can be lost: the receiver stops making requests after its
        # served EOF, so its idle budget finalizes it FAILED in bounded time (fail-closed)
        service = _make_service()
        tx_id, rid = _new_tx(service, receiver_idle_timeout=5.0)
        _pull_to_terminal(service, rid, "r1", confirm_capable=True)  # provisional, confirm never arrives

        run_monitor_once(service, now=time.time() + 30.0)

        outcome = service.get_transaction_outcome(tx_id)
        assert outcome is not None
        assert not outcome.completed
        assert outcome.refs[0].receiver_statuses == {"r1": DownloadStatus.FAILED}
        # the provisional record was cleaned up, not leaked
        assert service._tx_outcomes  # outcome recorded; ref is gone with the finished tx

    def test_active_receiver_is_not_idle_failed(self):
        service = _make_service()
        tx_id, rid = _new_tx(service, chunks=3, receiver_idle_timeout=5.0)
        reply = service._handle_download(pull_request(rid, "r1"))
        ref = service._ref_table[rid]
        tx = service._tx_table[tx_id]
        # receiver keeps making requests: refresh activity to "now" as seen by the monitor
        future = time.time() + 30.0
        with tx._stats_lock:
            tx._receiver_last_active["r1"] = future - 1.0

        run_monitor_once(service, now=future)

        assert service.get_transaction_outcome(tx_id) is None  # still live
        assert ref.snapshot_receiver_statuses() == {}


class TestAcquireBudget:
    def test_never_pulling_receiver_fails_at_acquire_deadline(self):
        service = _make_service()
        tx_id, rid = _new_tx(
            service,
            num_receivers=0,  # derived from receiver_ids
            receiver_ids=("r1", "r2"),
            receiver_acquire_timeout=5.0,
            min_receivers=1,
        )
        _pull_to_terminal(service, rid, "r1")  # r2 never shows up

        run_monitor_once(service, now=time.time() + 30.0)

        outcome = service.get_transaction_outcome(tx_id)
        assert outcome is not None
        statuses = outcome.refs[0].receiver_statuses
        assert statuses == {"r1": DownloadStatus.SUCCESS, "r2": DownloadStatus.FAILED}
        # strict certificate fails; the declared k-of-N quorum is satisfied
        assert not outcome.completed
        assert outcome.quorum_met
        assert outcome.min_receivers == 1

    def test_receiver_with_activity_is_not_acquire_failed(self):
        service = _make_service()
        tx_id, rid = _new_tx(service, chunks=3, num_receivers=0, receiver_ids=("r1",), receiver_acquire_timeout=5.0)
        service._handle_download(pull_request(rid, "r1"))  # first pull happened: acquire satisfied

        run_monitor_once(service, now=time.time() + 30.0)

        # no idle budget configured, so the slow-but-acquired receiver is left alone
        assert service.get_transaction_outcome(tx_id) is None
        assert service._ref_table[rid].snapshot_receiver_statuses() == {}

    def test_acquire_budget_needs_receiver_identities(self):
        # without receiver_ids there is no one to hold to the acquire deadline
        service = _make_service()
        tx_id, rid = _new_tx(service, num_receivers=2, receiver_acquire_timeout=5.0)

        run_monitor_once(service, now=time.time() + 30.0)

        assert service.get_transaction_outcome(tx_id) is None
        assert service._ref_table[rid].snapshot_receiver_statuses() == {}


class TestBudgetSemantics:
    def test_budgets_disabled_preserve_ttl_only_behavior(self):
        service = _make_service()
        tx_id, rid = _new_tx(service, num_receivers=2)
        service._handle_download(pull_request(rid, "r1"))

        run_monitor_once(service, now=time.time() + 500.0)  # inside the 1000s TTL

        assert service.get_transaction_outcome(tx_id) is None
        assert service._ref_table[rid].snapshot_receiver_statuses() == {}

    def test_activity_tracked_without_progress_cb(self):
        service = _make_service()
        tx_id, rid = _new_tx(service, chunks=2)
        service._handle_download(pull_request(rid, "r1"))

        activity = service._ref_table[rid].snapshot_receiver_activity()
        assert "r1" in activity

    def test_budget_failed_receiver_is_final_even_against_late_confirm(self):
        service = _make_service()
        tx_id, rid = _new_tx(service, num_receivers=2, receiver_idle_timeout=5.0)
        terminal = _pull_to_terminal(service, rid, "r1", confirm_capable=True)
        ref = service._ref_table[rid]

        run_monitor_once(service, now=time.time() + 30.0)  # idle budget fires
        assert ref.snapshot_receiver_statuses()["r1"] == DownloadStatus.FAILED
        assert ref.snapshot_pending_confirms() == {}

        # the straggler confirmation (correct nonce and all) cannot resurrect it
        assert ref.obj_confirmed("r1", DownloadStatus.SUCCESS, serve_nonce(terminal)) is False
        assert ref.snapshot_receiver_statuses()["r1"] == DownloadStatus.FAILED


class TestMultiRefAcquisition:
    def test_sequential_multi_ref_receiver_is_not_acquire_failed(self):
        # transaction-level acquisition: a healthy receiver pulling ref 1 must not be
        # acquire-failed on ref 2 it has not reached yet
        service = _make_service()
        tx_id = service.new_transaction(
            cell=Mock(), timeout=1000.0, num_receivers=0, receiver_ids=("r1",), receiver_acquire_timeout=5.0
        )
        rid1 = service.add_object(tx_id, MockDownloadable([b"chunk"] * 3))
        rid2 = service.add_object(tx_id, MockDownloadable([b"chunk"] * 3))
        service._handle_download(pull_request(rid1, "r1"))  # busy on ref 1, has not touched ref 2

        run_monitor_once(service, now=time.time() + 30.0)

        assert service.get_transaction_outcome(tx_id) is None  # still live
        assert service._ref_table[rid2].snapshot_receiver_statuses() == {}

    def test_confirmed_receiver_wins_over_stale_budget_snapshot(self):
        # truth-wins re-check: a receiver finalized (confirmed) between the budget snapshot
        # and enforcement must not be flipped to FAILED, and no failure is reported for it
        service = _make_service()
        tx_id, rid = _new_tx(service, receiver_idle_timeout=5.0)
        terminal = _pull_to_terminal(service, rid, "r1", confirm_capable=True)
        ref = service._ref_table[rid]
        service._handle_download(confirm_request(rid, "r1", DownloadStatus.SUCCESS, serve_nonce(terminal)))

        tx = service._tx_table[tx_id]
        with tx._stats_lock:
            tx_last_active = dict(tx._receiver_last_active)
        enforced = ref.enforce_budgets(
            time.time() + 30.0, None, 5.0, None, tx_acquired={"r1"}, tx_last_active=tx_last_active
        )

        assert enforced == []
        assert ref.snapshot_receiver_statuses() == {"r1": DownloadStatus.SUCCESS}


class TestMultiRefIdleEscape:
    def test_receiver_idle_after_finishing_one_ref_fails_sibling_ref_in_bounded_time(self):
        # P2 pin (reproduced by review): after finishing ref 1, the receiver is tx-acquired
        # (exempt from ref 2's acquire budget) and has no per-ref timestamp on ref 2 -- with
        # per-ref idle it escaped BOTH budgets and pinned the producer to the full TTL.
        # Idle is judged on transaction-level activity, so its silence now fails ref 2.
        service = _make_service()
        tx_id = service.new_transaction(
            cell=Mock(), timeout=1000.0, num_receivers=0, receiver_ids=("r1",), receiver_idle_timeout=5.0
        )
        rid1 = service.add_object(tx_id, MockDownloadable([b"chunk"]))
        rid2 = service.add_object(tx_id, MockDownloadable([b"chunk"]))
        _pull_to_terminal(service, rid1, "r1")  # finishes ref 1 (legacy final), never touches ref 2

        run_monitor_once(service, now=time.time() + 30.0)

        outcome = service.get_transaction_outcome(tx_id)
        assert outcome is not None, "must resolve via the idle budget, not the 1000s TTL"
        assert not outcome.completed
        statuses = {r.ref_id: r.receiver_statuses for r in outcome.refs}
        assert statuses[rid1] == {"r1": DownloadStatus.SUCCESS}
        assert statuses[rid2] == {"r1": DownloadStatus.FAILED}


class TestConcurrentSameIdCreation:
    def test_owner_and_table_stay_consistent_under_concurrent_creation(self):
        # P1 pin (reproduced by review): with registration and table insertion in separate
        # critical sections, two concurrent same-id constructors could leave the live
        # transaction without outcome ownership -- and record the retired transaction's
        # DELETED outcome as if it were the live attempt.
        import threading as th

        service = _make_service()
        for _ in range(30):
            barrier = th.Barrier(2)

            def create():
                barrier.wait()
                service.new_transaction(cell=Mock(), timeout=10.0, num_receivers=1, tx_id="TX-RACE2")

            threads = [th.Thread(target=create) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(5.0)

            live = service._tx_table["TX-RACE2"]
            with service._outcome_lock:
                owner = service._outcome_owners.get("TX-RACE2")
            assert owner is live, "the live transaction must own its outcome slot"
            # the retired constructor's DELETED outcome must never be recorded for the id
            assert service.get_transaction_outcome("TX-RACE2") is None


class TestConfigResolution:
    def test_budget_config_var_supplies_default(self):
        service = _make_service()
        with patch.object(ds_module.ConfigService, "get_float_var", return_value=42.0) as gv:
            tx_id = service.new_transaction(cell=Mock(), timeout=10.0, num_receivers=1)
        tx = service._tx_table[tx_id]
        assert tx.receiver_idle_timeout == 42.0
        assert tx.receiver_acquire_timeout == 42.0
        assert gv.call_args.kwargs.get("conf") == ds_module.SystemConfigs.APPLICATION_CONF

    def test_explicit_budget_overrides_config_var(self):
        service = _make_service()
        with patch.object(ds_module.ConfigService, "get_float_var", return_value=42.0):
            tx_id = service.new_transaction(cell=Mock(), timeout=10.0, num_receivers=1, receiver_idle_timeout=7.0)
        assert service._tx_table[tx_id].receiver_idle_timeout == 7.0

    def test_non_positive_config_var_disables_budget(self):
        service = _make_service()
        with patch.object(ds_module.ConfigService, "get_float_var", return_value=0.0):
            tx_id = service.new_transaction(cell=Mock(), timeout=10.0, num_receivers=1)
        tx = service._tx_table[tx_id]
        assert tx.receiver_idle_timeout is None
        assert tx.receiver_acquire_timeout is None


class TestTransactionValidation:
    def test_receiver_ids_derive_num_receivers(self):
        service = _make_service()
        tx_id = service.new_transaction(cell=Mock(), timeout=10.0, num_receivers=0, receiver_ids=("a", "b", "a"))
        tx = service._tx_table[tx_id]
        assert tx.receiver_ids == ("a", "b")  # deduped, order kept
        assert tx.num_receivers == 2

    def test_receiver_ids_num_receivers_mismatch_raises(self):
        service = _make_service()
        with pytest.raises(ValueError, match="does not match"):
            service.new_transaction(cell=Mock(), timeout=10.0, num_receivers=3, receiver_ids=("a", "b"))

    def test_min_receivers_validation(self):
        service = _make_service()
        with pytest.raises(ValueError, match="must be positive"):
            service.new_transaction(cell=Mock(), timeout=10.0, num_receivers=2, min_receivers=0)
        with pytest.raises(ValueError, match="exceeds num_receivers"):
            service.new_transaction(cell=Mock(), timeout=10.0, num_receivers=2, min_receivers=3)

    def test_negative_budget_rejected(self):
        service = _make_service()
        with pytest.raises(ValueError, match="must > 0"):
            service.new_transaction(cell=Mock(), timeout=10.0, num_receivers=1, receiver_idle_timeout=-1.0)


class TestQuorumSurface:
    def _refs_outcome(self, statuses, min_receivers, num_receivers=2):
        from nvflare.fuel.f3.streaming.transfer_outcome import RefOutcome, TransactionDoneStatus

        refs = [RefOutcome(ref_id="R1", receiver_statuses=statuses)]
        return compute_transfer_outcome(
            "T1", TransactionDoneStatus.FINISHED, num_receivers, refs, 100.0, min_receivers=min_receivers
        )

    def test_quorum_met_with_partial_fanout(self):
        outcome = self._refs_outcome({"a": DownloadStatus.SUCCESS, "b": DownloadStatus.FAILED}, min_receivers=1)
        assert not outcome.completed  # strict certificate unchanged
        assert outcome.quorum_met

    def test_quorum_not_met(self):
        outcome = self._refs_outcome({"a": DownloadStatus.SUCCESS, "b": DownloadStatus.FAILED}, min_receivers=2)
        assert not outcome.quorum_met

    def test_quorum_falls_back_to_completed_when_unset(self):
        success = self._refs_outcome({"a": DownloadStatus.SUCCESS, "b": DownloadStatus.SUCCESS}, min_receivers=None)
        assert success.completed and success.quorum_met
        partial = self._refs_outcome({"a": DownloadStatus.SUCCESS, "b": DownloadStatus.FAILED}, min_receivers=None)
        assert not partial.quorum_met

    def test_quorum_requires_same_receiver_across_all_refs(self):
        # disjoint success subsets per ref must NOT satisfy the quorum: no single receiver
        # holds the complete (multi-ref) payload
        from nvflare.fuel.f3.streaming.transfer_outcome import RefOutcome, TransactionDoneStatus

        refs = [
            RefOutcome(ref_id="R1", receiver_statuses={"a": DownloadStatus.SUCCESS, "b": DownloadStatus.FAILED}),
            RefOutcome(ref_id="R2", receiver_statuses={"a": DownloadStatus.FAILED, "b": DownloadStatus.SUCCESS}),
        ]
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.FINISHED, 2, refs, 100.0, min_receivers=1)
        assert not outcome.quorum_met  # each ref has one success, but no common receiver

        refs2 = [
            RefOutcome(ref_id="R1", receiver_statuses={"a": DownloadStatus.SUCCESS, "b": DownloadStatus.FAILED}),
            RefOutcome(ref_id="R2", receiver_statuses={"a": DownloadStatus.SUCCESS, "b": DownloadStatus.FAILED}),
        ]
        outcome2 = compute_transfer_outcome("T1", TransactionDoneStatus.FINISHED, 2, refs2, 100.0, min_receivers=1)
        assert outcome2.quorum_met  # receiver "a" holds the complete payload

    def test_quorum_fails_closed_with_no_refs(self):
        from nvflare.fuel.f3.streaming.transfer_outcome import TransactionDoneStatus

        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.FINISHED, 2, [], 100.0, min_receivers=1)
        assert not outcome.quorum_met
