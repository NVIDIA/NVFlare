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
import dataclasses
import threading
import time

import pytest

from nvflare.fuel.f3.streaming.download_service import DownloadStatus, TransactionDoneStatus, _Transaction
from nvflare.fuel.f3.streaming.transfer_outcome import (
    RefOutcome,
    TransferOutcomeReason,
    compute_transfer_outcome,
    terminal_state_for_done_status,
)
from nvflare.fuel.f3.streaming.transfer_progress import TransferProgressState
from tests.unit_test.fuel.f3.streaming.download_test_utils import (
    MockDownloadable,
    make_isolated_download_service,
    run_monitor_once,
)


def _stub_obj():
    return MockDownloadable([b"chunk"])


class TestOutcomeImmutability:
    """The recorded outcome is shared with pollers and outcome_cb consumers: it must be deep-frozen."""

    def test_outcome_is_deep_frozen(self):
        source_statuses = {"r1": DownloadStatus.SUCCESS}
        ref = RefOutcome(ref_id="R1", receiver_statuses=source_statuses)
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.FINISHED, 1, [ref], 100.0)

        # containers are frozen, not just the dataclass attributes
        assert isinstance(outcome.refs, tuple)
        with pytest.raises(TypeError):
            outcome.refs[0].receiver_statuses["r2"] = DownloadStatus.SUCCESS
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.refs = ()
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.refs[0].receiver_statuses = {}

        # the frozen view is a private copy: mutating the source dict after
        # construction cannot rewrite the recorded per-receiver truth
        source_statuses["r1"] = DownloadStatus.FAILED
        assert outcome.refs[0].receiver_statuses == {"r1": DownloadStatus.SUCCESS}
        assert outcome.completed


class TestComputeTransferOutcome:
    """Aggregation rules: COMPLETED only when every expected receiver succeeded; receiver truth wins."""

    def _refs(self, statuses_per_ref):
        return [RefOutcome(ref_id=f"R{i}", receiver_statuses=s) for i, s in enumerate(statuses_per_ref)]

    def test_finished_all_success_is_completed(self):
        refs = self._refs([{"r1": DownloadStatus.SUCCESS, "r2": DownloadStatus.SUCCESS}])
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.FINISHED, 2, refs, 100.0)
        assert outcome.status == TransferProgressState.COMPLETED
        assert outcome.reason == TransferOutcomeReason.ALL_RECEIVERS_SUCCEEDED
        assert outcome.completed
        assert outcome.done_status == TransactionDoneStatus.FINISHED

    def test_finished_with_failed_receiver_is_failed(self):
        # the distinction FINISHED alone cannot express: receiver-count reached, one receiver FAILED
        refs = self._refs([{"r1": DownloadStatus.SUCCESS, "r2": DownloadStatus.FAILED}])
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.FINISHED, 2, refs, 100.0)
        assert outcome.status == TransferProgressState.FAILED
        assert outcome.reason == TransferOutcomeReason.RECEIVER_FAILED
        assert not outcome.completed

    def test_finished_with_missing_receiver_is_failed(self):
        refs = self._refs([{"r1": DownloadStatus.SUCCESS}])
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.FINISHED, 2, refs, 100.0)
        assert outcome.status == TransferProgressState.FAILED
        assert outcome.reason == TransferOutcomeReason.RECEIVER_FAILED

    def test_failed_receiver_on_any_ref_fails_the_transfer(self):
        refs = self._refs(
            [
                {"r1": DownloadStatus.SUCCESS},
                {"r1": DownloadStatus.FAILED},
            ]
        )
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.FINISHED, 1, refs, 100.0)
        assert outcome.status == TransferProgressState.FAILED

    def test_finished_with_no_refs_fails_closed(self):
        # a mid-assembly race (tx terminated before add_object) must not certify success
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.FINISHED, 2, [], 100.0)
        assert outcome.status == TransferProgressState.FAILED
        assert outcome.reason == TransferOutcomeReason.NO_OBJECTS

    def test_deleted_after_full_success_is_completed(self):
        # routine cleanup via delete_transaction after all receivers succeeded: receiver truth wins
        refs = self._refs([{"r1": DownloadStatus.SUCCESS}])
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.DELETED, 1, refs, 100.0)
        assert outcome.status == TransferProgressState.COMPLETED
        assert outcome.reason == TransferOutcomeReason.ALL_RECEIVERS_SUCCEEDED
        assert outcome.done_status == TransactionDoneStatus.DELETED

    def test_timeout_after_full_success_is_completed(self):
        refs = self._refs([{"r1": DownloadStatus.SUCCESS}])
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.TIMEOUT, 1, refs, 100.0)
        assert outcome.status == TransferProgressState.COMPLETED

    def test_timeout_without_full_success_is_failed(self):
        refs = self._refs([{"r1": DownloadStatus.SUCCESS}])
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.TIMEOUT, 2, refs, 100.0)
        assert outcome.status == TransferProgressState.FAILED
        assert outcome.reason == TransferOutcomeReason.TIMEOUT

    def test_deleted_without_full_success_is_aborted(self):
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.DELETED, 1, [], 100.0)
        assert outcome.status == TransferProgressState.ABORTED
        assert outcome.reason == TransferOutcomeReason.DELETED

    def test_unknown_receiver_count_cannot_complete(self):
        refs = self._refs([{"r1": DownloadStatus.SUCCESS}])
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.FINISHED, 0, refs, 100.0)
        assert outcome.status == TransferProgressState.FAILED
        assert outcome.reason == TransferOutcomeReason.UNKNOWN_RECEIVER_COUNT

    def test_unknown_done_status_fails_closed(self):
        outcome = compute_transfer_outcome("T1", "bogus", 1, [], 100.0)
        assert outcome.status == TransferProgressState.FAILED
        assert outcome.reason == TransferOutcomeReason.UNKNOWN_DONE_STATUS
        assert outcome.done_status == "bogus"

    def test_unknown_done_status_with_successful_receivers_still_fails_closed(self):
        # status validation precedes receiver truth: an unknown/future termination
        # status must not certify success even when every receiver succeeded
        refs = self._refs([{"r1": DownloadStatus.SUCCESS}])
        outcome = compute_transfer_outcome("T1", "future-status", 1, refs, 100.0)
        assert outcome.status == TransferProgressState.FAILED
        assert outcome.reason == TransferOutcomeReason.UNKNOWN_DONE_STATUS

    def test_outcome_is_frozen(self):
        outcome = compute_transfer_outcome("T1", TransactionDoneStatus.DELETED, 1, [], 100.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.status = TransferProgressState.COMPLETED

    def test_terminal_state_mapping(self):
        assert terminal_state_for_done_status(TransactionDoneStatus.FINISHED) == TransferProgressState.COMPLETED
        assert terminal_state_for_done_status(TransactionDoneStatus.TIMEOUT) == TransferProgressState.FAILED
        assert terminal_state_for_done_status(TransactionDoneStatus.DELETED) == TransferProgressState.ABORTED
        assert terminal_state_for_done_status("bogus") is None


class TestTransactionOutcome:
    """transaction_done() computes/records the outcome first and never lets callbacks break termination."""

    def test_transaction_done_returns_outcome_with_receiver_map(self):
        tx = _Transaction(timeout=10.0, num_receivers=1)
        obj = _stub_obj()
        ref = tx.add_object(obj)
        ref.obj_downloaded("r1", DownloadStatus.SUCCESS)

        outcome = tx.transaction_done(TransactionDoneStatus.FINISHED)

        assert outcome.completed
        assert outcome.tx_id == tx.tid
        assert outcome.refs[0].receiver_statuses == {"r1": DownloadStatus.SUCCESS}
        assert obj.released

    def test_raising_release_cannot_leave_waiter_unresolved(self):
        """P1 pin: an unguarded custom release() used to escape transaction_done -- with
        recording moved to the end, that left the outcome unrecorded, the waiter pending
        forever, ownership registered, and (from the monitor) killed the monitor thread.
        Releases are individually guarded and recording runs in a finally."""
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        tx_id = service.new_transaction(cell=cell, timeout=10.0, num_receivers=1)
        obj = _stub_obj()
        obj.release = Mock(side_effect=RuntimeError("release blew up"))
        service.add_object(tx_id, obj)
        ref = service._tx_table[tx_id].snapshot_refs()[0]
        ref.obj_downloaded("r1", DownloadStatus.SUCCESS)
        waiter = service.get_transfer_waiter(tx_id)

        run_monitor_once(service, now=time.time())  # must not raise off the monitor thread

        outcome = waiter.wait(timeout=5.0)
        assert outcome is not None and outcome.completed, "recording must survive a raising release()"
        with service._outcome_lock:
            assert tx_id not in service._outcome_owners, "ownership must be consumed"

    def test_retry_waits_for_mid_settlement_generation(self):
        """P1 pin: the monitor pops a finishing transaction from _tx_table BEFORE running
        its callbacks; a retry reusing the tx_id in that gap must not take ownership until
        that settlement fully completes. Generations are strictly serialized -- every
        old-generation emission happens before the retry exists -- because the alternative
        (gating emissions on a superseded flag) is an unwinnable check-then-emit race."""
        from functools import partial
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        events = []
        cb_entered = threading.Event()
        cb_release = threading.Event()

        def slow_outcome_cb(outcome):
            cb_entered.set()
            cb_release.wait(10.0)  # hold settlement open while the retry tries to register
            events.append("old_outcome_cb")

        service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-MID", outcome_cb=slow_outcome_cb)
        obj = _stub_obj()
        service.add_object("TX-MID", obj)
        with service._tx_lock:
            old_tx = service._tx_table.pop("TX-MID")  # monitor termination step 1

        settler = threading.Thread(
            target=lambda: old_tx.transaction_done(
                TransactionDoneStatus.FINISHED, on_outcome=partial(service._record_outcome, tx=old_tx)
            )
        )
        retrier = threading.Thread(
            target=lambda: (
                service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-MID"),
                events.append("retry_registered"),
            )
        )
        try:
            settler.start()
            assert cb_entered.wait(5.0)

            # the retry lands mid-settlement: it must block, not register
            retrier.start()
            retrier.join(0.5)
            assert retrier.is_alive(), "retry must not take ownership while the old generation is settling"
        finally:
            cb_release.set()
            settler.join(5.0)
            retrier.join(5.0)
        assert not retrier.is_alive()

        # strict serialization: every old-generation emission precedes the retry
        assert events == ["old_outcome_cb", "retry_registered"]
        assert obj.released
        # the old generation's recorded outcome was purged at retry registration
        assert service.get_transaction_outcome("TX-MID") is None
        with service._tx_lock:
            with service._outcome_lock:
                assert service._tx_table["TX-MID"] is service._outcome_owners["TX-MID"]

    def test_on_outcome_fires_after_callbacks_and_release(self):
        # settle-then-record: recording (which releases waiters) happens only after the
        # callback chain and source release complete, so an upper layer acting on
        # waiter.wait() can never preempt them (e.g. by stopping the producer process)
        order = []
        tx = _Transaction(
            timeout=10.0,
            num_receivers=1,
            transaction_done_cb=lambda *a, **kw: order.append("done_cb"),
            outcome_cb=lambda outcome: order.append("outcome_cb"),
        )
        obj = _stub_obj()
        obj.release = lambda: order.append("release")  # observe source release order
        ref = tx.add_object(obj)
        ref.obj_downloaded("r1", DownloadStatus.FAILED)

        tx.transaction_done(TransactionDoneStatus.FINISHED, on_outcome=lambda outcome: order.append("recorded"))

        assert order == ["done_cb", "outcome_cb", "release", "recorded"]

    def test_raising_callbacks_do_not_break_recording_or_release(self):
        # a raising transaction_done_cb must not skip outcome recording, source
        # release, or (in production) kill the monitor thread
        recorded = []

        def bad_done_cb(*args, **kwargs):
            raise RuntimeError("done_cb boom")

        def bad_outcome_cb(outcome):
            raise RuntimeError("outcome_cb boom")

        tx = _Transaction(timeout=10.0, num_receivers=1, transaction_done_cb=bad_done_cb, outcome_cb=bad_outcome_cb)
        obj = _stub_obj()
        ref = tx.add_object(obj)
        ref.obj_downloaded("r1", DownloadStatus.SUCCESS)

        outcome = tx.transaction_done(TransactionDoneStatus.FINISHED, on_outcome=recorded.append)

        assert outcome.completed
        assert recorded and recorded[0] is outcome
        assert obj.released

    def test_transaction_done_cb_contract_unchanged(self):
        received = {}

        def done_cb(tx_id, status, base_objs, **cb_kwargs):
            received.update(tx_id=tx_id, status=status, base_objs=base_objs, kwargs=cb_kwargs)

        tx = _Transaction(timeout=10.0, num_receivers=1, transaction_done_cb=done_cb, cb_kwargs={"k": "v"})
        obj = _stub_obj()
        ref = tx.add_object(obj)
        ref.obj_downloaded("r1", DownloadStatus.FAILED)

        tx.transaction_done(TransactionDoneStatus.FINISHED)

        # a receiver failure does not change what transaction_done_cb sees
        assert received["tx_id"] == tx.tid
        assert received["status"] == TransactionDoneStatus.FINISHED
        assert received["base_objs"] == [[b"chunk"]]
        assert received["kwargs"] == {"k": "v"}


class TestServiceOutcomeTable:
    """DownloadService records and serves outcomes for terminated transactions."""

    def _add_tx(self, service, num_receivers=1, timeout=10.0):
        tx = _Transaction(timeout=timeout, num_receivers=num_receivers)
        with service._tx_lock:
            service._tx_table[tx.tid] = tx
        # mirror new_transaction(): a live tx owns its outcome slot.
        # _record_outcome() records only for the owning tx, so a tx placed directly in
        # _tx_table without ownership would never record its terminal outcome.
        with service._outcome_lock:
            service._outcome_owners[tx.tid] = tx
        obj = _stub_obj()
        rid = service.add_object(tx.tid, obj)
        with service._tx_lock:
            ref = service._ref_table[rid]
        return tx, ref, obj

    def test_monitor_records_completed_outcome(self):
        service = make_isolated_download_service()
        tx, ref, obj = self._add_tx(service, num_receivers=1)
        ref.obj_downloaded("r1", DownloadStatus.SUCCESS)

        run_monitor_once(service, now=time.time())

        outcome = service.get_transaction_outcome(tx.tid)
        assert outcome is not None
        assert outcome.completed
        assert obj.released

    def test_monitor_records_failed_outcome_on_receiver_failure(self):
        service = make_isolated_download_service()
        tx, ref, obj = self._add_tx(service, num_receivers=1)
        ref.obj_downloaded("r1", DownloadStatus.FAILED)

        run_monitor_once(service, now=time.time())

        outcome = service.get_transaction_outcome(tx.tid)
        assert outcome.status == TransferProgressState.FAILED
        assert outcome.reason == TransferOutcomeReason.RECEIVER_FAILED

    def test_monitor_records_timeout_outcome(self):
        service = make_isolated_download_service()
        tx, ref, obj = self._add_tx(service, num_receivers=2, timeout=10.0)
        now = time.time()
        tx.last_active_time = now - 11.0

        run_monitor_once(service, now=now)

        outcome = service.get_transaction_outcome(tx.tid)
        assert outcome.status == TransferProgressState.FAILED
        assert outcome.reason == TransferOutcomeReason.TIMEOUT

    def test_delete_before_success_records_aborted(self):
        service = make_isolated_download_service()
        tx, ref, obj = self._add_tx(service, num_receivers=1)

        service.delete_transaction(tx.tid)

        outcome = service.get_transaction_outcome(tx.tid)
        assert outcome.status == TransferProgressState.ABORTED
        assert outcome.reason == TransferOutcomeReason.DELETED

    def test_delete_after_success_records_completed(self):
        # the routine producer pattern: all receivers succeeded, then cleanup via
        # delete_transaction before the monitor tick — must not read as aborted
        service = make_isolated_download_service()
        tx, ref, obj = self._add_tx(service, num_receivers=1)
        ref.obj_downloaded("r1", DownloadStatus.SUCCESS)

        service.delete_transaction(tx.tid)

        outcome = service.get_transaction_outcome(tx.tid)
        assert outcome.completed
        assert outcome.done_status == TransactionDoneStatus.DELETED

    def test_reused_tx_id_does_not_surface_stale_outcome(self):
        # retry with the same explicit tx_id (design: tx_id = transfer_id, retries reuse it)
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()  # suppress real monitor thread start
        cell = Mock()
        try:
            first = service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-REUSE")
            service.delete_transaction(first)
            assert service.get_transaction_outcome("TX-REUSE") is not None

            second = service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-REUSE")
            assert second == "TX-REUSE"
            # the stale terminal outcome of the previous owner is purged
            assert service.get_transaction_outcome("TX-REUSE") is None
        finally:
            service.shutdown()

    def test_reusing_tx_id_from_own_settlement_callback_raises(self):
        # a settlement callback that synchronously retries its own tx_id would deadlock
        # (the retry must wait for a settlement that cannot complete until the callback
        # returns), so it gets an immediate error instead
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()  # suppress real monitor thread start
        cell = Mock()
        errors = []

        def retry_inline(outcome):
            try:
                service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-REENT")
            except RuntimeError as e:
                errors.append(str(e))

        service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-REENT", outcome_cb=retry_inline)
        service.delete_transaction("TX-REENT")
        assert len(errors) == 1 and "settlement callback" in errors[0]

    def test_dead_settled_owner_is_reclaimed(self):
        """Falsification pin (P-S3): if outcome recording itself fails, the generation is
        settled (event set) but its ownership was never consumed. A retry must reclaim
        that dead slot -- before the fix it hot-spun forever, because Event.wait returns
        True immediately for a set event so the bounded-wait escape never fired."""
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-DEAD")
        stranded_waiter = service.get_transfer_waiter("TX-DEAD")

        def exploding_record(outcome, tx):
            raise RuntimeError("recording blew up")

        service._record_outcome = exploding_record
        try:
            service.delete_transaction("TX-DEAD")  # settles; recording fails; owner never consumed
        finally:
            del service._record_outcome  # restore the inherited classmethod
        with service._outcome_lock:
            assert "TX-DEAD" in service._outcome_owners, "precondition: dead owner still registered"
        assert not stranded_waiter.done(), "precondition: the dead generation's waiter is stranded"

        registered = []
        retrier = threading.Thread(
            target=lambda: registered.append(
                service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-DEAD")
            )
        )
        retrier.start()
        retrier.join(5.0)
        assert not retrier.is_alive(), "retry must reclaim a settled-but-unconsumed owner, not spin"
        assert registered == ["TX-DEAD"]
        with service._tx_lock:
            with service._outcome_lock:
                assert service._tx_table["TX-DEAD"] is service._outcome_owners["TX-DEAD"]
        # the dead generation's waiters are abandoned with None at reclaim (its verdict
        # was lost) -- NOT inherited and later resolved with the retry's outcome
        assert stranded_waiter.done() and stranded_waiter.outcome is None

    def test_shutdown_settlement_serializes_same_id_retry(self):
        """Falsification pin (P-S4): shutdown used to clear ownership markers BEFORE its
        deferred settlements ran, so a same-id retry landing in that window found the id
        free and returned while the old generation's emissions were still to come.
        Ownership markers now survive until each settlement consumes them."""
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        events = []
        cb_entered = threading.Event()
        cb_release = threading.Event()

        def gated_done_cb(tid, status, base_objs, **kw):
            cb_entered.set()
            cb_release.wait(10.0)
            events.append("old_done_cb")

        service.new_transaction(
            cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-SHUT", transaction_done_cb=gated_done_cb
        )
        shutter = threading.Thread(target=service.shutdown)
        retrier = threading.Thread(
            target=lambda: (
                service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-SHUT"),
                events.append("retry_registered"),
            )
        )
        try:
            shutter.start()
            assert cb_entered.wait(5.0)  # shutdown is now mid-settlement, outside its locks

            retrier.start()
            retrier.join(0.5)
            assert retrier.is_alive(), "retry must serialize behind shutdown's in-flight settlement"
        finally:
            cb_release.set()
            shutter.join(5.0)
            retrier.join(5.0)
        assert not retrier.is_alive()
        assert events == ["old_done_cb", "retry_registered"]
        # shutdown still drops the old generation's verdict (nothing records after shutdown)
        # while the post-shutdown registration owns the slot cleanly
        with service._tx_lock:
            with service._outcome_lock:
                assert service._tx_table["TX-SHUT"] is service._outcome_owners["TX-SHUT"]

    def test_hung_retirement_of_live_transaction_fails_bounded(self):
        """P1 pin: retiring a LIVE prior transaction used to settle it inline on the
        caller's thread, so a hung callback in the old generation hung the retry forever
        -- the settlement bound only covered generations already settling elsewhere.
        Retirement now settles on its own thread and the retry takes the same bounded
        wait as any mid-settlement predecessor."""
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        service.SETTLEMENT_WAIT_TIMEOUT = 0.3
        cell = Mock()
        cb_release = threading.Event()

        def hung_done_cb(tid, status, base_objs, **kw):
            cb_release.wait(10.0)

        service.new_transaction(
            cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-HUNG", transaction_done_cb=hung_done_cb
        )
        errors = []

        def retry():
            try:
                service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-HUNG")
            except RuntimeError as e:
                errors.append(str(e))

        retrier = threading.Thread(target=retry)
        try:
            retrier.start()
            retrier.join(5.0)
            assert not retrier.is_alive(), "retry must fail after the bound, not hang on the hung retirement"
            assert len(errors) == 1 and "did not settle" in errors[0]
        finally:
            cb_release.set()
        # once the abandoned settlement eventually completes, the id is usable again
        for _ in range(50):
            with service._outcome_lock:
                consumed = "TX-HUNG" not in service._outcome_owners
            if consumed:
                break
            time.sleep(0.1)
        assert service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-HUNG") == "TX-HUNG"

    def test_inflight_budget_pass_drains_before_settlement(self):
        """P1 pin: the monitor re-checks table identity, releases _tx_lock, then runs
        budget enforcement -- a retry could retire and settle the transaction while
        enforcement was mid-flight, and enforcement then emitted FAILED terminal
        progress under the reused tx_id. Operations now register with the activity
        gate and settlement drains them before emitting or registering the retry."""
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        events = []
        cb_entered = threading.Event()
        cb_release = threading.Event()

        def gated_downloaded_to_one(receiver, status):
            cb_entered.set()
            cb_release.wait(10.0)
            events.append("old_downloaded_to_one")

        service.new_transaction(
            cell=cell,
            timeout=1000.0,
            num_receivers=1,
            tx_id="TX-BUDGET",
            receiver_ids=("r1",),
            receiver_acquire_timeout=5.0,
            progress_cb=lambda **kw: events.append(("old_progress", kw.get("state"))),
        )
        obj = _stub_obj()
        obj.downloaded_to_one = gated_downloaded_to_one
        service.add_object("TX-BUDGET", obj)

        monitor = threading.Thread(target=run_monitor_once, args=(service,), kwargs={"now": time.time() + 100.0})
        retrier = threading.Thread(
            target=lambda: (
                service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-BUDGET"),
                events.append("retry_registered"),
            )
        )
        try:
            monitor.start()
            assert cb_entered.wait(5.0), "budget enforcement must have finalized r1 as FAILED"

            # the retry lands while enforcement is mid-flight: it must drain first
            retrier.start()
            retrier.join(0.5)
            assert retrier.is_alive(), "retry must not register while a budget pass is in flight"
        finally:
            cb_release.set()
            monitor.join(5.0)
            retrier.join(5.0)
        assert not retrier.is_alive()

        # every old-generation emission -- including enforcement's FAILED terminal
        # progress -- happened strictly before the retry registered
        retry_at = events.index("retry_registered")
        old_emissions = [i for i, e in enumerate(events) if e != "retry_registered"]
        assert old_emissions and all(i < retry_at for i in old_emissions), f"stale emission after retry: {events}"
        assert ("old_progress", TransferProgressState.FAILED) in events

    def test_waiter_parked_during_shutdown_settlement_window_resolves(self):
        """P1 pin: shutdown resolves the waiters it sees and keeps ownership markers for
        serialization -- but a get_transfer_waiter call in the settlement window (after
        shutdown's locked teardown, before the deferred settlement consumes ownership)
        found the marker, parked, and hung forever: the record-forbidden branch consumed
        ownership without draining waiters. Every ownership-consuming path drains now."""
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        cb_entered = threading.Event()
        cb_release = threading.Event()

        def gated_done_cb(tid, status, base_objs, **kw):
            cb_entered.set()
            cb_release.wait(10.0)

        service.new_transaction(
            cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-WIN", transaction_done_cb=gated_done_cb
        )
        shutter = threading.Thread(target=service.shutdown)
        try:
            shutter.start()
            assert cb_entered.wait(5.0)  # locked teardown done; deferred settlement in flight

            # the ownership marker is still visible, so this waiter parks instead of
            # resolving immediately -- it must be drained when settlement consumes
            # ownership, not stranded
            waiter = service.get_transfer_waiter("TX-WIN")
            assert not waiter.done(), "precondition: the waiter parked inside the window"
        finally:
            cb_release.set()
            shutter.join(5.0)
        outcome = waiter.wait(timeout=5.0)
        assert waiter.done(), "window waiter must never hang past shutdown settlement"
        assert outcome is None, "shutdown drops verdicts: the window waiter resolves to None"

    def test_retry_gives_up_if_previous_generation_never_settles(self):
        # ownership was taken but settlement never runs (a hung terminator): a retry
        # must fail loudly after the bounded wait, never register ambiguously
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()  # suppress real monitor thread start
        cell = Mock()
        service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-STUCK")
        with service._tx_lock:
            service._tx_table.pop("TX-STUCK")  # popped for termination; settlement never follows

        service.SETTLEMENT_WAIT_TIMEOUT = 0.2
        with pytest.raises(RuntimeError, match="did not settle"):
            service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-STUCK")
        # and it did not register: the stuck generation still owns the slot
        with service._tx_lock:
            assert "TX-STUCK" not in service._tx_table

    def test_reusing_active_tx_id_retires_prior_transaction(self):
        # reusing a tx_id while the prior transaction is still LIVE must retire it,
        # not orphan it: a plain _tx_table overwrite would leave the old refs servable
        # in _ref_table forever (the monitor only sees _tx_table) and never release
        # the old sources via transaction_done
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()  # suppress real monitor thread start
        cell = Mock()
        done = []
        try:
            outcome_cbs = []
            first = service.new_transaction(
                cell=cell,
                timeout=10.0,
                num_receivers=1,
                tx_id="TX-DUP",
                transaction_done_cb=lambda tid, status, base_objs, **kw: done.append((tid, status)),
                outcome_cb=lambda outcome: outcome_cbs.append(outcome),
            )
            obj = _stub_obj()
            rid = service.add_object(first, obj)
            assert rid in service._ref_table
            gen1_waiter = service.get_transfer_waiter(first)

            second = service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-DUP")
            assert second == "TX-DUP"

            # waiters bind to the generation that records while they wait: the retired
            # generation resolved this waiter with its own (non-completed) verdict
            # before the retry registered
            gen1_outcome = gen1_waiter.wait(timeout=1.0)
            assert gen1_outcome is not None and not gen1_outcome.completed

            # the prior live transaction was retired and fully settled BEFORE the retry
            # registered: refs unservable, sources released, and ALL its terminal
            # emissions -- cleanup and outcome alike -- ran while the tx_id still
            # unambiguously named it (generations are strictly serialized)
            assert rid not in service._ref_table
            assert obj.released
            assert done == [("TX-DUP", TransactionDoneStatus.DELETED)]
            assert len(outcome_cbs) == 1 and not outcome_cbs[0].completed
            # its terminal outcome does not shadow the live retry
            assert service.get_transaction_outcome("TX-DUP") is None
            # the live tx is the new owner
            assert service._tx_table["TX-DUP"] is service._outcome_owners["TX-DUP"]
            assert service._tx_table["TX-DUP"] is not None
        finally:
            service.shutdown()

    def test_unknown_transaction_has_no_outcome(self):
        service = make_isolated_download_service()
        assert service.get_transaction_outcome("no-such-tx") is None

    def test_outcome_expires_after_ttl(self):
        service = make_isolated_download_service()
        tx, ref, obj = self._add_tx(service, num_receivers=1)
        ref.obj_downloaded("r1", DownloadStatus.SUCCESS)
        now = time.time()
        run_monitor_once(service, now=now)
        assert service.get_transaction_outcome(tx.tid) is not None

        # a later monitor pass past the TTL removes the record
        run_monitor_once(service, now=now + service.TX_OUTCOME_TTL + 1.0)
        with service._outcome_lock:
            assert tx.tid not in service._tx_outcomes

    def test_get_transaction_outcome_expires_lazily(self):
        service = make_isolated_download_service()
        tx, ref, obj = self._add_tx(service, num_receivers=1)
        ref.obj_downloaded("r1", DownloadStatus.SUCCESS)
        run_monitor_once(service, now=time.time() - service.TX_OUTCOME_TTL - 1.0)

        assert service.get_transaction_outcome(tx.tid) is None

    def test_shutdown_clears_outcomes_and_stops_recording(self):
        service = make_isolated_download_service()
        tx, ref, obj = self._add_tx(service, num_receivers=1)
        service.delete_transaction(tx.tid)
        assert service.get_transaction_outcome(tx.tid) is not None

        service.shutdown()

        assert service.get_transaction_outcome(tx.tid) is None
        # a monitor iteration that was mid-termination during shutdown cannot repopulate:
        # shutdown cleared outcome ownership, so the late recorder's tx no longer owns
        # the slot and its outcome drops
        from unittest.mock import Mock

        late = compute_transfer_outcome("T-LATE", TransactionDoneStatus.FINISHED, 1, [], time.time())
        service._record_outcome(late, tx=Mock())
        assert service.get_transaction_outcome("T-LATE") is None
