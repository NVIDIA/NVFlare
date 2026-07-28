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

    def test_new_transaction_rejects_live_id(self):
        """tx_ids are attempt-scoped: registering an id that names a LIVE transaction
        raises immediately -- and, unlike the old retire-on-reuse design, the live
        transaction is completely undisturbed (no callbacks, no release, still servable)."""
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        done, outcome_cbs = [], []
        service.new_transaction(
            cell=cell,
            timeout=10.0,
            num_receivers=1,
            tx_id="TX-DUP",
            transaction_done_cb=lambda tid, status, base_objs, **kw: done.append(status),
            outcome_cb=lambda outcome: outcome_cbs.append(outcome),
        )
        obj = _stub_obj()
        rid = service.add_object("TX-DUP", obj)

        with pytest.raises(ValueError, match="already in use"):
            service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-DUP")

        # the duplicate attempt left no trace on the live transaction
        assert rid in service._ref_table
        assert not obj.released
        assert done == [] and outcome_cbs == []
        with service._tx_lock:
            with service._outcome_lock:
                assert service._tx_table["TX-DUP"] is service._outcome_owners["TX-DUP"]

    def test_new_transaction_rejects_settling_and_receipted_id(self):
        """The duplicate check covers the id's whole lifetime: live (above), SETTLING
        (popped from _tx_table, ownership not yet consumed), and RECEIPTED (outcome
        retained for TX_OUTCOME_TTL). Only receipt expiry frees a used id."""
        from functools import partial
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-LIFE")
        with service._tx_lock:
            old_tx = service._tx_table.pop("TX-LIFE")  # monitor termination step 1

        # mid-settlement: ownership still registered
        with pytest.raises(ValueError, match="already in use"):
            service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-LIFE")

        old_tx.transaction_done(TransactionDoneStatus.FINISHED, on_outcome=partial(service._record_outcome, tx=old_tx))

        # receipted: the verdict is retained and the id stays excluded
        assert service.get_transaction_outcome("TX-LIFE") is not None
        with pytest.raises(ValueError, match="already in use"):
            service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-LIFE")

        # receipt expiry (TTL) frees the id -- checked INLINE by new_transaction, so an
        # expired-but-unswept receipt does not stretch the exclusion window
        recorded = service.get_transaction_outcome("TX-LIFE")
        assert recorded is not None
        with service._outcome_lock:
            service._tx_outcomes["TX-LIFE"] = dataclasses.replace(
                recorded, timestamp=recorded.timestamp - service.TX_OUTCOME_TTL - 1
            )
        assert service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-LIFE") == "TX-LIFE"

    def test_drain_leak_poisons_id_until_operation_exits(self):
        """P1 pin: a drain-timeout leak (operation hung past OP_DRAIN_TIMEOUT) may resume
        and emit later -- so its tx_id must stay excluded from registration even after
        the receipt expires, until every leaked operation has exited. Otherwise the
        leaked emission could land under a recycled id."""
        from unittest.mock import Mock

        import nvflare.fuel.f3.streaming.download_service as ds_module

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-LEAK")
        with service._tx_lock:
            tx = service._tx_table["TX-LEAK"]
        assert tx.begin_op()  # an in-flight operation (serve/budget pass) that will hang

        saved = ds_module.OP_DRAIN_TIMEOUT
        ds_module.OP_DRAIN_TIMEOUT = 0.2
        try:
            service.delete_transaction("TX-LEAK")  # drain times out; settles anyway (logged)
        finally:
            ds_module.OP_DRAIN_TIMEOUT = saved

        # force-expire the receipt: the id must STILL be excluded by the leak
        recorded = service.get_transaction_outcome("TX-LEAK")
        assert recorded is not None
        with service._outcome_lock:
            service._tx_outcomes["TX-LEAK"] = dataclasses.replace(
                recorded, timestamp=recorded.timestamp - service.TX_OUTCOME_TTL - 1
            )
        with pytest.raises(ValueError, match="has not fully terminated"):
            service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-LEAK")

        # the leaked operation exits: the id becomes usable
        tx.end_op()
        assert service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-LEAK") == "TX-LEAK"

    def test_settlement_callback_can_create_transactions_but_not_its_own_id(self):
        """Settlement runs outside all service locks, so callbacks may freely call back
        in and create NEW transactions; recreating the settling transaction's own id is
        rejected like any other in-use id (its ownership is consumed only after the
        callbacks, in on_outcome)."""
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        errors, created = [], []

        def cb(outcome):
            try:
                service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-SELF")
            except ValueError as e:
                errors.append(str(e))
            created.append(service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-OTHER"))

        service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-SELF", outcome_cb=cb)
        service.delete_transaction("TX-SELF")

        assert len(errors) == 1 and "has not fully terminated" in errors[0]
        assert created == ["TX-OTHER"]
        with service._tx_lock:
            assert "TX-OTHER" in service._tx_table

    def test_recording_failure_leaves_id_unusable_and_shutdown_frees_waiters(self):
        """If outcome recording itself fails abnormally, ownership is never consumed:
        the id stays excluded (fail-safe -- fresh uuids never collide with it) and its
        waiter is released by shutdown, the terminal never-hang backstop."""
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-DEAD")
        stranded = service.get_transfer_waiter("TX-DEAD")

        def exploding_record(outcome, tx):
            raise RuntimeError("recording blew up")

        service._record_outcome = exploding_record
        try:
            service.delete_transaction("TX-DEAD")  # settles; recording fails; owner never consumed
        finally:
            del service._record_outcome  # restore the inherited classmethod
        assert not stranded.done()
        with pytest.raises(ValueError, match="already in use"):
            service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-DEAD")

        service.shutdown()
        assert stranded.done() and stranded.outcome is None

    def test_shutdown_window_keeps_leaked_id_excluded(self):
        """P1 pin: shutdown clears the ownership and receipt exclusions up front, and
        leak registration used to run only AFTER the deferred settlement -- so a
        drain-leaked id looked free for that whole window and a replacement could
        register while the leaked operation was still able to emit. Shutdown now
        pre-registers every owner as a potential leak under the locks."""
        from unittest.mock import Mock

        import nvflare.fuel.f3.streaming.download_service as ds_module

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        cb_entered = threading.Event()
        cb_release = threading.Event()

        def gated_done_cb(tid, status, base_objs, **kw):
            cb_entered.set()
            cb_release.wait(10.0)

        service.new_transaction(
            cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-SHUTLEAK", transaction_done_cb=gated_done_cb
        )
        with service._tx_lock:
            tx = service._tx_table["TX-SHUTLEAK"]
        assert tx.begin_op()  # an in-flight operation that will outlive the drain

        saved = ds_module.OP_DRAIN_TIMEOUT
        ds_module.OP_DRAIN_TIMEOUT = 0.2
        shutter = threading.Thread(target=service.shutdown)
        try:
            shutter.start()
            assert cb_entered.wait(5.0)  # drain timed out; deferred settlement mid-window

            # the leaked operation could still emit: its id must NOT look free
            with pytest.raises(ValueError, match="has not fully terminated"):
                service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-SHUTLEAK")
        finally:
            cb_release.set()
            shutter.join(5.0)
            ds_module.OP_DRAIN_TIMEOUT = saved
        assert not shutter.is_alive()

        # still excluded after shutdown while the operation is in flight
        with pytest.raises(ValueError, match="has not fully terminated"):
            service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-SHUTLEAK")

        # the operation exits: the id frees
        tx.end_op()
        assert service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-SHUTLEAK") == "TX-SHUTLEAK"

    def test_outcome_computation_failure_still_records_fail_closed(self):
        """Greptile pin: if compute_transfer_outcome itself raises, settlement used to
        skip recording entirely -- ownership stayed consumed-never, and a waiter parked
        on the id hung until shutdown. The finally now records a fail-closed fallback
        verdict (empty refs certify nothing), so waiters always resolve."""
        from unittest.mock import Mock, patch

        import nvflare.fuel.f3.streaming.download_service as ds_module

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        obj = _stub_obj()
        outcome_cbs = []
        released_when_outcome_cb_ran = []

        def capture_outcome(outcome):
            outcome_cbs.append(outcome)
            released_when_outcome_cb_ran.append(obj.released)

        service.new_transaction(
            cell=cell,
            timeout=10.0,
            num_receivers=1,
            tx_id="TX-COMPUTE",
            receiver_ids=("r1",),
            min_receivers=1,
            outcome_cb=capture_outcome,
        )
        service.add_object("TX-COMPUTE", obj)
        waiters = [service.get_transfer_waiter("TX-COMPUTE") for _ in range(3)]

        def always_exploding(*args, **kwargs):
            raise RuntimeError("outcome computation blew up")

        # PERSISTENT failure (review round 10: a one-shot failure masks a fallback
        # that reuses the same failing function)
        with patch.object(ds_module, "compute_transfer_outcome", side_effect=always_exploding):
            service.delete_transaction("TX-COMPUTE")

        outcomes = [waiter.wait(timeout=5.0) for waiter in waiters]
        outcome = outcomes[0]
        assert outcome is not None, "waiters must resolve even when outcome computation raises persistently"
        assert all(o is outcome for o in outcomes), "all parked waiters must receive the same recorded receipt"
        late_waiter = service.get_transfer_waiter("TX-COMPUTE")
        assert late_waiter.done() and late_waiter.wait(timeout=0) is outcome
        assert not outcome.completed, "the fallback verdict must be fail-closed"
        with service._outcome_lock:
            assert "TX-COMPUTE" not in service._outcome_owners, "ownership must be consumed"
        # cleanup surfaces still ran: sources freed, per-object hook fired
        assert obj.released, "source release must run even when the verdict computation fails"
        assert obj.transaction_done_calls, "obj.transaction_done must run even when the verdict computation fails"
        # codex P2/P3: the outcome_cb contract holds on the fail-closed path, with
        # metadata preserved and an honest reason
        assert len(outcome_cbs) == 1
        assert released_when_outcome_cb_ran == [False], "outcome_cb must run before source release"
        assert outcome_cbs[0].reason == TransferOutcomeReason.COMPUTATION_FAILED
        assert outcome_cbs[0].receiver_ids == ("r1",)
        assert outcome_cbs[0].min_receivers == 1
        assert outcome.reason == TransferOutcomeReason.COMPUTATION_FAILED
        assert outcome.receiver_ids == ("r1",)
        assert outcome.min_receivers == 1
        # TX-COMPUTE's own termination marker is released (same-id re-registration
        # stays blocked by the retained receipt, which is intentional)
        with service._tx_lock:
            assert "TX-COMPUTE" not in service._terminating_txs

    @pytest.mark.parametrize(
        "terminal_path,expected_done_status",
        [
            ("finished", TransactionDoneStatus.FINISHED),
            ("timeout", TransactionDoneStatus.TIMEOUT),
        ],
    )
    def test_monitor_terminal_paths_survive_persistent_outcome_computation_failure(
        self, terminal_path, expected_done_status
    ):
        """A bad verdict computation must not kill the real monitor termination paths.

        FINISHED and TIMEOUT are classified by the monitor rather than by an explicit
        delete call, so both must still release sources, record a fail-closed receipt,
        resolve waiters, and retire ownership/termination markers.
        """
        from unittest.mock import Mock, patch

        import nvflare.fuel.f3.streaming.download_service as ds_module

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        tx_id = service.new_transaction(cell=cell, timeout=10.0, num_receivers=1)
        obj = _stub_obj()
        rid = service.add_object(tx_id, obj)
        if terminal_path == "finished":
            with service._tx_lock:
                ref = service._ref_table[rid]
            ref.obj_downloaded("r1", DownloadStatus.SUCCESS)
            monitor_now = time.time()
        else:
            with service._tx_lock:
                tx = service._tx_table[tx_id]
            monitor_now = tx.last_active_time + tx.timeout + 1.0
        waiter = service.get_transfer_waiter(tx_id)

        with patch.object(ds_module, "compute_transfer_outcome", side_effect=RuntimeError("persistent failure")):
            run_monitor_once(service, now=monitor_now)

        outcome = waiter.wait(timeout=5.0)
        assert outcome is not None
        assert outcome.status == TransferProgressState.FAILED
        assert outcome.reason == TransferOutcomeReason.COMPUTATION_FAILED
        assert outcome.done_status == expected_done_status
        assert obj.released
        assert obj.transaction_done_calls == [(tx_id, expected_done_status)]
        with service._outcome_lock:
            assert tx_id not in service._outcome_owners
        with service._tx_lock:
            assert tx_id not in service._terminating_txs

    def test_computation_failure_with_drain_leak_keeps_id_excluded_until_operation_exits(self):
        """The fail-closed settlement and drain-leak protections must compose: the
        waiter resolves even though an operation missed the drain deadline, but the old
        tx_id remains excluded after receipt expiry until that operation really exits.
        """
        from unittest.mock import Mock, patch

        import nvflare.fuel.f3.streaming.download_service as ds_module

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        tx_id = "TX-COMPUTE-LEAK"
        service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id=tx_id)
        service.add_object(tx_id, _stub_obj())
        with service._tx_lock:
            tx = service._tx_table[tx_id]
        assert tx.begin_op()
        waiter = service.get_transfer_waiter(tx_id)

        saved = ds_module.OP_DRAIN_TIMEOUT
        ds_module.OP_DRAIN_TIMEOUT = 0.05
        try:
            with patch.object(ds_module, "compute_transfer_outcome", side_effect=RuntimeError("persistent failure")):
                service.delete_transaction(tx_id)
        finally:
            ds_module.OP_DRAIN_TIMEOUT = saved

        outcome = waiter.wait(timeout=5.0)
        assert outcome is not None and outcome.reason == TransferOutcomeReason.COMPUTATION_FAILED
        with service._tx_lock:
            assert service._terminating_txs.get(tx_id) is tx

        # Receipt expiry alone is insufficient while an old operation can still emit.
        with service._outcome_lock:
            service._tx_outcomes[tx_id] = dataclasses.replace(
                service._tx_outcomes[tx_id], timestamp=time.time() - service.TX_OUTCOME_TTL - 1
            )
        with pytest.raises(ValueError, match="has not fully terminated"):
            service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id=tx_id)

        tx.end_op()
        assert service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id=tx_id) == tx_id

    def test_compound_computation_and_release_failures_still_settle_all_sources(self):
        """Real cleanup failures can stack: verdict computation may fail while one
        source's release hook also raises. Neither failure may skip sibling cleanup or
        strand the waiter/ownership marker.
        """
        from unittest.mock import Mock, patch

        import nvflare.fuel.f3.streaming.download_service as ds_module

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        cell = Mock()
        tx_id = service.new_transaction(cell=cell, timeout=10.0, num_receivers=1)
        first = _stub_obj()
        second = _stub_obj()
        first.release = Mock(side_effect=RuntimeError("first release failed"))
        service.add_object(tx_id, first)
        service.add_object(tx_id, second)
        waiter = service.get_transfer_waiter(tx_id)

        with patch.object(ds_module, "compute_transfer_outcome", side_effect=RuntimeError("persistent failure")):
            service.delete_transaction(tx_id)

        outcome = waiter.wait(timeout=5.0)
        assert outcome is not None and outcome.reason == TransferOutcomeReason.COMPUTATION_FAILED
        first.release.assert_called_once_with()
        assert second.released, "one raising release must not skip sibling cleanup"
        assert first.transaction_done_calls == [(tx_id, TransactionDoneStatus.DELETED)]
        assert second.transaction_done_calls == [(tx_id, TransactionDoneStatus.DELETED)]
        with service._outcome_lock:
            assert tx_id not in service._outcome_owners
        with service._tx_lock:
            assert tx_id not in service._terminating_txs

    def test_receipt_ttl_starts_at_recording_not_at_verdict(self):
        """P1 pin: the outcome timestamp is captured before the settlement callbacks
        run -- a settlement slower than TX_OUTCOME_TTL used to record a receipt that
        was EXPIRED at birth, which the inline expiry then handed to a same-id
        constructor in the gap before the terminator's marker sync. Receipts are now
        re-stamped at recording time ("kept 30 min" from recording)."""
        from unittest.mock import Mock

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        service.TX_OUTCOME_TTL = 0.2
        cell = Mock()

        def slow_done_cb(tid, status, base_objs, **kw):
            time.sleep(0.4)  # slower than the (patched) TTL

        service.new_transaction(
            cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-AGED", transaction_done_cb=slow_done_cb
        )
        service.delete_transaction("TX-AGED")

        # the receipt is fresh from RECORDING: queryable, and it excludes the id
        assert service.get_transaction_outcome("TX-AGED") is not None
        with pytest.raises(ValueError, match="already in use"):
            service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-AGED")

    def test_aged_receipt_with_leaked_op_cannot_expose_id(self):
        """P1 pin (reviewer's exact sequence): slow settlement past the TTL WITH a
        drain-leaked operation -- the id must stay excluded end to end; before the
        marker moved to _delete_tx, a constructor could slip between ownership
        consumption and the marker sync, and the old marker then overlapped the live
        replacement."""
        from unittest.mock import Mock

        import nvflare.fuel.f3.streaming.download_service as ds_module

        service = make_isolated_download_service()
        service._tx_monitor = Mock()
        service.TX_OUTCOME_TTL = 0.2
        cell = Mock()
        service.new_transaction(
            cell=cell,
            timeout=10.0,
            num_receivers=1,
            tx_id="TX-AGEDLEAK",
            transaction_done_cb=lambda tid, status, base_objs, **kw: time.sleep(0.4),
        )
        with service._tx_lock:
            tx = service._tx_table["TX-AGEDLEAK"]
        assert tx.begin_op()  # the leak

        saved = ds_module.OP_DRAIN_TIMEOUT
        ds_module.OP_DRAIN_TIMEOUT = 0.1
        try:
            service.delete_transaction("TX-AGEDLEAK")
        finally:
            ds_module.OP_DRAIN_TIMEOUT = saved

        # settlement complete, receipt aged past the (patched) TTL -- the leaked op
        # still excludes the id, and no overlap state can form
        time.sleep(0.25)
        with pytest.raises(ValueError, match="has not fully terminated"):
            service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-AGEDLEAK")
        with service._tx_lock:
            assert service._terminating_txs.get("TX-AGEDLEAK") is tx

        tx.end_op()
        assert service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-AGEDLEAK") == "TX-AGEDLEAK"

    def test_marker_installed_at_unlink_for_every_terminator(self):
        """The termination marker is installed by _delete_tx itself, so EVERY
        terminator's id is covered for its whole settlement window -- not just
        shutdown's."""
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
            cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-UNLINK", transaction_done_cb=gated_done_cb
        )
        deleter = threading.Thread(target=service.delete_transaction, args=("TX-UNLINK",))
        try:
            deleter.start()
            assert cb_entered.wait(5.0)
            with service._tx_lock:
                assert "TX-UNLINK" in service._terminating_txs, "marker must exist mid-settlement on the delete path"
            with pytest.raises(ValueError, match="has not fully terminated"):
                service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-UNLINK")
        finally:
            cb_release.set()
            deleter.join(5.0)
        assert not deleter.is_alive()

    def test_shutdown_window_rejects_id_during_clean_settlement(self):
        """P1 pin: the shutdown pre-registration used to be releasable whenever
        _active_ops == 0 -- but settlement callbacks are not operations, so a same-id
        constructor could pop the marker MID-transaction_done and register while the
        old outcome_cb was still about to fire under the id. The marker now needs
        settlement_complete AND zero operations to release."""
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
            cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-CLEANSET", transaction_done_cb=gated_done_cb
        )
        # NOTE: no in-flight operation -- the clean-settlement case the ops-only
        # release condition missed
        shutter = threading.Thread(target=service.shutdown)
        try:
            shutter.start()
            assert cb_entered.wait(5.0)  # settlement mid-callbacks, _active_ops == 0

            with pytest.raises(ValueError, match="has not fully terminated"):
                service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-CLEANSET")
        finally:
            cb_release.set()
            shutter.join(5.0)
        assert not shutter.is_alive()

        # settlement complete, no operations: the id frees
        assert service.new_transaction(cell=cell, timeout=10.0, num_receivers=1, tx_id="TX-CLEANSET") == "TX-CLEANSET"

    def test_waiter_during_shutdown_settlement_window_resolves_immediately(self):
        """shutdown() clears ownership atomically with the table teardown, so a
        get_transfer_waiter call landing in the deferred-settlement window finds the
        id unknown and resolves None IMMEDIATELY -- it never parks, so it can never
        be stranded by the settlements still running."""
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

            waiter = service.get_transfer_waiter("TX-WIN")
            assert waiter.done() and waiter.outcome is None, "window waiter must resolve immediately"
        finally:
            cb_release.set()
            shutter.join(5.0)
        assert not shutter.is_alive()

    def test_inflight_budget_pass_drains_before_settlement(self):
        """The monitor re-checks table identity, releases _tx_lock, then runs budget
        enforcement outside it. Settlement drains such in-flight operations (activity
        gate) before snapshotting the outcome: the enforcement's verdicts are COUNTED
        in the receipt, and none of its emissions land after the transaction settled."""
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
            outcome_cb=lambda outcome: events.append("outcome_recorded"),
        )
        obj = _stub_obj()
        obj.downloaded_to_one = gated_downloaded_to_one
        service.add_object("TX-BUDGET", obj)

        monitor = threading.Thread(target=run_monitor_once, args=(service,), kwargs={"now": time.time() + 100.0})
        deleter = threading.Thread(target=service.delete_transaction, args=("TX-BUDGET",))
        try:
            monitor.start()
            assert cb_entered.wait(5.0), "budget enforcement must have finalized r1 as FAILED"

            # settlement lands while enforcement is mid-flight: it must drain first
            deleter.start()
            deleter.join(0.5)
            assert deleter.is_alive(), "settlement must not proceed while a budget pass is in flight"
        finally:
            cb_release.set()
            monitor.join(5.0)
            deleter.join(5.0)
        assert not deleter.is_alive()

        # the enforcement's emissions all precede outcome recording, and its verdict
        # was counted: r1 books as FAILED in the receipt
        recorded_at = events.index("outcome_recorded")
        assert events.index("old_downloaded_to_one") < recorded_at
        assert events.index(("old_progress", TransferProgressState.FAILED)) < recorded_at
        outcome = service.get_transaction_outcome("TX-BUDGET")
        assert outcome is not None and not outcome.completed
        assert outcome.refs[0].receiver_statuses.get("r1") == DownloadStatus.FAILED

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
