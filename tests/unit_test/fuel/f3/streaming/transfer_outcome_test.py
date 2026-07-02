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

    def test_on_outcome_fires_before_user_callbacks(self):
        order = []
        tx = _Transaction(
            timeout=10.0,
            num_receivers=1,
            transaction_done_cb=lambda *a, **kw: order.append("done_cb"),
            outcome_cb=lambda outcome: order.append("outcome_cb"),
        )
        obj = _stub_obj()
        ref = tx.add_object(obj)
        ref.obj_downloaded("r1", DownloadStatus.FAILED)

        tx.transaction_done(TransactionDoneStatus.FINISHED, on_outcome=lambda outcome: order.append("recorded"))

        assert order == ["recorded", "done_cb", "outcome_cb"]

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
            # the stale terminal outcome of the previous incarnation is purged
            assert service.get_transaction_outcome("TX-REUSE") is None
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
        # a monitor iteration that was mid-termination during shutdown cannot repopulate
        late = compute_transfer_outcome("T-LATE", TransactionDoneStatus.FINISHED, 1, [], time.time())
        service._record_outcome(late)
        assert service.get_transaction_outcome("T-LATE") is None
