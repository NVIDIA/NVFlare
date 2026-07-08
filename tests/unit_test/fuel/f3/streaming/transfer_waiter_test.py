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

"""Tests for the awaitable transfer facade (plan: F3-4).

TransferWaiter is the "returns == delivered" primitive the upper layers (executor backends,
trainer engine) consume: wait() blocks -- event-driven, resolved inside the outcome-recording
path -- until the aggregate TransferOutcome is recorded; COMPLETED only when every expected
receiver succeeded. It composes with (never replaces) transaction_done_cb / outcome_cb.
"""

import threading
import time
from unittest.mock import Mock, patch

from nvflare.fuel.f3.streaming import download_service as ds_module
from nvflare.fuel.f3.streaming.download_service import DownloadStatus, TransferWaiter
from tests.unit_test.fuel.f3.streaming.download_test_utils import MockDownloadable, confirm_request
from tests.unit_test.fuel.f3.streaming.download_test_utils import make_confirm_test_service as _make_service
from tests.unit_test.fuel.f3.streaming.download_test_utils import pull_to_terminal as _pull_to_terminal
from tests.unit_test.fuel.f3.streaming.download_test_utils import run_monitor_once


def _new_tx(service, num_receivers=1, timeout=10.0):
    tx_id = service.new_transaction(cell=Mock(), timeout=timeout, num_receivers=num_receivers)
    rid = service.add_object(tx_id, MockDownloadable([b"chunk"]))
    return tx_id, rid


class TestTransferWaiter:
    def test_wait_blocks_until_outcome_and_returns_delivered(self):
        # the load-bearing property: a thread blocked in wait() is released by outcome
        # recording itself (event-driven, no polling), and gets the COMPLETED certificate
        service = _make_service()
        tx_id, rid = _new_tx(service)
        waiter = service.get_transfer_waiter(tx_id)
        assert not waiter.done()

        results = []
        t = threading.Thread(target=lambda: results.append(waiter.wait(timeout=10.0)))
        t.start()

        _pull_to_terminal(service, rid, "r1", confirm_capable=True)
        service._handle_download(confirm_request(rid, "r1", DownloadStatus.SUCCESS))
        run_monitor_once(service, now=time.time())

        t.join(5.0)
        assert not t.is_alive()
        assert len(results) == 1 and results[0] is not None
        assert results[0].completed
        assert results[0].tx_id == tx_id

    def test_waiter_after_termination_resolves_immediately(self):
        service = _make_service()
        tx_id, rid = _new_tx(service)
        _pull_to_terminal(service, rid, "r1")
        run_monitor_once(service, now=time.time())

        waiter = service.get_transfer_waiter(tx_id)
        assert waiter.done()
        outcome = waiter.wait(timeout=0)
        assert outcome is not None and outcome.completed

    def test_wait_timeout_returns_none_then_resolves(self):
        service = _make_service()
        tx_id, rid = _new_tx(service)
        waiter = service.get_transfer_waiter(tx_id)

        assert waiter.wait(timeout=0.05) is None
        assert waiter.outcome is None

        _pull_to_terminal(service, rid, "r1")
        run_monitor_once(service, now=time.time())
        outcome = waiter.wait(timeout=5.0)
        assert outcome is not None and outcome.completed

    def test_failed_outcome_is_returned_not_masked(self):
        service = _make_service()
        tx_id, rid = _new_tx(service)
        waiter = service.get_transfer_waiter(tx_id)

        _pull_to_terminal(service, rid, "r1", confirm_capable=True)
        service._handle_download(
            confirm_request(rid, "r1", DownloadStatus.FAILED)
        )  # receiver truth: finalization failed
        run_monitor_once(service, now=time.time())

        outcome = waiter.wait(timeout=5.0)
        assert outcome is not None
        assert not outcome.completed

    def test_linger_applies_to_finished_outcomes_only(self):
        # linger preserves the process/tombstone window so lost terminal replies can be
        # replayed: it applies to ANY FINISHED outcome (completed or receiver-failed --
        # the served-SUCCESS receivers of a partial fan-out are exactly who needs it),
        # and not to timed-out transactions (no receiver is retrying a served reply).
        service = _make_service()

        # completed FINISHED: linger applied
        tx_id, rid = _new_tx(service)
        _pull_to_terminal(service, rid, "r1")
        run_monitor_once(service, now=time.time())
        waiter = service.get_transfer_waiter(tx_id)
        with patch.object(ds_module.time, "sleep") as mock_sleep:
            outcome = waiter.wait(timeout=5.0, linger=0.2)
        assert outcome.completed
        mock_sleep.assert_called_once_with(0.2)

        # receiver-failed FINISHED: linger still applied (partial fan-out healing window)
        tx_id2, rid2 = _new_tx(service)
        _pull_to_terminal(service, rid2, "r1", confirm_capable=True)
        service._handle_download(confirm_request(rid2, "r1", DownloadStatus.FAILED))
        run_monitor_once(service, now=time.time())
        waiter2 = service.get_transfer_waiter(tx_id2)
        with patch.object(ds_module.time, "sleep") as mock_sleep:
            outcome2 = waiter2.wait(timeout=5.0, linger=0.2)
        assert not outcome2.completed
        mock_sleep.assert_called_once_with(0.2)

        # TIMEOUT termination: no linger
        tx_id3, rid3 = _new_tx(service)
        service._tx_table[tx_id3].last_active_time = time.time() - 100.0
        run_monitor_once(service, now=time.time())
        waiter3 = service.get_transfer_waiter(tx_id3)
        with patch.object(ds_module.time, "sleep") as mock_sleep:
            outcome3 = waiter3.wait(timeout=5.0, linger=5.0)
        assert outcome3 is not None and not outcome3.completed
        mock_sleep.assert_not_called()

    def test_waiter_for_unknown_tx_resolves_immediately(self):
        # invariant: waiters can never hang -- nothing will ever record an outcome for an
        # unknown (or already-forgotten) transaction id
        service = _make_service()
        waiter = service.get_transfer_waiter("no-such-tx")
        assert waiter.done()
        assert waiter.wait(timeout=0) is None
        assert "no-such-tx" not in service._tx_waiters

    def test_waiter_after_shutdown_resolves_immediately(self):
        service = _make_service()
        tx_id, _ = _new_tx(service)
        service.shutdown()

        waiter = service.get_transfer_waiter(tx_id)
        assert waiter.done()
        assert waiter.wait(timeout=0) is None
        assert not service._tx_waiters

    def test_shutdown_unblocks_waiters_with_none(self):
        service = _make_service()
        tx_id, _ = _new_tx(service)
        waiter = service.get_transfer_waiter(tx_id)

        results = []
        t = threading.Thread(target=lambda: results.append(waiter.wait(timeout=10.0)))
        t.start()
        service.shutdown()
        t.join(5.0)

        assert not t.is_alive(), "shutdown must never leave a waiter hanging"
        assert results == [None]

    def test_acquired_receivers_reflects_first_pulls(self):
        service = _make_service()
        tx_id, rid = _new_tx(service, num_receivers=2)
        assert service.get_acquired_receivers(tx_id) == set()

        _pull_to_terminal(service, rid, "r1")
        assert service.get_acquired_receivers(tx_id) == {"r1"}

    def test_budget_failure_resolves_waiter_in_bounded_time(self):
        # end-to-end with F3-3: a stalled receiver's budget failure terminates the tx and
        # releases the waiter -- the producer is never pinned to the full TTL
        service = _make_service()
        tx_id = service.new_transaction(cell=Mock(), timeout=1000.0, num_receivers=2, receiver_idle_timeout=5.0)
        rid = service.add_object(tx_id, MockDownloadable([b"chunk"]))
        waiter = service.get_transfer_waiter(tx_id)

        _pull_to_terminal(service, rid, "healthy")
        _pull_to_terminal(service, rid, "stalled", confirm_capable=True)  # confirm never arrives
        run_monitor_once(service, now=time.time() + 30.0)

        outcome = waiter.wait(timeout=5.0)
        assert outcome is not None
        assert not outcome.completed
        assert outcome.refs[0].receiver_statuses["stalled"] == DownloadStatus.FAILED

    def test_waiter_is_a_transfer_waiter(self):
        service = _make_service()
        tx_id, _ = _new_tx(service)
        waiter = service.get_transfer_waiter(tx_id)
        assert isinstance(waiter, TransferWaiter)
        assert waiter.transaction_id == tx_id
