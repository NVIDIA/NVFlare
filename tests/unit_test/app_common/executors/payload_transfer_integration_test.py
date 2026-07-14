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

"""Integration tests: the Client API payload seam against the REAL #4865 payload layer.

payload_transfer_test.py covers the seam with fakes in both regimes; this module drives
the seam through the ACTUAL ObjectDownloader/DownloadService/TransferWaiter code — the
producer side at the wire level (the trainer's pulls and receiver-truth confirmations
delivered to the service handler, the way the #4865 suite's own tests do), and the
consumer side through the real ``download_object`` loop over a loopback cell.

Every test here is skipped until the #4865 surface is present (payload_layer_available()),
so this file is green both before and after the merge: pre-merge it documents exactly what
activates; post-merge it proves the seam's contract claims — "returns == delivered",
receiver truth failing an attempt, waiter None-arm folding, and attempt termination —
against the shipped layer instead of fakes.
"""

import time
from unittest.mock import Mock

import pytest

from nvflare.apis.shareable import Shareable
from nvflare.app_common.executors.client_api import payload_transfer as pt
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey

pytestmark = pytest.mark.skipif(
    not pt.payload_layer_available(),
    reason="requires the #4865 F3 payload layer (TransferWaiter, receiver identity/budgets)",
)

TRAINER_FQCN = "site-1.job-1.client_api_trainer_1"
CJ_FQCN = "site-1.job-1"


@pytest.fixture
def harness(monkeypatch):
    """An isolated real DownloadService, with the seam and the real ObjectDownloader
    routed to it, receiver-confirm pinned ON (as the #4865 suite's conftest does)."""
    from nvflare.fuel.f3.streaming import download_service as ds_module
    from nvflare.fuel.f3.streaming import obj_downloader as od_module
    from tests.unit_test.fuel.f3.streaming import download_test_utils as utils

    # isolated real service with the 5s monitor suppressed so we drive monitor passes
    # explicitly via run_monitor_once (renamed from make_confirm_test_service pre-merge)
    service = utils.make_service_no_monitor()
    monkeypatch.setattr(od_module, "DownloadService", service)
    monkeypatch.setattr(pt, "DownloadService", service)
    monkeypatch.setattr(ds_module, "_receiver_confirm_cached", True)
    return service, utils


def _task_shareable() -> Shareable:
    task = Shareable()
    task["weights"] = [1.0, 2.0, 3.0]
    return task


def _pull_one_shot(service, utils, ref_id, requester):
    """Drives the trainer side of one ShareableDownloadable transfer at the wire level.

    Returns (data, terminal_reply): the payload served in the single OK round, and the
    EOF reply carrying the confirm nonce.
    """
    from nvflare.fuel.f3.streaming.download_service import ProduceRC, _PropKey

    first = service._handle_download(utils.pull_request(ref_id, requester, confirm_capable=True))
    assert first.payload.get(_PropKey.STATUS) == ProduceRC.OK
    data = first.payload.get(_PropKey.DATA)
    state = first.payload.get(_PropKey.STATE)

    terminal = service._handle_download(utils.pull_request(ref_id, requester, confirm_capable=True, state=state))
    assert terminal.payload.get(_PropKey.STATUS) == ProduceRC.EOF
    return data, terminal


class TestTaskAttemptAgainstRealLayer:
    def test_returns_equals_delivered_happy_path(self, harness):
        """The seam's core promise: attempt.wait() is True only after the declared
        receiver pulled the payload AND certified storage (receiver-confirmed)."""
        from nvflare.fuel.f3.streaming.download_service import DownloadStatus

        service, utils = harness
        task = _task_shareable()
        attempt = pt.TaskPayloadAttempt(Mock(), task, TRAINER_FQCN)

        # nothing delivered yet: the waiter must not certify (None arm folds to False)
        assert attempt.wait(timeout=0.05) is False
        assert not attempt.completed() and not attempt.failed()

        data, terminal = _pull_one_shot(service, utils, attempt.ref_id, TRAINER_FQCN)
        assert data == task, "the one-shot Downloadable must serve the exact task payload"

        # served is NOT delivered: only the receiver's own confirmation certifies
        service._handle_download(
            utils.confirm_request(attempt.ref_id, TRAINER_FQCN, DownloadStatus.SUCCESS, utils.serve_nonce(terminal))
        )
        utils.run_monitor_once(service, now=time.time())

        assert attempt.wait(timeout=5.0) is True
        assert attempt.completed() and not attempt.failed()
        assert attempt.failure_reason() is None

    def test_receiver_truth_fails_the_attempt(self, harness):
        """A receiver that pulled everything but failed finalization (its own truth)
        must fail the attempt — the pre-#4865 served-EOF success would have hidden this."""
        from nvflare.fuel.f3.streaming.download_service import DownloadStatus

        service, utils = harness
        attempt = pt.TaskPayloadAttempt(Mock(), _task_shareable(), TRAINER_FQCN)

        _, terminal = _pull_one_shot(service, utils, attempt.ref_id, TRAINER_FQCN)
        service._handle_download(
            utils.confirm_request(attempt.ref_id, TRAINER_FQCN, DownloadStatus.FAILED, utils.serve_nonce(terminal))
        )
        utils.run_monitor_once(service, now=time.time())

        assert attempt.wait(timeout=5.0) is False
        assert attempt.failed() and not attempt.completed()
        assert "task payload transfer failed" in attempt.failure_reason()

    def test_terminate_resolves_the_waiter_as_not_delivered(self, harness):
        """Terminating a live attempt (retire/teardown path) settles it promptly:
        no receiver certified, so the verdict is not-delivered — and termination of an
        already-settled attempt stays a no-op."""
        service, utils = harness
        attempt = pt.TaskPayloadAttempt(Mock(), _task_shareable(), TRAINER_FQCN)

        attempt.terminate()
        utils.run_monitor_once(service, now=time.time())

        assert attempt.wait(timeout=5.0) is False
        assert attempt.failed()
        attempt.terminate()  # idempotent

    def test_attempts_get_fresh_attempt_scoped_tx_ids(self, harness):
        service, utils = harness
        first = pt.TaskPayloadAttempt(Mock(), _task_shareable(), TRAINER_FQCN)
        second = pt.TaskPayloadAttempt(Mock(), _task_shareable(), TRAINER_FQCN)
        assert first.tx_id != second.tx_id
        assert first.ref_id != second.ref_id


class _LoopbackCell:
    """Delivers consumer-side messages straight to the producer's service handler,
    stamping the ORIGIN header the way the cellnet transport would. download_object
    pulls over send_request and sends its receiver-truth confirmation over
    fire_and_forget — both land on the same service handler."""

    def __init__(self, service, my_fqcn):
        self.service = service
        self.my_fqcn = my_fqcn

    def send_request(self, channel, topic, target, request, timeout=None, **kwargs):
        request.set_header(MessageHeaderKey.ORIGIN, self.my_fqcn)
        return self.service._handle_download(request)

    def fire_and_forget(self, channel, topic, targets, message, **kwargs):
        message.set_header(MessageHeaderKey.ORIGIN, self.my_fqcn)
        self.service._handle_download(message)


class TestResultPullAgainstRealLayer:
    def test_fetch_result_payload_end_to_end_certifies_the_producer(self, harness):
        """The result-up direction: the trainer-side producer registers the result, the
        CJ pulls it through the REAL download_object loop, and — because download_object
        confirms receiver truth itself — the producer's waiter certifies delivery the
        moment fetch_result_payload returns (the trainer's flare.send() unblock signal)."""
        service, utils = harness
        result = Shareable()
        result["trained"] = [9.0]

        # trainer side: one result attempt with the CJ as the declared receiver
        tx_id = service.new_transaction(cell=Mock(), timeout=10.0, num_receivers=1, receiver_ids=(CJ_FQCN,))
        ref_id = service.add_object(tx_id, pt.ShareableDownloadable(result))

        cj_cell = _LoopbackCell(service, CJ_FQCN)
        objs = pt.fetch_result_payload(cj_cell, TRAINER_FQCN, [ref_id])
        assert objs == [result]

        # the pull's own confirmation is the certificate: the producer settles COMPLETED
        utils.run_monitor_once(service, now=time.time())
        waiter = service.get_transfer_waiter(tx_id)
        outcome = waiter.wait(timeout=5.0)
        assert outcome is not None and outcome.completed

    def test_fetch_failure_is_confirmed_to_the_producer_as_failed(self, harness):
        """A consumer-side pull failure must raise AND reach the producer as receiver
        truth (FAILED), so the trainer's send() fails instead of resolving delivered."""
        service, utils = harness

        class _FailingDownloadable(pt.ShareableDownloadable):
            def produce(self, state, requester):
                from nvflare.fuel.f3.streaming.download_service import ProduceRC

                return ProduceRC.ERROR, None, {}

        tx_id = service.new_transaction(cell=Mock(), timeout=10.0, num_receivers=1, receiver_ids=(CJ_FQCN,))
        ref_id = service.add_object(tx_id, _FailingDownloadable(Shareable()))

        cj_cell = _LoopbackCell(service, CJ_FQCN)
        with pytest.raises(pt.PayloadTransferError, match=ref_id):
            pt.fetch_result_payload(cj_cell, TRAINER_FQCN, [ref_id])

        utils.run_monitor_once(service, now=time.time())
        outcome = service.get_transfer_waiter(tx_id).wait(timeout=5.0)
        assert outcome is not None and not outcome.completed
