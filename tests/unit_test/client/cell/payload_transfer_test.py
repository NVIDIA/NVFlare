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

"""Tests for the Client API payload seam (nvflare/client/cell/payload_transfer.py).

The F3 layer itself (ObjectDownloader / DownloadService / download_object) is faked at the
seam's imports; payload_transfer_integration_test.py drives the seam against the real
layer. What is verified here is the seam's own policy and adapters:

- attempts declare the receiver identity and the acquire budget, set the TTL backstop, and
  deliberately set NO per-receiver idle budget (the one-shot round's healthy quiet period
  spans the whole inner via-downloader tensor transfer — see the module docstring);
- the terminal verdict is read through the transaction's TransferWaiter (non-blocking
  completed()/failed() polls, event-driven wait() with the None arm folded into
  not-delivered);
- terminate() is idempotent and never raises; a failing add_object terminates the
  already-registered transaction instead of leaking it.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.client.cell import payload_transfer as pt
from nvflare.fuel.f3.streaming.download_service import ProduceRC


class FakeWaiter:
    """Fakes TransferWaiter: done()/outcome for the poll accessors, wait() for the verdict."""

    def __init__(self):
        self._event = threading.Event()
        self._outcome = None
        self.wait_calls = []
        self.wait_result = None  # what wait() returns when the event is not pre-resolved

    def resolve(self, outcome):
        self._outcome = outcome
        self._event.set()

    @property
    def outcome(self):
        return self._outcome

    def done(self):
        return self._event.is_set()

    def wait(self, timeout=None, linger=None):
        self.wait_calls.append((timeout, linger))
        if self._event.is_set():
            return self._outcome
        return self.wait_result


class FakeDownloader:
    """Fakes the ObjectDownloader surface the seam consumes."""

    instances = []
    add_object_error = None

    def __init__(self, **kwargs):
        FakeDownloader.instances.append(self)
        self.kwargs = kwargs
        self.tx_id = f"tx-{len(FakeDownloader.instances)}"
        self.added = []
        self.waiter = FakeWaiter()

    def add_object(self, obj, ref_id=None):
        if FakeDownloader.add_object_error is not None:
            raise FakeDownloader.add_object_error
        self.added.append(obj)
        return f"{self.tx_id}-ref-{len(self.added)}"

    def get_waiter(self):
        return self.waiter


class FakeDownloadService:
    deleted = []

    @classmethod
    def delete_transaction(cls, transaction_id):
        cls.deleted.append(transaction_id)


def _outcome(completed: bool, reason=None):
    return SimpleNamespace(completed=completed, reason=reason)


@pytest.fixture
def fake_layer(monkeypatch):
    """Fakes the F3 layer at the seam's own imports."""
    FakeDownloader.instances = []
    FakeDownloader.add_object_error = None
    FakeDownloadService.deleted = []
    monkeypatch.setattr(pt, "ObjectDownloader", FakeDownloader)
    monkeypatch.setattr(pt, "DownloadService", FakeDownloadService)
    return FakeDownloadService


class TestTaskPayloadAttempt:
    def test_attempt_declares_receiver_identity_and_budgets(self, fake_layer):
        cell = MagicMock()
        attempt = pt.TaskPayloadAttempt(cell, {"w": 1}, "site-1.job-1.trainer_1")

        downloader = FakeDownloader.instances[0]
        assert downloader.kwargs["cell"] is cell
        # declared identity enables the acquire budget and identity-checked completion
        assert downloader.kwargs["receiver_ids"] == ("site-1.job-1.trainer_1",)
        assert downloader.kwargs["num_receivers"] == 1
        assert downloader.kwargs["timeout"] == pt.TRANSFER_TTL
        assert downloader.kwargs["receiver_acquire_timeout"] == pt.TASK_ACQUIRE_TIMEOUT
        # DELIBERATELY unset: an idle wall around the one-shot round would expire during a
        # healthy inner (via-downloader) tensor transfer and fail it mid-flight
        assert "receiver_idle_timeout" not in downloader.kwargs
        assert attempt.tx_id == downloader.tx_id
        assert attempt.ref_id == f"{downloader.tx_id}-ref-1"
        # the payload rides a Downloadable wrapper, one object = one lifecycle unit
        assert isinstance(downloader.added[0], pt.ShareableDownloadable)
        assert downloader.added[0].base_obj == {"w": 1}

    def test_fresh_attempts_get_fresh_tx_ids(self, fake_layer):
        first = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        second = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        # attempt-scoped, never reused: a retry is a new attempt under a new tx_id
        assert first.tx_id != second.tx_id

    def test_failing_add_object_terminates_the_registered_transaction(self, fake_layer):
        FakeDownloader.add_object_error = RuntimeError("service shutting down")
        with pytest.raises(RuntimeError, match="service shutting down"):
            pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        # the constructor registered the transaction before add_object failed; it must be
        # terminated, not leaked until the TTL backstop reclaims it
        assert FakeDownloadService.deleted == [FakeDownloader.instances[0].tx_id]

    def test_waiter_resolution_latches_verdict(self, fake_layer):
        attempt = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        waiter = FakeDownloader.instances[0].waiter
        assert not attempt.completed() and not attempt.failed()
        assert attempt.failure_reason() is None

        waiter.resolve(_outcome(completed=True))
        assert attempt.completed() and not attempt.failed()
        assert attempt.failure_reason() is None

    def test_failed_outcome_carries_reason(self, fake_layer):
        attempt = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        FakeDownloader.instances[0].waiter.resolve(_outcome(completed=False, reason="acquire budget"))
        assert attempt.failed() and not attempt.completed()
        assert "acquire budget" in attempt.failure_reason()

    def test_service_shutdown_resolves_failed_with_reason(self, fake_layer):
        # waiter terminally resolved with no outcome: the service shut down before the
        # attempt settled — must read as failed, never as still-in-flight
        attempt = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        FakeDownloader.instances[0].waiter.resolve(None)
        assert attempt.failed() and not attempt.completed()
        assert "shut down" in attempt.failure_reason()

    def test_wait_true_only_on_completed_outcome(self, fake_layer):
        attempt = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        waiter = FakeDownloader.instances[0].waiter

        waiter.wait_result = _outcome(completed=True)
        assert attempt.wait(timeout=1.0) is True
        waiter.wait_result = _outcome(completed=False)
        assert attempt.wait(timeout=1.0) is False

    def test_wait_handles_the_none_arm(self, fake_layer, caplog):
        # waiter.wait resolves None on timeout or service shutdown; the seam must fold
        # that into not-delivered, never into a hang or a raise
        attempt = pt.TaskPayloadAttempt(MagicMock(), {}, "r")

        assert attempt.wait(timeout=1.0, linger=0.5) is False
        assert "waiter resolved None" in caplog.text
        assert FakeDownloader.instances[0].waiter.wait_calls == [(1.0, 0.5)]

    def test_terminate_uses_delete_transaction(self, fake_layer):
        attempt = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        attempt.terminate()
        attempt.terminate()  # idempotent from the seam's side
        assert FakeDownloadService.deleted == [attempt.tx_id, attempt.tx_id]

    def test_terminate_never_raises(self, fake_layer, monkeypatch, caplog):
        attempt = pt.TaskPayloadAttempt(MagicMock(), {}, "r")

        def boom(transaction_id):
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(FakeDownloadService, "delete_transaction", boom)
        attempt.terminate()
        assert "cleanup failed" in caplog.text


class TestShareableDownloadable:
    def test_one_shot_produce_then_eof(self):
        obj = {"weights": [1, 2, 3]}
        downloadable = pt.ShareableDownloadable(obj)

        rc, data, state = downloadable.produce(None, requester="r")
        assert (rc, data) == (ProduceRC.OK, obj)
        rc, data, _ = downloadable.produce(state, requester="r")
        assert (rc, data) == (ProduceRC.EOF, None)

    def test_release_drops_the_payload_reference(self):
        downloadable = pt.ShareableDownloadable({"big": "model"})
        downloadable.release()
        assert downloadable.base_obj is None


class TestFetchResultPayload:
    def _fake_download_object(self, results_by_ref, fail_refs=()):
        calls = []

        def fake(
            from_fqcn,
            ref_id,
            per_request_timeout,
            cell,
            consumer,
            secure=False,
            optional=False,
            abort_signal=None,
            max_retries=3,
            progress_cb=None,
            progress_interval=30.0,
        ):
            calls.append((from_fqcn, ref_id, per_request_timeout, abort_signal))
            if ref_id in fail_refs:
                consumer.download_failed(ref_id, "producer error")
                return
            consumer.consume(ref_id, {}, results_by_ref[ref_id])
            consumer.download_completed(ref_id)

        return fake, calls

    def test_pulls_objects_in_manifest_order(self, monkeypatch):
        fake, calls = self._fake_download_object({"r1": {"a": 1}, "r2": {"b": 2}})
        monkeypatch.setattr(pt, "download_object", fake)
        cell = MagicMock()

        objs = pt.fetch_result_payload(cell, "trainer-fqcn", ["r1", "r2"])

        assert objs == [{"a": 1}, {"b": 2}]
        assert [c[0] for c in calls] == ["trainer-fqcn", "trainer-fqcn"]
        assert all(c[2] == pt.RESULT_PULL_PER_REQUEST_TIMEOUT for c in calls)

    def test_abort_signal_reaches_every_pull(self, monkeypatch):
        fake, calls = self._fake_download_object({"r1": {"a": 1}})
        monkeypatch.setattr(pt, "download_object", fake)
        signal = object()

        pt.fetch_result_payload(MagicMock(), "trainer-fqcn", ["r1"], abort_signal=signal)

        assert calls[0][3] is signal

    def test_failed_pull_raises_with_reason(self, monkeypatch):
        fake, _ = self._fake_download_object({"r1": {"a": 1}}, fail_refs=("r1",))
        monkeypatch.setattr(pt, "download_object", fake)

        with pytest.raises(pt.PayloadTransferError, match="producer error"):
            pt.fetch_result_payload(MagicMock(), "trainer-fqcn", ["r1"])

    def test_incomplete_pull_raises(self, monkeypatch):
        def never_completes(from_fqcn, ref_id, per_request_timeout, cell, consumer, **kwargs):
            consumer.consume(ref_id, {}, {"partial": True})

        monkeypatch.setattr(pt, "download_object", never_completes)
        with pytest.raises(pt.PayloadTransferError, match="did not complete"):
            pt.fetch_result_payload(MagicMock(), "trainer-fqcn", ["r1"])
