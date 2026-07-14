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

"""Tests for the Client API payload-transfer seam (payload_transfer.py).

The seam is coded against the F3 payload-layer contract of PR #4865
(docs/design/f3_backend_interface_contract.md), which is not merged yet. These tests
verify both regimes:

- pre-#4865 (today's main): producer-side attempts are guarded and raise
  PayloadLayerUnavailable with a message naming the missing layer;
- post-#4865 (simulated by faking the contract surface at the seam's imports): attempts
  declare the receiver identity and budgets, latch outcomes from outcome_cb, fold the
  waiter's None arm into not-delivered, and terminate through the contract-listed
  DownloadService.delete_transaction.

The consumer side (download_object) already exists on main with the contract signature,
so fetch_result_payload is tested against a fake of that exact signature.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.app_common.executors.client_api import payload_transfer as pt
from nvflare.fuel.f3.streaming.download_service import ProduceRC


class FakeWaiter:
    def __init__(self):
        self.outcome_to_return = None
        self.wait_calls = []

    def wait(self, timeout=None, linger=None):
        self.wait_calls.append((timeout, linger))
        return self.outcome_to_return


class FakeDownloader:
    """Fakes the #4865 ObjectDownloader surface the contract exposes to backends."""

    instances = []

    def __init__(self, **kwargs):
        FakeDownloader.instances.append(self)
        self.kwargs = kwargs
        self.tx_id = f"tx-{len(FakeDownloader.instances)}"
        self.added = []
        self.waiter = FakeWaiter()

    def add_object(self, obj, ref_id=None):
        self.added.append(obj)
        return f"{self.tx_id}-ref-{len(self.added)}"

    def get_waiter(self):
        return self.waiter


class FakeDownloadService:
    """Presence of get_transfer_waiter is the seam's availability signal."""

    deleted = []

    @classmethod
    def get_transfer_waiter(cls, transaction_id):
        raise AssertionError("the seam should prefer downloader.get_waiter()")

    @classmethod
    def delete_transaction(cls, transaction_id):
        cls.deleted.append(transaction_id)


def _outcome(completed: bool, reason=None):
    return SimpleNamespace(completed=completed, reason=reason)


@pytest.fixture
def available_layer(monkeypatch):
    """Simulates the merged #4865 contract surface at the seam's own imports."""
    FakeDownloader.instances = []
    FakeDownloadService.deleted = []
    monkeypatch.setattr(pt, "ObjectDownloader", FakeDownloader)
    monkeypatch.setattr(pt, "DownloadService", FakeDownloadService)
    return FakeDownloadService


class TestAvailabilityGuard:
    def test_availability_probe_matches_the_service_surface(self):
        # regime-agnostic: False on pre-#4865 main, flips to True the moment the merged
        # layer ships the TransferWaiter accessor — with no change here or in the seam
        from nvflare.fuel.f3.streaming.download_service import DownloadService

        assert pt.payload_layer_available() == hasattr(DownloadService, "get_transfer_waiter")

    @pytest.mark.skipif(pt.payload_layer_available(), reason="the #4865 payload layer is present in this build")
    def test_attempt_creation_raises_with_clear_hint_pre_4865(self):
        with pytest.raises(pt.PayloadLayerUnavailable, match="4865"):
            pt.TaskPayloadAttempt(MagicMock(), {"w": 1}, "site-1.job-1.trainer_1")

    def test_available_when_service_has_waiter_accessor(self, available_layer):
        assert pt.payload_layer_available() is True


class TestTaskPayloadAttempt:
    def test_attempt_declares_receiver_identity_and_budgets(self, available_layer):
        cell = MagicMock()
        attempt = pt.TaskPayloadAttempt(cell, {"w": 1}, "site-1.job-1.trainer_1")

        downloader = FakeDownloader.instances[0]
        assert downloader.kwargs["cell"] is cell
        # declared identity enables the acquire budget and identity-checked completion
        assert downloader.kwargs["receiver_ids"] == ("site-1.job-1.trainer_1",)
        assert downloader.kwargs["num_receivers"] == 1
        assert downloader.kwargs["timeout"] == pt.TASK_TRANSFER_INACTIVITY_TIMEOUT
        assert downloader.kwargs["receiver_acquire_timeout"] == pt.TASK_ACQUIRE_TIMEOUT
        assert downloader.kwargs["receiver_idle_timeout"] == pt.TASK_IDLE_TIMEOUT
        assert attempt.tx_id == downloader.tx_id
        assert attempt.ref_id == f"{downloader.tx_id}-ref-1"
        # the payload rides a Downloadable wrapper, one object = one lifecycle unit
        assert isinstance(downloader.added[0], pt.ShareableDownloadable)
        assert downloader.added[0].base_obj == {"w": 1}

    def test_fresh_attempts_get_fresh_tx_ids(self, available_layer):
        first = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        second = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        # attempt-scoped, never reused: a retry is a new attempt under a new tx_id
        assert first.tx_id != second.tx_id

    def test_outcome_cb_latches_verdict(self, available_layer):
        attempt = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        outcome_cb = FakeDownloader.instances[0].kwargs["outcome_cb"]
        assert not attempt.completed() and not attempt.failed()

        outcome_cb(_outcome(completed=True))
        assert attempt.completed() and not attempt.failed()
        assert attempt.failure_reason() is None

    def test_failed_outcome_carries_reason(self, available_layer):
        attempt = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        FakeDownloader.instances[0].kwargs["outcome_cb"](_outcome(completed=False, reason="acquire budget"))
        assert attempt.failed() and not attempt.completed()
        assert "acquire budget" in attempt.failure_reason()

    def test_wait_true_only_on_completed_outcome(self, available_layer):
        attempt = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        waiter = FakeDownloader.instances[0].waiter

        waiter.outcome_to_return = _outcome(completed=True)
        assert attempt.wait(timeout=1.0) is True
        waiter.outcome_to_return = _outcome(completed=False)
        assert attempt.wait(timeout=1.0) is False

    def test_wait_handles_the_none_arm(self, available_layer, caplog):
        # contract: waiter.wait resolves None on timeout or service shutdown; the seam
        # must fold that into not-delivered, never into a hang or a raise
        attempt = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        FakeDownloader.instances[0].waiter.outcome_to_return = None

        assert attempt.wait(timeout=1.0, linger=0.5) is False
        assert "waiter resolved None" in caplog.text
        assert FakeDownloader.instances[0].waiter.wait_calls == [(1.0, 0.5)]

    def test_terminate_uses_contract_listed_deletion(self, available_layer):
        attempt = pt.TaskPayloadAttempt(MagicMock(), {}, "r")
        attempt.terminate()
        attempt.terminate()  # idempotent from the seam's side
        assert FakeDownloadService.deleted == [attempt.tx_id, attempt.tx_id]

    def test_terminate_never_raises(self, available_layer, monkeypatch, caplog):
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
            calls.append((from_fqcn, ref_id, per_request_timeout))
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
