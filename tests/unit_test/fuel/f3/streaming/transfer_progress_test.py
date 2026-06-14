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

import pytest

from nvflare.fuel.f3.streaming.transfer_progress import (
    DEFAULT_STREAMING_IDLE_TIMEOUT,
    DEFAULT_STREAMING_MAX_PEER_SILENCE,
    DIRECTION_RESULT_UPLOAD,
    DIRECTION_TASK_PAYLOAD_DOWNLOAD,
    STREAMING_IDLE_TIMEOUT,
    STREAMING_MAX_PEER_SILENCE,
    TransferProgressState,
    TransferProgressTracker,
    resolve_streaming_progress_config,
)


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _update(tracker, **kwargs):
    values = {
        "job_id": "job-1",
        "task_id": "task-1",
        "transfer_id": "transfer-1",
        "direction": DIRECTION_TASK_PAYLOAD_DOWNLOAD,
        "sequence": 0,
        "bytes_done": 0,
    }
    values.update(kwargs)
    return tracker.update(**values)


def test_tracker_records_first_progress_with_fake_clock():
    clock = FakeClock()
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=clock)

    result = _update(tracker, sequence=1, bytes_done=10)

    assert result.accepted is True
    assert result.progressed is True
    record = result.record
    assert record.key == ("job-1", "task-1", "transfer-1", DIRECTION_TASK_PAYLOAD_DOWNLOAD, None)
    assert record.receiver_id is None
    assert record.sequence == 1
    assert record.bytes_done == 10
    assert record.items_done is None
    assert record.started_time == 1000.0
    assert record.last_progress_time == 1000.0
    assert record.terminal is False


def test_tracker_rejects_unknown_direction():
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=FakeClock())

    with pytest.raises(ValueError, match="direction"):
        _update(tracker, direction="unexpected_direction")


def test_tracker_records_transfer_id_kind():
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=FakeClock())

    result = _update(tracker, sequence=1, bytes_done=10, transfer_id_kind="download_ref")

    assert result.record.transfer_id_kind == "download_ref"


def test_forward_progress_ignores_receiver_id_for_backward_compatible_keying():
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=FakeClock())

    first = _update(tracker, sequence=1, bytes_done=10, receiver_id="receiver-a")
    second = _update(tracker, sequence=2, bytes_done=20, receiver_id="receiver-b")

    assert first.record is second.record
    assert second.record.receiver_id is None
    assert len(list(tracker.records(direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD))) == 1
    assert (
        tracker.get_record(
            job_id="job-1",
            task_id="task-1",
            transfer_id="transfer-1",
            direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
            receiver_id="ignored-receiver",
        )
        is second.record
    )


def test_stale_sequence_and_counter_regression_are_ignored():
    clock = FakeClock()
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=clock)
    _update(tracker, sequence=3, bytes_done=100, items_done=4)

    clock.advance(10.0)
    stale = _update(tracker, sequence=2, bytes_done=200, items_done=5)
    regressed_bytes = _update(tracker, sequence=4, bytes_done=99, items_done=5)
    regressed_items = _update(tracker, sequence=4, bytes_done=120, items_done=3)

    record = tracker.get_record(
        job_id="job-1", task_id="task-1", transfer_id="transfer-1", direction="task_payload_download"
    )
    assert stale.accepted is False
    assert stale.reason == "stale_sequence"
    assert regressed_bytes.accepted is False
    assert regressed_bytes.reason == "bytes_regressed"
    assert regressed_items.accepted is False
    assert regressed_items.reason == "items_regressed"
    assert record.sequence == 3
    assert record.bytes_done == 100
    assert record.items_done == 4
    assert record.last_progress_time == 1000.0


def test_repeated_unchanged_counters_are_not_progress():
    clock = FakeClock()
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=clock)
    _update(tracker, sequence=1, bytes_done=100)

    clock.advance(10.0)
    repeated = _update(tracker, sequence=2, bytes_done=100)

    assert repeated.accepted is True
    assert repeated.progressed is False
    assert repeated.record.sequence == 2
    assert repeated.record.last_progress_time == 1000.0


def test_monotonic_progress_extends_transfer_without_moving_time_backwards():
    clock = FakeClock(now=2000.0)
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=clock)
    _update(tracker, sequence=1, bytes_done=100)

    result = _update(tracker, sequence=2, bytes_done=200, timestamp=1999.0)

    assert result.accepted is True
    assert result.progressed is True
    assert result.record.bytes_done == 200
    assert result.record.last_progress_time == 2000.0


def test_item_progress_is_optional_and_monotonic():
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=FakeClock())
    _update(tracker, sequence=1, bytes_done=0, items_done=None)

    item_progress = _update(tracker, sequence=2, bytes_done=0, items_done=1)
    omitted_items = _update(tracker, sequence=3, bytes_done=0, items_done=None)

    assert item_progress.progressed is True
    assert item_progress.record.items_done == 1
    assert omitted_items.accepted is True
    assert omitted_items.progressed is False
    assert omitted_items.record.items_done == 1


def test_sibling_progress_does_not_mask_stalled_transfer():
    clock = FakeClock()
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=clock)
    _update(tracker, transfer_id="transfer-a", sequence=1, bytes_done=100)
    _update(tracker, transfer_id="transfer-b", sequence=1, bytes_done=100)

    clock.advance(30.0)
    _update(tracker, transfer_id="transfer-a", sequence=2, bytes_done=200)
    clock.advance(31.0)

    assert (
        tracker.is_stalled(
            job_id="job-1", task_id="task-1", transfer_id="transfer-a", direction="task_payload_download"
        )
        is False
    )
    assert (
        tracker.is_stalled(
            job_id="job-1", task_id="task-1", transfer_id="transfer-b", direction="task_payload_download"
        )
        is True
    )
    assert [record.transfer_id for record in tracker.stalled_records()] == ["transfer-b"]


def test_result_upload_progress_is_isolated_by_receiver_id():
    clock = FakeClock()
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=clock)
    _update(
        tracker,
        direction=DIRECTION_RESULT_UPLOAD,
        receiver_id="receiver-a",
        sequence=1,
        bytes_done=100,
    )
    _update(
        tracker,
        direction=DIRECTION_RESULT_UPLOAD,
        receiver_id="receiver-b",
        sequence=1,
        bytes_done=100,
    )

    clock.advance(30.0)
    _update(
        tracker,
        direction=DIRECTION_RESULT_UPLOAD,
        receiver_id="receiver-a",
        sequence=2,
        bytes_done=200,
    )
    clock.advance(31.0)

    receiver_a = tracker.get_record(
        job_id="job-1",
        task_id="task-1",
        transfer_id="transfer-1",
        direction=DIRECTION_RESULT_UPLOAD,
        receiver_id="receiver-a",
    )
    receiver_b = tracker.get_record(
        job_id="job-1",
        task_id="task-1",
        transfer_id="transfer-1",
        direction=DIRECTION_RESULT_UPLOAD,
        receiver_id="receiver-b",
    )

    assert receiver_a.key == ("job-1", "task-1", "transfer-1", DIRECTION_RESULT_UPLOAD, "receiver-a")
    assert receiver_b.key == ("job-1", "task-1", "transfer-1", DIRECTION_RESULT_UPLOAD, "receiver-b")
    assert (
        tracker.is_stalled(
            job_id="job-1",
            task_id="task-1",
            transfer_id="transfer-1",
            direction=DIRECTION_RESULT_UPLOAD,
            receiver_id="receiver-a",
        )
        is False
    )
    assert (
        tracker.is_stalled(
            job_id="job-1",
            task_id="task-1",
            transfer_id="transfer-1",
            direction=DIRECTION_RESULT_UPLOAD,
            receiver_id="receiver-b",
        )
        is True
    )
    assert [record.receiver_id for record in tracker.stalled_records(direction=DIRECTION_RESULT_UPLOAD)] == [
        "receiver-b"
    ]


def test_result_upload_receiver_sequences_and_counters_are_monotonic_per_receiver():
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=FakeClock())
    _update(tracker, direction=DIRECTION_RESULT_UPLOAD, receiver_id="receiver-a", sequence=5, bytes_done=500)

    receiver_b = _update(
        tracker, direction=DIRECTION_RESULT_UPLOAD, receiver_id="receiver-b", sequence=1, bytes_done=10
    )
    stale_a = _update(tracker, direction=DIRECTION_RESULT_UPLOAD, receiver_id="receiver-a", sequence=4, bytes_done=600)

    assert receiver_b.accepted is True
    assert receiver_b.record.sequence == 1
    assert receiver_b.record.bytes_done == 10
    assert stale_a.accepted is False
    assert stale_a.reason == "stale_sequence"
    assert (
        tracker.get_record(
            job_id="job-1",
            task_id="task-1",
            transfer_id="transfer-1",
            direction=DIRECTION_RESULT_UPLOAD,
            receiver_id="receiver-a",
        ).bytes_done
        == 500
    )


def test_result_upload_single_receiver_uses_none_receiver_key():
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=FakeClock())

    result = _update(tracker, direction=DIRECTION_RESULT_UPLOAD, sequence=1, bytes_done=10)

    assert result.record.key == ("job-1", "task-1", "transfer-1", DIRECTION_RESULT_UPLOAD, None)
    assert (
        tracker.get_record(
            job_id="job-1",
            task_id="task-1",
            transfer_id="transfer-1",
            direction=DIRECTION_RESULT_UPLOAD,
            receiver_id=None,
        )
        is result.record
    )
    assert len(list(tracker.records(direction=DIRECTION_RESULT_UPLOAD, receiver_id=None))) == 1


@pytest.mark.parametrize(
    "state",
    [TransferProgressState.COMPLETED, TransferProgressState.FAILED, TransferProgressState.ABORTED],
)
def test_terminal_states_are_retained_and_ignore_later_updates_until_pruned(state):
    clock = FakeClock()
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=clock)
    _update(tracker, sequence=1, bytes_done=100)

    clock.advance(1.0)
    terminal = _update(tracker, sequence=2, bytes_done=100, state=state)
    later = _update(tracker, sequence=3, bytes_done=200)

    assert terminal.accepted is True
    assert terminal.record.terminal is True
    assert terminal.record.state == state
    assert later.accepted is False
    assert later.reason == "terminal"
    assert tracker.prune(before_time=clock.now - 1.0) == 0
    assert tracker.get_record(
        job_id="job-1", task_id="task-1", transfer_id="transfer-1", direction="task_payload_download"
    )
    assert tracker.prune(before_time=clock.now) == 1
    assert (
        tracker.get_record(
            job_id="job-1", task_id="task-1", transfer_id="transfer-1", direction="task_payload_download"
        )
        is None
    )


def test_prune_can_be_limited_to_one_direction():
    clock = FakeClock()
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=clock)
    _update(tracker, sequence=1, bytes_done=100)
    _update(
        tracker,
        task_id="task-2",
        transfer_id="result-1",
        direction=DIRECTION_RESULT_UPLOAD,
        receiver_id="server",
        sequence=1,
        bytes_done=100,
    )

    clock.advance(1.0)
    _update(tracker, sequence=2, bytes_done=100, state=TransferProgressState.COMPLETED)
    _update(
        tracker,
        task_id="task-2",
        transfer_id="result-1",
        direction=DIRECTION_RESULT_UPLOAD,
        receiver_id="server",
        sequence=2,
        bytes_done=100,
        state=TransferProgressState.COMPLETED,
    )

    assert tracker.prune(before_time=clock.now, direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD) == 1
    assert len(list(tracker.records(direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD))) == 0
    assert len(list(tracker.records(direction=DIRECTION_RESULT_UPLOAD))) == 1


def test_mark_terminal_can_create_abort_record_for_unknown_transfer():
    tracker = TransferProgressTracker(idle_timeout=60.0, clock=FakeClock())

    result = tracker.mark_terminal(
        job_id="job-1",
        task_id="task-1",
        transfer_id="transfer-1",
        direction="task_payload_download",
        state=TransferProgressState.ABORTED,
    )

    assert result.accepted is True
    assert result.record.terminal is True
    assert result.record.state == TransferProgressState.ABORTED


def test_resolve_streaming_progress_config_defaults():
    config = resolve_streaming_progress_config()

    assert config.streaming_idle_timeout == DEFAULT_STREAMING_IDLE_TIMEOUT
    assert config.streaming_max_peer_silence == DEFAULT_STREAMING_MAX_PEER_SILENCE


def test_resolve_streaming_progress_config_derives_max_peer_silence_when_idle_is_raised():
    config = resolve_streaming_progress_config({STREAMING_IDLE_TIMEOUT: 1200.0})

    assert config.streaming_idle_timeout == 1200.0
    assert config.streaming_max_peer_silence == 1800.0


def test_resolve_streaming_progress_config_honors_explicit_max_peer_silence():
    config = resolve_streaming_progress_config({STREAMING_IDLE_TIMEOUT: 1200.0, STREAMING_MAX_PEER_SILENCE: 1300.0})

    assert config.streaming_idle_timeout == 1200.0
    assert config.streaming_max_peer_silence == 1300.0


def test_resolve_streaming_progress_config_keeps_default_max_peer_silence_when_idle_is_not_raised():
    config = resolve_streaming_progress_config({STREAMING_IDLE_TIMEOUT: 300.0})

    assert config.streaming_idle_timeout == 300.0
    assert config.streaming_max_peer_silence == DEFAULT_STREAMING_MAX_PEER_SILENCE


@pytest.mark.parametrize("key", [STREAMING_IDLE_TIMEOUT, STREAMING_MAX_PEER_SILENCE])
def test_resolve_streaming_progress_config_rejects_non_positive_values(key):
    with pytest.raises(ValueError):
        resolve_streaming_progress_config({key: 0.0})


@pytest.mark.parametrize("key", [STREAMING_IDLE_TIMEOUT, STREAMING_MAX_PEER_SILENCE])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_resolve_streaming_progress_config_rejects_non_finite_values(key, value):
    with pytest.raises(ValueError, match="finite"):
        resolve_streaming_progress_config({key: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_transfer_progress_tracker_rejects_non_finite_idle_timeout(value):
    with pytest.raises(ValueError, match="finite"):
        TransferProgressTracker(idle_timeout=value)


def test_transfer_progress_tracker_set_idle_timeout_validates_and_coerces():
    tracker = TransferProgressTracker(idle_timeout=60.0)

    tracker.set_idle_timeout(120)

    assert tracker.idle_timeout == 120.0
    with pytest.raises(ValueError, match="finite"):
        tracker.set_idle_timeout(float("inf"))
