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

import threading
import time
from unittest.mock import MagicMock

import pytest

from nvflare.client.flare_agent import FlareAgent, _ReverseResultUploadProgressTracker, _TaskContext
from nvflare.fuel.f3.streaming.download_service import TransactionDoneStatus
from nvflare.fuel.f3.streaming.transfer_progress import DIRECTION_RESULT_UPLOAD, TransferProgressState
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposers.via_downloader import DownloadTransactionInfo, _tls, clear_download_initiated


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _make_tracker(clock=None, idle_timeout=10.0):
    return _ReverseResultUploadProgressTracker(idle_timeout=idle_timeout, clock=clock or FakeClock())


def _register(tracker, tx_id="tx-1", pairs=(("ref-1", None),), created_time=1000.0):
    tracker.register_transaction(tx_id=tx_id, expected_pairs=pairs, created_time=created_time)


def _progress(
    tracker,
    *,
    tx_id="tx-1",
    ref_id="ref-1",
    receiver_id=None,
    sequence=1,
    bytes_done=1,
    items_done=None,
    state=TransferProgressState.ACTIVE,
    timestamp=1000.0,
):
    accepted, reason = tracker.update(
        tx_id=tx_id,
        transfer_id=ref_id,
        receiver_id=receiver_id,
        sequence=sequence,
        bytes_done=bytes_done,
        items_done=items_done,
        state=state,
        timestamp=timestamp,
    )
    assert accepted, reason


def test_result_upload_positive_waits_past_fixed_timeout_while_progress_is_recent():
    clock = FakeClock()
    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(tracker)
    _progress(tracker, sequence=1, bytes_done=100, timestamp=clock.now)

    for sequence in range(2, 8):
        clock.advance(9.0)
        _progress(tracker, sequence=sequence, bytes_done=sequence * 100, timestamp=clock.now)
        decision = tracker.decide()
        assert decision.done is False

    decision = tracker.decide(callback_fired=True, callback_status=TransactionDoneStatus.FINISHED)

    assert decision.done is True
    assert decision.success is True


def test_result_upload_no_start_times_out():
    clock = FakeClock()
    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(tracker, created_time=clock.now)

    clock.advance(10.0)
    decision = tracker.decide()

    assert decision.done is True
    assert decision.success is False
    assert "did not start" in decision.reason


def test_result_upload_failure_decision_deletes_source_transaction(monkeypatch):
    clock = FakeClock()
    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(tracker, created_time=clock.now)
    clock.advance(10.0)

    agent = FlareAgent.__new__(FlareAgent)
    agent.logger = MagicMock()
    agent.asked_to_stop = False
    agent.pipe_handler = MagicMock()
    agent.pipe_handler.asked_to_stop = False
    agent.pipe = MagicMock()
    agent.pipe.closed = False
    agent._result_upload_poll_interval = 10.0
    deleted = []
    monkeypatch.setattr(
        "nvflare.client.flare_agent.DownloadService.delete_transaction", lambda tx_id: deleted.append(tx_id)
    )

    result = agent._wait_for_reverse_result_upload(
        tracker,
        threading.Event(),
        threading.Event(),
        [None],
        wait_start=clock.now,
        transactions=[DownloadTransactionInfo("tx-1", (("ref-1", None),), 1000.0)],
    )

    assert result is False
    assert deleted == ["tx-1"]


def test_result_upload_started_pair_stalls_independently():
    clock = FakeClock()
    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(tracker, created_time=clock.now)
    _progress(tracker, bytes_done=100, timestamp=clock.now)

    clock.advance(10.0)
    decision = tracker.decide()

    assert decision.done is True
    assert decision.success is False
    assert "stalled" in decision.reason


def test_result_upload_multi_ref_completion_requires_all_refs():
    clock = FakeClock()
    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(tracker, pairs=(("ref-a", None), ("ref-b", None)), created_time=clock.now)

    _progress(
        tracker,
        ref_id="ref-a",
        sequence=1,
        bytes_done=100,
        state=TransferProgressState.COMPLETED,
        timestamp=clock.now,
    )
    decision = tracker.decide()
    assert decision.done is False

    _progress(
        tracker,
        ref_id="ref-b",
        sequence=1,
        bytes_done=200,
        state=TransferProgressState.COMPLETED,
        timestamp=clock.now,
    )
    decision = tracker.decide()
    assert decision.done is False
    assert decision.reason == "completion_grace"


def test_result_upload_completed_ref_does_not_mask_failed_sibling():
    clock = FakeClock()
    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(tracker, pairs=(("ref-a", None), ("ref-b", None)), created_time=clock.now)

    _progress(
        tracker,
        ref_id="ref-a",
        sequence=1,
        bytes_done=100,
        state=TransferProgressState.COMPLETED,
        timestamp=clock.now,
    )
    _progress(
        tracker,
        ref_id="ref-b",
        sequence=1,
        bytes_done=50,
        state=TransferProgressState.FAILED,
        timestamp=clock.now,
    )
    decision = tracker.decide()

    assert decision.done is True
    assert decision.success is False
    assert "ref-b" in decision.reason


def test_result_upload_completed_ref_does_not_mask_stalled_sibling():
    clock = FakeClock()
    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(tracker, pairs=(("ref-a", None), ("ref-b", None)), created_time=clock.now)

    _progress(
        tracker,
        ref_id="ref-a",
        sequence=1,
        bytes_done=100,
        state=TransferProgressState.COMPLETED,
        timestamp=clock.now,
    )
    _progress(tracker, ref_id="ref-b", sequence=1, bytes_done=50, timestamp=clock.now)
    clock.advance(10.0)
    decision = tracker.decide()

    assert decision.done is True
    assert decision.success is False
    assert "ref-b" in decision.reason
    assert "stalled" in decision.reason


def test_result_upload_progressing_ref_does_not_mask_unstarted_sibling():
    clock = FakeClock()
    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(tracker, pairs=(("ref-a", None), ("ref-b", None)), created_time=clock.now)
    _progress(tracker, ref_id="ref-a", sequence=1, bytes_done=100, timestamp=clock.now)

    clock.advance(9.0)
    _progress(tracker, ref_id="ref-a", sequence=2, bytes_done=200, timestamp=clock.now)
    clock.advance(1.0)
    decision = tracker.decide()

    assert decision.done is True
    assert decision.success is False
    assert "ref-b" in decision.reason
    assert "did not start" in decision.reason


def test_result_upload_receiver_isolation_for_same_ref():
    clock = FakeClock()
    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(
        tracker,
        pairs=(("ref-1", "server"), ("ref-1", "peer")),
        created_time=clock.now,
    )
    _progress(tracker, receiver_id="server", sequence=1, bytes_done=100, timestamp=clock.now)

    clock.advance(9.0)
    _progress(tracker, receiver_id="server", sequence=2, bytes_done=200, timestamp=clock.now)
    clock.advance(1.0)
    decision = tracker.decide()

    assert decision.done is True
    assert decision.success is False
    assert "receiver=peer" in decision.reason


def test_result_upload_completion_grace_waits_for_callback_then_succeeds_without_it():
    clock = FakeClock()
    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(tracker, created_time=clock.now)
    _progress(
        tracker,
        sequence=1,
        bytes_done=100,
        state=TransferProgressState.COMPLETED,
        timestamp=clock.now,
    )

    decision = tracker.decide()
    assert decision.done is False
    assert decision.reason == "completion_grace"

    clock.advance(29.0)
    assert tracker.decide().done is False
    decision = tracker.decide(callback_fired=True, callback_status=TransactionDoneStatus.FINISHED)
    assert decision.done is True
    assert decision.success is True

    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(tracker, created_time=clock.now)
    _progress(
        tracker,
        sequence=1,
        bytes_done=100,
        state=TransferProgressState.COMPLETED,
        timestamp=clock.now,
    )
    assert tracker.decide().done is False
    clock.advance(30.0)
    decision = tracker.decide()
    assert decision.done is True
    assert decision.success is True


def test_result_upload_terminal_failure_fails():
    clock = FakeClock()
    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(tracker, created_time=clock.now)
    _progress(
        tracker,
        sequence=1,
        state=TransferProgressState.FAILED,
        bytes_done=100,
        timestamp=clock.now,
    )

    decision = tracker.decide()

    assert decision.done is True
    assert decision.success is False


@pytest.fixture(autouse=True)
def _no_os_exit(monkeypatch):
    monkeypatch.setattr("nvflare.client.flare_agent.os._exit", lambda code: None)


def _make_cell_pipe():
    from nvflare.fuel.utils.pipe.cell_pipe import CellPipe

    pipe = MagicMock(spec=CellPipe)
    pipe.pass_through_on_send = True
    pipe.closed = False
    pipe.cell = MagicMock()
    pipe.cell.get_fobs_context.return_value = {}
    return pipe


def _make_agent(pipe, download_complete_timeout=0.01):
    agent = FlareAgent.__new__(FlareAgent)
    agent.logger = MagicMock()
    agent.pipe = pipe
    agent.submit_result_timeout = 30.0
    agent._download_complete_timeout = download_complete_timeout
    agent._streaming_idle_timeout = 10.0
    agent._result_upload_poll_interval = 0.001
    agent._launch_once = False
    agent.asked_to_stop = False
    agent.pipe_handler = MagicMock()
    agent.pipe_handler.asked_to_stop = False
    agent.pipe_handler.send_to_peer.return_value = True
    agent.task_result_to_shareable = MagicMock(return_value=MagicMock())
    return agent


def test_no_large_result_proceeds_immediately_without_waiting():
    clear_download_initiated()
    pipe = _make_cell_pipe()
    agent = _make_agent(pipe, download_complete_timeout=100.0)

    result = agent._do_submit_result(_TaskContext("tid-1", "validate", "msg-1"), None, "OK")

    assert result is True
    agent.pipe_handler.send_to_peer.assert_called_once()


def test_tracking_unavailable_falls_back_to_download_complete_timeout():
    clear_download_initiated()
    pipe = _make_cell_pipe()
    agent = _make_agent(pipe, download_complete_timeout=0.01)

    agent._wait_for_download_complete_fixed = MagicMock(return_value=True)
    agent._wait_for_reverse_result_upload = MagicMock(return_value=True)

    def _send(reply, timeout):
        _tls.download_initiated = True
        _tls.download_transactions = []
        return True

    agent.pipe_handler.send_to_peer.side_effect = _send

    result = agent._do_submit_result(_TaskContext("tid-1", "train", "msg-1"), None, "OK")

    assert result is True
    agent._wait_for_download_complete_fixed.assert_called_once()
    agent._wait_for_reverse_result_upload.assert_not_called()
    assert any("falling back" in call[0][0] for call in agent.logger.info.call_args_list)


def test_do_submit_result_waits_for_result_upload_progress_until_completion():
    clear_download_initiated()
    pipe = _make_cell_pipe()
    agent = _make_agent(pipe, download_complete_timeout=0.001)
    agent._streaming_idle_timeout = 0.1
    agent._result_upload_poll_interval = 0.001

    callbacks_ready = threading.Event()
    callbacks = {}
    result = {}

    def _send(reply, timeout):
        ctx = pipe.cell.update_fobs_context.call_args.args[0]
        callbacks["progress"] = ctx[fobs.FOBSContextKey.STREAM_PROGRESS_CB]
        callbacks["complete"] = ctx[fobs.FOBSContextKey.DOWNLOAD_COMPLETE_CB]
        _tls.download_initiated = True
        _tls.download_transactions = [
            DownloadTransactionInfo("tx-1", (("ref-1", None),), time.time()),
        ]
        callbacks_ready.set()
        return True

    agent.pipe_handler.send_to_peer.side_effect = _send

    def _run_submit():
        result["ok"] = agent._do_submit_result(_TaskContext("tid-1", "train", "msg-1"), None, "OK")

    thread = threading.Thread(target=_run_submit)
    start = time.time()
    thread.start()
    assert callbacks_ready.wait(timeout=1.0)

    for sequence in range(1, 5):
        callbacks["progress"](
            direction=DIRECTION_RESULT_UPLOAD,
            tx_id="tx-1",
            transfer_id="ref-1",
            sequence=sequence,
            bytes_done=sequence * 100,
            state=TransferProgressState.ACTIVE,
            timestamp=time.time(),
        )
        time.sleep(0.01)

    callbacks["progress"](
        direction=DIRECTION_RESULT_UPLOAD,
        tx_id="tx-1",
        transfer_id="ref-1",
        sequence=5,
        bytes_done=500,
        state=TransferProgressState.COMPLETED,
        timestamp=time.time(),
    )
    callbacks["complete"]("tx-1", TransactionDoneStatus.FINISHED, None)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result["ok"] is True
    assert time.time() - start > agent._download_complete_timeout


def test_reverse_download_ttl_preserves_legacy_fallback_timeout():
    clear_download_initiated()
    pipe = _make_cell_pipe()
    agent = _make_agent(pipe, download_complete_timeout=1800.0)
    captured = {}

    def _send(reply, timeout):
        captured["dl_ttl"] = reply._dl_ttl
        return True

    agent.pipe_handler.send_to_peer.side_effect = _send

    assert agent._do_submit_result(_TaskContext("tid-1", "validate", "msg-1"), None, "OK") is True
    assert captured["dl_ttl"] == 1800.0


def test_reverse_result_upload_receiver_ids_are_stamped_for_fobs_encode():
    clear_download_initiated()
    pipe = _make_cell_pipe()
    agent = _make_agent(pipe, download_complete_timeout=1800.0)
    captured = {}

    def _send(reply, timeout):
        captured["receiver_ids"] = reply._receiver_ids
        return True

    agent.pipe_handler.send_to_peer.side_effect = _send

    result = agent._do_submit_result(
        _TaskContext(
            "tid-1",
            "train",
            "msg-1",
            result_receiver_ids=("site-1.job-1", "site-2.job-1"),
        ),
        None,
        "OK",
    )

    assert result is True
    assert captured["receiver_ids"] == ("site-1.job-1", "site-2.job-1")
    update_calls = [call.args[0] for call in pipe.cell.update_fobs_context.call_args_list]
    assert any(call.get("receiver_ids") == ("site-1.job-1", "site-2.job-1") for call in update_calls)


def test_reverse_result_upload_wait_abandons_when_pipe_handler_stops(monkeypatch):
    tracker = _make_tracker(idle_timeout=10.0)
    _register(tracker)
    agent = FlareAgent.__new__(FlareAgent)
    agent.logger = MagicMock()
    agent.asked_to_stop = False
    agent.pipe_handler = MagicMock()
    agent.pipe_handler.asked_to_stop = True
    agent.pipe = MagicMock()
    agent.pipe.closed = False
    agent._result_upload_poll_interval = 10.0
    deleted = []
    monkeypatch.setattr(
        "nvflare.client.flare_agent.DownloadService.delete_transaction", lambda tx_id: deleted.append(tx_id)
    )

    result = agent._wait_for_reverse_result_upload(
        tracker,
        threading.Event(),
        threading.Event(),
        [None],
        wait_start=1000.0,
        transactions=[DownloadTransactionInfo("tx-1", (("ref-1", None),), 1000.0)],
    )

    assert result is False
    assert deleted == ["tx-1"]
    decision = tracker.decide()
    assert decision.done is True
    assert decision.success is False
    assert "aborted" in decision.reason
    assert any("abandoning result_upload wait" in call[0][0] for call in agent.logger.warning.call_args_list)


def test_flare_agent_progress_callback_uses_result_upload_direction_only():
    clock = FakeClock()
    tracker = _make_tracker(clock=clock, idle_timeout=10.0)
    _register(tracker, created_time=clock.now)
    event = threading.Event()
    agent = FlareAgent.__new__(FlareAgent)
    agent.logger = MagicMock()

    agent._update_reverse_result_upload_progress(
        tracker,
        event,
        direction="task_payload_download",
        transfer_id="ref-1",
        sequence=1,
        bytes_done=100,
    )
    assert event.is_set() is False

    agent._update_reverse_result_upload_progress(
        tracker,
        event,
        direction=DIRECTION_RESULT_UPLOAD,
        tx_id="tx-1",
        transfer_id="ref-1",
        receiver_id=None,
        sequence=1,
        bytes_done=100,
        state=TransferProgressState.ACTIVE,
        timestamp=clock.now,
    )
    assert event.is_set() is True
