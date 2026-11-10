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
import weakref
from types import SimpleNamespace

import pytest

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming import deadline_scheduler
from nvflare.fuel.f3.streaming.byte_receiver import FAILED_NON_RELIABLE_TASK_TTL, RxTask
from nvflare.fuel.f3.streaming.deadline_scheduler import DeadlineScheduler
from nvflare.fuel.f3.streaming.stream_const import StreamDataType, StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import StreamError
from tests.timing_utils import WAIT_TIMEOUT, ManualClock, ObservedLock, wait_for

# A broken lock order must fail even if it prevents test cleanup from starting.
pytestmark = pytest.mark.timeout(60)


@pytest.fixture
def cleanup(monkeypatch):
    clock = ManualClock()
    scheduler = DeadlineScheduler()
    monkeypatch.setattr(deadline_scheduler, "time", clock)
    monkeypatch.setattr(RxTask, "cleanup_scheduler", scheduler)
    monkeypatch.setattr(RxTask, "rx_task_map", {})
    monkeypatch.setattr(RxTask, "_shutting_down", False)
    try:
        yield scheduler, clock
    finally:
        RxTask.shutdown()
        assert not scheduler.snapshot()["worker_alive"]


def _final(sid, origin="peer", reliable=True, retry_timeout=60, retry_wait=5):
    return Message(
        {
            MessageHeaderKey.ORIGIN: origin,
            StreamHeaderKey.STREAM_ID: sid,
            StreamHeaderKey.SEQUENCE: 0,
            StreamHeaderKey.DATA_TYPE: StreamDataType.FINAL,
            StreamHeaderKey.RELIABLE: reliable,
            StreamHeaderKey.RETRY_TIMEOUT: retry_timeout,
            StreamHeaderKey.RETRY_WAIT: retry_wait,
            StreamHeaderKey.SIZE: 4,
        },
        b"data",
    )


def _advance(cleanup, seconds):
    scheduler, clock = cleanup
    clock.advance(seconds)
    with scheduler._condition:
        scheduler._condition.notify_all()


def test_ten_thousand_completions_use_one_worker_and_keep_full_retry_window(cleanup, monkeypatch):
    def forbidden_timer(*args, **kwargs):
        pytest.fail("receive cleanup must not allocate a Timer thread")

    monkeypatch.setattr(threading, "Timer", forbidden_timer)
    before = set(threading.enumerate())
    cell = SimpleNamespace(fire_and_forget=lambda *a, **kw: {})
    for sid in range(10000):
        message = _final(sid)
        task = RxTask.find_or_create_task(message, cell)
        assert task.process_chunk(message)
        assert task.completed_task_ttl == 65
    stats = RxTask.get_cleanup_stats()
    assert stats["pending"] == stats["queue_depth"] == stats["retained_tasks"] == 10000
    assert set(threading.enumerate()) - before == {cleanup[0]._thread}
    _advance(cleanup, 64)
    barrier = threading.Event()
    cleanup[0].schedule(0, barrier.set)
    assert barrier.wait(WAIT_TIMEOUT)
    assert len(RxTask.rx_task_map) == 10000
    _advance(cleanup, 1)
    wait_for(lambda: RxTask.get_cleanup_stats()["expired"] == 10001)
    assert RxTask.rx_task_map == {}
    assert RxTask.get_cleanup_stats()["pending"] == 0


def test_retry_reacks_original_task_without_resetting_deadline(cleanup):
    sent = []
    cell = SimpleNamespace(fire_and_forget=lambda *args, **kw: sent.append(args[3]))
    message = _final(1, retry_timeout=80, retry_wait=10)
    task = RxTask.find_or_create_task(message, cell)
    assert task.process_chunk(message)
    handle = task.cleanup_deadline
    assert task.read(4) == b"data"
    _advance(cleanup, 89)
    assert RxTask.find_or_create_task(message, cell) is task
    assert not task.process_chunk(message)
    assert len(sent) == 2
    assert sent[-1].get_header(StreamHeaderKey.OFFSET) == 4
    assert sent[-1].get_header(StreamHeaderKey.SEQUENCE) == 0
    assert task.cleanup_deadline is handle
    _advance(cleanup, 1)
    wait_for(lambda: not RxTask.rx_task_map)


@pytest.mark.parametrize("reliable,ttl", [(True, 65), (False, FAILED_NON_RELIABLE_TASK_TTL)])
def test_failed_tombstone_survives_until_its_original_deadline(cleanup, reliable, ttl):
    sent = []
    message = _final(2, reliable=reliable)
    task = RxTask.find_or_create_task(message, SimpleNamespace(fire_and_forget=lambda *a, **kw: sent.append(a[3])))
    task.stop(StreamError("failed"), notify=False)
    assert task.cleanup_deadline.interval == ttl
    assert not task.process_chunk(message)
    assert len(sent) == (2 if reliable else 0)
    _advance(cleanup, ttl)
    wait_for(lambda: not RxTask.rx_task_map)


def test_non_reliable_success_needs_no_deadline_or_worker(cleanup):
    message = _final(3, reliable=False)
    task = RxTask.find_or_create_task(message, SimpleNamespace(fire_and_forget=lambda *a, **kw: {}))
    assert task.process_chunk(message)
    assert RxTask.rx_task_map == {}
    assert not RxTask.get_cleanup_stats()["worker_alive"]
    assert RxTask.get_cleanup_stats()["pending"] == 0


@pytest.mark.parametrize("failed", [False, True])
def test_worker_start_failure_releases_task_without_interrupting_delivery(cleanup, monkeypatch, caplog, failed):
    message = _final(9, reliable=not failed)
    task = RxTask.find_or_create_task(message, SimpleNamespace(fire_and_forget=lambda *a, **kw: {}))
    start = threading.Thread.start

    def fail_cleanup_worker(thread):
        if thread.name == "f3-rx-cleanup":
            raise RuntimeError("can't start new thread")
        return start(thread)

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", fail_cleanup_worker)
        if failed:
            task.stop(StreamError("failed"), notify=False)
        else:
            # A single-chunk completion must still reach the caller's new-stream
            # callback path, even after its final ACK has already been sent.
            assert task.process_chunk(message)
            assert task.read(4) == b"data"
    assert RxTask.rx_task_map == {}
    assert task.cleanup_deadline is None
    assert "cleanup worker unavailable" in caplog.text
    assert RxTask.get_cleanup_stats()["pending"] == 0
    done = threading.Event()
    cleanup[0].schedule(0, done.set)
    assert done.wait(WAIT_TIMEOUT)


def test_expiry_racing_removal_cannot_remove_replacement(cleanup):
    task = RxTask.find_or_create_task(_final(4), SimpleNamespace())
    task.stop_lock = ObservedLock(task.stop_lock)
    with task.stop_lock:
        task._schedule_remove_task(ttl=0)
        assert task.stop_lock.blocked.wait(WAIT_TIMEOUT)
        task._remove_task()
        replacement = RxTask.find_or_create_task(_final(4), SimpleNamespace())
        assert replacement is not task
    wait_for(lambda: cleanup[0].snapshot()["expired"] == 1)
    assert RxTask.rx_task_map[("peer", 4)] is replacement
    assert task.cleanup_deadline is None


def test_process_shutdown_releases_tasks_and_rejects_late_arrivals(cleanup):
    message = _final(5)
    task = RxTask.find_or_create_task(message, SimpleNamespace(fire_and_forget=lambda *a, **kw: {}))
    assert task.process_chunk(message)
    reference = weakref.ref(task)
    del task
    RxTask.shutdown()
    assert reference() is None
    assert RxTask.rx_task_map == {}
    assert RxTask.find_or_create_task(message, SimpleNamespace()) is None
    # A completion already in flight at shutdown must not repopulate the queue/map.
    late = RxTask(6, "peer", SimpleNamespace(), reliable=True)
    RxTask.rx_task_map[("peer", 6)] = late
    late._schedule_remove_task()
    assert RxTask.rx_task_map == {}
    assert RxTask.get_cleanup_stats()["pending"] == 0
    assert not RxTask.get_cleanup_stats()["worker_alive"]


@pytest.mark.parametrize(
    "retry_timeout,retry_wait",
    [(float("inf"), 5), (60, float("nan")), (float("-inf"), 5), (-10, 5), (1e308, 1e308)],
)
def test_invalid_local_retry_window_falls_back_without_stranding_other_tasks(
    cleanup, monkeypatch, caplog, retry_timeout, retry_wait
):
    monkeypatch.setattr(CommConfigurator, "get_streaming_retry_timeout", lambda self, default: retry_timeout)
    monkeypatch.setattr(CommConfigurator, "get_streaming_retry_wait", lambda self, default: retry_wait)
    cell = SimpleNamespace(fire_and_forget=lambda *a, **kw: {})
    message = _final(7)
    task = RxTask.find_or_create_task(message, cell)
    assert task.completed_task_ttl == 65
    assert "invalid local retry window" in caplog.text
    assert task.process_chunk(message)
    failed = RxTask.find_or_create_task(_final(8, reliable=False), cell)
    failed.stop(StreamError("failed"), notify=False)
    _advance(cleanup, 5)
    wait_for(lambda: ("peer", 8) not in RxTask.rx_task_map)
    assert RxTask.rx_task_map[("peer", 7)] is task
    assert RxTask.get_cleanup_stats()["worker_alive"]
    _advance(cleanup, 60)
    wait_for(lambda: not RxTask.rx_task_map)
