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

import pytest

from nvflare.fuel.f3.streaming import deadline_scheduler
from nvflare.fuel.f3.streaming.deadline_scheduler import DeadlineScheduler
from tests.timing_utils import WAIT_TIMEOUT, ManualClock, ThreadGroup, wait_for


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.setattr(deadline_scheduler, "time", ManualClock())
    scheduler = DeadlineScheduler()
    try:
        yield scheduler
    finally:
        scheduler.shutdown()
        assert not scheduler.snapshot()["worker_alive"]


def _advance(scheduler, seconds):
    deadline_scheduler.time.advance(seconds)
    with scheduler._condition:
        scheduler._condition.notify_all()


def test_deadlines_preserve_order_and_never_expire_early(scheduler):
    called = []
    scheduler.schedule(10, lambda: called.append("last"))
    scheduler.schedule(5, lambda: called.append("first"))
    scheduler.schedule(5, lambda: called.append("second"))
    _advance(scheduler, 4)
    barrier = threading.Event()
    scheduler.schedule(0, barrier.set)
    assert barrier.wait(WAIT_TIMEOUT)
    assert called == []

    _advance(scheduler, 1)
    wait_for(lambda: len(called) == 2)
    assert called == ["first", "second"]
    assert scheduler.snapshot()["pending"] == 1
    _advance(scheduler, 7)
    wait_for(lambda: scheduler.snapshot()["expired"] == 4)
    assert called == ["first", "second", "last"]
    assert scheduler.snapshot()["max_cleanup_lag"] == 2


def test_cancellation_releases_callback_and_bounds_dead_entries(scheduler):
    class Target:
        def callback(self):
            raise AssertionError("cancelled callback ran")

    scheduler.schedule(100, lambda: None)  # keep a live earlier head during cancellation
    target = Target()
    reference = weakref.ref(target)
    entry = scheduler.schedule(200, target.callback)
    del target
    entry.cancel()
    entry.cancel()
    assert reference() is None
    for _ in range(1000):
        scheduler.schedule(200, lambda: None).cancel()
    stats = scheduler.snapshot()
    assert stats["pending"] == 1
    assert stats["cancelled"] == 1001
    assert stats["queue_depth"] <= 64


def test_callback_failure_does_not_stop_other_deadlines(scheduler, caplog):
    def fail():
        raise ValueError("cleanup failed")

    done = threading.Event()
    scheduler.schedule(0, fail)
    scheduler.schedule(0, done.set)
    assert done.wait(WAIT_TIMEOUT)
    wait_for(lambda: scheduler.snapshot()["expired"] == 2)
    assert "Receive-task deadline cleanup failed" in caplog.text


def test_shutdown_waits_for_claimed_callback_and_discards_queued_work(scheduler, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    joining = threading.Event()
    called = []

    def callback():
        entered.set()
        release.wait()
        called.append("claimed")

    def shutdown():
        scheduler.shutdown()
        returned.set()

    with ThreadGroup(release) as workers:
        scheduler.schedule(0, callback)
        assert entered.wait(WAIT_TIMEOUT)
        scheduler.schedule(1, lambda: called.append("queued"))
        real_join = scheduler._thread.join

        def observed_join():
            joining.set()
            real_join()

        monkeypatch.setattr(scheduler._thread, "join", observed_join)
        workers.start(shutdown)
        assert joining.wait(WAIT_TIMEOUT)
        assert not returned.is_set()
        assert scheduler.schedule(0, lambda: called.append("late")) is None
    assert returned.is_set()
    assert called == ["claimed"]
    assert scheduler.snapshot()["pending"] == scheduler.snapshot()["queue_depth"] == 0
    assert scheduler.snapshot()["cancelled"] == 1


def test_failed_worker_start_can_be_retried(scheduler, monkeypatch):
    start = threading.Thread.start

    def fail(thread):
        raise RuntimeError("can't start new thread")

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", fail)
        with pytest.raises(RuntimeError, match="can't start new thread"):
            scheduler.schedule(1, lambda: None)
    assert scheduler.snapshot()["pending"] == scheduler.snapshot()["queue_depth"] == 0
    assert threading.Thread.start is start
    done = threading.Event()
    scheduler.schedule(0, done.set)
    assert done.wait(WAIT_TIMEOUT)


@pytest.mark.parametrize("interval", [float("inf"), float("-inf"), float("nan"), -1, 10**400, "bad", None])
def test_invalid_interval_does_not_admit_work_or_start_worker(scheduler, interval):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        scheduler.schedule(interval, lambda: None)
    assert scheduler.snapshot()["pending"] == scheduler.snapshot()["queue_depth"] == 0
    assert not scheduler.snapshot()["worker_alive"]
    # Invalid input must not prevent the next valid cleanup from running.
    done = threading.Event()
    scheduler.schedule(0, done.set)
    assert done.wait(WAIT_TIMEOUT)


def test_large_finite_deadline_bounds_wait_and_allows_earlier_cleanup(scheduler, monkeypatch):
    waited = threading.Event()
    waits = []
    called = []
    original_wait = scheduler._condition.wait

    def observe_wait(timeout=None):
        if timeout is not None:
            waits.append(timeout)
            waited.set()
            # Bound the regression too: record a bad timeout rather than letting
            # the old worker raise OverflowError or sleep beyond the test guard.
            timeout = min(timeout, threading.TIMEOUT_MAX)
        return original_wait(timeout)

    monkeypatch.setattr(scheduler._condition, "wait", observe_wait)
    scheduler.schedule(2 * threading.TIMEOUT_MAX, lambda: called.append("long"))
    assert waited.wait(WAIT_TIMEOUT)
    assert all(0 < timeout <= threading.TIMEOUT_MAX for timeout in waits)
    done = threading.Event()
    scheduler.schedule(0, done.set)
    assert done.wait(WAIT_TIMEOUT)
    wait_for(lambda: scheduler.snapshot()["expired"] == 1)
    assert called == []
    assert scheduler.snapshot()["pending"] == 1
    assert scheduler.snapshot()["worker_alive"]
