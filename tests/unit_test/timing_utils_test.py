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
from types import SimpleNamespace

import pytest

from tests.timing_utils import (
    WAIT_TIMEOUT,
    CheckedThread,
    ManualClock,
    ObservedLock,
    ThreadGroup,
    isolate_time,
    join_thread,
    stop_processes,
    wait_for,
)


def test_clock_patch_does_not_change_stdlib_or_unrelated_workers(monkeypatch):
    real_time = time.time
    first = SimpleNamespace(time=time)
    second = SimpleNamespace(time=time)
    clock = isolate_time(monkeypatch, first, second)
    monkeypatch.setattr(clock, "time", lambda: 12.0)
    assert first.time.time() == second.time.time() == 12.0
    assert time.time is real_time
    assert time.time() > 12.0


def test_manual_clock_advances_only_on_explicit_request():
    clock = ManualClock()
    clock.sleep(0.01)
    assert clock.monotonic() == clock.time() == 1000.0
    clock.advance(15.0)
    assert clock.monotonic() == clock.time() == 1015.0


def test_wait_guard_is_not_frozen_by_a_shared_time_patch(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(AssertionError, match="missing notification"):
        wait_for(lambda: False, timeout=0.01, message="missing notification")


def test_worker_error_is_reported_on_join():
    def fail():
        raise ValueError("worker failed")

    thread = CheckedThread(target=fail)
    thread.start()
    with pytest.raises(ValueError, match="worker failed"):
        join_thread(thread)


def test_missing_worker_completion_is_not_accepted():
    thread = SimpleNamespace(name="blocked worker", join=lambda timeout: None, is_alive=lambda: True)
    with pytest.raises(AssertionError, match="blocked worker did not stop"):
        join_thread(thread, timeout=0)


def test_thread_group_releases_and_joins_workers_when_body_fails():
    release = threading.Event()
    completed = threading.Event()

    def work():
        assert release.wait(WAIT_TIMEOUT)
        completed.set()

    with pytest.raises(ValueError, match="test body failed"):
        with ThreadGroup(release) as workers:
            thread = workers.start(work)
            raise ValueError("test body failed")

    assert completed.is_set()
    assert not thread.is_alive()


def test_thread_group_joins_other_workers_before_reporting_a_worker_error():
    release = threading.Event()
    completed = threading.Event()

    def fail():
        raise ValueError("background failure")

    with pytest.raises(ValueError, match="background failure"):
        with ThreadGroup(release) as workers:
            workers.start(fail)
            other = workers.start(lambda: (release.wait(WAIT_TIMEOUT), completed.set()))
    assert completed.is_set()
    assert not other.is_alive()


def test_partial_thread_start_releases_and_joins_started_worker(monkeypatch):
    original_start = CheckedThread.start
    release = threading.Event()
    started = []

    def start(thread):
        if started:
            raise RuntimeError("controlled startup failure")
        original_start(thread)
        started.append(thread)

    monkeypatch.setattr(CheckedThread, "start", start)
    with pytest.raises(RuntimeError, match="controlled startup failure"):
        with ThreadGroup(release) as workers:
            workers.start(release.wait)
            workers.start(lambda: None)
    assert len(started) == 1
    assert not started[0].is_alive()


def test_observed_lock_requires_an_actual_failed_acquisition():
    lock = ObservedLock()
    with ThreadGroup() as workers:
        with lock:
            # Thread creation alone must not be accepted as proof of contention.
            assert not lock.blocked.is_set()

            def acquire():
                with lock:
                    pass

            workers.start(acquire)
            assert lock.blocked.wait(WAIT_TIMEOUT)


@pytest.mark.parametrize("raw_lock", [threading.Lock, threading.RLock])
def test_observed_lock_preserves_nonblocking_and_reentrant_behavior(raw_lock):
    lock = ObservedLock(raw_lock())
    assert lock.acquire(blocking=False)
    try:
        acquired_again = lock.acquire(blocking=False)
        assert acquired_again is (raw_lock is threading.RLock)
        if acquired_again:
            lock.release()
    finally:
        lock.release()


def test_stop_processes_escalates_only_for_owned_survivors():
    calls = []

    class Child:
        pid = 123
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            calls.append("join")

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")
            self.alive = False

    child = Child()
    stop_processes(child, timeout=0)
    assert calls == ["join", "terminate", "join", "kill", "join"]
    assert not child.is_alive()


def test_stop_processes_attempts_other_children_after_a_cleanup_error():
    closed = []

    class Child:
        pid = 123
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            pass

        def terminate(self):
            raise RuntimeError("terminate failed")

        def kill(self):
            closed.append(self)
            self.alive = False

    first, second = Child(), Child()
    with pytest.raises(RuntimeError, match="terminate failed"):
        stop_processes(first, second, timeout=0)
    assert closed == [first, second]
