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

"""Real wait guards and explicitly advanced policy clocks for tests."""

import importlib
import threading
import time
from types import SimpleNamespace

# A test's shared time-module patch must not freeze its own failure guard.
_monotonic = time.monotonic
_sleep = time.sleep
WAIT_TIMEOUT = 10.0


def isolate_time(monkeypatch, *modules):
    """Keep existing clock mocks local to a test's cooperating modules.

    Patching ``some_module.time.monotonic`` otherwise changes the singleton
    stdlib time module, including logging, transports and unrelated workers.
    Cooperating modules share one copy so their timestamp origins still agree.
    Patch the selected module's time reference afterward; patching stdlib
    time.time or time.sleep will not affect the isolated copy.
    """
    local_time = SimpleNamespace(**vars(time))
    for module in modules:
        if isinstance(module, str):
            module = importlib.import_module(module)
        monkeypatch.setattr(module, "time", local_time)
    return local_time


def wait_for(predicate, timeout=WAIT_TIMEOUT, message="condition did not become true"):
    deadline = _monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        remaining = deadline - _monotonic()
        if remaining <= 0:
            raise AssertionError(message)
        _sleep(min(0.01, remaining))


def expire_deadline_when_ready(monkeypatch, module, ready):
    """Let a real child reach readiness before expiring its isolated policy clock.

    The initial deadline starts at zero. Subsequent reads stay there until the
    test observes readiness, then jump to the real monotonic clock. Cleanup uses
    that advancing real clock. An independent real guard bounds missing readiness.
    """
    clock = isolate_time(monkeypatch, module)
    started = _monotonic()
    first_read = True
    expired = False

    def monotonic():
        nonlocal first_read, expired
        if first_read:
            first_read = False
            return 0.0
        if expired or ready():
            expired = True
            return _monotonic()
        if _monotonic() - started >= WAIT_TIMEOUT:
            expired = True
            raise AssertionError("child did not reach readiness before timeout injection")
        return 0.0

    monkeypatch.setattr(clock, "monotonic", monotonic)


def join_thread(thread, timeout=WAIT_TIMEOUT):
    thread.join(timeout=timeout)
    assert not thread.is_alive(), f"test-owned thread {thread.name} did not stop"
    if isinstance(thread, CheckedThread):
        thread.raise_if_failed()


class CheckedThread(threading.Thread):
    """Report worker failures on the test thread, where pytest can observe them."""

    error = None

    def run(self):
        try:
            super().run()
        except BaseException as error:
            self.error = error

    def raise_if_failed(self):
        if self.error is not None:
            raise self.error


class ThreadGroup:
    """Release test gates and join every owned worker, including after an assertion fails."""

    def __init__(self, *release_events, timeout=WAIT_TIMEOUT):
        self.release_events = release_events
        self.timeout = timeout
        self.threads = []

    def __enter__(self):
        return self

    def start(self, target, *args, **kwargs):
        thread = CheckedThread(target=target, args=args, kwargs=kwargs)
        self.threads.append(thread)
        try:
            thread.start()
        except BaseException:
            if thread.ident is None:
                self.threads.remove(thread)
            raise
        return thread

    def __exit__(self, exc_type, exc_value, traceback):
        for event in self.release_events:
            event.set()
        deadline = _monotonic() + self.timeout
        errors = []
        for thread in self.threads:
            try:
                join_thread(thread, timeout=max(0, deadline - _monotonic()))
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0] from exc_value


class ObservedLock:
    """Report actual lock contention instead of guessing that a thread has reached a lock.

    ``blocked`` is set only after a nonblocking acquisition fails. Waiting for
    an event immediately before a protected method call leaves a scheduling
    gap and can accept an implementation that never acquires its lock.
    """

    def __init__(self, lock=None):
        self.lock = lock if lock is not None else threading.Lock()
        self.blocked = threading.Event()

    def acquire(self, blocking=True, timeout=-1):
        if self.lock.acquire(blocking=False):
            return True
        self.blocked.set()
        if not blocking:
            return False
        return self.lock.acquire(timeout=timeout)

    def release(self):
        self.lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def stop_processes(*processes, timeout=WAIT_TIMEOUT):
    """Join, terminate, then kill only the multiprocessing children owned by the test."""
    pending = [process for process in processes if process is not None and process.pid is not None]
    errors = []
    for action in (None, "terminate", "kill"):
        deadline = _monotonic() + timeout
        for process in pending:
            try:
                if action is not None and process.is_alive():
                    getattr(process, action)()
                process.join(timeout=max(0, deadline - _monotonic()))
            except ProcessLookupError:
                pass
            except Exception as error:
                errors.append(error)
        pending = [process for process in pending if process.is_alive()]
        if not pending:
            break
    if pending:
        raise AssertionError(f"test-owned child processes did not stop: {[process.pid for process in pending]}")
    if errors:
        raise RuntimeError(f"test child cleanup failed: {errors}") from errors[0]


class ManualClock:
    """Replace a module's time reference, leaving other modules' clocks alone.

    Time is advanced by the test, never by a background worker. Real sleep and
    the other time functions remain available for cooperative thread scheduling.
    """

    def __init__(self, now=1000.0):
        self._now = now
        self._lock = threading.Lock()

    def time(self):
        with self._lock:
            return self._now

    monotonic = time

    def advance(self, seconds):
        with self._lock:
            self._now += seconds

    def __getattr__(self, name):
        return getattr(time, name)
