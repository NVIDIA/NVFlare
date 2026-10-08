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
