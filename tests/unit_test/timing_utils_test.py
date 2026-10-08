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

import time
from types import SimpleNamespace

import pytest

from tests.timing_utils import CheckedThread, ManualClock, isolate_time, join_thread, wait_for


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
