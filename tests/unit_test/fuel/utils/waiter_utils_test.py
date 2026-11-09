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

from nvflare.fuel.utils import waiter_utils


class FakeClock:
    def __init__(self, wall_step=0.0):
        self.elapsed = 0.0
        self.polls = 0
        self.wall_step = wall_step

    def monotonic(self):
        return self.elapsed

    def time(self):
        return 100.0 + (self.wall_step if self.polls else 0.0)

    def sleep(self, seconds):
        self.elapsed += seconds
        self.polls += 1
        if self.polls > 10:
            raise AssertionError("wait exceeded its bounded polling budget")


def test_timeout_uses_elapsed_time_after_backward_clock_step(monkeypatch):
    clock = FakeClock(wall_step=-3600)
    monkeypatch.setattr(waiter_utils, "time", clock)

    result = waiter_utils.conditional_wait(waiter=None, timeout=0.25, abort_signal=None)

    assert result == waiter_utils.WaiterRC.TIMEOUT
    assert clock.polls == 3
    assert clock.elapsed == pytest.approx(0.3)


def test_forward_clock_step_does_not_preempt_condition_callback(monkeypatch):
    clock = FakeClock(wall_step=3600)
    monkeypatch.setattr(waiter_utils, "time", clock)

    result = waiter_utils.conditional_wait(waiter=None, timeout=0.5, abort_signal=None, condition_cb=lambda: 42)

    assert result == 42
    assert clock.polls == 1
