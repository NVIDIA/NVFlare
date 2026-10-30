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

"""CPU accounting policy with controlled usage counters and child handles."""

from decimal import Decimal

from nvflare.private.fed.resource_stats.cpu_consumed import CpuConsumedAccountant


def _snapshots(start, end):
    values = iter((tuple(Decimal(item) for item in start), tuple(Decimal(item) for item in end)))
    return lambda: next(values)


class _Child:
    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def test_self_and_waited_children_are_added_once_without_parent_cpu():
    # The first delta is SELF, the second is the OS's already-inclusive
    # RUSAGE_CHILDREN. Individual worker reports must not be added again.
    accountant = CpuConsumedAccountant(usage_reader=_snapshots(("100", "40"), ("101.5", "42.25")))
    accountant.register_child(_Child(0))
    accountant.register_child(_Child(0))
    assert accountant.finish() == {"cpu_seconds": "3.75"}


def test_known_live_or_unwaited_worker_is_not_reported_as_complete():
    accountant = CpuConsumedAccountant(usage_reader=_snapshots(("0", "0"), ("1", "0.25")))
    accountant.register_child(_Child(None))
    assert accountant.finish() == {
        "status": "partial",
        "issues": ["attribution_incomplete"],
        "cpu_seconds": "1.25",
    }

    accountant = CpuConsumedAccountant(usage_reader=_snapshots(("0", "0"), ("1", "1")))
    accountant.register_child(_Child(0), descendants_may_be_unwaited=True)
    assert accountant.finish() == {
        "status": "partial",
        "issues": ["attribution_incomplete"],
        "cpu_seconds": "2",
    }


def test_missing_counter_and_prior_attempt_have_independent_status():
    def broken():
        raise OSError("counter unavailable")

    assert CpuConsumedAccountant(usage_reader=broken).finish() == {
        "status": "unavailable",
        "issues": ["observation_incomplete"],
    }
    restored = CpuConsumedAccountant(usage_reader=_snapshots(("0", "0"), ("1", "0")), prior_observation_incomplete=True)
    assert restored.finish() == {"status": "partial", "issues": ["observation_incomplete"], "cpu_seconds": "1"}
