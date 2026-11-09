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

"""CPU accounting policy with controlled usage counters and attribution gaps."""


from nvflare.private.fed.resource_stats import cpu_consumed as cpu_module
from nvflare.private.fed.resource_stats.cpu_consumed import CpuConsumedAccountant


def _snapshots(start, end):
    values = iter((tuple((float(item) for item in start)), tuple((float(item) for item in end))))
    return lambda: next(values)


def test_self_and_waited_children_are_added_once_without_parent_cpu():
    # The first delta is SELF, the second is the OS's already-inclusive
    # RUSAGE_CHILDREN. Individual worker reports must not be added again.
    accountant = CpuConsumedAccountant(usage_reader=_snapshots(("100", "40"), ("101.5", "42.25")))
    assert accountant.finish() == {"cpu_seconds": 3.75}


def test_managed_children_gap_keeps_the_observed_subtotal_partial(monkeypatch):
    accountant = CpuConsumedAccountant(usage_reader=_snapshots(("0", "0"), ("1", "0.25")))
    monkeypatch.setattr(cpu_module, "_active", accountant)
    cpu_module.mark_unwaited_managed_children()
    assert accountant.finish() == {"status": "partial", "issues": ["attribution_incomplete"], "cpu_seconds": 1.25}


def test_missing_counter_and_prior_attempt_have_independent_status():
    def broken():
        raise OSError("counter unavailable")

    assert CpuConsumedAccountant(usage_reader=broken).finish() == {
        "status": "unavailable",
        "issues": ["observation_incomplete"],
    }
    restored = CpuConsumedAccountant(usage_reader=_snapshots(("0", "0"), ("1", "0")), prior_observation_incomplete=True)
    assert restored.finish() == {"status": "partial", "issues": ["observation_incomplete"], "cpu_seconds": 1}
