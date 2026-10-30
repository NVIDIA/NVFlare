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

from decimal import Decimal

import pytest

from nvflare.private.fed.resource_stats import collector
from nvflare.private.fed.resource_stats.contract import (
    U128_MAX,
    ContractError,
    capacity_to_resource_time,
    derive_job_totals,
    participant_measurements,
    validate_record,
)
from nvflare.private.fed.resource_stats.handoff import read_terminal_handoff, write_terminal_handoff


def _collect(monkeypatch, tmp_path, *, ticks=(0, 10_000_000_000), capacity_failure=False, cpu_failure=False):
    events = []
    times = iter(ticks)
    usages = iter(((Decimal(100), Decimal(40)), (Decimal(102), Decimal(40))))

    def clock():
        events.append("clock")
        value = next(times)
        if isinstance(value, Exception):
            raise value
        return value

    def capacity():
        events.append("capacity")
        if capacity_failure:
            raise OSError("capacity unavailable")
        return {"cpu": {"groups": [{"units": "8"}]}, "memory": {"bytes": "4096"}, "gpu": {"groups": []}}

    def usage():
        events.append("cpu")
        if cpu_failure:
            raise OSError("CPU accounting unavailable")
        return next(usages)

    original = collector.CpuConsumedAccountant
    monkeypatch.setattr(collector, "CpuConsumedAccountant", lambda **kwargs: original(usage_reader=usage, **kwargs))
    job = collector.JobResourceCollector(tmp_path, clock_ns=clock, capacity_probe=capacity)
    handoff = job.finish(message_traffic={"sent_bytes": "0", "received_bytes": "0"})
    write_terminal_handoff(tmp_path, handoff)
    report = collector.assemble_participant_summary(
        job_id="job-1", participant_name="site-1", child_handoff=read_terminal_handoff(tmp_path)
    )
    return report, events


def _entry(report, name="site-1"):
    return {
        "participant_name": name,
        "role": "client",
        "received_at": report["reported_at"],
        **participant_measurements(report),
    }


def test_one_execution_clock_flows_through_handoff_report_and_query(monkeypatch, tmp_path):
    report, events = _collect(monkeypatch, tmp_path, ticks=(9_000_000_000_000_000_001, 9_000_000_010_000_000_001))
    assert events == ["capacity", "clock", "cpu", "cpu", "clock"]
    assert report["duration_seconds"] == "10"
    assert report["resource_utilization"]["cpu_consumed"] == {"cpu_seconds": "2"}
    assert "duration_seconds" not in report["resource_capacity"]
    totals = derive_job_totals([_entry(report)])
    assert totals["resource_utilization"]["cpu_consumed"] == {"cpu_seconds": "2"}
    assert totals["resource_time"]["cpu"]["groups"] == [{"unit_seconds": "80"}]
    assert "duration_seconds" not in totals["resource_time"]


@pytest.mark.parametrize("capacity_failure,cpu_failure", [(True, False), (False, True), (True, True)])
def test_failed_probe_does_not_remove_shared_duration(monkeypatch, tmp_path, capacity_failure, cpu_failure):
    report, _ = _collect(monkeypatch, tmp_path, capacity_failure=capacity_failure, cpu_failure=cpu_failure)
    assert report["duration_seconds"] == "10"
    totals = derive_job_totals([_entry(report)])
    assert (totals["resource_time"].get("status") == "unavailable") == capacity_failure
    assert (totals["resource_utilization"]["cpu_consumed"].get("status") == "unavailable") == cpu_failure


@pytest.mark.parametrize("ticks", [(10, 9), (None, 10), (0, True), (0, -1), (OSError("clock"), 10), (0, 2**160)])
def test_bad_clock_preserves_raw_quantities_without_inventing_duration(monkeypatch, tmp_path, ticks):
    report, _ = _collect(monkeypatch, tmp_path, ticks=ticks)
    assert "duration_seconds" not in report
    assert report["resource_utilization"]["cpu_consumed"] == {"cpu_seconds": "2"}
    assert report["resource_capacity"]["cpu"]["groups"] == [{"units": "8"}]
    assert derive_job_totals([_entry(report)])["resource_time"]["status"] == "unavailable"


def test_zero_shared_duration_is_retained(monkeypatch, tmp_path):
    report, _ = _collect(monkeypatch, tmp_path, ticks=(10, 10))
    assert report["duration_seconds"] == "0"
    assert derive_job_totals([_entry(report)])["resource_time"]["cpu"]["groups"] == [{"unit_seconds": "0"}]


def test_collector_cannot_extend_a_finalized_window(tmp_path):
    ticks = iter((0, 10_000_000_000))
    job = collector.JobResourceCollector(
        tmp_path, clock_ns=lambda: next(ticks), capacity_probe=lambda: {"cpu": {"groups": [{"units": "8"}]}}
    )
    assert job.finish()["duration_seconds"] == "10"
    with pytest.raises(RuntimeError, match="already finalized"):
        job.finish()


def test_totals_sum_explicit_time_without_requiring_participant_duration(monkeypatch, tmp_path):
    report, _ = _collect(monkeypatch, tmp_path)
    first, second, third = (_entry(report, name) for name in ("site-1", "site-2", "site-3"))
    second["duration_seconds"] = "20"
    second["resource_time"] = capacity_to_resource_time(second["resource_capacity"], "20")
    second["resource_utilization"]["cpu_consumed"] = {"status": "unavailable", "issues": ["observation_incomplete"]}
    third.pop("duration_seconds")
    totals = derive_job_totals([first, second, third])
    assert totals["resource_utilization"]["cpu_consumed"] == {"status": "partial", "cpu_seconds": "4"}
    assert totals["resource_time"]["cpu"]["groups"] == [{"unit_seconds": "320"}]
    assert "status" not in totals["resource_time"]
    assert "duration_seconds" not in str(totals)


@pytest.mark.parametrize("field", ["resource_capacity", "cpu_consumed"])
@pytest.mark.parametrize("key", ["duration_seconds", "measured_seconds"])
def test_nested_durations_are_rejected(monkeypatch, tmp_path, field, key):
    report, _ = _collect(monkeypatch, tmp_path)
    measurement = report if field == "resource_capacity" else report["resource_utilization"]
    measurement[field][key] = "10"
    with pytest.raises(ContractError, match="unexpected fields"):
        validate_record(report)


def test_unused_capacity_timeline_is_rejected(monkeypatch, tmp_path):
    report, _ = _collect(monkeypatch, tmp_path)
    report["resource_capacity"] = {"intervals": [{"duration_seconds": "10", "cpu": {"groups": [{"units": "8"}]}}]}
    with pytest.raises(ContractError):
        validate_record(report)


@pytest.mark.parametrize("duration", [None, True, "-1", "NaN", "1e3", 10])
def test_malformed_handoff_duration_does_not_erase_cpu_work(monkeypatch, tmp_path, duration):
    _collect(monkeypatch, tmp_path)
    handoff = read_terminal_handoff(tmp_path)
    handoff["duration_seconds"] = duration
    write_terminal_handoff(tmp_path, handoff)
    checked = read_terminal_handoff(tmp_path)
    assert "duration_seconds" not in checked
    assert checked["resource_utilization"]["cpu_consumed"] == {"cpu_seconds": "2"}
    assert checked["resource_capacity"]["cpu"]["groups"] == [{"units": "8"}]


def test_handoff_does_not_recalculate_explicit_time_from_participant_duration(monkeypatch, tmp_path):
    _collect(monkeypatch, tmp_path)
    handoff = read_terminal_handoff(tmp_path)
    handoff["duration_seconds"] = str(U128_MAX)
    write_terminal_handoff(tmp_path, handoff)
    checked = read_terminal_handoff(tmp_path)
    assert checked["duration_seconds"] == str(U128_MAX)
    assert checked["resource_capacity"]["cpu"]["groups"] == [{"units": "8"}]
    assert checked["resource_time"]["cpu"]["groups"] == [{"unit_seconds": "80"}]
    assert checked["resource_utilization"]["cpu_consumed"] == {"cpu_seconds": "2"}


"""Validate current participant, job, and study records at the report boundary."""
