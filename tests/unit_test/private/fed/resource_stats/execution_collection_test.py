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

"""Execution windows are shared, immutable and independent of file scans."""

from decimal import Decimal

import pytest

from nvflare.private.fed.resource_stats import collector
from nvflare.private.fed.resource_stats import cpu_consumed as cpu_module
from nvflare.private.fed.resource_stats.collector import ExecutionResourceCollector
from nvflare.private.fed.resource_stats.contract import U128_MAX
from nvflare.private.fed.resource_stats.cpu_consumed import CpuConsumedAccountant


def _usage(*values):
    snapshots = iter(tuple(Decimal(item) for item in value) for value in values)
    return lambda: next(snapshots)


@pytest.fixture(autouse=True)
def isolate_accounting_owners(monkeypatch):
    monkeypatch.setattr(cpu_module, "_active", None)


def test_execution_installs_its_own_managed_descendant_coverage(monkeypatch):
    accountant = CpuConsumedAccountant(usage_reader=_usage(("0", "0"), ("2", "0")))
    monkeypatch.setattr(collector, "CpuConsumedAccountant", lambda **_kwargs: accountant)
    instance = ExecutionResourceCollector(capacity_probe=lambda: {}, clock_ns=lambda: 0)
    cpu_module.mark_unwaited_managed_children()
    assert instance.stop()["cpu_consumed"] == {
        "status": "partial",
        "issues": ["attribution_incomplete"],
        "cpu_seconds": "2",
    }


def test_coordinator_self_cpu_excludes_normally_reaped_worker_cpu():
    accountant = CpuConsumedAccountant(usage_reader=_usage(("100", "40"), ("102", "52")), include_children=False)
    assert accountant.finish() == {"cpu_seconds": "2"}


def test_selecting_task_mode_retains_the_existing_cpu_baseline():
    accountant = CpuConsumedAccountant(usage_reader=_usage(("100", "40"), ("102", "52")))
    accountant.exclude_children()
    assert accountant.finish() == {"cpu_seconds": "2"}


def test_execution_stop_is_one_snapshot_and_does_not_scan_files(monkeypatch):
    clock = iter([0, 10_000_000_000])
    accountant = CpuConsumedAccountant(usage_reader=_usage(("0", "0"), ("2", "0")))
    monkeypatch.setattr(collector, "CpuConsumedAccountant", lambda **_kwargs: accountant)

    def unexpected_scan(_path):
        raise AssertionError("execution collectors must not scan files")

    monkeypatch.setattr(collector, "observe_retained_content", unexpected_scan)
    instance = ExecutionResourceCollector(
        clock_ns=lambda: next(clock),
        capacity_probe=lambda: {
            "cpu": {"groups": [{"units": "2"}]},
            "memory": {"bytes": "1024"},
            "gpu": {"groups": []},
        },
    )
    first = instance.stop()
    assert first["duration_seconds"] == "10"
    assert first["cpu_consumed"] == {"cpu_seconds": "2"}
    first["resource_capacity"]["cpu"]["groups"][0]["units"] = "999"
    assert instance.stop()["resource_capacity"]["cpu"]["groups"] == [{"units": "2"}]


def test_bad_capacity_and_clock_do_not_erase_cpu_work(monkeypatch):
    accountant = CpuConsumedAccountant(usage_reader=_usage(("0", "0"), ("2", "1")))
    monkeypatch.setattr(collector, "CpuConsumedAccountant", lambda **_kwargs: accountant)

    def bad_probe():
        raise OSError("unavailable")

    instance = ExecutionResourceCollector(capacity_probe=bad_probe, clock_ns=lambda: -1)
    snapshot = instance.stop()
    assert "duration_seconds" not in snapshot
    assert snapshot["resource_capacity"]["status"] == "unavailable"
    assert snapshot["cpu_consumed"] == {"cpu_seconds": "3"}


def test_task_mode_keeps_one_window_and_installs_reducer_before_workers(tmp_path, monkeypatch):
    accountant = CpuConsumedAccountant(usage_reader=_usage(("0", "0"), ("2", "12")))
    monkeypatch.setattr(collector, "CpuConsumedAccountant", lambda **_kwargs: accountant)
    monkeypatch.setattr(collector, "_active_collector", None)
    installed = []
    monkeypatch.setattr(collector, "install_task_resource_reducer", installed.append)
    ticks = iter([0, 60_000_000_000])
    instance = collector.JobResourceCollector(
        tmp_path,
        clock_ns=lambda: next(ticks),
        capacity_probe=lambda: {
            "cpu": {"groups": [{"units": "2"}]},
            "memory": {"bytes": "1024"},
            "gpu": {"groups": []},
        },
    )
    collector.configure_task_collection()
    reducer = instance._task_reducer
    collector.configure_task_collection()
    assert instance._task_reducer is reducer
    assert installed == [None, reducer]
    assert instance.stop()["duration_seconds"] == "60"
    assert instance.stop()["cpu_consumed"] == {"cpu_seconds": "2"}


@pytest.mark.parametrize("task_based", [False, True])
def test_task_statistics_setup_failure_is_best_effort(tmp_path, monkeypatch, task_based):
    accountant = CpuConsumedAccountant(usage_reader=_usage(("0", "0"), ("2", "12")))
    monkeypatch.setattr(collector, "CpuConsumedAccountant", lambda **_kwargs: accountant)
    monkeypatch.setattr(collector, "_active_collector", None)

    def failed_reducer(_self):
        raise OSError("optional statistics setup failed")

    monkeypatch.setattr(collector.TaskResourceReducer, "__init__", failed_reducer)
    monkeypatch.setattr(collector, "observe_workspace_filesystem", lambda _path: {"capacity_bytes": "1024"})
    monkeypatch.setattr(collector, "observe_retained_content", lambda _path: {"bytes": "0"})
    ticks = iter([0, 60_000_000_000])
    capacity = {
        "cpu": {"groups": [{"units": "2"}]},
        "memory": {"bytes": "1024"},
        "gpu": {"groups": []},
    }
    instance = collector.JobResourceCollector(
        tmp_path, task_based=task_based, capacity_probe=lambda: capacity, clock_ns=lambda: next(ticks)
    )
    if not task_based:
        collector.configure_task_collection()
    report = instance.finish(message_traffic={"sent_bytes": "7", "received_bytes": "11"})
    assert report["resource_capacity"] == capacity
    assert "duration_seconds" not in report
    assert report["resource_time"] == {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "cpu": {"groups": [{"unit_seconds": "120"}]},
        "memory": {"byte_seconds": "61440"},
        "gpu": {"groups": []},
    }
    assert report["resource_utilization"]["cpu_consumed"] == {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "cpu_seconds": "2",
    }
    assert report["resource_utilization"]["message_traffic"] == {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "sent_bytes": "7",
        "received_bytes": "11",
    }
    assert report["resource_utilization"]["retained_content"] == {"bytes": "0"}


def test_final_worker_fold_keeps_cpu_and_gpu_when_memory_total_overflows():
    coordinator = {
        "cpu": {"groups": [{"unit_seconds": "1"}]},
        "memory": {"byte_seconds": str(U128_MAX)},
        "gpu": {"groups": []},
    }
    workers = {
        "cpu": {"groups": [{"unit_seconds": "2"}]},
        "memory": {"byte_seconds": "1"},
        "gpu": {"groups": [{"kind": "cuda_device_unclassified", "instance_seconds": "15"}]},
    }
    result = collector._fold_measurements("resource_time", coordinator, workers)
    assert result == {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "cpu": {"groups": [{"unit_seconds": "3"}]},
        "memory": {"byte_seconds": str(U128_MAX)},
        "gpu": {"groups": [{"kind": "cuda_device_unclassified", "instance_seconds": "15"}]},
    }


@pytest.mark.parametrize(
    "gpu_value",
    [{"groups": [{"kind": "full_gpu", "count": 2}]}, None, {"groups": [{"kind": "full_gpu", "count": True}]}],
)
def test_gpu_completion_follows_cpu_and_duration_boundary(monkeypatch, gpu_value):
    events = []

    class PendingGpu:
        def __init__(self):
            events.append("start-gpu")

        def result(self):
            events.append("wait-gpu")
            return gpu_value

        def close(self):
            events.append("close-gpu")

    def usage():
        events.append("cpu")
        return Decimal(0), Decimal(0)

    def clock():
        events.append("clock")
        return 0

    monkeypatch.setattr(collector.gpu_probe_process, "GpuProbe", PendingGpu)
    monkeypatch.setattr(collector.cgroup_linux, "probe_cpu", lambda *a, **kw: {"groups": [{"units": "2"}]})
    monkeypatch.setattr(collector.cgroup_linux, "probe_memory", lambda *a, **kw: {"bytes": "1024"})
    monkeypatch.setattr(
        collector, "CpuConsumedAccountant", lambda **kw: CpuConsumedAccountant(usage_reader=usage, **kw)
    )
    instance = ExecutionResourceCollector(clock_ns=clock)
    assert events == ["start-gpu", "clock", "cpu"]
    snapshot = instance.stop()
    assert events == ["start-gpu", "clock", "cpu", "cpu", "clock", "wait-gpu", "close-gpu"]
    assert snapshot["resource_capacity"]["memory"] == {"bytes": "1024"}
    if gpu_value is not None and type(gpu_value["groups"][0]["count"]) is int:
        assert snapshot["resource_capacity"]["gpu"]["groups"][0]["count"] == "2"
    else:
        assert "gpu" not in snapshot["resource_capacity"]
        assert snapshot["resource_capacity"]["status"] == "partial"
    instance.stop()
    assert events.count("wait-gpu") == events.count("close-gpu") == 1
