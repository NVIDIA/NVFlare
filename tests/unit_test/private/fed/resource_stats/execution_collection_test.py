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


import pytest

from nvflare.private.fed.resource_stats import collector
from nvflare.private.fed.resource_stats import cpu_consumed as cpu_module
from nvflare.private.fed.resource_stats.collector import ExecutionResourceCollector
from nvflare.private.fed.resource_stats.cpu_consumed import CpuConsumedAccountant


def _usage(*values):
    snapshots = iter((tuple((float(item) for item in value)) for value in values))
    return lambda: next(snapshots)


@pytest.fixture(autouse=True)
def isolate_accounting_owners(monkeypatch):
    monkeypatch.setattr(cpu_module, "_active", None)


def test_execution_installs_its_own_managed_descendant_coverage(monkeypatch):
    accountant = CpuConsumedAccountant(usage_reader=_usage(("0", "0"), ("2", "0")))
    monkeypatch.setattr(collector, "CpuConsumedAccountant", lambda **_kwargs: accountant)
    instance = ExecutionResourceCollector(capacity_probe=lambda: {}, clock_ns=lambda: 0)
    cpu_module.mark_unwaited_managed_children()
    assert instance.stop()["cpu_consumed"] == {"status": "partial", "cpu_seconds": 2}
    assert [issue["code"] for issue in instance.stop()["issues"] if issue["metric"] == "cpu_consumed"] == [
        "attribution_incomplete"
    ]


def test_coordinator_self_cpu_excludes_normally_reaped_worker_cpu():
    accountant = CpuConsumedAccountant(usage_reader=_usage(("100", "40"), ("102", "52")), include_children=False)
    assert accountant.finish() == {"cpu_seconds": 2}


def test_selecting_task_mode_retains_the_existing_cpu_baseline():
    accountant = CpuConsumedAccountant(usage_reader=_usage(("100", "40"), ("102", "52")))
    accountant.exclude_children()
    assert accountant.finish() == {"cpu_seconds": 2}


def test_execution_stop_is_one_snapshot_and_does_not_scan_files(monkeypatch):
    clock = iter([0, 10_000_000_000])
    accountant = CpuConsumedAccountant(usage_reader=_usage(("0", "0"), ("2", "0")))
    monkeypatch.setattr(collector, "CpuConsumedAccountant", lambda **_kwargs: accountant)

    def unexpected_scan(_path):
        raise AssertionError("execution collectors must not scan files")

    monkeypatch.setattr(collector, "observe_retained_content", unexpected_scan)
    instance = ExecutionResourceCollector(
        clock_ns=lambda: next(clock),
        capacity_probe=lambda: {"cpu": {"units": 2}, "memory": {"bytes": 1024}, "gpu": {"count": 0}},
    )
    first = instance.stop()
    assert first["duration_seconds"] == 10
    assert first["cpu_consumed"] == {"cpu_seconds": 2}
    first["resource_capacity"]["cpu"]["units"] = "999"
    assert instance.stop()["resource_capacity"]["cpu"] == {"units": 2}


def test_bad_capacity_and_clock_do_not_erase_cpu_work(monkeypatch):
    accountant = CpuConsumedAccountant(usage_reader=_usage(("0", "0"), ("2", "1")))
    monkeypatch.setattr(collector, "CpuConsumedAccountant", lambda **_kwargs: accountant)

    def bad_probe():
        raise OSError("unavailable")

    instance = ExecutionResourceCollector(capacity_probe=bad_probe, clock_ns=lambda: -1)
    snapshot = instance.stop()
    assert "duration_seconds" not in snapshot
    assert snapshot["resource_capacity"]["status"] == "unavailable"
    assert snapshot["cpu_consumed"] == {"cpu_seconds": 3}


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
        capacity_probe=lambda: {"cpu": {"units": 2}, "memory": {"bytes": 1024}, "gpu": {"count": 0}},
    )
    collector.configure_task_collection()
    reducer = instance._task_reducer
    collector.configure_task_collection()
    assert instance._task_reducer is reducer
    assert installed == [None, reducer]
    assert instance.stop()["duration_seconds"] == 60
    assert instance.stop()["cpu_consumed"] == {"cpu_seconds": 2}


@pytest.mark.parametrize("task_based", [False, True])
def test_task_statistics_setup_failure_is_best_effort(tmp_path, monkeypatch, task_based):
    accountant = CpuConsumedAccountant(usage_reader=_usage(("0", "0"), ("2", "12")))
    monkeypatch.setattr(collector, "CpuConsumedAccountant", lambda **_kwargs: accountant)
    monkeypatch.setattr(collector, "_active_collector", None)

    def failed_reducer(_self):
        raise OSError("optional statistics setup failed")

    monkeypatch.setattr(collector.TaskResourceReducer, "__init__", failed_reducer)
    monkeypatch.setattr(collector, "observe_workspace_filesystem", lambda _path: {"capacity_bytes": 1024})
    monkeypatch.setattr(collector, "observe_retained_content", lambda _path: {"bytes": 0})
    ticks = iter([0, 60_000_000_000])
    capacity = {"cpu": {"units": 2}, "memory": {"bytes": 1024}, "gpu": {"count": 0}}
    instance = collector.JobResourceCollector(
        tmp_path, task_based=task_based, capacity_probe=lambda: capacity, clock_ns=lambda: next(ticks)
    )
    if not task_based:
        collector.configure_task_collection()
    report = instance.finish(message_traffic={"sent_bytes": 7, "received_bytes": 11})
    assert report["resource_capacity"] == capacity
    assert "duration_seconds" not in report
    assert report["resource_time"] == {
        "status": "partial",
        "cpu": {"unit_seconds": 120},
        "memory": {"byte_seconds": 61440},
        "gpu": {"instance_seconds": 0},
    }
    assert [issue["code"] for issue in report["issues"] if issue["metric"] == "resource_time"] == [
        "observation_incomplete"
    ]
    assert report["resource_utilization"]["cpu_consumed"] == {"status": "partial", "cpu_seconds": 2}
    assert [issue["code"] for issue in report["issues"] if issue["metric"] == "cpu_consumed"] == [
        "observation_incomplete"
    ]
    assert report["resource_utilization"]["message_traffic"] == {
        "status": "partial",
        "sent_bytes": 7,
        "received_bytes": 11,
    }
    assert [issue["code"] for issue in report["issues"] if issue["metric"] == "message_traffic"] == [
        "observation_incomplete"
    ]
    assert report["resource_utilization"]["retained_content"] == {"bytes": 0}


def test_final_worker_fold_keeps_cpu_and_gpu_when_memory_total_overflows():
    coordinator = {"cpu": {"unit_seconds": 1}, "memory": {"byte_seconds": 1e308}, "gpu": {"instance_seconds": 0}}
    workers = {"cpu": {"unit_seconds": 2}, "memory": {"byte_seconds": 1e308}, "gpu": {"instance_seconds": 15}}
    issues = []
    result = collector._fold_measurements("resource_time", coordinator, workers, issues)
    assert result == {
        "status": "partial",
        "cpu": {"unit_seconds": 3},
        "memory": {"byte_seconds": 1e308},
        "gpu": {"instance_seconds": 15},
    }
    assert issues == [{"metric": "resource_time", "code": "observation_incomplete"}]


@pytest.mark.parametrize(
    "gpu_value", [{"count": value} for value in (2, 0, "2", True)] + [None, OSError("query failed")]
)
def test_gpu_snapshot_precedes_cpu_and_duration_baselines(monkeypatch, gpu_value):
    events = []
    counters = [0.0, 0.0]
    elapsed_ns = 0
    expected_count = gpu_value.get("count") if isinstance(gpu_value, dict) else None

    def gpu_query():
        nonlocal elapsed_ns
        events.append("gpu")
        counters[0] += 7.0
        counters[1] += 11.0
        elapsed_ns += 2_000_000_000
        if isinstance(gpu_value, Exception):
            raise gpu_value
        return gpu_value

    def usage():
        events.append("cpu")
        return tuple(counters)

    def clock():
        events.append("clock")
        return elapsed_ns

    monkeypatch.setattr(collector.gpu_nvidia, "probe_gpu", gpu_query)
    monkeypatch.setattr(collector.cgroup_linux, "probe_cpu", lambda *a, **kw: {"units": 2})
    monkeypatch.setattr(collector.cgroup_linux, "probe_memory", lambda *a, **kw: {"bytes": 1024})
    monkeypatch.setattr(
        collector, "CpuConsumedAccountant", lambda **kw: CpuConsumedAccountant(usage_reader=usage, **kw)
    )
    instance = ExecutionResourceCollector(clock_ns=clock)
    assert events == ["gpu", "clock", "cpu"]

    # Later environment/provider changes do not resize the captured startup count.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "changed-after-startup")
    if isinstance(gpu_value, dict):
        gpu_value["count"] = 99
    counters[0] += 1.0
    counters[1] += 2.0
    elapsed_ns += 1_000_000_000
    snapshot = instance.stop()

    assert events == ["gpu", "clock", "cpu", "cpu", "clock"]
    assert snapshot["duration_seconds"] == 1
    assert snapshot["cpu_consumed"] == {"cpu_seconds": 3}
    assert snapshot["resource_capacity"]["memory"] == {"bytes": 1024}
    if type(expected_count) is int:
        assert snapshot["resource_capacity"]["gpu"] == {"count": expected_count}
    else:
        assert "gpu" not in snapshot["resource_capacity"]
        assert snapshot["resource_capacity"]["status"] == "partial"
        assert snapshot["issues"] == [{"metric": "resource_capacity", "code": "observation_incomplete"}]
    assert instance.stop() == snapshot
    assert events == ["gpu", "clock", "cpu", "cpu", "clock"]
