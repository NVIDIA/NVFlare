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

import json
import os

import pytest

from nvflare.private.fed.resource_stats import collector
from nvflare.private.fed.resource_stats.collector import (
    ExecutionResourceCollector,
    JobResourceCollector,
    assemble_participant_summary,
    observe_retained_content,
)
from nvflare.private.fed.resource_stats.handoff import (
    RESOURCE_STATS_DIR,
    STAGING_DIR,
    read_terminal_handoff,
    remove_terminal_handoff,
    terminal_handoff_path,
    write_terminal_handoff,
)
from nvflare.private.fed.resource_stats.measurements import canonical_json_bytes, capacity_to_resource_time
from nvflare.private.fed.resource_stats.probes import cgroup_linux, gpu_nvidia
from tests.timing_utils import isolate_time


@pytest.fixture(autouse=True)
def no_native_gpu_queries(monkeypatch):
    monkeypatch.setattr(gpu_nvidia, "probe_gpu", lambda: None)


def test_capacity_retains_numeric_cpu_memory_and_gpu_count():
    instance = ExecutionResourceCollector(
        capacity_probe=lambda: {"cpu": {"units": 4.0}, "memory": {"bytes": 1024}, "gpu": {"count": 2}},
        clock_ns=lambda: 0,
    )

    result = instance.stop()["resource_capacity"]

    assert result["cpu"] == {"units": 4}
    assert result["gpu"] == {"count": 2}
    assert "status" not in result


@pytest.mark.parametrize("count", [-1, 1.5, True, "2"])
def test_gpu_capacity_rejects_non_integer_or_negative_counts(count):
    instance = ExecutionResourceCollector(capacity_probe=lambda: {"gpu": {"count": count}}, clock_ns=lambda: 0)

    snapshot = instance.stop()

    assert snapshot["resource_capacity"] == {"status": "unavailable"}
    assert {"metric": "resource_capacity", "code": "observation_incomplete"} in snapshot["issues"]


def test_fallback_capacity_keeps_numeric_values_and_full_observed_duration():
    instance = ExecutionResourceCollector(
        capacity_probe=lambda: {
            "status": "partial",
            "issues": ["observation_incomplete"],
            "cpu": {"units": 8},
            "memory": {"bytes": 4096},
            "gpu": {"count": 0},
        },
        clock_ns=lambda: 0,
    )

    snapshot = instance.stop()
    result = snapshot["resource_capacity"]

    assert result == {
        "status": "partial",
        "cpu": {"units": 8},
        "memory": {"bytes": 4096},
        "gpu": {"count": 0},
    }
    assert {"metric": "resource_capacity", "code": "observation_incomplete"} in snapshot["issues"]
    assert capacity_to_resource_time(result, 10, snapshot["issues"])["cpu"] == {"unit_seconds": 80}


def test_grouped_cpu_shape_is_not_an_alternate_capacity_format():
    instance = ExecutionResourceCollector(
        capacity_probe=lambda: {"cpu": {"groups": [{"units": 8}]}}, clock_ns=lambda: 0
    )

    assert instance.stop()["resource_capacity"] == {"status": "unavailable"}


def test_cpu_only_host_without_loadable_driver_has_partial_resource_capacity(monkeypatch):
    # Missing inventory does not establish zero GPUs, even on a CPU-only host.
    monkeypatch.setattr(gpu_nvidia, "probe_gpu", lambda: None)
    monkeypatch.setattr(cgroup_linux, "probe_cpu", lambda _discovery, *, issues: {"units": 4})
    monkeypatch.setattr(cgroup_linux, "probe_memory", lambda _discovery, *, issues: {"bytes": 1073741824})

    capacity = collector.probe_capacity()
    assert "gpu" not in capacity

    instance = ExecutionResourceCollector(capacity_probe=lambda: capacity, clock_ns=lambda: 0)
    snapshot = instance.stop()
    result = snapshot["resource_capacity"]

    assert result["status"] == "partial"
    assert {"metric": "resource_capacity", "code": "observation_incomplete"} in snapshot["issues"]
    assert "duration_seconds" not in result
    assert "gpu" not in result


@pytest.mark.parametrize("failed_resource", ["cpu", "memory", "gpu"])
def test_probe_failure_does_not_erase_independent_capacity(monkeypatch, failed_resource):
    expected = {"cpu": {"units": 4}, "memory": {"bytes": 4096}, "gpu": {"count": 0}}
    monkeypatch.setattr(cgroup_linux, "discover_cgroups", lambda: ([], []))
    for resource, value in expected.items():
        module = gpu_nvidia if resource == "gpu" else cgroup_linux
        function = "probe_gpu" if resource == "gpu" else f"probe_{resource}"
        monkeypatch.setattr(module, function, lambda *args, value=value, **kwargs: value)

    def fail_probe(*args, **kwargs):
        raise RuntimeError("simulated probe failure")

    module = gpu_nvidia if failed_resource == "gpu" else cgroup_linux
    function = "probe_gpu" if failed_resource == "gpu" else f"probe_{failed_resource}"
    monkeypatch.setattr(module, function, fail_probe)

    capacity = collector.probe_capacity()

    assert capacity == {
        **{resource: value for resource, value in expected.items() if resource != failed_resource},
        "status": "partial",
        "issues": ["observation_incomplete"],
    }


def test_probe_fallback_remains_partial_through_handoff_and_public_summary(monkeypatch, tmp_path):
    def cpu_fallback(_discovery, *, issues):
        issues.add("observation_incomplete")
        return {"units": 8}

    monkeypatch.setattr(cgroup_linux, "probe_cpu", cpu_fallback)
    monkeypatch.setattr(cgroup_linux, "probe_memory", lambda _discovery, *, issues: {"bytes": 4096})
    monkeypatch.setattr(gpu_nvidia, "probe_gpu", lambda: {"count": 0})
    ticks = iter([0, 10_000_000_000])
    job_collector = JobResourceCollector(tmp_path, clock_ns=lambda: next(ticks))
    write_terminal_handoff(tmp_path, job_collector.finish(message_traffic={"sent_bytes": 0, "received_bytes": 0}))

    report = assemble_participant_summary(
        job_id="job-1",
        participant_name="site-1",
        child_handoff=read_terminal_handoff(tmp_path),
    )

    assert report["resource_capacity"] == {
        "status": "partial",
        "cpu": {"units": 8},
        "memory": {"bytes": 4096},
        "gpu": {"count": 0},
    }
    assert [issue["code"] for issue in report["issues"] if issue["metric"] == "resource_capacity"] == [
        "observation_incomplete"
    ]
    assert report["resource_utilization"]["message_traffic"] == {"sent_bytes": 0, "received_bytes": 0}


def test_handoff_to_public_report_round_trip(tmp_path):
    ticks = iter([1_000_000_000, 3_000_000_000])
    job_collector = JobResourceCollector(
        tmp_path,
        clock_ns=lambda: next(ticks),
        capacity_probe=lambda: {"cpu": {"units": 2}, "memory": {"bytes": 4096}, "gpu": {"count": 0}},
    )

    path = write_terminal_handoff(
        tmp_path, job_collector.finish(message_traffic={"sent_bytes": 1200, "received_bytes": 700})
    )
    assert path == terminal_handoff_path(tmp_path)
    handoff = read_terminal_handoff(tmp_path)
    report = assemble_participant_summary(
        job_id="job-1",
        participant_name="site-1",
        child_handoff=handoff,
    )
    encoded = canonical_json_bytes(report)

    assert json.loads(encoded)["participant_name"] == "site-1"
    assert report["duration_seconds"] == 2
    assert "status" not in report["workspace_filesystem"]
    assert report["resource_utilization"]["retained_content"] == {"bytes": 0}
    assert report["resource_utilization"]["message_traffic"] == {"sent_bytes": 1200, "received_bytes": 700}
    assert "participant_key" not in encoded.decode()


def test_observe_retained_content_sums_regular_files_under_the_run_dir(tmp_path):
    (tmp_path / "output.bin").write_bytes(b"x" * 100)
    nested = tmp_path / "app" / "checkpoints"
    nested.mkdir(parents=True)
    (nested / "model.pt").write_bytes(b"y" * 250)

    assert observe_retained_content(tmp_path) == {"bytes": 350}


def test_observe_retained_content_excludes_only_the_resource_stats_directory(tmp_path):
    (tmp_path / "output.bin").write_bytes(b"x" * 100)
    stats_dir = tmp_path / "resource_stats" / "participants"
    stats_dir.mkdir(parents=True)
    (stats_dir / "site-1.json").write_bytes(b"z" * 900)

    assert observe_retained_content(tmp_path) == {"bytes": 100}


def test_observe_retained_content_does_not_exclude_a_nested_directory_of_the_same_name(tmp_path):
    # Only the top-level resource_stats directory is platform bookkeeping; a
    # job-produced directory that happens to share that name deeper in the
    # tree is ordinary job output and must still be counted.
    nested_same_name = tmp_path / "app" / "resource_stats"
    nested_same_name.mkdir(parents=True)
    (nested_same_name / "custom_output.bin").write_bytes(b"w" * 40)

    assert observe_retained_content(tmp_path) == {"bytes": 40}


def test_observe_retained_content_is_reported_zero_for_an_empty_directory(tmp_path):
    assert observe_retained_content(tmp_path) == {"bytes": 0}


@pytest.mark.parametrize("existing_file", [False, True])
def test_retained_content_is_unavailable_without_a_run_directory(tmp_path, existing_file):
    run = tmp_path / "run"
    if existing_file:
        run.write_bytes(b"")
    assert observe_retained_content(run) == {"status": "unavailable", "issues": ["observation_incomplete"]}


@pytest.mark.parametrize("directory", [False, True])
def test_retained_content_ignores_symlinked_entries(tmp_path, directory):
    outside = tmp_path / "outside"
    if directory:
        outside.mkdir()
        (outside / "large.bin").write_bytes(b"o" * 10_000)
    else:
        outside.write_bytes(b"o" * 10_000)
    run = tmp_path / "run"
    run.mkdir()
    (run / "linked").symlink_to(outside, target_is_directory=directory)
    assert observe_retained_content(run) == {"bytes": 0}


def test_observe_retained_content_rejects_a_symlinked_run_directory(tmp_path):
    actual_run_dir = tmp_path / "actual"
    actual_run_dir.mkdir()
    (actual_run_dir / "output.bin").write_bytes(b"x" * 100)
    linked_run_dir = tmp_path / "linked"
    linked_run_dir.symlink_to(actual_run_dir, target_is_directory=True)

    assert observe_retained_content(linked_run_dir) == {
        "status": "unavailable",
        "issues": ["observation_incomplete"],
    }


def test_observe_retained_content_marks_a_disappearing_file_partial(tmp_path, monkeypatch):
    (tmp_path / "kept.bin").write_bytes(b"k" * 60)
    (tmp_path / "removed.bin").write_bytes(b"r" * 40)

    real_scandir = os.scandir

    def flaky_scandir(top):
        with real_scandir(top) as entries:
            for entry in entries:
                if entry.name == "removed.bin":
                    os.unlink(entry.name, dir_fd=top)
                yield entry

    monkeypatch.setattr(collector.os, "scandir", flaky_scandir)

    assert observe_retained_content(tmp_path) == {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "bytes": 60,
    }


def test_observe_retained_content_marks_an_iteration_error_partial(tmp_path, monkeypatch):
    (tmp_path / "kept.bin").write_bytes(b"k" * 60)
    real_scandir = os.scandir

    def flaky_scandir(top):
        with real_scandir(top) as entries:
            yield from entries
        raise PermissionError("simulated traversal failure")

    monkeypatch.setattr(collector.os, "scandir", flaky_scandir)

    assert observe_retained_content(tmp_path) == {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "bytes": 60,
    }


@pytest.mark.parametrize("file_count", [3, 30])
def test_retained_entry_budget_returns_subtotal_and_closes_iterator(tmp_path, monkeypatch, file_count):
    for number in range(file_count):
        (tmp_path / f"{number}.bin").write_bytes(b"x")
    real_scandir = os.scandir
    closed = []

    def tracked_scandir(path):
        with real_scandir(path) as entries:
            try:
                yield from entries
            finally:
                closed.append(path)

    monkeypatch.setattr(collector, "_MAX_RETAINED_ENTRIES", 3)
    monkeypatch.setattr(collector.os, "scandir", tracked_scandir)

    result = observe_retained_content(tmp_path)
    assert result == (
        {"bytes": 3} if file_count == 3 else {"status": "partial", "issues": ["observation_incomplete"], "bytes": 3}
    )
    assert len(closed) == 1
    with pytest.raises(OSError):
        os.fstat(closed[0])


def test_retained_depth_budget_keeps_shallower_files(tmp_path, monkeypatch):
    (tmp_path / "root.bin").write_bytes(b"xx")
    first = tmp_path / "first"
    second = first / "second"
    second.mkdir(parents=True)
    (first / "first.bin").write_bytes(b"xxx")
    (second / "omitted.bin").write_bytes(b"xxxxx")
    monkeypatch.setattr(collector, "_MAX_RETAINED_DEPTH", 1)

    assert observe_retained_content(tmp_path) == {"status": "partial", "issues": ["observation_incomplete"], "bytes": 5}


@pytest.mark.parametrize("observed_entries", [0, 2])
def test_retained_deadline_stops_between_entries(tmp_path, monkeypatch, observed_entries):
    for number in range(5):
        (tmp_path / f"{number}.bin").write_bytes(b"x")
    ticks = iter([0] + [0] * observed_entries + [1])
    isolate_time(monkeypatch, collector)
    monkeypatch.setattr(collector.time, "monotonic", lambda: next(ticks))

    result = observe_retained_content(tmp_path)

    assert result == (
        {"status": "partial", "issues": ["observation_incomplete"], "bytes": 2}
        if observed_entries
        else {"status": "unavailable", "issues": ["observation_incomplete"]}
    )


def test_retained_unreadable_subdirectory_keeps_known_files(tmp_path, monkeypatch):
    (tmp_path / "kept.bin").write_bytes(b"xxx")
    inaccessible = tmp_path / "inaccessible"
    inaccessible.mkdir()
    real_open = os.open

    def limited_open(path, flags, **kwargs):
        if path == inaccessible.name:
            raise PermissionError("denied")
        return real_open(path, flags, **kwargs)

    monkeypatch.setattr(collector.os, "open", limited_open)

    assert observe_retained_content(tmp_path) == {"status": "partial", "issues": ["observation_incomplete"], "bytes": 3}


def test_restored_collector_keeps_new_totals_but_marks_prior_interval_incomplete(tmp_path):
    ticks = iter([1_000_000_000, 3_000_000_000])
    job_collector = JobResourceCollector(
        tmp_path,
        clock_ns=lambda: next(ticks),
        capacity_probe=lambda: {"cpu": {"units": 2}, "memory": {"bytes": 4096}, "gpu": {"count": 0}},
        prior_observation_incomplete=True,
    )

    handoff = job_collector.finish()
    resource_capacity = handoff["resource_capacity"]

    assert resource_capacity == {
        "status": "partial",
        "cpu": {"units": 2},
        "memory": {"bytes": 4096},
        "gpu": {"count": 0},
    }

    assert {"metric": "resource_capacity", "code": "observation_incomplete"} in handoff["issues"]


def test_shared_duration_survives_missing_gpu_capacity(monkeypatch, tmp_path):
    resource_ticks = iter((0, 10_000_000_000))
    usages = iter(((float(0), float(0)), (float(2), float(0))))
    # Inject the OS counter through the accountant constructor without changing
    # the collector's production wiring or its shared clock's two reads.
    original = collector.CpuConsumedAccountant
    monkeypatch.setattr(
        collector, "CpuConsumedAccountant", lambda **kwargs: original(usage_reader=lambda: next(usages), **kwargs)
    )
    job_collector = JobResourceCollector(
        tmp_path,
        clock_ns=lambda: next(resource_ticks),
        capacity_probe=lambda: {"cpu": {"units": 8}, "memory": {"bytes": 4096}},
    )

    handoff = job_collector.finish()

    assert handoff["resource_capacity"]["status"] == "partial"
    assert "gpu" not in handoff["resource_capacity"]
    assert handoff["resource_utilization"]["cpu_consumed"] == {"cpu_seconds": 2}
    assert handoff["duration_seconds"] == 10


def test_multinode_slurm_rank_zero_measurement_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("NVFL_NNODES", "4")
    ticks = iter([1_000_000_000, 3_000_000_000])
    job_collector = JobResourceCollector(
        tmp_path,
        clock_ns=lambda: next(ticks),
        capacity_probe=lambda: {"cpu": {"units": 8}, "memory": {"bytes": 4096}, "gpu": {"count": 0}},
    )

    handoff = job_collector.finish()
    assert handoff["resource_capacity"] == {"status": "unavailable"}
    assert {"metric": "resource_capacity", "code": "unsupported"} in handoff["issues"]


def test_missing_handoff_is_explicitly_unavailable():
    report = assemble_participant_summary(
        job_id="job-1",
        participant_name="site-1",
        child_handoff=None,
    )

    assert report["resource_capacity"] == {"status": "unavailable"}
    assert [issue["code"] for issue in report["issues"] if issue["metric"] == "resource_capacity"] == [
        "observation_incomplete"
    ]
    assert report["resource_utilization"]["message_traffic"] == {"status": "unavailable"}
    assert [issue["code"] for issue in report["issues"] if issue["metric"] == "message_traffic"] == [
        "observation_incomplete"
    ]


def test_final_handoff_preserves_partial_traffic_without_merging_parent_values(make_handoff, tmp_path):
    handoff = make_handoff()
    handoff["resource_utilization"]["message_traffic"] = {"status": "partial", "sent_bytes": 5, "received_bytes": 9}
    handoff["issues"] = [issue for issue in handoff["issues"] if issue["metric"] != "message_traffic"] + [
        {"metric": "message_traffic", "code": "observation_incomplete"}
    ]
    write_terminal_handoff(tmp_path, handoff)
    report = assemble_participant_summary(
        job_id="job-1", participant_name="site-1", child_handoff=read_terminal_handoff(tmp_path)
    )
    assert report["resource_utilization"]["message_traffic"] == {
        "status": "partial",
        "sent_bytes": 5,
        "received_bytes": 9,
    }
    assert [issue["code"] for issue in report["issues"] if issue["metric"] == "message_traffic"] == [
        "observation_incomplete"
    ]


def test_handoff_reader_rejects_symlink(make_handoff, tmp_path):
    target_run = tmp_path / "target"
    target = write_terminal_handoff(target_run, make_handoff())
    link_run = tmp_path / "link"
    link = terminal_handoff_path(link_run)
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this platform")

    assert read_terminal_handoff(link_run) is None


def test_handoff_parent_does_not_follow_intermediate_staging_symlink(make_handoff, tmp_path):
    outside_run = tmp_path / "outside"
    outside_handoff = write_terminal_handoff(outside_run, make_handoff())
    run_dir = tmp_path / "run"
    stats_dir = run_dir / RESOURCE_STATS_DIR
    stats_dir.mkdir(parents=True)
    try:
        (stats_dir / STAGING_DIR).symlink_to(outside_handoff.parent, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this platform")

    assert read_terminal_handoff(run_dir) is None
    remove_terminal_handoff(run_dir)

    assert outside_handoff.exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are unavailable on this platform")
def test_handoff_reader_rejects_fifo_without_blocking(tmp_path):
    run_dir = tmp_path / "fifo"
    path = terminal_handoff_path(run_dir)
    path.parent.mkdir(parents=True)
    os.mkfifo(path)

    assert read_terminal_handoff(run_dir) is None


@pytest.mark.parametrize("replace_root", [False, True])
def test_retained_directory_replaced_with_symlink_before_open(tmp_path, monkeypatch, replace_root):
    run = tmp_path / "run"
    child = run / "child"
    child.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "unrelated.bin").write_bytes(b"x" * 4096)
    target = run if replace_root else child
    real_open = os.open
    replaced = []

    def racing_open(path, flags, **kwargs):
        if path == (run if replace_root else child.name) and not replaced:
            target.rename(tmp_path / "original")
            target.symlink_to(outside, target_is_directory=True)
            replaced.append(True)
        return real_open(path, flags, **kwargs)

    monkeypatch.setattr(collector.os, "open", racing_open)
    assert observe_retained_content(run) == (
        {"status": "unavailable", "issues": ["observation_incomplete"]}
        if replace_root
        else {"status": "partial", "issues": ["observation_incomplete"], "bytes": 0}
    )
    assert replaced == [True]


def test_retained_scan_pins_parent_when_its_path_is_replaced(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "child").mkdir(parents=True)
    (run / "child" / "kept.bin").write_bytes(b"xxx")
    outside = tmp_path / "outside"
    (outside / "child").mkdir(parents=True)
    (outside / "child" / "unrelated.bin").write_bytes(b"x" * 4096)
    real_open = os.open
    descriptors = []

    def racing_open(path, flags, **kwargs):
        if path == "child":
            run.rename(tmp_path / "original")
            run.symlink_to(outside, target_is_directory=True)
        descriptor = real_open(path, flags, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(collector.os, "open", racing_open)
    assert observe_retained_content(run) == {"bytes": 3}
    assert len(descriptors) == 2
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_retained_scan_closes_directory_when_scandir_fails(tmp_path, monkeypatch):
    real_open = os.open
    descriptors = []

    def tracked_open(path, flags, **kwargs):
        descriptor = real_open(path, flags, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def denied(_descriptor):
        raise PermissionError("denied")

    monkeypatch.setattr(collector.os, "open", tracked_open)
    monkeypatch.setattr(collector.os, "scandir", denied)
    assert observe_retained_content(tmp_path) == {"status": "unavailable", "issues": ["observation_incomplete"]}
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_retained_scan_is_unavailable_without_safe_directory_operations(tmp_path, monkeypatch):
    (tmp_path / "file.bin").write_bytes(b"xxx")
    monkeypatch.setattr(collector, "_RETAINED_DIR_FD_SUPPORTED", False)
    assert observe_retained_content(tmp_path) == {"status": "unavailable", "issues": ["unsupported"]}


@pytest.mark.parametrize(
    "traffic",
    [
        {"sent_bytes": 0, "received_bytes": 0},
        {"sent_bytes": 5, "received_bytes": 9},
        {"status": "unavailable", "issues": ["observation_incomplete"]},
    ],
)
def test_prior_observation_gap_preserves_traffic_subtotals_and_independent_files(tmp_path, traffic):
    job = JobResourceCollector(tmp_path, prior_observation_incomplete=True, capacity_probe=lambda: {})
    report = job.finish(message_traffic=traffic)
    expected = {key: value for key, value in traffic.items() if key != "issues"}
    expected["status"] = "unavailable" if traffic.get("status") == "unavailable" else "partial"
    assert report["resource_utilization"]["message_traffic"] == expected
    assert report["issues"].count({"metric": "message_traffic", "code": "observation_incomplete"}) == 1
    assert report["resource_utilization"]["retained_content"] == {"bytes": 0}
    assert "status" not in report["workspace_filesystem"]
