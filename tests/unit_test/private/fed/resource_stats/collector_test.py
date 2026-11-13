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

import os

import pytest

from nvflare.private.fed.resource_stats import collector
from nvflare.private.fed.resource_stats.collector import JobResourceCollector, assemble_participant_summary
from nvflare.private.fed.resource_stats.handoff import (
    RESOURCE_STATS_DIR,
    STAGING_DIR,
    read_terminal_handoff,
    remove_terminal_handoff,
    terminal_handoff_path,
    write_terminal_handoff,
)
from nvflare.private.fed.resource_stats.probes import cgroup_linux, gpu_nvidia


@pytest.fixture(autouse=True)
def no_native_gpu_queries(monkeypatch):
    monkeypatch.setattr(gpu_nvidia, "probe_gpu", lambda: None)


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
    job_collector = JobResourceCollector(clock_ns=lambda: next(ticks))
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


def test_restored_collector_keeps_new_totals_but_marks_prior_interval_incomplete(tmp_path):
    ticks = iter([1_000_000_000, 3_000_000_000])
    job_collector = JobResourceCollector(
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


def test_multinode_slurm_rank_zero_measurement_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("NVFL_NNODES", "4")
    ticks = iter([1_000_000_000, 3_000_000_000])
    job_collector = JobResourceCollector(
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


@pytest.mark.parametrize(
    "traffic",
    [
        {"sent_bytes": 0, "received_bytes": 0},
        {"sent_bytes": 5, "received_bytes": 9},
        {"status": "unavailable", "issues": ["observation_incomplete"]},
    ],
)
def test_prior_observation_gap_preserves_traffic_subtotals_(tmp_path, traffic):
    job = JobResourceCollector(prior_observation_incomplete=True, capacity_probe=lambda: {})
    report = job.finish(message_traffic=traffic)
    expected = {key: value for key, value in traffic.items() if key != "issues"}
    expected["status"] = "unavailable" if traffic.get("status") == "unavailable" else "partial"
    assert report["resource_utilization"]["message_traffic"] == expected
    assert report["issues"].count({"metric": "message_traffic", "code": "observation_incomplete"}) == 1
