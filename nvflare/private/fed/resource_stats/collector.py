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

"""Collect job measurements and assemble one final participant report."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Optional

from .cpu_consumed import CpuConsumedAccountant, install_cpu_accountant
from .measurements import (
    SCHEMA_VERSION,
    aggregate_scope_issues,
    capacity_to_resource_time,
    merge_worker_measurement,
    ordered_issues,
    scoped_measurement,
    utc_timestamp,
)
from .probes import cgroup_linux, gpu_nvidia
from .task_rollup import TaskResourceReducer, install_task_resource_reducer

_SLURM_NODE_COUNT_ENV = "NVFL_NNODES"

CapacitySnapshot = Mapping[str, Any]
CapacityProbe = Callable[[], CapacitySnapshot]
_active_collector: JobResourceCollector | None = None


def _reset_collection_after_fork():
    global _active_collector
    _active_collector = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_collection_after_fork)


def configure_task_collection():
    """Select disjoint worker accounting at the existing job-config boundary."""

    current = _active_collector
    if current is not None:
        try:
            current.enable_task_collection()
        except Exception:
            current._task_collection_failed = True


def probe_capacity() -> dict[str, Any]:
    """Keep independent usable observations, marking any unchecked limits."""

    capacity = {}
    issues: set[str] = set()
    try:
        discovery = cgroup_linux.discover_cgroups()
    except Exception:
        discovery = (None, None)
        issues.add("observation_incomplete")
    probes = [
        ("cpu", lambda: cgroup_linux.probe_cpu(discovery, issues=issues)),
        ("memory", lambda: cgroup_linux.probe_memory(discovery, issues=issues)),
        ("gpu", gpu_nvidia.probe_gpu),
    ]
    for name, probe in probes:
        try:
            value = probe()
            if value is None:
                continue
            capacity[name] = value
        except Exception:
            # An unavailable native query must not erase the others.
            issues.add("observation_incomplete")
    if issues:
        capacity.update(status="partial", issues=sorted(issues))
    return capacity


def _unavailable(issue: str) -> dict[str, Any]:
    return {"status": "unavailable", "issues": [issue]}


def _startup_capacity_snapshot(capacity_probe, prior_observation_incomplete):
    """Keep one startup observation and its quality, without resampling."""

    try:
        result = deepcopy(dict(capacity_probe()))
    except Exception:
        return _unavailable("observation_incomplete")
    if prior_observation_incomplete or not {"cpu", "memory", "gpu"} <= result.keys():
        result["status"] = "partial" if any(name in result for name in ("cpu", "memory", "gpu")) else "unavailable"
        result["issues"] = sorted(set(result.get("issues", [])) | {"observation_incomplete"})
    return result


class ExecutionResourceCollector:
    """Capture one execution window and its launch-visible resource capacity.

    The launcher establishes limits and visibility before collection. CPU/RAM
    bounds and selected GPU inventory are observed before the CPU and duration
    baselines. Later application choices do not resize the startup snapshot.
    """

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        capacity_probe: CapacityProbe = probe_capacity,
        prior_observation_incomplete: bool = False,
        include_children: bool = True,
        known_attribution_incomplete: bool = False,
    ):
        self._clock_ns = clock_ns
        self._snapshot: Optional[dict[str, Any]] = None
        self._startup_capacity = _startup_capacity_snapshot(capacity_probe, prior_observation_incomplete)
        # Capacity probing finishes before both baselines, excluding the
        # inventory command's wait and CPU work from execution measurements.
        self._start_ns = self._read_clock()
        self._cpu_consumed = CpuConsumedAccountant(
            prior_observation_incomplete=prior_observation_incomplete,
            known_attribution_incomplete=known_attribution_incomplete,
            include_children=include_children,
        )
        install_cpu_accountant(self._cpu_consumed)

    def _read_clock(self) -> int | None:
        try:
            return self._clock_ns()
        except Exception:
            pass
        return None

    def mark_cpu_consumption_incomplete(self) -> None:
        """Mark an unfinished job callback as a CPU accounting coverage gap."""

        self._cpu_consumed.mark_attribution_incomplete()

    def stop(self) -> dict[str, Any]:
        """Freeze CPU work and duration, retaining the one startup capacity snapshot."""

        if self._snapshot is not None:
            return deepcopy(self._snapshot)
        cpu_consumed = self._cpu_consumed.finish()
        end_ns = self._read_clock()
        timing = {}
        if self._start_ns is not None and end_ns is not None and end_ns >= self._start_ns:
            try:
                duration = (end_ns - self._start_ns) / 1_000_000_000
                if math.isfinite(duration):
                    timing["duration_seconds"] = duration
            except OverflowError:
                pass
        issues = []
        self._snapshot = {
            **timing,
            "resource_capacity": scoped_measurement("resource_capacity", self._startup_capacity, issues),
            "cpu_consumed": scoped_measurement("cpu_consumed", cpu_consumed, issues),
        }
        self._snapshot["issues"] = ordered_issues(issues)
        return deepcopy(self._snapshot)


class JobResourceCollector(ExecutionResourceCollector):
    """Roll up the job process and its owned task workers into one final report."""

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        capacity_probe: CapacityProbe = probe_capacity,
        prior_observation_incomplete: bool = False,
    ):
        self._prior_observation_incomplete = prior_observation_incomplete
        raw_node_count = os.environ.get(_SLURM_NODE_COUNT_ENV)
        try:
            self._multi_node_incomplete = raw_node_count is not None and int(raw_node_count) != 1
        except ValueError:
            self._multi_node_incomplete = True
        self._finished = False
        super().__init__(
            clock_ns=clock_ns,
            capacity_probe=capacity_probe,
            prior_observation_incomplete=prior_observation_incomplete,
            known_attribution_incomplete=self._multi_node_incomplete,
        )
        self._task_reducer = None
        self._task_collection_failed = False
        install_task_resource_reducer(None)
        global _active_collector
        _active_collector = self

    def enable_task_collection(self):
        """Keep the original window, but count each worker's CPU separately."""

        if self._task_reducer is None and not self._task_collection_failed:
            self._cpu_consumed.exclude_children()
            reducer = TaskResourceReducer()
            install_task_resource_reducer(reducer)
            self._task_reducer = reducer

    def finish(self, *, message_traffic: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Build one terminal handoff from the final execution measurements."""

        if self._finished:
            raise RuntimeError("resource collection is already finalized")
        self._finished = True
        snapshot = self.stop()
        timing = {name: snapshot[name] for name in ("duration_seconds",) if name in snapshot}
        issues = list(snapshot["issues"])
        resource_capacity = snapshot["resource_capacity"]
        cpu_consumed = snapshot["cpu_consumed"]
        if self._multi_node_incomplete:
            # The current Slurm worker runs the collector on rank zero only.
            # Rank-zero CPU, RAM, and GPU values do not describe the other allocated
            # nodes, so publishing those numbers would understate the site.
            resource_capacity = {"status": "unavailable"}
            issues = [issue for issue in issues if issue["metric"] != "resource_capacity"]
            issues.append({"metric": "resource_capacity", "code": "unsupported"})
        resource_time = capacity_to_resource_time(resource_capacity, timing.get("duration_seconds"), issues)
        traffic = scoped_measurement(
            "message_traffic",
            message_traffic if message_traffic is not None else _unavailable("observation_incomplete"),
            issues,
        )
        if self._prior_observation_incomplete and traffic.get("status") != "unavailable":
            traffic["status"] = "partial"
            issues.append({"metric": "message_traffic", "code": "observation_incomplete"})
        if self._task_reducer is not None or self._task_collection_failed:
            worker_totals = {
                name: {"status": "unavailable"} for name in ("resource_time", "cpu_consumed", "message_traffic")
            }
            worker_totals["duration_contained"] = False
            worker_totals["issues"] = [
                {"metric": name, "code": "observation_incomplete"}
                for name in ("resource_time", "cpu_consumed", "message_traffic")
            ]
            if not self._task_collection_failed:
                try:
                    worker_totals = self._task_reducer.snapshot()
                except Exception:
                    pass
            issues.extend(worker_totals["issues"])
            if worker_totals["resource_time"] is not None:
                resource_time = merge_worker_measurement(
                    "resource_time", resource_time, worker_totals["resource_time"], issues
                )
            if worker_totals["cpu_consumed"] is not None:
                cpu_consumed = merge_worker_measurement(
                    "cpu_consumed", cpu_consumed, worker_totals["cpu_consumed"], issues
                )
            if worker_totals["message_traffic"] is not None:
                traffic = merge_worker_measurement("message_traffic", traffic, worker_totals["message_traffic"], issues)
            if not worker_totals["duration_contained"]:
                timing = {}
        measurements = {
            "resource_capacity": resource_capacity,
            "resource_time": resource_time,
            "cpu_consumed": cpu_consumed,
            "message_traffic": traffic,
        }
        issues = aggregate_scope_issues(issues, measurements)
        return {
            **timing,
            "resource_capacity": resource_capacity,
            "resource_time": resource_time,
            "issues": issues,
            "resource_utilization": {
                "cpu_consumed": cpu_consumed,
                "message_traffic": traffic,
            },
        }


def assemble_participant_summary(
    *,
    job_id: str,
    participant_name: str,
    child_handoff: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Wrap this launch's observations with parent-owned job and participant identity."""

    if child_handoff is None:
        child_handoff = {
            "resource_capacity": {"status": "unavailable"},
            "resource_time": {"status": "unavailable"},
            "resource_utilization": {name: {"status": "unavailable"} for name in ("cpu_consumed", "message_traffic")},
            "issues": [
                {"metric": name, "code": "observation_incomplete"}
                for name in ("cpu_consumed", "message_traffic", "resource_capacity", "resource_time")
            ],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "nvflare.resource_stats.participant_summary",
        "job_id": job_id,
        "participant_name": participant_name,
        "reported_at": utc_timestamp(),
        **{
            name: child_handoff[name]
            for name in ("resource_capacity", "resource_time", "resource_utilization", "issues")
        },
        **({"duration_seconds": child_handoff["duration_seconds"]} if "duration_seconds" in child_handoff else {}),
    }
