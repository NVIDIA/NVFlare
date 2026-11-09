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

import datetime
import math
import os
import stat
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from . import accumulator, filesystem_probe, handoff
from .contract import (
    SCHEMA_VERSION,
    _capacity_time_from_validated,
    aggregate_cpu_consumed,
    aggregate_message_traffic,
    aggregate_scope_issues,
    normalize_measurement,
    normalize_scope_issues,
    scoped_measurement,
)
from .cpu_consumed import CpuConsumedAccountant, install_cpu_accountant
from .probes import cgroup_linux, gpu_nvidia
from .task_rollup import TaskResourceReducer, install_task_resource_reducer

_SLURM_NODE_COUNT_ENV = "NVFL_NNODES"

# Fixed collection budgets, not operator settings. Time is checked between
# entries; Python cannot interrupt a blocked filesystem syscall.
_MAX_RETAINED_ENTRIES = 100_000
_MAX_RETAINED_DEPTH = 64
_MAX_RETAINED_SCAN_SECONDS = 1.0
_RETAINED_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_RETAINED_DIR_FD_SUPPORTED = (
    hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and os.open in os.supports_dir_fd
    and os.scandir in os.supports_fd
)

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


def utc_now() -> str:
    """Return a schema-compatible UTC wall-clock timestamp."""

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _checked_capacity_dimension(name: str, value: Any) -> Any:
    return normalize_measurement("resource_capacity", {"status": "partial", name: value})[name]


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
            capacity[name] = _checked_capacity_dimension(name, value)
        except Exception:
            # A native query or malformed dimension must not erase the others.
            issues.add("observation_incomplete")
    if issues:
        capacity.update(status="partial", issues=sorted(issues))
    return capacity


def observe_workspace_filesystem(workspace_path: str | Path) -> dict[str, Any]:
    """Observe only the filesystem containing the existing job workspace."""

    try:
        stats = os.statvfs(os.fspath(workspace_path))
        capacity = stats.f_blocks * stats.f_frsize
    except (OSError, TypeError, ValueError):
        return {"status": "unavailable", "issues": ["observation_incomplete"]}
    if stats.f_frsize <= 0 or stats.f_blocks <= 0 or capacity <= 0:
        return {"status": "unavailable", "issues": ["observation_incomplete"]}
    return {"capacity_bytes": capacity}


def observe_retained_content(run_dir: str | Path) -> dict[str, Any]:
    """Observe the logical size of regular files in one job run directory.

    Includes logs, configuration, and inputs; excludes the top-level
    resource_stats directory. Symlinks and non-regular entries are skipped.
    Depth-first streaming avoids materializing a large directory's entries.
    A budget or traversal error yields an honest partial subtotal; failure
    before any directory entry or empty directory can be observed is unavailable.
    This remains a non-atomic self-report, including ordinary mounted subtrees.
    """

    if not _RETAINED_DIR_FD_SUPPORTED:
        return {"status": "unavailable", "issues": ["unsupported"]}

    deadline = time.monotonic() + _MAX_RETAINED_SCAN_SECONDS
    try:
        root_fd = os.open(run_dir, _RETAINED_DIR_FLAGS)
        try:
            root_entries = os.scandir(root_fd)
        except BaseException:
            os.close(root_fd)
            raise
    except OSError:
        return {"status": "unavailable", "issues": ["observation_incomplete"]}

    total = 0
    entries_seen = 0
    incomplete = False
    observed = False
    stack = [(root_entries, root_fd)]
    try:
        while stack:
            if time.monotonic() >= deadline:
                incomplete = True
                break
            iterator, directory_fd = stack[-1]
            depth = len(stack) - 1
            try:
                entry = next(iterator)
            except StopIteration:
                observed = True
                iterator.close()
                os.close(directory_fd)
                stack.pop()
                continue
            except OSError:
                incomplete = True
                iterator.close()
                os.close(directory_fd)
                stack.pop()
                continue

            if entries_seen >= _MAX_RETAINED_ENTRIES:
                incomplete = True
                break
            entries_seen += 1
            observed = True
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError:
                incomplete = True
                continue

            if stat.S_ISREG(entry_stat.st_mode):
                if entry_stat.st_size < 0:
                    incomplete = True
                    break
                total += entry_stat.st_size
            elif stat.S_ISDIR(entry_stat.st_mode):
                if depth == 0 and entry.name == handoff.RESOURCE_STATS_DIR:
                    continue
                if depth >= _MAX_RETAINED_DEPTH:
                    incomplete = True
                    continue
                try:
                    # Resolve relative to the still-open parent. O_NOFOLLOW
                    # rejects a directory replaced by a symlink after stat.
                    child_fd = os.open(entry.name, _RETAINED_DIR_FLAGS, dir_fd=directory_fd)
                    try:
                        child_entries = os.scandir(child_fd)
                    except BaseException:
                        os.close(child_fd)
                        raise
                    stack.append((child_entries, child_fd))
                except OSError:
                    incomplete = True
    finally:
        for iterator, directory_fd in stack:
            iterator.close()
            os.close(directory_fd)

    if incomplete:
        if not observed:
            return {"status": "unavailable", "issues": ["observation_incomplete"]}
        return {"status": "partial", "issues": ["observation_incomplete"], "bytes": total}
    return {"bytes": total}


def _unavailable(issue: str) -> dict[str, Any]:
    return {"status": "unavailable", "issues": [issue]}


def _fold_measurements(name, current, contribution, issues=None):
    issues = [] if issues is None else issues
    aggregate = {
        "resource_time": TaskResourceReducer.merge_resource_time,
        "cpu_consumed": aggregate_cpu_consumed,
        "message_traffic": aggregate_message_traffic,
    }[name]
    try:
        return (
            aggregate(current, contribution, issues) if name == "resource_time" else aggregate([current, contribution])
        )
    except Exception:
        # Overflow or a broken optional source must not erase the valid subtotal.
        result = deepcopy(current)
        if result.get("status") != "unavailable":
            result["status"] = "partial"
        issues.append({"metric": name, "code": "observation_incomplete"})
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
        self._accumulator = accumulator.ResourceCapacityRecorder()
        self._clock_ns = clock_ns
        self._snapshot: Optional[dict[str, Any]] = None
        self._startup_capacity = {}
        self._prior_observation_incomplete = prior_observation_incomplete
        try:
            capacity = capacity_probe()
            if not isinstance(capacity, Mapping):
                raise TypeError("capacity probe must return a mapping")
            self._startup_capacity = deepcopy(dict(capacity))
        except Exception:
            pass
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
            value = self._clock_ns()
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
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
        try:
            self._accumulator.observe(self._startup_capacity)
            if self._prior_observation_incomplete:
                self._accumulator.mark_prior_observation_incomplete()
        except Exception:
            pass
        issues = []
        self._snapshot = {
            **timing,
            "resource_capacity": scoped_measurement("resource_capacity", self._accumulator.finish(), issues),
            "cpu_consumed": scoped_measurement("cpu_consumed", cpu_consumed, issues),
        }
        self._snapshot["issues"] = normalize_scope_issues(
            issues, {name: self._snapshot[name] for name in ("resource_capacity", "cpu_consumed")}
        )
        return deepcopy(self._snapshot)


class JobResourceCollector(ExecutionResourceCollector):
    """Roll up the job process and its owned task workers into one final report."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        capacity_probe: CapacityProbe = probe_capacity,
        prior_observation_incomplete: bool = False,
        task_based: bool = False,
    ):
        self.run_dir = Path(run_dir)
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
            include_children=not task_based,
            known_attribution_incomplete=self._multi_node_incomplete,
        )
        self._task_reducer = None
        self._task_collection_failed = False
        if task_based:
            try:
                self.enable_task_collection()
            except Exception:
                self._task_collection_failed = True
        else:
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
        """Build one terminal handoff, with one filesystem observation and scan."""

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
        resource_time = _capacity_time_from_validated(resource_capacity, timing.get("duration_seconds"), issues)
        traffic = scoped_measurement(
            "message_traffic",
            message_traffic if message_traffic is not None else _unavailable("observation_incomplete"),
            issues,
        )
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
                resource_time = _fold_measurements(
                    "resource_time", resource_time, worker_totals["resource_time"], issues
                )
            if worker_totals["cpu_consumed"] is not None:
                cpu_consumed = _fold_measurements("cpu_consumed", cpu_consumed, worker_totals["cpu_consumed"], issues)
            if worker_totals["message_traffic"] is not None:
                traffic = _fold_measurements("message_traffic", traffic, worker_totals["message_traffic"], issues)
            if not worker_totals["duration_contained"]:
                timing = {}
        filesystem = filesystem_probe.observe_filesystem_isolated(self.run_dir)
        workspace = scoped_measurement("workspace_filesystem", filesystem["workspace_filesystem"], issues)
        retained = scoped_measurement("retained_content", filesystem["retained_content"], issues)
        measurements = {
            "resource_capacity": resource_capacity,
            "resource_time": resource_time,
            "cpu_consumed": cpu_consumed,
            "message_traffic": traffic,
            "workspace_filesystem": workspace,
            "retained_content": retained,
        }
        issues = aggregate_scope_issues(issues, measurements)
        return {
            **timing,
            "internal_version": handoff.INTERNAL_HANDOFF_VERSION,
            "kind": handoff.INTERNAL_HANDOFF_KIND,
            "resource_capacity": resource_capacity,
            "resource_time": resource_time,
            "workspace_filesystem": workspace,
            "issues": issues,
            "resource_utilization": {
                "cpu_consumed": cpu_consumed,
                "retained_content": retained,
                "message_traffic": traffic,
            },
        }


def assemble_participant_summary(
    *,
    job_id: str,
    participant_name: str,
    child_handoff: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble a handoff already checked by read_terminal_handoff.

    The parent supplies identity; the handoff reader already checked and
    normalized each measurement.
    """

    issues = (
        list(child_handoff["issues"])
        if child_handoff is not None
        else [
            {"metric": name, "code": "observation_incomplete"}
            for name in (
                "resource_capacity",
                "resource_time",
                "workspace_filesystem",
                "cpu_consumed",
                "retained_content",
                "message_traffic",
            )
        ]
    )
    measurements = {
        name: child_handoff[name] if child_handoff is not None else {"status": "unavailable"}
        for name in ("resource_capacity", "resource_time", "workspace_filesystem")
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "nvflare.resource_stats.participant_summary",
        "job_id": job_id,
        "participant_name": participant_name,
        "reported_at": utc_now(),
        "issues": normalize_scope_issues(issues),
        **measurements,
        "resource_utilization": (
            child_handoff["resource_utilization"]
            if child_handoff is not None
            else {name: {"status": "unavailable"} for name in ("cpu_consumed", "retained_content", "message_traffic")}
        ),
        **(
            {"duration_seconds": child_handoff["duration_seconds"]}
            if child_handoff and "duration_seconds" in child_handoff
            else {}
        ),
    }
    return report
