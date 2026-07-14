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
import math
import os
import sys
import threading
import time
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ProcessType, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.app_common.aggregators.weighted_aggregation_helper import AggregationStatsKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.fuel.f3.stats_pool import HistPool, StatsPoolManager
from nvflare.widgets.widget import Widget

_MB = 1024.0 * 1024.0
_MSG_SIZE_POOL_PREFIXES = ("sent_msg_sizes", "received_msg_sizes")
_STREAM_SIZE_POOL_PREFIXES = ("sent_stream_sizes", "received_stream_sizes")


class JobStatusCode:
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    # Kept as an API alias for callers that used the first implementation.
    PARTIAL_SUCCESS = PARTIAL
    FAILURE = "FAILURE"


_STATUS_DISPLAY = {
    JobStatusCode.SUCCESS: "SUCCESS",
    JobStatusCode.PARTIAL: "PARTIAL SUCCESS",
    JobStatusCode.FAILURE: "FAILURE",
}


def _to_float(value) -> Optional[float]:
    # bool is excluded: telemetry fields are measurements, and coercing True to 1.0 would
    # fabricate plausible-looking samples out of malformed input
    if value is None or isinstance(value, (bool, dict, list, set, tuple)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _numeric_stats(values: List[float]) -> Optional[dict]:
    values = [f for f in (_to_float(v) for v in values) if f is not None]
    if not values:
        return None
    return {
        "count": len(values),
        "mean": mean(values),
        "stddev": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _sanitize_stats_dict(value) -> Optional[dict]:
    """Validate an externally supplied stats dict (count/mean/stddev/min/max), coercing to floats."""
    if not isinstance(value, dict):
        return None
    count = _to_float(value.get("count"))
    mean_v = _to_float(value.get("mean"))
    if not count or count < 1 or mean_v is None:
        return None
    stddev = _to_float(value.get("stddev"))
    min_v = _to_float(value.get("min"))
    max_v = _to_float(value.get("max"))
    return {
        "count": int(count),
        "mean": mean_v,
        "stddev": stddev if stddev is not None else 0.0,
        "min": min_v if min_v is not None else mean_v,
        "max": max_v if max_v is not None else mean_v,
    }


def _combine_numeric_stats(stats_list: List[dict]) -> Optional[dict]:
    """Pool multiple sample stats (count/mean/stddev/min/max) into one distribution summary."""
    stats_list = [s for s in (_sanitize_stats_dict(x) for x in stats_list) if s]
    if not stats_list:
        return None
    total = sum(s["count"] for s in stats_list)
    combined_mean = sum(s["count"] * s["mean"] for s in stats_list) / total
    second_moment = sum(s["count"] * (s["stddev"] ** 2 + s["mean"] ** 2) for s in stats_list) / total
    return {
        "count": total,
        "mean": combined_mean,
        "stddev": max(0.0, second_moment - combined_mean**2) ** 0.5,
        "min": min(s["min"] for s in stats_list),
        "max": max(s["max"] for s in stats_list),
    }


_TELEMETRY_FLOAT_KEYS = (
    "execution_time",
    "client_processing_time",
    "client_framework_overhead",
    "input_size_mb",
    "update_size_mb",
)
_RESOURCE_STATS_KEYS = ("cpu_percent", "memory_rss_mb", "gpu_percent", "gpu_memory_mb")
_BATTERY_FLOAT_KEYS = ("battery_start_percent", "battery_end_percent", "battery_used_percent")


def _sanitize_telemetry(value) -> dict:
    """Validate the client-supplied telemetry header; drop anything malformed instead of failing the report."""
    if not isinstance(value, dict):
        return {}
    telemetry = {}
    round_num = value.get("round")
    if isinstance(round_num, (int, str)) and not isinstance(round_num, bool):
        telemetry["round"] = round_num
    for key in ("task_name", "return_code", "error"):
        v = value.get(key)
        if isinstance(v, str):
            telemetry[key] = v
    for key in _TELEMETRY_FLOAT_KEYS:
        v = _to_float(value.get(key))
        if v is not None:
            telemetry[key] = v
    resources = value.get("resources")
    if isinstance(resources, dict):
        clean = {}
        for key in _RESOURCE_STATS_KEYS:
            stats = _sanitize_stats_dict(resources.get(key))
            if stats:
                clean[key] = stats
        for key in _BATTERY_FLOAT_KEYS:
            v = _to_float(resources.get(key))
            if v is not None:
                clean[key] = v
        plugged = resources.get("battery_plugged")
        if isinstance(plugged, bool):
            clean["battery_plugged"] = plugged
        telemetry["resources"] = clean
    else:
        telemetry["resources"] = {}
    return telemetry


_AGGR_STATS_COUNT_KEYS = (
    AggregationStatsKey.ACCEPTED_CONTRIBUTIONS,
    AggregationStatsKey.KEYS_AGGREGATED,
    AggregationStatsKey.KEYS_SEEN,
    AggregationStatsKey.FULLY_MATCHED_KEYS,
    AggregationStatsKey.PARTIALLY_MATCHED_KEYS,
    AggregationStatsKey.SKIPPED_KEYS,
)


def _sanitize_aggregation_stats(value) -> Optional[dict]:
    """Validate an AGGREGATION_STATS dict published on the FLContext.

    The prop is an open contract that any (custom) aggregator may publish, so it gets the same
    treatment as client telemetry: counts are coerced to non-negative ints, contributor names to
    a list of strings (a bare string becomes a one-element list, not an iterable of characters),
    and anything malformed is dropped so one bad value cannot corrupt or abort the report.
    """
    if not isinstance(value, dict):
        return None
    stats = {}
    round_num = value.get(AggregationStatsKey.ROUND)
    if isinstance(round_num, (int, str)) and not isinstance(round_num, bool):
        stats[AggregationStatsKey.ROUND] = round_num
    contributors = value.get(AggregationStatsKey.CONTRIBUTORS)
    if isinstance(contributors, str):
        contributors = [contributors]
    if isinstance(contributors, (list, tuple, set, frozenset)):
        stats[AggregationStatsKey.CONTRIBUTORS] = sorted(str(c) for c in contributors)
    else:
        stats[AggregationStatsKey.CONTRIBUTORS] = []
    for key in _AGGR_STATS_COUNT_KEYS:
        v = _to_float(value.get(key))
        stats[key] = int(v) if v is not None and v >= 0 else None
    return stats


def _mean_std(values: List[float], precision: int = 1) -> str:
    stats = _numeric_stats(values)
    if not stats:
        return "N/A"
    if stats["count"] == 1:
        return f"{stats['mean']:.{precision}f}"
    return f"{stats['mean']:.{precision}f} ± {stats['stddev']:.{precision}f}"


def _format_stats(stats: Optional[dict], precision: int = 1) -> str:
    if not stats:
        return "N/A"
    if stats["count"] == 1:
        return f"{stats['mean']:.{precision}f}"
    return f"{stats['mean']:.{precision}f} ± {stats['stddev']:.{precision}f}"


def _format_duration(seconds: Optional[float]) -> str:
    return "N/A" if seconds is None else f"{seconds:.3f} sec"


def _format_mb(size_mb: Optional[float]) -> str:
    if size_mb is None:
        return "N/A"
    if size_mb >= 1024.0:
        return f"{size_mb / 1024.0:.2f} GB"
    return f"{size_mb:.3f} MB"


def _format_table(headers: List[str], rows: List[List[object]]) -> List[str]:
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in str_rows:
        lines.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return lines


def _validate_report_filename(filename, arg_name: str):
    """Reject filenames that would escape the job run directory (fails fast at config time)."""
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{arg_name} must be a non-empty string but got {filename!r}")
    if os.path.isabs(filename):
        raise ValueError(f"{arg_name} must be relative to the job run directory but got absolute path '{filename}'")
    if ".." in filename.split(os.sep) or ".." in filename.split("/"):
        raise ValueError(f"{arg_name} must not contain '..' components: '{filename}'")


def _prepare_output_path(run_dir: str, filename: str) -> str:
    """Resolve an output path strictly under the run directory and ensure its subdirectory exists."""
    root = os.path.realpath(run_dir)
    path = os.path.realpath(os.path.join(root, filename))
    if os.path.commonpath([root, path]) != root:
        raise ValueError(f"report filename '{filename}' must resolve under the job run directory")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


class _UnknownSizeError(Exception):
    """Raised when a payload contains references whose true size cannot be determined."""

    pass


def _estimate_size_bytes(obj, seen=None) -> int:
    """Estimate logical payload bytes without serializing or materializing lazy tensors.

    Only containers are memoized (to guard against reference cycles); leaf values are counted on
    every occurrence so that shared immutables (interned ints/strings, [x] * n lists) are not
    collapsed to a single count.

    Raises:
        _UnknownSizeError: when the payload contains lazy/download references whose size cannot
            be determined — better to report N/A than a few proxy-object bytes for a real model.
    """
    if obj is None:
        return 0

    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))

    if isinstance(obj, (dict, list, tuple, set, frozenset)):
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        if isinstance(obj, dict):
            return sum(_estimate_size_bytes(k, seen) + _estimate_size_bytes(v, seen) for k, v in obj.items())
        return sum(_estimate_size_bytes(v, seen) for v in obj)

    nbytes = getattr(obj, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    numel = getattr(obj, "numel", None)
    element_size = getattr(obj, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return int(numel()) * int(element_size())
        except Exception:
            pass
    if obj.__class__.__name__.endswith(("LazyRef", "DownloadRef")):
        size = getattr(obj, "size", None)
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            return size
        # disk-offloaded tensors (_LazyRef) point at a local file; a stat is cheap
        file_path = getattr(obj, "file_path", None)
        if isinstance(file_path, str):
            try:
                return os.path.getsize(file_path)
            except OSError:
                raise _UnknownSizeError()
        raise _UnknownSizeError()
    try:
        return sys.getsizeof(obj)
    except Exception:
        return 0


def _estimate_payload_mb(obj) -> Optional[float]:
    """Payload size in MB, or None when it cannot be determined (e.g. unresolvable lazy refs)."""
    try:
        return _estimate_size_bytes(obj) / _MB
    except _UnknownSizeError:
        return None


def _round_from_shareable(shareable) -> Optional[object]:
    """Extract the round from a task/result Shareable's header or cookie only.

    Deliberately does NOT fall back to the FLContext CURRENT_ROUND prop (on either side):
    workflows like ScatterAndGather set it sticky, so a later non-round workflow in the same
    job (e.g. cross-site eval) would otherwise have its tasks attributed to a stale round.
    """
    if not isinstance(shareable, Shareable):
        return None
    round_num = shareable.get_header(AppConstants.CURRENT_ROUND, None)
    return shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND, round_num)


def _round_sort_key(round_num) -> tuple:
    """Order rounds numerically, with non-numeric round labels after numeric ones."""
    if isinstance(round_num, (int, float)) and not isinstance(round_num, bool):
        return (0, float(round_num), "")
    return (1, 0.0, str(round_num))


class _ResourceSampler:
    """Samples process/system resources while one client task executes."""

    def __init__(self, interval: float):
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None
        self.process = None
        self.psutil = None
        self.pynvml = None
        self.gpu_handles = []

    def start(self):
        try:
            import psutil

            self.psutil = psutil
            self.process = psutil.Process()
            self.process.cpu_percent(interval=None)
        except Exception:
            self.psutil = None
            self.process = None

        pynvml = None
        nvml_initialized = False
        try:
            import pynvml

            pynvml.nvmlInit()
            nvml_initialized = True
            self.pynvml = pynvml
            self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            if pynvml and nvml_initialized:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            self.pynvml = None
            self.gpu_handles = []

        self._sample(include_cpu=False)
        self.thread = threading.Thread(target=self._run, name="job-stats-resource-sampler", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self._sample(include_cpu=True)

    def _sample(self, include_cpu: bool):
        sample = {"time": time.time()}
        if self.process:
            try:
                sample["memory_rss_mb"] = self.process.memory_info().rss / _MB
                if include_cpu:
                    sample["cpu_percent"] = self.process.cpu_percent(interval=None)
            except Exception:
                pass
        if self.psutil:
            try:
                battery = self.psutil.sensors_battery()
                if battery:
                    sample["battery_percent"] = float(battery.percent)
                    sample["battery_plugged"] = bool(battery.power_plugged)
            except Exception:
                pass
        if self.pynvml:
            pids = self._process_family_pids()
            gpu_percent = []
            gpu_memory_mb = []
            for handle in self.gpu_handles:
                utilization, memory_mb = self._sample_process_gpu(handle, pids)
                if utilization is not None:
                    gpu_percent.append(utilization)
                if memory_mb is not None:
                    gpu_memory_mb.append(memory_mb)
            if gpu_percent:
                sample["gpu_percent"] = mean(gpu_percent)
            if gpu_memory_mb:
                sample["gpu_memory_mb"] = sum(gpu_memory_mb)
        self.samples.append(sample)

    def _process_family_pids(self) -> set:
        """This process plus its descendants.

        Subprocess-mode training (ScriptRunner/launcher external process — the common Client API
        setup) does its GPU work in a launched CHILD process, so scoping to os.getpid() alone
        would miss it entirely.
        """
        pids = {os.getpid()}
        if self.process:
            try:
                pids.update(child.pid for child in self.process.children(recursive=True))
            except Exception:
                pass
        return pids

    def _sample_process_gpu(self, handle, pids: set) -> Tuple[Optional[float], Optional[float]]:
        """Return GPU utilization and memory for this client process family on one device.

        NVML's device utilization/memory APIs include unrelated processes. Prefer process
        utilization/accounting APIs and process lists scoped to this process and its children;
        if the installed driver does not expose them, leave the metric unavailable instead of
        mislabeling device-wide activity as the client's usage.
        """
        per_pid_util = {}
        process_util_fn = getattr(self.pynvml, "nvmlDeviceGetProcessUtilization", None)
        if callable(process_util_fn):
            try:
                for s in process_util_fn(handle, 0):
                    pid = getattr(s, "pid", None)
                    if pid not in pids:
                        continue
                    timestamp = getattr(s, "timeStamp", 0)
                    value = _to_float(getattr(s, "smUtil", None))
                    if value is None or not 0.0 <= value <= 100.0:
                        continue
                    prev = per_pid_util.get(pid)
                    if prev is None or timestamp >= prev[0]:
                        per_pid_util[pid] = (timestamp, value)
            except Exception:
                per_pid_util = {}
        if not per_pid_util:
            accounting_fn = getattr(self.pynvml, "nvmlDeviceGetAccountingStats", None)
            if callable(accounting_fn):
                for pid in pids:
                    try:
                        value = _to_float(getattr(accounting_fn(handle, pid), "gpuUtilization", None))
                        if value is not None and 0.0 <= value <= 100.0:
                            per_pid_util[pid] = (0, value)
                    except Exception:
                        continue
        # utilization of the whole process family on this device (may exceed 100 with
        # multiple worker processes, like process-level CPU%)
        utilization = sum(v for _, v in per_pid_util.values()) if per_pid_util else None

        per_pid_memory = {}
        unavailable = getattr(self.pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
        for names in (
            (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses_v2",
                "nvmlDeviceGetComputeRunningProcesses",
            ),
            (
                "nvmlDeviceGetGraphicsRunningProcesses_v3",
                "nvmlDeviceGetGraphicsRunningProcesses_v2",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ),
        ):
            process_fn = next(
                (getattr(self.pynvml, name, None) for name in names if callable(getattr(self.pynvml, name, None))), None
            )
            if not process_fn:
                continue
            try:
                for process in process_fn(handle):
                    pid = getattr(process, "pid", None)
                    if pid not in pids:
                        continue
                    used = getattr(process, "usedGpuMemory", None)
                    if used is None or used == unavailable:
                        continue
                    value = _to_float(used)
                    if value is not None and value >= 0.0:
                        # a process may appear in both the compute and graphics lists
                        per_pid_memory[pid] = max(per_pid_memory.get(pid, 0.0), value / _MB)
            except Exception:
                continue

        memory_mb = sum(per_pid_memory.values()) if per_pid_memory else None
        return utilization, memory_mb

    def stop(self) -> dict:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=max(1.0, self.interval * 2.0))
        self._sample(include_cpu=True)
        if self.pynvml:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass

        result = {}
        for key in ("cpu_percent", "memory_rss_mb", "gpu_percent", "gpu_memory_mb"):
            stats = _numeric_stats([s[key] for s in self.samples if key in s])
            if stats:
                result[key] = stats
        battery = [s for s in self.samples if "battery_percent" in s]
        if battery:
            start = battery[0]["battery_percent"]
            end = battery[-1]["battery_percent"]
            result["battery_start_percent"] = start
            result["battery_end_percent"] = end
            result["battery_used_percent"] = max(0.0, start - end)
            result["battery_plugged"] = battery[-1].get("battery_plugged")
        return result


class _RoundRecord:
    def __init__(self, round_num, start_mono: float, workflow: int = 0):
        self.round_num = round_num
        self.workflow = workflow
        # durations come from the monotonic clock (immune to NTP/clock adjustments);
        # wall-clock timestamps are kept only for display in the JSON report
        self.start_mono = start_mono
        self.end_mono: Optional[float] = None
        self.started_at = time.time()
        self.ended_at: Optional[float] = None
        self.aggr_stats: Optional[dict] = None
        self.targeted_clients = set()
        self.accepted_clients = set()
        self.rejected_clients = set()

    def close(self, end_mono: float):
        self.end_mono = end_mono
        self.ended_at = time.time()

    @property
    def is_open(self) -> bool:
        return self.end_mono is None

    @property
    def duration(self) -> Optional[float]:
        return None if self.end_mono is None else self.end_mono - self.start_mono


class _TaskRecord:
    def __init__(
        self,
        client: str,
        task_id: str,
        task_name: str,
        round_num,
        assigned_mono: float,
        payload_mb: Optional[float],
        workflow: int = 0,
    ):
        self.client = client
        self.task_id = task_id
        self.task_name = task_name
        self.round_num = round_num
        self.workflow = workflow
        self.assigned_mono = assigned_mono
        self.assigned_at = time.time()
        self.received_at = None
        self.payload_mb = payload_mb
        self.update_mb = None
        self.client_execution_time = None
        self.client_processing_time = None
        self.client_framework_overhead = None
        self.server_elapsed_time = None
        self.communication_time = None
        self.return_code = None
        self.error = None
        self.resources = {}

    def complete(self, result: Optional[Shareable], telemetry: dict):
        self.received_at = time.time()
        self.server_elapsed_time = time.monotonic() - self.assigned_mono
        self.client_execution_time = telemetry.get("execution_time")
        self.client_processing_time = telemetry.get("client_processing_time")
        self.client_framework_overhead = telemetry.get("client_framework_overhead")
        self.update_mb = telemetry.get("update_size_mb")
        if self.update_mb is None and isinstance(result, Shareable):
            self.update_mb = _estimate_payload_mb(result)
        local_processing = self.client_processing_time
        if local_processing is None:
            local_processing = self.client_execution_time
        if local_processing is not None:
            self.communication_time = max(0.0, self.server_elapsed_time - local_processing)
        self.resources = telemetry.get("resources") or {}
        if isinstance(result, Shareable):
            self.return_code = result.get_return_code(default=ReturnCode.OK)
            self.error = result.get_header(ReservedHeaderKey.ERROR)

    def to_dict(self) -> dict:
        return {
            "workflow": self.workflow,
            "round": self.round_num,
            "client": self.client,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "assigned_at": self.assigned_at,
            "received_at": self.received_at,
            "server_elapsed_time": self.server_elapsed_time,
            "client_computation_time": self.client_execution_time,
            "client_processing_time": self.client_processing_time,
            "client_framework_overhead": self.client_framework_overhead,
            "communication_time": self.communication_time,
            "task_payload_mb": self.payload_mb,
            "model_update_mb": self.update_mb,
            "return_code": self.return_code,
            "error": self.error,
            "resources": self.resources,
        }


class JobStatsReporter(Widget):
    def __init__(
        self,
        filename: str = "job_stats_run_summary.log",
        json_filename: Optional[str] = "job_stats_run_summary.json",
        error_filename: Optional[str] = None,
        collect_resources: bool = True,
        resource_sample_interval: float = 1.0,
    ):
        """Collect complete job, round, client, task, communication, and optional resource statistics.

        Configure this component in both the server and client application configs.  The client instance
        measures executor time/resources and attaches telemetry to each task result; the server instance
        produces the final report.  When only configured on the server, server-observed timing and logical
        payload sizes are still reported, while client computation/resources are N/A.

        The client telemetry header (AppConstants.JOB_STATS_CLIENT_TELEMETRY) is attached at
        AFTER_TASK_EXECUTION, i.e. BEFORE task-result filters run, so privacy filters can inspect or
        strip it, and the reported model-update size reflects the executor's actual result (before any
        filter- or streaming-driven transformation). Client processing time therefore covers task-data
        filtering plus execution; result filtering and transmission are accounted to communication/wait
        time on the server side. The server sanitizes all received telemetry before use.

        Args:
            filename: human-readable report written to the server run directory.
            json_filename: structured report filename, written by default. Set to None to disable JSON output.
            error_filename: optional separate error report, created only when errors occurred.
            collect_resources: whether client instances sample CPU/GPU/memory/battery during task execution.
            resource_sample_interval: resource sampling interval in seconds.
        """
        super().__init__()
        if resource_sample_interval <= 0:
            raise ValueError("resource_sample_interval must be greater than 0")
        _validate_report_filename(filename, "filename")
        if json_filename is not None:
            _validate_report_filename(json_filename, "json_filename")
        if error_filename is not None:
            _validate_report_filename(error_filename, "error_filename")
        self._filename = filename
        self._json_filename = json_filename
        self._error_filename = error_filename
        self._collect_resources = collect_resources
        self._resource_sample_interval = resource_sample_interval
        self._lock = threading.Lock()
        self._reset()

    def _reset(self):
        self._job_start_time = None
        self._job_end_time = None
        self._job_start_mono = None
        self._job_end_mono = None
        self._process_type = None
        self._is_client = False
        self._all_clients = set()
        self._workflow_seq = 0
        self._rounds: Dict[tuple, _RoundRecord] = {}  # (workflow_seq, round_num) -> record
        self._open_tasks: Dict[Tuple[str, str], _TaskRecord] = {}
        self._completed_task_keys = set()  # (client, task_id) pairs whose result was already processed
        self._completed_tasks: List[_TaskRecord] = []
        self._client_errors: Dict[str, List[str]] = {}
        self._error_details: List[dict] = []
        self._disconnected_clients = set()
        self._fatal_error_reason = None
        self._client_task_info = {}
        self._client_task_telemetry = {}

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        try:
            if event_type == EventType.START_RUN:
                self._handle_start_run(fl_ctx)
                return

            if self._is_client:
                if event_type == EventType.BEFORE_TASK_DATA_FILTER:
                    self._handle_client_task_received(fl_ctx)
                elif event_type == EventType.BEFORE_TASK_EXECUTION:
                    self._handle_client_task_start(fl_ctx)
                elif event_type == EventType.AFTER_TASK_EXECUTION:
                    # Attach telemetry here, BEFORE task-result filters run, so that privacy
                    # filters remain the last components to see/scrub the outgoing Shareable
                    # (including the telemetry header itself).
                    self._handle_client_task_completed(fl_ctx)
                elif event_type == EventType.AFTER_SEND_TASK_RESULT:
                    self._handle_client_after_send(fl_ctx)
                elif event_type == EventType.END_RUN:
                    self._stop_client_samplers(clear=True)
                return

            if event_type == EventType.START_WORKFLOW:
                self._handle_workflow_change()
            elif event_type == EventType.END_WORKFLOW:
                self._close_open_rounds()
            elif event_type == AppEventType.ROUND_STARTED:
                self._handle_round_started(fl_ctx)
            elif event_type == AppEventType.ROUND_DONE:
                self._close_round(fl_ctx.get_prop(AppConstants.CURRENT_ROUND))
            elif event_type == AppEventType.AFTER_AGGREGATION:
                self._handle_after_aggregation(fl_ctx)
            elif event_type == AppEventType.AFTER_CONTRIBUTION_ACCEPT:
                self._handle_contribution_accept(fl_ctx)
            elif event_type == EventType.AFTER_TASK_DATA_FILTER:
                self._handle_task_assigned(fl_ctx)
            elif event_type == EventType.BEFORE_PROCESS_SUBMISSION:
                self._handle_submission(fl_ctx)
            elif event_type == EventType.CLIENT_DISCONNECTED:
                self._handle_disconnect(fl_ctx)
            elif event_type == EventType.CLIENT_RECONNECTED:
                self._handle_reconnect(fl_ctx)
            elif event_type == EventType.FATAL_SYSTEM_ERROR:
                reason = fl_ctx.get_prop(FLContextKey.EVENT_DATA, "")
                with self._lock:
                    self._fatal_error_reason = str(reason) if reason else "fatal system error"
            elif event_type == EventType.END_RUN:
                self._handle_end_run(fl_ctx)
        except Exception:
            self.log_exception(fl_ctx, f"error handling event {event_type} for job stats", fire_event=False)

    @staticmethod
    def _get_peer_name(fl_ctx: FLContext) -> Optional[str]:
        peer_ctx = fl_ctx.get_peer_context()
        return peer_ctx.get_identity_name() if isinstance(peer_ctx, FLContext) else None

    def _handle_start_run(self, fl_ctx: FLContext):
        self._stop_client_samplers()
        with self._lock:
            self._reset()
            self._job_start_time = time.time()
            self._job_start_mono = time.monotonic()
            self._process_type = fl_ctx.get_prop(FLContextKey.PROCESS_TYPE)
            self._is_client = self._process_type in (ProcessType.CLIENT_PARENT, ProcessType.CLIENT_JOB)
            if not self._is_client:
                engine = fl_ctx.get_engine()
                get_clients = getattr(engine, "get_clients", None)
                if callable(get_clients):
                    clients = get_clients()
                    if clients:
                        self._all_clients = {c.name for c in clients}

    def _handle_client_task_received(self, fl_ctx: FLContext):
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        if not task_id:
            return
        task_data = fl_ctx.get_prop(FLContextKey.TASK_DATA)
        entry = {
            "task_name": fl_ctx.get_prop(FLContextKey.TASK_NAME),
            "round": _round_from_shareable(task_data),
            "processing_start_time": time.monotonic(),
            "input_size_mb": _estimate_payload_mb(task_data),
            "sampler": None,
        }
        with self._lock:
            self._client_task_info[task_id] = entry

    def _handle_client_task_start(self, fl_ctx: FLContext):
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        if not task_id:
            return
        task_data = fl_ctx.get_prop(FLContextKey.TASK_DATA)
        sampler = None
        if self._collect_resources:
            sampler = _ResourceSampler(self._resource_sample_interval)
            sampler.start()
        now = time.monotonic()
        with self._lock:
            info = self._client_task_info.setdefault(task_id, {})
            info.update(
                {
                    "task_name": fl_ctx.get_prop(FLContextKey.TASK_NAME),
                    "round": _round_from_shareable(task_data),
                    "processing_start_time": info.get("processing_start_time", now),
                    "execution_start_time": now,
                    "input_size_mb": _estimate_payload_mb(task_data),
                    "sampler": sampler,
                }
            )

    def _finalize_client_task(self, task_id: str) -> dict:
        with self._lock:
            telemetry = self._client_task_telemetry.get(task_id)
            if telemetry is not None:
                return telemetry
            info = self._client_task_info.get(task_id, {})
            start_time = info.get("execution_start_time")
            sampler = info.get("sampler")
            if info:
                info["sampler"] = None
        # stop the sampler (joins its thread) outside the lock
        resources = sampler.stop() if sampler else {}
        with self._lock:
            telemetry = self._client_task_telemetry.get(task_id)
            if telemetry is not None:
                return telemetry
            telemetry = {
                "round": info.get("round"),
                "task_name": info.get("task_name"),
                "execution_time": time.monotonic() - start_time if start_time is not None else None,
                "input_size_mb": info.get("input_size_mb"),
                "resources": resources,
            }
            self._client_task_telemetry[task_id] = telemetry
            return telemetry

    def _handle_client_task_completed(self, fl_ctx: FLContext):
        """Finalize telemetry and attach it to the pre-filter task result at AFTER_TASK_EXECUTION.

        This runs before task-result filters, so the result (including any tensors later moved by
        streaming components) still reflects the true model update size, and privacy filters can
        inspect or strip the telemetry header before anything leaves the client.
        """
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        if not task_id:
            return
        telemetry = dict(self._finalize_client_task(task_id))
        with self._lock:
            info = self._client_task_info.get(task_id, {})
            processing_start = info.get("processing_start_time")
        if processing_start is not None:
            telemetry["client_processing_time"] = time.monotonic() - processing_start
            execution_time = telemetry.get("execution_time")
            if execution_time is not None:
                telemetry["client_framework_overhead"] = max(0.0, telemetry["client_processing_time"] - execution_time)
        result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
        if isinstance(result, Shareable):
            update_mb = _estimate_payload_mb(result)
            if update_mb is not None:
                telemetry["update_size_mb"] = update_mb
            telemetry["return_code"] = result.get_return_code(default=ReturnCode.OK)
            error = result.get_header(ReservedHeaderKey.ERROR)
            if error:
                telemetry["error"] = str(error)
            result.set_header(AppConstants.JOB_STATS_CLIENT_TELEMETRY, telemetry)
        # the telemetry is attached; per-task state is no longer needed (this also covers
        # aux DO_TASK executions, for which no send events ever fire)
        with self._lock:
            self._client_task_info.pop(task_id, None)
            self._client_task_telemetry.pop(task_id, None)

    def _handle_client_after_send(self, fl_ctx: FLContext):
        # safety net for tasks that never reached AFTER_TASK_EXECUTION (data-filter or
        # executor errors): stop any sampler and drop the per-task state
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        if not task_id:
            return
        with self._lock:
            info = self._client_task_info.pop(task_id, None)
            self._client_task_telemetry.pop(task_id, None)
        sampler = info.get("sampler") if info else None
        if sampler:
            sampler.stop()

    def _stop_client_samplers(self, clear: bool = False):
        with self._lock:
            infos = list(getattr(self, "_client_task_info", {}).values())
            samplers = [info.get("sampler") for info in infos]
            for info in infos:
                info["sampler"] = None
            if clear:
                self._client_task_info.clear()
                self._client_task_telemetry.clear()
        for sampler in samplers:
            if sampler:
                sampler.stop()

    def _handle_workflow_change(self):
        now = time.monotonic()
        with self._lock:
            self._workflow_seq += 1
            for rec in self._rounds.values():
                if rec.is_open:
                    rec.close(now)

    def _close_open_rounds(self):
        now = time.monotonic()
        with self._lock:
            for rec in self._rounds.values():
                if rec.is_open:
                    rec.close(now)

    def _handle_round_started(self, fl_ctx: FLContext):
        round_num = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        if round_num is None:
            return
        now = time.monotonic()
        with self._lock:
            key = (self._workflow_seq, round_num)
            for rec in self._rounds.values():
                if rec.is_open and (rec.workflow, rec.round_num) != key:
                    rec.close(now)
            if key not in self._rounds:
                self._rounds[key] = _RoundRecord(round_num, now, workflow=self._workflow_seq)

    def _close_round(self, round_num):
        if round_num is None:
            return
        with self._lock:
            rec = self._rounds.get((self._workflow_seq, round_num))
            if rec:
                rec.close(time.monotonic())

    def _get_or_create_round(self, round_num) -> _RoundRecord:
        key = (self._workflow_seq, round_num)
        rec = self._rounds.get(key)
        if not rec:
            rec = _RoundRecord(round_num, time.monotonic(), workflow=self._workflow_seq)
            self._rounds[key] = rec
        return rec

    def _handle_after_aggregation(self, fl_ctx: FLContext):
        # sanitize before use: the AGGREGATION_STATS prop may be published by any custom
        # aggregator, so it is as untrusted as client telemetry
        stats = _sanitize_aggregation_stats(fl_ctx.get_prop(AppConstants.AGGREGATION_STATS))
        round_num = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        if stats is not None and stats.get(AggregationStatsKey.ROUND) is not None:
            round_num = stats[AggregationStatsKey.ROUND]
        if round_num is None:
            return
        with self._lock:
            rec = self._get_or_create_round(round_num)
            rec.close(time.monotonic())
            if stats is not None:
                rec.aggr_stats = stats
                contributors = stats.get(AggregationStatsKey.CONTRIBUTORS) or []
                rec.accepted_clients.update(contributors)
                self._all_clients.update(contributors)

    def _handle_contribution_accept(self, fl_ctx: FLContext):
        round_num = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        client_name = self._get_peer_name(fl_ctx)
        if round_num is None or not client_name:
            return
        accepted = fl_ctx.get_prop(AppConstants.AGGREGATION_ACCEPTED)
        if accepted is None:
            return
        with self._lock:
            rec = self._get_or_create_round(round_num)
            self._all_clients.add(client_name)
            if accepted:
                rec.accepted_clients.add(client_name)
                rec.rejected_clients.discard(client_name)
            else:
                rec.rejected_clients.add(client_name)
                task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
                task_rec = next(
                    (
                        task
                        for task in reversed(self._completed_tasks)
                        if task.client == client_name and (not task_id or task.task_id == task_id)
                    ),
                    None,
                )
                self._record_error(client_name, task_rec, "CONTRIBUTION_REJECTED", "aggregator rejected result")

    def _handle_task_assigned(self, fl_ctx: FLContext):
        task_name = fl_ctx.get_prop(FLContextKey.TASK_NAME)
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        task_data = fl_ctx.get_prop(FLContextKey.TASK_DATA)
        client_name = self._get_peer_name(fl_ctx)
        if not task_name or not task_id or not client_name:
            return
        # Only trust the round carried by the task data itself; the FLContext CURRENT_ROUND
        # prop may be a stale sticky value from an earlier workflow in the same job.
        round_num = _round_from_shareable(task_data)
        payload_mb = _estimate_payload_mb(task_data)
        with self._lock:
            if round_num is None and self._rounds:
                # Some legacy workflows keep the round only on their controller context.
                # Associate the assignment with the most recently started open round.
                open_rounds = [rec for rec in self._rounds.values() if rec.is_open]
                if open_rounds:
                    round_num = max(open_rounds, key=lambda rec: rec.start_mono).round_num
            self._all_clients.add(client_name)
            self._open_tasks[(client_name, task_id)] = _TaskRecord(
                client_name, task_id, task_name, round_num, time.monotonic(), payload_mb, workflow=self._workflow_seq
            )
            if round_num is not None:
                self._get_or_create_round(round_num).targeted_clients.add(client_name)

    def _record_error(self, client_name: str, rec: Optional[_TaskRecord], code: str, message=None):
        self._client_errors.setdefault(client_name, []).append(str(code))
        self._error_details.append(
            {
                "client": client_name,
                "round": rec.round_num if rec else None,
                "task_id": rec.task_id if rec else None,
                "task_name": rec.task_name if rec else None,
                "code": str(code),
                "message": str(message) if message else None,
            }
        )

    def _handle_submission(self, fl_ctx: FLContext):
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        client_name = self._get_peer_name(fl_ctx)
        if not task_id or not client_name:
            return
        result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
        telemetry = {}
        if isinstance(result, Shareable):
            telemetry = _sanitize_telemetry(result.get_header(AppConstants.JOB_STATS_CLIENT_TELEMETRY))
        task_key = (client_name, task_id)
        with self._lock:
            rec = self._open_tasks.pop(task_key, None)
            if not rec:
                if task_key in self._completed_task_keys:
                    # a client re-sent a result the server already processed; do not
                    # double-count it as another task execution
                    return
                round_num = telemetry.get("round")
                if round_num is None:
                    round_num = _round_from_shareable(result)
                rec = _TaskRecord(
                    client_name,
                    task_id,
                    fl_ctx.get_prop(FLContextKey.TASK_NAME) or telemetry.get("task_name") or "unknown",
                    round_num,
                    time.monotonic(),
                    telemetry.get("input_size_mb"),
                    workflow=self._workflow_seq,
                )
            self._completed_task_keys.add(task_key)
            rec.complete(result, telemetry)
            self._completed_tasks.append(rec)
            if rec.round_num is not None:
                self._get_or_create_round(rec.round_num).targeted_clients.add(client_name)
            if rec.return_code and rec.return_code != ReturnCode.OK:
                self._record_error(client_name, rec, rec.return_code, rec.error or telemetry.get("error"))

    def _handle_disconnect(self, fl_ctx: FLContext):
        client_name = fl_ctx.get_prop(FLContextKey.DISCONNECTED_CLIENT_NAME)
        if client_name:
            with self._lock:
                self._disconnected_clients.add(client_name)
                # record through _record_error so _client_errors and _error_details stay
                # symmetric; _handle_reconnect scrubs both when the disconnect was transient
                self._record_error(client_name, None, "CLIENT_DISCONNECTED", "client disconnected during the run")

    def _handle_reconnect(self, fl_ctx: FLContext):
        client_name = fl_ctx.get_prop(FLContextKey.RECONNECTED_CLIENT_NAME)
        if not client_name:
            return
        with self._lock:
            # the disconnect was transient; do not hold it against the job outcome
            self._disconnected_clients.discard(client_name)
            self._error_details = [
                e
                for e in self._error_details
                if not (e["client"] == client_name and e["code"] == "CLIENT_DISCONNECTED")
            ]
            codes = self._client_errors.get(client_name)
            if codes:
                codes = [c for c in codes if c != "CLIENT_DISCONNECTED"]
                if codes:
                    self._client_errors[client_name] = codes
                else:
                    self._client_errors.pop(client_name, None)

    def _handle_end_run(self, fl_ctx: FLContext):
        now_mono = time.monotonic()
        with self._lock:
            self._job_end_time = time.time()
            self._job_end_mono = now_mono
            for rec in self._rounds.values():
                if rec.is_open:
                    rec.close(now_mono)
            for rec in self._open_tasks.values():
                self._record_error(rec.client, rec, "NO_RESULT", "task assigned but no result was received")
        # A bug in summary generation must degrade the report, never lose it entirely.
        summary = None
        try:
            summary = self.get_summary(fl_ctx)
            report = self.format_report(summary)
        except Exception as e:
            self.log_exception(fl_ctx, "job stats summary generation failed", fire_event=False)
            report = (
                "NVFLARE Job Stats Run Summary\n\n"
                f"Report generation failed with an internal error: {type(e).__name__}: {e}\n"
                "See the server log for the full traceback.\n"
            )
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
        os.makedirs(run_dir, exist_ok=True)
        report_path = _prepare_output_path(run_dir, self._filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        if summary is not None and self._json_filename:
            json_path = _prepare_output_path(run_dir, self._json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
        if summary is not None and self._error_filename and summary["errors"]:
            error_path = _prepare_output_path(run_dir, self._error_filename)
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(self.format_errors(summary["errors"]))
        self.log_info(fl_ctx, f"job stats run summary saved to {report_path}")

    def _completed_ok_clients_by_round(self) -> Dict[tuple, set]:
        """Map (workflow, round) -> clients that returned an OK result for that round."""
        completed = {}
        for rec in self._completed_tasks:
            if rec.round_num is None:
                continue
            if rec.return_code in (None, ReturnCode.OK):
                completed.setdefault((rec.workflow, rec.round_num), set()).add(rec.client)
        return completed

    @staticmethod
    def _round_participants(rec: _RoundRecord, completed_by_round: Dict[tuple, set]) -> set:
        """Clients that participated in a round.

        When the round saw per-client acceptance information (accepted or rejected names),
        participation means the contribution was accepted. Otherwise — relay/cyclic flows that
        never report acceptance, or aggregators that publish stats without contributor names —
        participation is measured by tasks completed with an OK result instead. The condition
        deliberately keys on the NAME sets, not on aggr_stats being present, so a stats dict
        without contributor names cannot zero out participation for a successful round.
        """
        if rec.accepted_clients or rec.rejected_clients:
            return rec.accepted_clients
        return completed_by_round.get((rec.workflow, rec.round_num), set())

    def _participation_counts(self) -> Tuple[int, int]:
        completed_by_round = self._completed_ok_clients_by_round()
        targeted = sum(len(rec.targeted_clients) for rec in self._rounds.values())
        participated = sum(
            len(self._round_participants(rec, completed_by_round) & rec.targeted_clients)
            for rec in self._rounds.values()
        )
        return targeted, participated

    def _determine_status(self, fl_ctx: Optional[FLContext]) -> Tuple[str, str]:
        fatal_reason = self._fatal_error_reason
        aborted = False
        if fl_ctx:
            if fl_ctx.get_prop(FLContextKey.FATAL_SYSTEM_ERROR, False) and not fatal_reason:
                fatal_reason = "fatal system error"
            abort_signal = fl_ctx.get_prop(FLContextKey.RUN_ABORT_SIGNAL)
            aborted = bool(abort_signal and getattr(abort_signal, "triggered", False))
        if fatal_reason:
            return JobStatusCode.FAILURE, fatal_reason
        if aborted:
            return JobStatusCode.FAILURE, "job was aborted"

        # participation uses accepted contributions when the workflow reports acceptance, and
        # OK-completed tasks otherwise (e.g. relay/cyclic flows never fire acceptance events),
        # so a fully successful non-aggregation job is not misread as having zero participation
        targeted, participated = self._participation_counts()
        if targeted and participated == 0:
            return JobStatusCode.FAILURE, "no targeted client produced an accepted contribution or completed task"

        completed_by_round = self._completed_ok_clients_by_round()
        issues = []
        if self._error_details:
            issues.append(f"{len(self._error_details)} task error(s)")
        if self._disconnected_clients:
            issues.append(f"disconnected clients: {sorted(self._disconnected_clients)}")
        missing_rounds = sorted(
            {
                rec.round_num
                for rec in self._rounds.values()
                if rec.targeted_clients
                and not rec.targeted_clients.issubset(self._round_participants(rec, completed_by_round))
            },
            key=_round_sort_key,
        )
        if missing_rounds:
            issues.append(f"rounds with missing/rejected contributions: {missing_rounds}")
        if issues:
            return JobStatusCode.PARTIAL, "; ".join(issues)
        if not self._rounds and not self._completed_tasks and not self._open_tasks:
            # the run ended cleanly but no FL work was ever observed; calling that a
            # success would hide a job that silently did nothing
            return JobStatusCode.PARTIAL, "no rounds or client tasks were observed during the run"
        return JobStatusCode.SUCCESS, "all targeted client contributions were accepted"

    @staticmethod
    def _round_pattern(rec: _RoundRecord) -> Optional[tuple]:
        stats = rec.aggr_stats
        if not stats:
            return None
        accepted = stats.get(AggregationStatsKey.ACCEPTED_CONTRIBUTIONS)
        if accepted is None:
            accepted = len(rec.accepted_clients)
        return (
            accepted,
            stats.get(AggregationStatsKey.KEYS_AGGREGATED),
            stats.get(AggregationStatsKey.FULLY_MATCHED_KEYS),
            stats.get(AggregationStatsKey.PARTIALLY_MATCHED_KEYS),
            stats.get(AggregationStatsKey.SKIPPED_KEYS),
        )

    def _summarize_round_consistency(self, ordered_rounds: List[_RoundRecord]) -> dict:
        patterns = [self._round_pattern(rec) for rec in ordered_rounds]
        known = [p for p in patterns if p is not None]
        result = {
            "total_rounds": len(ordered_rounds),
            "stable_rounds": None,
            "inconsistent_rounds": [],
            "keys_aggregated": None,
            "fully_matched_keys": None,
            "partially_matched_keys": None,
            "skipped_keys": None,
        }
        if not known:
            return result
        counts = {}
        for pattern in known:
            counts[pattern] = counts.get(pattern, 0) + 1
        dominant = max(counts, key=counts.get)
        result["stable_rounds"] = counts[dominant]
        result["inconsistent_rounds"] = [
            rec.round_num
            for rec, pattern in zip(ordered_rounds, patterns)
            if pattern is not None and pattern != dominant
        ]
        for index, key in enumerate(
            (None, "keys_aggregated", "fully_matched_keys", "partially_matched_keys", "skipped_keys")
        ):
            if key:
                # _mean_std coerces via _to_float and skips anything non-numeric
                result[key] = _mean_std([p[index] for p in known])
        return result

    def _get_consistency_summary(self, ordered_rounds: List[_RoundRecord]) -> dict:
        """Summarize aggregation consistency without comparing unrelated workflows."""
        by_workflow = {}
        for rec in ordered_rounds:
            by_workflow.setdefault(rec.workflow, []).append(rec)

        workflow_summaries = []
        for workflow, records in by_workflow.items():
            item = self._summarize_round_consistency(records)
            item["workflow"] = workflow
            workflow_summaries.append(item)

        if len(workflow_summaries) <= 1:
            result = (
                dict(workflow_summaries[0]) if workflow_summaries else self._summarize_round_consistency(ordered_rounds)
            )
            result["workflows"] = workflow_summaries
            return result

        known_summaries = [item for item in workflow_summaries if item["stable_rounds"] is not None]
        return {
            "total_rounds": sum(item["total_rounds"] for item in workflow_summaries),
            "stable_rounds": (sum(item["stable_rounds"] for item in known_summaries) if known_summaries else None),
            "inconsistent_rounds": [
                {"workflow": item["workflow"], "round": round_num}
                for item in workflow_summaries
                for round_num in item["inconsistent_rounds"]
            ],
            # Per-key distributions are meaningful only within a workflow. They are retained
            # in the workflow summaries instead of being pooled across unrelated models.
            "keys_aggregated": None,
            "fully_matched_keys": None,
            "partially_matched_keys": None,
            "skipped_keys": None,
            "workflows": workflow_summaries,
        }

    @staticmethod
    def _get_transport_comm_stats() -> dict:
        # Message-size pools already include every stream chunk sent over the wire, so
        # stream-pool totals must NOT be added to the message totals (that would double-count
        # streamed payloads). Stream totals are reported separately as an informational figure.
        totals = {
            "sent_msg_sizes": [0.0, 0],
            "received_msg_sizes": [0.0, 0],
            "sent_stream_sizes": [0.0, 0],
            "received_stream_sizes": [0.0, 0],
        }
        found = False
        for name, pool in dict(StatsPoolManager.pools).items():
            if not isinstance(pool, HistPool):
                continue
            base_name = name.split("@", 1)[0]
            if base_name not in totals:
                continue
            found = True
            # the cell may still be recording messages (END_RUN acks, heartbeats); hold the
            # pool's own lock so a concurrent first-time category insert cannot break iteration
            lock = getattr(pool, "update_lock", None)
            try:
                if lock is not None:
                    with lock:
                        pool_total = sum(b.total for bins in pool.cat_bins.values() for b in bins if b)
                        pool_count = sum(b.count for bins in pool.cat_bins.values() for b in bins if b)
                else:
                    pool_total = sum(b.total for bins in pool.cat_bins.values() for b in bins if b)
                    pool_count = sum(b.count for bins in pool.cat_bins.values() for b in bins if b)
            except RuntimeError:
                # dict changed size during iteration (no lock available); skip this pool
                continue
            totals[base_name][0] += pool_total
            totals[base_name][1] += pool_count
        if not found:
            return {}
        return {
            "total_sent_mb": totals["sent_msg_sizes"][0],
            "total_received_mb": totals["received_msg_sizes"][0],
            "item_count_sent": totals["sent_msg_sizes"][1],
            "item_count_received": totals["received_msg_sizes"][1],
            "streamed_sent_mb": totals["sent_stream_sizes"][0],
            "streamed_received_mb": totals["received_stream_sizes"][0],
        }

    def _build_communication_summary(self) -> dict:
        clients = {}
        all_records = self._completed_tasks + list(self._open_tasks.values())
        for rec in all_records:
            item = clients.setdefault(rec.client, {"task_payloads": [], "model_updates": []})
            if rec.payload_mb is not None:
                item["task_payloads"].append(rec.payload_mb)
            if rec.update_mb is not None:
                item["model_updates"].append(rec.update_mb)

        client_rows = {}
        for client, values in sorted(clients.items()):
            client_rows[client] = {
                "server_sent_client_received_mb": sum(values["task_payloads"]),
                "client_sent_server_received_mb": sum(values["model_updates"]),
                "model_update_size_mb": _numeric_stats(values["model_updates"]),
                "download_size_mb": _numeric_stats(values["task_payloads"]),
            }
        total_sent = sum(v["server_sent_client_received_mb"] for v in client_rows.values())
        total_received = sum(v["client_sent_server_received_mb"] for v in client_rows.values())
        return {
            "server": {"total_sent_mb": total_sent, "total_received_mb": total_received},
            "clients": client_rows,
            "model_update_size_mb": _numeric_stats(
                [rec.update_mb for rec in self._completed_tasks if rec.update_mb is not None]
            ),
            "download_size_mb": _numeric_stats([rec.payload_mb for rec in all_records if rec.payload_mb is not None]),
            "transport": self._get_transport_comm_stats(),
        }

    def _build_client_round_timing(self) -> List[dict]:
        groups = {}
        for rec in self._completed_tasks:
            key = (rec.workflow, rec.round_num, rec.client)
            groups.setdefault(key, []).append(rec)
        rows = []
        for (workflow, round_num, client), records in sorted(
            groups.items(), key=lambda x: (x[0][0], _round_sort_key(x[0][1]), x[0][2])
        ):
            computation = [r.client_execution_time for r in records if r.client_execution_time is not None]
            overhead = [r.client_framework_overhead for r in records if r.client_framework_overhead is not None]
            communication = [r.communication_time for r in records if r.communication_time is not None]
            rows.append(
                {
                    "workflow": workflow,
                    "round": round_num,
                    "client": client,
                    "computation_time": sum(computation) if computation else None,
                    "client_framework_overhead": sum(overhead) if overhead else None,
                    "communication_time": sum(communication) if communication else None,
                    "task_count": len(records),
                }
            )
        return rows

    def _build_resource_summary(self) -> dict:
        result = {}
        by_client = {}
        for rec in self._completed_tasks:
            if rec.resources:
                by_client.setdefault(rec.client, []).append(rec.resources)
        for client, resources in sorted(by_client.items()):
            row = {}
            for key in ("cpu_percent", "memory_rss_mb", "gpu_percent", "gpu_memory_mb"):
                combined = _combine_numeric_stats([r[key] for r in resources if r.get(key)])
                if combined:
                    row[key] = combined
            battery_pairs = [
                (r["battery_start_percent"], r["battery_end_percent"])
                for r in resources
                if "battery_start_percent" in r and "battery_end_percent" in r
            ]
            if battery_pairs:
                row["battery_start_percent"] = battery_pairs[0][0]
                row["battery_end_percent"] = battery_pairs[-1][1]
                row["battery_used_percent"] = sum(max(0.0, start - end) for start, end in battery_pairs)
            plugged_values = [r["battery_plugged"] for r in resources if isinstance(r.get("battery_plugged"), bool)]
            if plugged_values:
                # distinguishes "0% drain because charging" from genuine idle
                row["battery_plugged"] = plugged_values[-1]
            result[client] = row
        return result

    def get_summary(self, fl_ctx: Optional[FLContext] = None) -> dict:
        """Build the structured server-side job statistics summary."""
        with self._lock:
            status, status_reason = self._determine_status(fl_ctx)
            ordered_keys = sorted(self._rounds, key=lambda k: (k[0], _round_sort_key(k[1])))
            ordered_rounds = [self._rounds[k] for k in ordered_keys]
            completed_by_round = self._completed_ok_clients_by_round()
            round_rows = []
            for rec in ordered_rounds:
                stats = rec.aggr_stats or {}
                participants = self._round_participants(rec, completed_by_round)
                targeted = len(rec.targeted_clients)
                accepted = len(participants & rec.targeted_clients) if targeted else len(participants)
                missing = sorted(rec.targeted_clients - participants)
                round_rows.append(
                    {
                        "workflow": rec.workflow,
                        "round": rec.round_num,
                        "targeted_clients": sorted(rec.targeted_clients),
                        "accepted_client_names": sorted(participants),
                        "accepted_clients": accepted,
                        "total_clients": targeted,
                        "participation_rate_percent": 100.0 * accepted / targeted if targeted else None,
                        "missing_clients": missing,
                        "rejected_clients": sorted(rec.rejected_clients),
                        "keys_aggregated": stats.get(AggregationStatsKey.KEYS_AGGREGATED),
                        "keys_seen": stats.get(AggregationStatsKey.KEYS_SEEN),
                        "fully_matched_keys": stats.get(AggregationStatsKey.FULLY_MATCHED_KEYS),
                        "partially_matched_keys": stats.get(AggregationStatsKey.PARTIALLY_MATCHED_KEYS),
                        "skipped_keys": stats.get(AggregationStatsKey.SKIPPED_KEYS),
                        "started_at": rec.started_at,
                        "ended_at": rec.ended_at,
                        "server_training_time": rec.duration,
                        "duration": rec.duration,
                        "aggregation_stats": dict(stats),
                    }
                )

            # The final global model is the result of the last COMPLETED aggregation; a round
            # that was started (or aborted) without aggregating must not be presented as final.
            final_aggregation = None
            final_source = next((row for row in reversed(round_rows) if row["aggregation_stats"]), None)
            if final_source:
                final_aggregation = dict(final_source)
                final_aggregation["source_round"] = final_aggregation["round"]
                final_aggregation["round"] = "Final"
                final_aggregation["server_training_time"] = None
                final_aggregation["duration"] = None

            targeted, accepted = self._participation_counts()
            incomplete = list(self._open_tasks.values())
            failed_clients = sorted(
                set(self._client_errors)
                | self._disconnected_clients
                | {r.client for r in incomplete}
                | {c for rec in ordered_rounds for c in rec.rejected_clients}
            )
            # report the basis actually used by _round_participants: acceptance when a round
            # carried accepted/rejected names, completed-OK tasks otherwise
            bases = {
                "accepted_contributions" if (rec.accepted_clients or rec.rejected_clients) else "completed_tasks"
                for rec in ordered_rounds
                if rec.targeted_clients or rec.accepted_clients or rec.rejected_clients
            }
            participation_basis = "mixed" if len(bases) > 1 else next(iter(bases), "completed_tasks")

            job_duration = None
            if self._job_start_mono is not None:
                job_duration = (self._job_end_mono or time.monotonic()) - self._job_start_mono

            task_rows = [rec.to_dict() for rec in self._completed_tasks + incomplete]
            computations = [
                r.client_execution_time for r in self._completed_tasks if r.client_execution_time is not None
            ]
            overheads = [
                r.client_framework_overhead for r in self._completed_tasks if r.client_framework_overhead is not None
            ]
            communications = [r.communication_time for r in self._completed_tasks if r.communication_time is not None]
            task_groups = {}
            for rec in self._completed_tasks:
                task_groups.setdefault(rec.task_name, []).append(rec.server_elapsed_time)

            client_rounds = self._build_client_round_timing()
            round_durations = {(rec.workflow, rec.round_num): rec.duration for rec in ordered_rounds}
            comm_per_client_round = [
                row["communication_time"] for row in client_rounds if row["communication_time"] is not None
            ]
            comm_overhead_ratios = [
                100.0 * row["communication_time"] / round_durations[(row["workflow"], row["round"])]
                for row in client_rounds
                if row["communication_time"] is not None and round_durations.get((row["workflow"], row["round"]))
            ]

            return {
                "job_id": fl_ctx.get_job_id() if fl_ctx else None,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "job_start_time": self._job_start_time,
                "job_end_time": self._job_end_time,
                "status": status,
                "status_reason": status_reason,
                "rounds": round_rows,
                "final_aggregation": final_aggregation,
                "consistency": self._get_consistency_summary(ordered_rounds),
                "participation": {
                    "total_rounds": len(ordered_rounds),
                    "known_clients": sorted(self._all_clients),
                    "total_clients": len(self._all_clients),
                    "participation_basis": participation_basis,
                    "targeted_client_rounds": targeted,
                    "accepted_client_rounds": accepted,
                    "avg_clients_per_round": _mean_std([r["accepted_clients"] for r in round_rows]),
                    "participation_rate_percent": 100.0 * accepted / targeted if targeted else None,
                    "failure_rate_percent": 100.0 * (targeted - accepted) / targeted if targeted else None,
                    "failed_clients": failed_clients,
                    "client_errors": {c: list(codes) for c, codes in self._client_errors.items()},
                    "disconnected_clients": sorted(self._disconnected_clients),
                    "incomplete_tasks": len(incomplete),
                    "error_log_location": (
                        self._error_filename if self._error_filename and self._error_details else None
                    ),
                },
                "timing": {
                    "total_job_time": job_duration,
                    "server_round_time": _numeric_stats([r.duration for r in ordered_rounds if r.duration is not None]),
                    "avg_round_time": _mean_std([r.duration for r in ordered_rounds if r.duration is not None]),
                    "client_computation_time": _numeric_stats(computations),
                    "client_framework_overhead": _numeric_stats(overheads),
                    "communication_time": _numeric_stats(communications),
                    "total_client_computation_time": sum(computations) if computations else None,
                    "total_client_framework_overhead": sum(overheads) if overheads else None,
                    "total_communication_time": sum(communications) if communications else None,
                    "avg_client_task_time": _mean_std(computations),
                    "communication_time_per_client_round": _numeric_stats(comm_per_client_round),
                    "communication_overhead_percent": _numeric_stats(comm_overhead_ratios),
                    "client_rounds": client_rounds,
                    "tasks": task_rows,
                    "task_breakdown": {
                        name: _numeric_stats([d for d in durations if d is not None])
                        for name, durations in sorted(task_groups.items())
                    },
                },
                "communication": self._build_communication_summary(),
                "resources": self._build_resource_summary(),
                "errors": list(self._error_details),
                "report_files": {
                    "text": self._filename,
                    "json": self._json_filename,
                    "errors": self._error_filename if self._error_details else None,
                },
            }

    def format_report(self, summary: dict) -> str:
        """Render the structured summary as a human-readable run summary log."""

        def na(value):
            return "N/A" if value is None else value

        lines = ["=" * 100, "NVFLARE Job Stats Run Summary", "=" * 100]
        lines.append(f"Job ID       : {na(summary.get('job_id'))}")
        lines.append(f"Generated At : {summary.get('generated_at')}")
        lines.append("")
        lines.append("Job Status")
        lines.append("-" * 100)
        lines.extend(
            _format_table(
                ["Metric", "Value"],
                [
                    ["Status Code", _STATUS_DISPLAY.get(summary["status"], summary["status"])],
                    ["Details", summary["status_reason"]],
                ],
            )
        )
        lines.append("")

        rounds = summary["rounds"]
        final_aggregation = summary.get("final_aggregation")
        suffix = " + final" if final_aggregation else ""
        lines.append(f"Aggregation Stats Summary ({len(rounds)} rounds{suffix})")
        lines.append("-" * 100)
        if rounds:
            rows = []
            display_rows = list(rounds)
            if final_aggregation:
                display_rows.append(final_aggregation)
            show_workflow = len({item["workflow"] for item in display_rows}) > 1
            for item in display_rows:
                keys = (
                    "N/A" if item["keys_aggregated"] is None else f"{item['keys_aggregated']} / {na(item['keys_seen'])}"
                )
                rate = item["participation_rate_percent"]
                row = [
                    item["round"],
                    f"{item['accepted_clients']}/{item['total_clients']}",
                    f"{rate:.1f}%" if rate is not None else "N/A",
                    ", ".join(item["missing_clients"]) or "None",
                    _format_duration(item["server_training_time"]),
                    keys,
                    na(item["fully_matched_keys"]),
                    na(item["partially_matched_keys"]),
                    na(item["skipped_keys"]),
                ]
                if show_workflow:
                    row.insert(0, item["workflow"])
                rows.append(row)
            headers = [
                "Round",
                "Accepted Clients",
                "Participation",
                "Missing/Rejected",
                "Server Time",
                "Keys Aggregated / Seen",
                "Fully Matched Keys",
                "Partially Matched Keys",
                "Skipped Keys",
            ]
            if show_workflow:
                headers.insert(0, "Workflow")
            lines.extend(_format_table(headers, rows))
        else:
            lines.append("No round data collected.")
        lines.append("")

        cons = summary["consistency"]
        lines.append("Aggregation Consistency Across Rounds")
        lines.append("-" * 100)
        workflow_consistency = cons.get("workflows") or []
        consistency_sections = workflow_consistency if len(workflow_consistency) > 1 else [cons]
        for index, item in enumerate(consistency_sections):
            if len(consistency_sections) > 1:
                if index:
                    lines.append("")
                lines.append(f"Workflow {item['workflow']}")
            if item["stable_rounds"] is not None:
                inconsistent = ", ".join(f"Round {r}" for r in item["inconsistent_rounds"]) or "None"
                lines.extend(
                    _format_table(
                        ["Metric", "Value / Notes"],
                        [
                            ["Total Rounds", item["total_rounds"]],
                            ["Stable Rounds", f"{item['stable_rounds']} / {item['total_rounds']}"],
                            ["Keys Aggregated", item["keys_aggregated"]],
                            ["Fully Matched Keys", item["fully_matched_keys"]],
                            ["Partially Matched Keys", item["partially_matched_keys"]],
                            ["Skipped Keys", item["skipped_keys"]],
                            ["Inconsistent Rounds", inconsistent],
                        ],
                    )
                )
            else:
                lines.append("No aggregation key stats available (aggregator did not publish them).")
        lines.append("")

        part = summary["participation"]
        lines.append("Client Participation and Failures")
        lines.append("-" * 100)
        participation_rate = part["participation_rate_percent"]
        failure_rate = part["failure_rate_percent"]
        failed = "None"
        if part["failed_clients"]:
            failed = ", ".join(part["failed_clients"])
            if part["total_clients"]:
                failed = f"[{failed}] ({len(part['failed_clients'])}/{part['total_clients']})"
        part_rows = [
            ["Total Rounds", part["total_rounds"]],
            ["Total Clients", part["total_clients"]],
            ["Avg Clients Participated / Round", part["avg_clients_per_round"]],
            ["Targeted Client-Rounds", part["targeted_client_rounds"]],
            ["Accepted Client-Rounds", part["accepted_client_rounds"]],
            ["Participation Rate", f"{participation_rate:.1f}%" if participation_rate is not None else "N/A"],
            ["Failure/Dropout Rate", f"{failure_rate:.1f}%" if failure_rate is not None else "N/A"],
            ["Client Failures", failed],
            ["Disconnected Clients", ", ".join(part["disconnected_clients"]) or "None"],
            ["Incomplete Tasks", part["incomplete_tasks"]],
        ]
        if part.get("error_log_location"):
            part_rows.append(["Error Log Location", part["error_log_location"]])
        lines.extend(_format_table(["Metric", "Value / Notes"], part_rows))
        lines.append("")

        def stats_cell(stats: Optional[dict], unit: str, precision: int = 3) -> str:
            value = _format_stats(stats, precision)
            return value if value == "N/A" else f"{value}{unit}"

        timing = summary["timing"]
        lines.append("Computation and Communication Time")
        lines.append("-" * 100)
        timing_rows = [
            ["Total Job Time", _format_duration(timing["total_job_time"])],
            ["Avg Server Round Time (mean ± std)", stats_cell(timing["server_round_time"], " sec")],
            ["Avg Client Computation Time (mean ± std)", stats_cell(timing["client_computation_time"], " sec")],
            ["Total Client Computation Time", _format_duration(timing["total_client_computation_time"])],
            ["Total Client Framework Overhead", _format_duration(timing["total_client_framework_overhead"])],
            ["Total Communication/Wait Time", _format_duration(timing["total_communication_time"])],
            [
                "Time in Communication / Round (per client, mean ± std)",
                stats_cell(timing["communication_time_per_client_round"], " sec"),
            ],
            [
                "Communication Overhead",
                stats_cell(timing["communication_overhead_percent"], "% of round time", 1),
            ],
        ]
        if timing["task_breakdown"]:
            timing_rows.append(["Task Timing Breakdown (server elapsed, mean ± std)", ""])
            for task_name, stats in timing["task_breakdown"].items():
                timing_rows.append([f"  {task_name}", stats_cell(stats, " sec")])
        lines.extend(_format_table(["Metric", "Value"], timing_rows))
        if timing["client_rounds"]:
            lines.append("")
            lines.append("Per-Client Per-Round Time")
            client_rounds = timing["client_rounds"]
            show_workflow = len({row["workflow"] for row in client_rounds}) > 1
            headers = ["Round", "Client", "Tasks", "Computation", "Client Overhead", "Communication/Wait"]
            rows = [
                [
                    row["round"],
                    row["client"],
                    row["task_count"],
                    _format_duration(row["computation_time"]),
                    _format_duration(row["client_framework_overhead"]),
                    _format_duration(row["communication_time"]),
                ]
                for row in client_rounds
            ]
            if show_workflow:
                headers.insert(0, "Workflow")
                for row, values in zip(rows, client_rounds):
                    row.insert(0, values["workflow"])
            lines.extend(_format_table(headers, rows))
        if timing["tasks"]:
            lines.append("")
            lines.append("Every Task Execution")
            tasks = timing["tasks"]
            show_workflow = len({task["workflow"] for task in tasks}) > 1
            headers = [
                "Round",
                "Client",
                "Task",
                "Task ID",
                "Server Elapsed",
                "Client Compute",
                "Client Overhead",
                "Comm/Wait",
                "RC",
            ]
            rows = [
                [
                    task["round"],
                    task["client"],
                    task["task_name"],
                    task["task_id"],
                    _format_duration(task["server_elapsed_time"]),
                    _format_duration(task["client_computation_time"]),
                    _format_duration(task["client_framework_overhead"]),
                    _format_duration(task["communication_time"]),
                    task["return_code"],
                ]
                for task in tasks
            ]
            if show_workflow:
                headers.insert(0, "Workflow")
                for row, values in zip(rows, tasks):
                    row.insert(0, values["workflow"])
            lines.extend(_format_table(headers, rows))
        lines.append("")

        communication = summary["communication"]
        lines.append("Communication Statistics (logical task/model payloads)")
        lines.append("-" * 100)
        server_comm = communication["server"]
        lines.extend(
            _format_table(
                ["Server Metric", "Value"],
                [
                    ["Total Data Sent (server)", _format_mb(server_comm["total_sent_mb"])],
                    ["Total Data Received (server)", _format_mb(server_comm["total_received_mb"])],
                    [
                        "Avg Upload Size per Client (model update, mean ± std)",
                        stats_cell(communication["model_update_size_mb"], " MB"),
                    ],
                    [
                        "Avg Download Size per Client (task payload, mean ± std)",
                        stats_cell(communication["download_size_mb"], " MB"),
                    ],
                ],
            )
        )
        if communication["clients"]:
            lines.append("")
            lines.extend(
                _format_table(
                    [
                        "Client",
                        "Received from Server",
                        "Sent to Server",
                        "Upload MB (mean ± std)",
                        "Download MB (mean ± std)",
                    ],
                    [
                        [
                            client,
                            _format_mb(values["server_sent_client_received_mb"]),
                            _format_mb(values["client_sent_server_received_mb"]),
                            _format_stats(values["model_update_size_mb"], 3),
                            _format_stats(values["download_size_mb"], 3),
                        ]
                        for client, values in communication["clients"].items()
                    ],
                )
            )
        transport = communication["transport"]
        if transport:
            lines.append("")
            lines.append(
                "Server transport pools (all protocol messages, including stream chunks): "
                f"sent {_format_mb(transport['total_sent_mb'])}, received {_format_mb(transport['total_received_mb'])}"
                f" (of which streamed payloads: sent {_format_mb(transport['streamed_sent_mb'])},"
                f" received {_format_mb(transport['streamed_received_mb'])})."
            )
        lines.append("")

        lines.append("System Resource Utilization by Client")
        lines.append("-" * 100)
        resources = summary["resources"]
        if resources:
            lines.extend(
                _format_table(
                    [
                        "Client",
                        "Avg CPU % (mean ± std)",
                        "Process GPU % (mean ± std)",
                        "Avg Memory MB (mean ± std)",
                        "Process GPU Memory MB (mean ± std)",
                        "Battery Used %",
                    ],
                    [
                        [
                            client,
                            self._format_resource(values.get("cpu_percent")),
                            self._format_resource(values.get("gpu_percent")),
                            self._format_resource(values.get("memory_rss_mb")),
                            self._format_resource(values.get("gpu_memory_mb")),
                            self._format_battery(values),
                        ]
                        for client, values in resources.items()
                    ],
                )
            )
        else:
            lines.append("No client resource telemetry received. Configure JobStatsReporter on clients to collect it.")
        lines.append("")

        lines.append("Errors")
        lines.append("-" * 100)
        if summary["errors"]:
            lines.extend(self.format_errors(summary["errors"]).rstrip().splitlines())
        else:
            lines.append("None")
        lines.append("")
        lines.append(
            "Communication/wait time is server assignment-to-result elapsed time minus client-local processing "
            "(task-data filtering + execution); client overhead covers client-side framework work before execution. "
            "Result filtering and transmission are included in communication/wait time."
        )
        return "\n".join(lines)

    @staticmethod
    def _format_resource(value: Optional[dict]) -> str:
        return "N/A" if not value else f"{value['mean']:.1f} ± {value['stddev']:.1f}"

    @staticmethod
    def _format_battery(values: dict) -> str:
        used = values.get("battery_used_percent")
        if used is None:
            return "N/A"
        plugged = values.get("battery_plugged")
        suffix = " (plugged in)" if plugged is True else " (on battery)" if plugged is False else ""
        return f"{used:.1f}{suffix}"

    @staticmethod
    def format_errors(errors: List[dict]) -> str:
        if not errors:
            return "No errors.\n"
        return (
            "\n".join(
                f"client={e['client']} round={e['round']} task={e['task_name']} task_id={e['task_id']} "
                f"code={e['code']} message={e['message'] or 'N/A'}"
                for e in errors
            )
            + "\n"
        )
