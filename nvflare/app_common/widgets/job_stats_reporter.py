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
import threading
import time
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.aggregators.weighted_aggregation_helper import AggregationStatsKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.fuel.f3.stats_pool import HistPool, StatsPoolManager
from nvflare.widgets.widget import Widget


class JobStatusCode:
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL SUCCESS"
    FAILURE = "FAILURE"


_STATUS_DISPLAY = {
    JobStatusCode.SUCCESS: "✅ SUCCESS",
    JobStatusCode.PARTIAL_SUCCESS: "⚠️ PARTIAL SUCCESS",
    JobStatusCode.FAILURE: "❌ FAILURE",
}

_MSG_SIZE_POOL_PREFIXES = ("sent_msg_sizes", "received_msg_sizes")
_STREAM_SIZE_POOL_PREFIXES = ("sent_stream_sizes", "received_stream_sizes")


def _mean_std(values: List[float], precision: int = 1) -> str:
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.{precision}f}"
    return f"{mean(values):.{precision}f} ± {pstdev(values):.{precision}f}"


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    return f"{seconds:.1f} seconds"


def _format_mb(size_mb: Optional[float]) -> str:
    if size_mb is None:
        return "N/A"
    if size_mb >= 1024.0:
        return f"{size_mb / 1024.0:.2f} GB"
    return f"{size_mb:.2f} MB"


def _format_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    lines = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return lines


class _RoundRecord:
    def __init__(self, round_num, start_time: float):
        self.round_num = round_num
        self.start_time = start_time
        self.end_time: Optional[float] = None
        self.aggr_stats: Optional[dict] = None
        self.accepted_clients = set()
        self.rejected_clients = set()

    @property
    def duration(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


class JobStatsReporter(Widget):
    def __init__(self, filename: str = "job_stats_run_summary.log", json_filename: Optional[str] = None):
        """Collects per-round job statistics on the server and writes a run summary log at the end of the job.

        The summary contains the job status, a per-round aggregation stats table (accepted clients, keys
        aggregated/seen, fully/partially matched keys, skipped keys), aggregation consistency across rounds,
        client participation and failures, computation timing, and communication statistics.

        The aggregation key stats are published to the FLContext (AppConstants.AGGREGATION_STATS) by the
        built-in weighted aggregation path (InTimeAccumulateWeightedAggregator and FedAvg); with custom
        aggregators that do not publish them, those columns are reported as N/A.

        To use, add this component to the server app config (config_fed_server.json):

        .. code-block:: json

            {
              "id": "job_stats_reporter",
              "path": "nvflare.app_common.widgets.job_stats_reporter.JobStatsReporter",
              "args": {}
            }

        Args:
            filename: name of the summary log file, written to the job's run dir on the server.
            json_filename: optional name of a JSON file with the same summary in structured form.
        """
        super().__init__()
        self._filename = filename
        self._json_filename = json_filename
        self._lock = threading.Lock()
        self._reset()

    def _reset(self):
        self._job_start_time: Optional[float] = None
        self._job_end_time: Optional[float] = None
        self._all_clients = set()
        self._rounds: Dict[int, _RoundRecord] = {}
        self._open_tasks: Dict[Tuple[str, str], Tuple[str, float]] = {}  # (client, task_id) -> (task_name, time)
        self._task_durations: Dict[str, List[float]] = {}  # task_name -> durations of completed tasks
        self._submission_counts: Dict[str, int] = {}  # client -> num results received
        self._client_errors: Dict[str, List[str]] = {}  # client -> return codes of error results
        self._disconnected_clients = set()
        self._fatal_error_reason: Optional[str] = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        try:
            if event_type == EventType.START_RUN:
                self._handle_start_run(fl_ctx)
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
                client_name = fl_ctx.get_prop(FLContextKey.DISCONNECTED_CLIENT_NAME)
                if client_name:
                    with self._lock:
                        self._disconnected_clients.add(client_name)
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
        if isinstance(peer_ctx, FLContext):
            return peer_ctx.get_identity_name()
        return None

    def _handle_start_run(self, fl_ctx: FLContext):
        with self._lock:
            self._reset()
            self._job_start_time = time.time()
            engine = fl_ctx.get_engine()
            if engine:
                clients = engine.get_clients()
                if clients:
                    self._all_clients = {c.name for c in clients}

    def _handle_round_started(self, fl_ctx: FLContext):
        round_num = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        if round_num is None:
            return
        now = time.time()
        with self._lock:
            # close any still-open earlier round (FedAvg-style flows never fire ROUND_DONE)
            for rec in self._rounds.values():
                if rec.end_time is None and rec.round_num != round_num:
                    rec.end_time = now
            if round_num not in self._rounds:
                self._rounds[round_num] = _RoundRecord(round_num, now)

    def _close_round(self, round_num):
        if round_num is None:
            return
        with self._lock:
            rec = self._rounds.get(round_num)
            if rec:
                rec.end_time = time.time()

    def _get_or_create_round(self, round_num) -> _RoundRecord:
        # must be called with the lock held
        rec = self._rounds.get(round_num)
        if not rec:
            rec = _RoundRecord(round_num, time.time())
            self._rounds[round_num] = rec
        return rec

    def _handle_after_aggregation(self, fl_ctx: FLContext):
        stats = fl_ctx.get_prop(AppConstants.AGGREGATION_STATS)
        round_num = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        if isinstance(stats, dict):
            round_num = stats.get(AggregationStatsKey.ROUND, round_num)
        if round_num is None:
            return
        with self._lock:
            rec = self._get_or_create_round(round_num)
            rec.end_time = time.time()
            if isinstance(stats, dict):
                rec.aggr_stats = stats
                contributors = stats.get(AggregationStatsKey.CONTRIBUTORS)
                if contributors:
                    rec.accepted_clients.update(contributors)
                    self._all_clients.update(contributors)

    def _handle_contribution_accept(self, fl_ctx: FLContext):
        round_num = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        client_name = self._get_peer_name(fl_ctx)
        if round_num is None or not client_name:
            return
        accepted = fl_ctx.get_prop(AppConstants.AGGREGATION_ACCEPTED, True)
        with self._lock:
            rec = self._get_or_create_round(round_num)
            self._all_clients.add(client_name)
            if accepted:
                rec.accepted_clients.add(client_name)
            else:
                rec.rejected_clients.add(client_name)

    def _handle_task_assigned(self, fl_ctx: FLContext):
        task_name = fl_ctx.get_prop(FLContextKey.TASK_NAME)
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        client_name = self._get_peer_name(fl_ctx)
        if not task_name or not task_id or not client_name:
            return
        with self._lock:
            self._all_clients.add(client_name)
            self._open_tasks[(client_name, task_id)] = (task_name, time.time())

    def _handle_submission(self, fl_ctx: FLContext):
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        client_name = self._get_peer_name(fl_ctx)
        if not task_id or not client_name:
            return
        result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
        now = time.time()
        with self._lock:
            self._submission_counts[client_name] = self._submission_counts.get(client_name, 0) + 1
            dispatched = self._open_tasks.pop((client_name, task_id), None)
            if dispatched:
                task_name, dispatch_time = dispatched
                self._task_durations.setdefault(task_name, []).append(now - dispatch_time)
            if isinstance(result, Shareable):
                rc = result.get_return_code(default=ReturnCode.OK)
                if rc != ReturnCode.OK:
                    self._client_errors.setdefault(client_name, []).append(str(rc))

    def _handle_end_run(self, fl_ctx: FLContext):
        with self._lock:
            self._job_end_time = time.time()
            for rec in self._rounds.values():
                if rec.end_time is None:
                    rec.end_time = self._job_end_time
        summary = self.get_summary(fl_ctx)
        report = self.format_report(summary)
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
        os.makedirs(run_dir, exist_ok=True)
        report_path = os.path.join(run_dir, self._filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        if self._json_filename:
            with open(os.path.join(run_dir, self._json_filename), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
        self.log_info(fl_ctx, f"job stats run summary saved to {report_path}")

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

        issues = []
        if self._client_errors:
            issues.append(f"error results from clients: {sorted(self._client_errors)}")
        if self._disconnected_clients:
            issues.append(f"disconnected clients: {sorted(self._disconnected_clients)}")
        if self._all_clients:
            short_rounds = [
                rec.round_num
                for rec in self._rounds.values()
                if rec.accepted_clients and len(rec.accepted_clients) < len(self._all_clients)
            ]
            if short_rounds:
                issues.append(f"rounds with missing client contributions: {sorted(short_rounds)}")
        if issues:
            return JobStatusCode.PARTIAL_SUCCESS, "; ".join(issues)
        return JobStatusCode.SUCCESS, "all rounds completed with full participation"

    @staticmethod
    def _round_pattern(rec: _RoundRecord) -> Optional[tuple]:
        stats = rec.aggr_stats
        if not stats:
            return None
        return (
            len(rec.accepted_clients) or stats.get(AggregationStatsKey.ACCEPTED_CONTRIBUTIONS),
            stats.get(AggregationStatsKey.KEYS_AGGREGATED),
            stats.get(AggregationStatsKey.FULLY_MATCHED_KEYS),
            stats.get(AggregationStatsKey.PARTIALLY_MATCHED_KEYS),
            stats.get(AggregationStatsKey.SKIPPED_KEYS),
        )

    def _get_consistency_summary(self, ordered_rounds: List[_RoundRecord]) -> dict:
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
        for p in known:
            counts[p] = counts.get(p, 0) + 1
        dominant = max(counts, key=counts.get)
        result["stable_rounds"] = counts[dominant]
        result["inconsistent_rounds"] = [
            rec.round_num for rec, p in zip(ordered_rounds, patterns) if p is not None and p != dominant
        ]
        for i, key in enumerate(
            ["", "keys_aggregated", "fully_matched_keys", "partially_matched_keys", "skipped_keys"]
        ):
            if key:
                result[key] = _mean_std([float(p[i]) for p in known if p[i] is not None])
        return result

    @staticmethod
    def _get_comm_stats() -> dict:
        sent_mb, sent_count, received_mb, received_count = 0.0, 0, 0.0, 0
        found = False
        for name, pool in dict(StatsPoolManager.pools).items():
            if not isinstance(pool, HistPool):
                continue
            base_name = name.split("@", 1)[0]
            if base_name not in _MSG_SIZE_POOL_PREFIXES and base_name not in _STREAM_SIZE_POOL_PREFIXES:
                continue
            found = True
            total = 0.0
            count = 0
            for bins in pool.cat_bins.values():
                for b in bins:
                    if b:
                        total += b.total
                        count += b.count
            if base_name.startswith("sent"):
                sent_mb += total
                sent_count += count
            else:
                received_mb += total
                received_count += count
        if not found:
            return {}
        return {
            "total_sent_mb": sent_mb,
            "total_received_mb": received_mb,
            "num_msgs_sent": sent_count,
            "num_msgs_received": received_count,
            "avg_sent_msg_mb": sent_mb / sent_count if sent_count else None,
            "avg_received_msg_mb": received_mb / received_count if received_count else None,
        }

    def get_summary(self, fl_ctx: Optional[FLContext] = None) -> dict:
        """Build the structured job stats summary from the collected data."""
        with self._lock:
            status, status_reason = self._determine_status(fl_ctx)
            ordered_rounds = [self._rounds[r] for r in sorted(self._rounds)]

            round_rows = []
            for rec in ordered_rounds:
                stats = rec.aggr_stats or {}
                accepted = len(rec.accepted_clients) or stats.get(AggregationStatsKey.ACCEPTED_CONTRIBUTIONS)
                round_rows.append(
                    {
                        "round": rec.round_num,
                        "accepted_clients": accepted,
                        "total_clients": len(self._all_clients) or None,
                        "keys_aggregated": stats.get(AggregationStatsKey.KEYS_AGGREGATED),
                        "keys_seen": stats.get(AggregationStatsKey.KEYS_SEEN),
                        "fully_matched_keys": stats.get(AggregationStatsKey.FULLY_MATCHED_KEYS),
                        "partially_matched_keys": stats.get(AggregationStatsKey.PARTIALLY_MATCHED_KEYS),
                        "skipped_keys": stats.get(AggregationStatsKey.SKIPPED_KEYS),
                        "duration": rec.duration,
                    }
                )

            participated_counts = [float(len(rec.accepted_clients)) for rec in ordered_rounds if rec.accepted_clients]
            num_slots = len(ordered_rounds) * len(self._all_clients)
            num_missing = sum(
                len(self._all_clients) - len(rec.accepted_clients) for rec in ordered_rounds if rec.accepted_clients
            )
            failed_clients = sorted(set(self._client_errors) | self._disconnected_clients)

            job_duration = None
            if self._job_start_time is not None:
                end = self._job_end_time or time.time()
                job_duration = end - self._job_start_time

            all_task_durations = [d for durations in self._task_durations.values() for d in durations]

            return {
                "job_id": fl_ctx.get_job_id() if fl_ctx else None,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": status,
                "status_reason": status_reason,
                "rounds": round_rows,
                "consistency": self._get_consistency_summary(ordered_rounds),
                "participation": {
                    "total_rounds": len(ordered_rounds),
                    "total_clients": len(self._all_clients),
                    "avg_clients_per_round": _mean_std(participated_counts),
                    "failure_rate_percent": (100.0 * num_missing / num_slots) if num_slots else None,
                    "failed_clients": failed_clients,
                    "client_errors": {c: list(rcs) for c, rcs in self._client_errors.items()},
                    "disconnected_clients": sorted(self._disconnected_clients),
                    "incomplete_tasks": len(self._open_tasks),
                },
                "timing": {
                    "total_job_time": job_duration,
                    "avg_round_time": _mean_std([r.duration for r in ordered_rounds if r.duration is not None]),
                    "avg_client_task_time": _mean_std(all_task_durations),
                    "task_breakdown": {name: _mean_std(d) for name, d in sorted(self._task_durations.items())},
                },
                "communication": self._get_comm_stats(),
            }

    def format_report(self, summary: dict) -> str:
        """Render the structured summary as the human-readable run summary log."""

        def na(v):
            return "N/A" if v is None else v

        lines = []
        lines.append("=" * 78)
        lines.append("\U0001f4cb NVFLARE Job Stats Run Summary")
        lines.append("=" * 78)
        lines.append(f"Job ID       : {na(summary.get('job_id'))}")
        lines.append(f"Generated At : {summary.get('generated_at')}")
        lines.append("")

        lines.append("Job Status")
        lines.append("-" * 78)
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
        lines.append(f"Aggregation Stats Summary ({len(rounds)} rounds)")
        lines.append("-" * 78)
        if rounds:
            rows = []
            for r in rounds:
                accepted = na(r["accepted_clients"])
                if r["accepted_clients"] is not None and r["total_clients"]:
                    accepted = f"{r['accepted_clients']}/{r['total_clients']}"
                keys = "N/A"
                if r["keys_aggregated"] is not None:
                    keys = f"{r['keys_aggregated']} / {na(r['keys_seen'])}"
                rows.append(
                    [
                        r["round"],
                        accepted,
                        keys,
                        na(r["fully_matched_keys"]),
                        na(r["partially_matched_keys"]),
                        na(r["skipped_keys"]),
                    ]
                )
            lines.extend(
                _format_table(
                    [
                        "Round",
                        "Accepted Clients",
                        "Keys Aggregated / Seen",
                        "Fully Matched Keys",
                        "Partially Matched Keys",
                        "Skipped Keys",
                    ],
                    rows,
                )
            )
        else:
            lines.append("No round data collected.")
        lines.append("")

        cons = summary["consistency"]
        lines.append("Aggregation Consistency Across Rounds")
        lines.append("-" * 78)
        if cons["stable_rounds"] is not None:
            inconsistent = ", ".join(f"Round {r}" for r in cons["inconsistent_rounds"]) or "None"
            lines.extend(
                _format_table(
                    ["Metric", "Value / Notes"],
                    [
                        ["Total Rounds", cons["total_rounds"]],
                        ["Stable Rounds", f"{cons['stable_rounds']} / {cons['total_rounds']}"],
                        ["Keys Aggregated", cons["keys_aggregated"]],
                        ["Fully Matched Keys", cons["fully_matched_keys"]],
                        ["Partially Matched Keys", cons["partially_matched_keys"]],
                        ["Skipped Keys", cons["skipped_keys"]],
                        ["Inconsistent Rounds", inconsistent],
                    ],
                )
            )
        else:
            lines.append("No aggregation key stats available (aggregator did not publish them).")
        lines.append("")

        part = summary["participation"]
        lines.append("\U0001f504 Client Participation and Failures")
        lines.append("-" * 78)
        failure_rate = part["failure_rate_percent"]
        failed = ", ".join(part["failed_clients"]) if part["failed_clients"] else "None"
        if part["failed_clients"] and part["total_clients"]:
            failed = f"[{', '.join(part['failed_clients'])}] ({len(part['failed_clients'])}/{part['total_clients']})"
        part_rows = [
            ["Total Rounds", part["total_rounds"]],
            ["Total Clients", part["total_clients"]],
            ["Avg Clients Participated / Round", part["avg_clients_per_round"]],
            ["Failure Rate", f"{failure_rate:.1f}%" if failure_rate is not None else "N/A"],
            ["Client Failures", failed],
        ]
        if part["disconnected_clients"]:
            part_rows.append(["Disconnected Clients", ", ".join(part["disconnected_clients"])])
        if part["incomplete_tasks"]:
            part_rows.append(["Tasks Assigned But Never Completed", part["incomplete_tasks"]])
        for client, rcs in sorted(part["client_errors"].items()):
            part_rows.append([f"  Errors from {client}", ", ".join(rcs)])
        lines.extend(_format_table(["Metric", "Value / Notes"], part_rows))
        lines.append("")

        timing = summary["timing"]
        lines.append("⏱️ Computation Time")
        lines.append("-" * 78)
        timing_rows = [
            ["Total Job Time", _format_duration(timing["total_job_time"])],
            ["Avg Server Round Time", f"{timing['avg_round_time']} sec"],
            ["Avg Client Task Time (assign to result)", f"{timing['avg_client_task_time']} sec"],
        ]
        if timing["task_breakdown"]:
            timing_rows.append(["Task Timing Breakdown", ""])
            for task_name, stats in timing["task_breakdown"].items():
                timing_rows.append([f"  {task_name}", f"{stats} sec"])
        lines.extend(_format_table(["Metric", "Value"], timing_rows))
        lines.append(
            "Note: client task time is measured on the server from task assignment to result receipt;"
            " it includes communication time."
        )
        lines.append("")

        comm = summary["communication"]
        lines.append("\U0001f4e1 Communication Statistics")
        lines.append("-" * 78)
        if comm:
            lines.extend(
                _format_table(
                    ["Metric", "Value"],
                    [
                        ["Total Bytes Sent (server)", _format_mb(comm["total_sent_mb"])],
                        ["Total Bytes Received (server)", _format_mb(comm["total_received_mb"])],
                        ["Messages Sent / Received", f"{comm['num_msgs_sent']} / {comm['num_msgs_received']}"],
                        ["Avg Sent Message Size", _format_mb(comm["avg_sent_msg_mb"])],
                        ["Avg Received Message Size", _format_mb(comm["avg_received_msg_mb"])],
                    ],
                )
            )
        else:
            lines.append("No communication stats available (cellnet stats pools not found).")
        lines.append("")

        lines.append("⚙️ System Resource Utilization")
        lines.append("-" * 78)
        lines.append("Not collected: clients do not report CPU/GPU/memory usage to the server.")
        lines.append("")
        return "\n".join(lines)
