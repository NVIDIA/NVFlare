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

"""Human rendering for ``nvflare resources``."""

from __future__ import annotations

import math
import unicodedata

from nvflare.private.fed.resource_stats.measurements import sum_job_totals

_GIB = 2**30
_HOUR = 3600
_MISSING = "—"

_MEASUREMENT_LABELS = {
    "resource_time": "capacity-time",
    "cpu_consumed": "CPU time used",
    "message_traffic": "message traffic",
}


def _terminal_label(value: str) -> str:
    """Show metadata literally without letting controls alter terminal output."""

    escaped = []
    for char in value:
        code = ord(char)
        if unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}:
            if char in {"\n", "\r", "\t"}:
                escaped.append({"\n": r"\n", "\r": r"\r", "\t": r"\t"}[char])
            elif code <= 0xFF:
                escaped.append(f"\\x{code:02x}")
            elif code <= 0xFFFF:
                escaped.append(f"\\u{code:04x}")
            else:
                escaped.append(f"\\U{code:08x}")
        else:
            escaped.append(char)
    return "".join(escaped)


def _number(value, divisor=1) -> str:
    if value is None:
        return _MISSING
    try:
        amount = value / divisor
    except OverflowError:
        return _MISSING
    if not math.isfinite(amount):
        return _MISSING
    if value > 0 and amount < 0.01:
        return "<0.01"
    return f"{amount:.2f}".rstrip("0").rstrip(".")


def _duration(value) -> str:
    if value is None:
        return _MISSING
    seconds = int(value)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes}m{seconds}s"
    if minutes:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


def _average(value, duration_seconds) -> str:
    if value is None or duration_seconds is None or duration_seconds <= 0:
        return _MISSING
    return _number(value, duration_seconds)


def _marked(value: str, status: str | None) -> str:
    return f"{value}*" if value != _MISSING and status == "partial" else value


def _measurement(value: dict, field: str) -> dict:
    source = value if field in {"resource_capacity", "resource_time"} else value.get("resource_utilization", {})
    return source.get(field, {"status": "unavailable"})


def _capacity_time(value: dict, resource: str) -> str:
    field = {"cpu": "unit_seconds", "memory": "byte_seconds", "gpu": "instance_seconds"}[resource]
    amount = value.get("resource_time", {}).get(resource, {}).get(field)
    return _number(amount, _GIB * _HOUR if resource == "memory" else _HOUR)


def _capacity_row(value: dict) -> list[str]:
    status = value.get("resource_time", {}).get("status", "reported")
    return [_duration(value.get("duration_seconds"))] + [
        _marked(_capacity_time(value, resource), status) for resource in ("cpu", "memory", "gpu")
    ]


def _cpu_used(value: dict) -> str:
    consumed = value.get("resource_utilization", {}).get("cpu_consumed", {})
    return _marked(_average(consumed.get("cpu_seconds"), value.get("duration_seconds")), consumed.get("status"))


def _quantity(label: str, value: str, unit: str) -> str:
    return f"{label} {value}" if value == _MISSING else f"{label} {value} {unit}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    values = [headers] + [[str(value) for value in row] for row in rows]
    widths = [max(len(row[index]) for row in values) for index in range(len(headers))]
    return ["  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip() for row in values]


def _data_gaps(participants: list[dict]) -> list[str]:
    lines = []
    for entry in participants:
        name = entry["participant_name"]
        if "status" in entry:
            lines.append(f"  {name}: {entry['status']} report")
            continue
        gaps = []
        if "duration_seconds" not in entry:
            gaps.append("measurement duration unavailable")
        for field, label in _MEASUREMENT_LABELS.items():
            value = _measurement(entry, field)
            status = value.get("status", "reported")
            if status == "reported":
                continue
            gaps.append(f"{label} {'incomplete' if status == 'partial' else 'unavailable'}")
        if gaps:
            lines.append(f"  {name}: " + "; ".join(gaps))
    return lines


def _total_resource_quantities(totals: dict) -> list[str]:
    status = totals["resource_time"].get("status", "reported")
    return [
        _quantity(label, _marked(_capacity_time(totals, resource), status), unit)
        for label, resource, unit in (
            ("CPU visible", "cpu", "core-h"),
            ("Memory visible", "memory", "GiB-h"),
            ("GPUs available", "gpu", "GPU-h"),
        )
    ]


def _utilization_row(value: dict, show_message_traffic: bool, *, average_cpu=False) -> list[str]:
    utilization = value.get("resource_utilization", {})
    consumed = utilization.get("cpu_consumed", {})
    row = [
        (
            _cpu_used(value)
            if average_cpu
            else _marked(_number(consumed.get("cpu_seconds"), _HOUR), consumed.get("status"))
        )
    ]
    if show_message_traffic:
        traffic = utilization.get("message_traffic", {})
        row.extend(
            _marked(_number(traffic.get(field), _GIB), traffic.get("status"))
            for field in ("sent_bytes", "received_bytes")
        )
    return row


def _total_other_quantities(totals: dict, show_message_traffic: bool) -> list[str]:
    labels = [("CPU time used", "core-h")]
    if show_message_traffic:
        labels.extend([("Payload sent", "GiB"), ("Payload received", "GiB")])
    return [
        _quantity(label, value, unit)
        for (label, unit), value in zip(labels, _utilization_row(totals, show_message_traffic))
    ]


def _totals_lines(totals, show_message_traffic):
    return [
        "",
        "Recorded capacity-time totals",
        "  " + " | ".join(_total_resource_quantities(totals)),
        "",
        "Recorded resource utilization totals",
        "  " + " | ".join(_total_other_quantities(totals, show_message_traffic)),
    ]


def _startup_capacity(participant: dict) -> list[str]:
    capacity = participant.get("resource_capacity", {})
    return [
        _quantity("CPU:", _number(capacity.get("cpu", {}).get("units")), "visible units"),
        _quantity("RAM:", _number(capacity.get("memory", {}).get("bytes"), _GIB), "GiB"),
        _quantity("GPU:", str(capacity.get("gpu", {}).get("count", _MISSING)), "GPUs"),
    ]


def render_job_resources(summary: dict, participant: dict | None = None, *, job_name: str | None = None) -> str:
    participants = sorted(summary["participants"], key=lambda entry: (entry["role"], entry["participant_name"]))
    accepted = sum(("status" not in entry) for entry in participants)
    selected = participant.get("participant_name") if participant else None
    visible_entries = [entry for entry in participants if not selected or entry["participant_name"] == selected]
    show_message_traffic = any(
        ("status" not in entry) and "sent_bytes" in _measurement(entry, "message_traffic") for entry in visible_entries
    )
    title = f"job {_terminal_label(job_name)} (ID: {summary['job_id']})" if job_name else f"job ID {summary['job_id']}"
    lines = [f"Recorded resources for {title}."]
    coverage = f"Participants with data: {accepted}/{len(participants)}"
    if selected:
        coverage += f" | showing {selected}"
    lines.extend([coverage, "", "Recorded capacity-time per participant"])

    rows = []
    for entry in visible_entries:
        metrics = _capacity_row(entry) if ("status" not in entry) else [_MISSING] * 4
        rows.append([entry["participant_name"], *metrics])
    metric_headers = [
        "TIME",
        "CPU VISIBLE core-h",
        "MEM VISIBLE GiB-h",
        "GPU-h",
    ]
    lines.extend(_table(["SITE", *metric_headers], rows))
    other_headers = ["SITE", "AVG CPU USED (cores)"]
    if show_message_traffic:
        other_headers.extend(["PAYLOAD SENT GiB", "PAYLOAD RECEIVED GiB"])
    other_rows = []
    for entry in visible_entries:
        values = _utilization_row(entry if ("status" not in entry) else {}, show_message_traffic, average_cpu=True)
        other_rows.append([entry["participant_name"], *values])
    lines.extend(["", "Recorded resource utilization per participant", *_table(other_headers, other_rows)])
    job_totals = sum_job_totals(visible_entries)
    lines.extend(_totals_lines(job_totals, show_message_traffic))
    gaps = _data_gaps(visible_entries)
    if gaps:
        lines.extend(["", "Data gaps:", *gaps])
    show_legend = any("*" in line or _MISSING in line for line in lines)
    if show_legend:
        lines.extend(["", "* = incomplete; — = unavailable."])
    if selected and participant:
        lines.extend(
            [
                "",
                f"Job-process startup capacity for {selected}",
                "",
            ]
        )
        lines.extend(_startup_capacity(participant))
    return "\n".join(lines)


def render_study_resources(summary: dict) -> str:
    coverage = summary["coverage"]
    lines = [
        f"Resources recorded for finalized jobs in study {summary['selection']['study_name']}.",
        f"Jobs: {coverage['selected_jobs']} found | {coverage['included_jobs']} with resource data | "
        f"{coverage['unavailable_jobs']} without | {coverage['nonterminal_jobs']} not finalized (excluded)",
        "",
    ]
    show_message_traffic = any(
        job["resource_data"] == "included" and "sent_bytes" in _measurement(job["totals"], "message_traffic")
        for job in summary["jobs"]
    )
    capacity_rows, utilization_rows = [], []
    gaps = []
    for job in sorted(summary["jobs"], key=lambda job: job["job_id"]):
        if job["resource_data"] == "included":
            totals = job["totals"]
            resource_time = totals["resource_time"]
            status = resource_time.get("status", "reported")
            capacity = [_marked(_capacity_time(totals, resource), status) for resource in ("gpu", "cpu", "memory")]
            utilization = _utilization_row(totals, show_message_traffic)
            incomplete = [
                f"{label} {'incomplete' if _measurement(totals, field).get('status', 'reported') == 'partial' else 'unavailable'}"
                for field, label in _MEASUREMENT_LABELS.items()
                if _measurement(totals, field).get("status", "reported") != "reported"
            ]
            if incomplete:
                gaps.append(f"  {job['job_id']}: " + ", ".join(incomplete))
        else:
            capacity = [_MISSING] * 3
            utilization = [_MISSING] * (1 + 2 * int(show_message_traffic))
            reason = (
                "no valid resource summary" if job["resource_data"] == "unavailable" else "not finalized (excluded)"
            )
            gaps.append(f"  {job['job_id']}: {reason}")
        identity = [job["job_id"], _terminal_label(job["job_name"]), job["job_status"]]
        capacity_rows.append([*identity, *capacity])
        utilization_rows.append([*identity, *utilization])
    identity_headers = ["JOB ID", "NAME", "JOB STATUS"]
    usage_headers = ["CPU USED core-h"]
    if show_message_traffic:
        usage_headers.extend(["PAYLOAD SENT GiB", "PAYLOAD RECEIVED GiB"])
    lines.extend(
        [
            "Recorded resource capacity-time per job",
            *_table([*identity_headers, "GPU-h", "CPU VISIBLE core-h", "MEM VISIBLE GiB-h"], capacity_rows),
            "",
            "Recorded resource utilization per job",
            *_table([*identity_headers, *usage_headers], utilization_rows),
        ]
    )
    if any(job["resource_data"] == "included" for job in summary["jobs"]):
        totals = summary["totals"]
        lines.extend(_totals_lines(totals, "sent_bytes" in _measurement(totals, "message_traffic")))
    else:
        lines.extend(["", "Recorded capacity-time totals", "  No jobs with resource data"])
    if gaps:
        lines.extend(["", "Data gaps:", *gaps])
    show_legend = any("*" in line or _MISSING in line for line in lines)
    if show_legend:
        lines.extend(["", "* = incomplete; — = unavailable."])
    return "\n".join(lines)
