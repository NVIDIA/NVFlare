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

import re
from copy import deepcopy

import pytest

from nvflare.private.fed.resource_stats.measurements import (
    canonical_json_bytes,
    capacity_to_resource_time,
    sum_job_totals,
    sum_study_totals,
)
from nvflare.private.fed.resource_stats.report_reader import read_resource_report
from nvflare.tool.resources.resources_output import render_job_resources, render_study_resources
from tests.resource_stats_report_test_utils import _accepted_entry, _resource_summary


def _job_summary(participant):
    if "resource_time" not in participant:
        participant["resource_time"] = capacity_to_resource_time(
            participant["resource_capacity"], participant.get("duration_seconds"), scope_issues=participant["issues"]
        )
    return _resource_summary(
        participants=[_accepted_entry(participant["participant_name"], participant["role"], record=participant)]
    )


def _study_summary(*summaries):
    jobs = [
        {
            "job_id": f"job-{index}",
            "job_name": "hello-pt",
            "job_status": "FINISHED:COMPLETED",
            "resource_data": "included",
            "totals": sum_job_totals(summary["participants"]),
        }
        for index, summary in enumerate(summaries, 1)
    ]
    return {
        "selection": {"study_name": "default"},
        "coverage": {
            "selected_jobs": len(jobs),
            "included_jobs": len(jobs),
            "unavailable_jobs": 0,
            "nonterminal_jobs": 0,
        },
        "jobs": jobs,
        "totals": sum_study_totals(jobs),
    }


def _participant():
    return {
        "duration_seconds": 3600,
        "participant_name": "site-1",
        "role": "client",
        "resource_capacity": {"cpu": {"units": 8}, "memory": {"bytes": 64 * 2**30}, "gpu": {"count": 2}},
        "resource_utilization": {
            "cpu_consumed": {"cpu_seconds": 1800},
            "message_traffic": {"status": "unavailable"},
        },
        "issues": [
            {"metric": "message_traffic", "code": "observation_incomplete"},
        ],
    }


def _with_missing_server(summary):
    summary["participants"].append({"participant_name": "server", "role": "server", "status": "missing"})
    return summary


def _site_cell(output: str, header: str, site: str = "site-1") -> str:
    headings = next(line for line in output.splitlines() if header in line)
    lines = output.splitlines()
    values = next(line for line in lines[lines.index(headings) + 1 :] if line.startswith(site))
    columns = re.split(r"\s{2,}", headings.strip())
    cells = re.split(r"\s{2,}", values.strip())
    return cells[next(index for index, column in enumerate(columns) if column.startswith(header))]


@pytest.mark.parametrize("scope", ["job", "study"])
@pytest.mark.parametrize("incomplete", [False, True])
def test_cli_only_shows_a_short_legend_when_needed(scope, incomplete):
    participant = _participant()
    participant["resource_utilization"]["message_traffic"] = {"sent_bytes": 0, "received_bytes": 0}
    participant["issues"] = [issue for issue in participant["issues"] if issue["metric"] != "message_traffic"]
    if incomplete:
        participant["resource_capacity"].update(status="partial")
        participant["issues"] = [issue for issue in participant["issues"] if issue["metric"] != "resource_capacity"] + [
            {"metric": "resource_capacity", "code": "observation_incomplete"}
        ]
    summary = _job_summary(participant)
    output = render_job_resources(summary) if scope == "job" else render_study_resources(_study_summary(summary))

    if incomplete:
        assert "Data gaps:" in output
        assert output.endswith("* = incomplete; — = unavailable.")
    else:
        assert "Data gaps:" not in output
        assert "* = incomplete" not in output


@pytest.mark.parametrize("measured, cpu_hours, average", [(3600, "8", "0.5"), (7200, "16", "0.25"), (0, "0", "—")])
def test_shows_execution_capacity_time_and_startup_detail(measured, cpu_hours, average):
    participant = _participant()
    participant["duration_seconds"] = measured
    summary = _job_summary(participant)

    output = render_job_resources(summary, job_name="hello-pt")
    detail = render_job_resources(summary, participant)

    assert "Recorded capacity-time per participant" in output
    assert _site_cell(output, "CPU VISIBLE") == cpu_hours
    assert _site_cell(output, "AVG CPU USED") == average
    assert _site_cell(output, "MEM VISIBLE GiB-h") == {3600: "64", 7200: "128", 0: "0"}[measured]
    assert _site_cell(output, "GPU-h") == {3600: "2", 7200: "4", 0: "0"}[measured]
    assert "Recorded capacity-time totals" in output
    assert f"CPU visible {cpu_hours} core-h" in output
    assert "CPU: 8 visible units" in detail
    assert "GPU: 2 GPUs" in detail


def test_zero_visible_devices_is_an_observed_zero():
    participant = _participant()
    participant["resource_capacity"]["gpu"] = {"count": 0}

    output = render_job_resources(_job_summary(participant))

    assert _site_cell(output, "GPU-h") == "0"
    assert "GPUs available 0 GPU-h" in output


def test_partial_capacity_keeps_values_and_marks_only_affected_metrics():
    participant = _participant()
    participant["resource_capacity"].update(status="partial")
    participant["issues"] = [issue for issue in participant["issues"] if issue["metric"] != "resource_capacity"] + [
        {"metric": "resource_capacity", "code": "observation_incomplete"}
    ]

    output = render_job_resources(_job_summary(participant))

    assert _site_cell(output, "CPU VISIBLE") == "8*"
    assert _site_cell(output, "MEM VISIBLE GiB-h") == "64*"
    assert _site_cell(output, "GPU-h") == "2*"
    assert _site_cell(output, "AVG CPU USED") == "0.5"
    assert "capacity-time incomplete" in output
    assert "CPU time used incomplete" not in output
    assert "observation_incomplete" not in output


def test_job_header_falls_back_to_id_when_name_metadata_is_not_supplied():
    output = render_job_resources(_job_summary(_participant()))
    assert output.startswith("Recorded resources for job ID job-1.")


def test_message_traffic_columns_are_only_shown_when_bound():
    participant = _participant()
    summary = _job_summary(participant)

    initial_output = render_job_resources(summary)
    assert "PAYLOAD SENT GiB" not in initial_output

    participant["resource_utilization"]["message_traffic"] = {
        "status": "partial",
        "sent_bytes": 5 * 2**30,
        "received_bytes": 2 * 2**30,
    }
    participant["issues"] = [issue for issue in participant["issues"] if issue["metric"] != "message_traffic"] + [
        {"metric": "message_traffic", "code": "observation_incomplete"}
    ]
    summary = _with_missing_server(_job_summary(participant))
    output = render_job_resources(summary)

    assert "PAYLOAD SENT GiB" in output
    assert "PAYLOAD RECEIVED GiB" in output
    assert "5*" in output  # Useful observed subtotal, not a complete send total.
    assert "2*" in output
    assert "Data gaps" in output
    assert "server" in output  # The expected server report is missing.

    partial_output = render_job_resources(_with_missing_server(_job_summary(participant)))
    assert "2*" in partial_output


def test_missing_report_does_not_invent_received_bytes_from_another_sender():
    site = _participant()
    site["resource_utilization"]["message_traffic"] = {"sent_bytes": 2**30, "received_bytes": 3 * 2**30}
    site["issues"] = [issue for issue in site["issues"] if issue["metric"] != "message_traffic"]
    summary = _with_missing_server(_job_summary(site))

    output = render_job_resources(summary)
    traffic_heading = next(index for index, line in enumerate(output.splitlines()) if "PAYLOAD RECEIVED GiB" in line)
    server_row = next(line for line in output.splitlines()[traffic_heading + 1 :] if line.startswith("server"))
    assert re.split(r"\s{2,}", server_row.strip()) == ["server", "—", "—", "—"]
    site_row = next(line for line in output.splitlines()[traffic_heading + 1 :] if line.startswith("site-1"))
    assert re.split(r"\s{2,}", site_row.strip()) == ["site-1", "0.5", "1", "3"]


@pytest.mark.parametrize("cpu_partial", [False, True])
@pytest.mark.parametrize(
    "capacity, duration", [("complete", 10), ("no_gpu", 10), ("unavailable", 10), ("cpu_only", None), ("complete", 0)]
)
def test_cpu_average_preserves_its_own_quality_and_shared_duration(capacity, duration, cpu_partial):
    participant = _participant()
    if capacity != "complete":
        value = participant["resource_capacity"]
        if capacity == "unavailable":
            value.clear()
        else:
            value.pop("gpu")
            if capacity == "cpu_only":
                value.pop("memory")
        value["status"] = "unavailable" if capacity == "unavailable" else "partial"
        participant["issues"].append({"metric": "resource_capacity", "code": "observation_incomplete"})
    if duration is None:
        participant.pop("duration_seconds")
    else:
        participant["duration_seconds"] = duration
    participant["resource_utilization"]["cpu_consumed"] = {"cpu_seconds": 2}
    if cpu_partial:
        participant["resource_utilization"]["cpu_consumed"]["status"] = "partial"
        participant["issues"].append({"metric": "cpu_consumed", "code": "attribution_incomplete"})

    output = render_job_resources(_job_summary(participant))

    assert _site_cell(output, "AVG CPU USED") == ("—" if not duration else "0.2*" if cpu_partial else "0.2")
    if capacity == "no_gpu":
        assert _site_cell(output, "CPU VISIBLE") == "0.02*"
        assert _site_cell(output, "MEM VISIBLE GiB-h") == "0.18*"
        assert _site_cell(output, "GPU-h") == "—"
        assert "capacity-time incomplete" in output
    elif capacity == "unavailable":
        assert _site_cell(output, "CPU VISIBLE") == "—"
    elif capacity == "complete":
        assert _site_cell(output, "CPU VISIBLE") == ("0.02" if duration else "0")
    if duration is None:
        assert "measurement duration unavailable" in output


def test_unavailable_metric_has_short_text_and_preserves_json_issue_codes():
    participant = _participant()
    participant["resource_utilization"]["message_traffic"] = {"status": "unavailable"}

    participant["issues"] = [{"metric": "message_traffic", "code": "malformed_source"}]

    output = render_job_resources(_job_summary(participant))

    assert "message traffic unavailable" in output
    assert "malformed_source" not in output
    assert participant["issues"] == [{"metric": "message_traffic", "code": "malformed_source"}]


def test_job_renderer_sorts_participants_with_numeric_counters():
    summary = _with_missing_server(_job_summary(_participant()))
    summary["participants"][0]["resource_utilization"]["message_traffic"] = {
        "sent_bytes": 1073741824,
        "received_bytes": 2147483648,
    }
    summary["participants"][0]["issues"] = [
        issue for issue in summary["participants"][0]["issues"] if issue["metric"] != "message_traffic"
    ]
    alternate = deepcopy(summary)
    alternate["participants"].reverse()
    alternate["participants"][1]["resource_utilization"]["message_traffic"]["sent_bytes"] = 1073741824
    alternate["participants"][1]["resource_utilization"]["message_traffic"]["received_bytes"] = 2147483648

    assert render_job_resources(alternate) == render_job_resources(summary)


def test_study_capacity_and_optional_traffic_columns():
    study = _study_summary(_job_summary(_participant()))
    job_totals = study["jobs"][0]["totals"]

    initial_output = render_study_resources(study)
    assert "JOB ID" in initial_output
    assert "NAME" in initial_output
    assert "hello-pt" in initial_output
    assert "Recorded capacity-time totals" in initial_output
    assert "GPU-h" in initial_output
    assert "CPU USED core-h" in initial_output
    assert "0.5" in initial_output
    assert "PAYLOAD SENT GiB" not in initial_output

    job_totals["resource_utilization"]["message_traffic"] = {
        "status": "partial",
        "sent_bytes": 5 * 2**30,
        "received_bytes": 2 * 2**30,
    }
    job_totals["issues"] = [{"metric": "message_traffic", "code": "observation_incomplete"}]
    study["totals"] = sum_study_totals(study["jobs"])
    output = render_study_resources(study)
    assert "PAYLOAD SENT GiB" in output
    assert "PAYLOAD RECEIVED GiB" in output
    assert "5*" in output
    assert "2*" in output


@pytest.mark.parametrize(
    "name, displayed",
    [
        ("肺部训练 — étude 🚀", "肺部训练 — étude 🚀"),
        ("name\nnext\r\t\x1b[2J", r"name\nnext\r\t\x1b[2J"),
        ("name\x7f\x9b\u202e\u2028\u2066", r"name\x7f\x9b\u202e\u2028\u2066"),
    ],
)
def test_human_output_escapes_terminal_controls_without_changing_query_names(name, displayed):
    summary = _job_summary(_participant())
    row = {
        "job_id": "job-1",
        "job_name": name,
        "job_status": "FINISHED:COMPLETED",
        "resource_data": "included",
        "totals": sum_job_totals(summary["participants"]),
    }
    study = {
        "selection": {"study_name": "default"},
        "coverage": {"selected_jobs": 1, "included_jobs": 1, "unavailable_jobs": 0, "nonterminal_jobs": 0},
        "jobs": [row],
        "totals": sum_study_totals([row]),
    }

    job_output = render_job_resources(summary, job_name=name)
    study_output = render_study_resources(study)

    assert displayed in job_output
    assert displayed in study_output
    assert "\x1b" not in job_output + study_output
    assert "\u202e" not in job_output + study_output
    assert row["job_name"] == name


def test_unknown_job_status_is_not_described_as_running():
    row = {"job_id": "job-1", "job_name": "job", "job_status": "UNKNOWN", "resource_data": "nonterminal"}
    study = {
        "selection": {"study_name": "default"},
        "coverage": {"selected_jobs": 1, "included_jobs": 0, "unavailable_jobs": 0, "nonterminal_jobs": 1},
        "jobs": [row],
        "totals": sum_study_totals([row]),
    }

    output = render_study_resources(study)

    assert "not finalized (excluded)" in output
    assert "still running" not in output


def test_fractional_cpu_capacity_and_gpu_hours_do_not_become_fractional_device_counts():
    participant = _participant()
    participant["resource_capacity"]["cpu"]["units"] = 1.5
    participant["resource_capacity"]["gpu"]["count"] = 1
    participant["duration_seconds"] = 1800
    participant["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = 225
    output = render_job_resources(_job_summary(participant))
    assert _site_cell(output, "CPU VISIBLE") == "0.75"
    assert _site_cell(output, "AVG CPU USED") == "0.12"
    assert _site_cell(output, "GPU-h") == "0.5"
    assert "CPU visible 0.75 core-h" in output
    assert "GPU: 1 GPUs" in render_job_resources(_job_summary(participant), participant)


def test_positive_small_quantities_are_not_displayed_as_zero():
    participant = _participant()
    participant["duration_seconds"] = 1
    participant["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = 1e-06
    output = render_job_resources(_job_summary(participant))
    assert _site_cell(output, "AVG CPU USED") == "<0.01"
    assert _site_cell(output, "GPU-h") == "<0.01"
    assert "CPU time used <0.01 core-h" in output


def test_capacity_and_utilization_are_separate_in_job_tables_and_totals():
    participant = _participant()
    participant["resource_utilization"]["message_traffic"] = {"sent_bytes": 1, "received_bytes": 2}
    participant["issues"] = [issue for issue in participant["issues"] if issue["metric"] != "message_traffic"]
    output = render_job_resources(_job_summary(participant))
    capacity, usage = output.split("Recorded resource utilization per participant")
    assert "CPU VISIBLE" in capacity
    assert "AVG CPU USED" not in capacity
    assert "PAYLOAD SENT" not in capacity
    assert "AVG CPU USED" in usage
    capacity_totals, usage_totals = usage.split("Recorded resource utilization totals")
    assert "Recorded capacity-time totals" in capacity_totals
    assert "CPU time used" not in capacity_totals
    assert "CPU time used" in usage_totals
    assert "CPU visible" not in usage_totals


def test_large_native_integer_bytes_remain_in_json_when_display_is_unavailable():
    participant = _participant()
    huge_bytes = 10**400
    participant["resource_utilization"]["message_traffic"] = {"sent_bytes": huge_bytes, "received_bytes": 0}
    summary = _job_summary(participant)
    summary["kind"] = "nvflare.resource_stats.resource_summary"
    read_resource_report(canonical_json_bytes(summary))

    output = render_job_resources(summary)

    assert _site_cell(output, "PAYLOAD SENT") == "—"
    assert summary["participants"][0]["resource_utilization"]["message_traffic"]["sent_bytes"] == huge_bytes


@pytest.mark.parametrize(
    "cpu_seconds, duration, expected",
    [(1e308, 1e-308, "—"), (10**400, 1, "—"), (5e-324, 1e308, "<0.01")],
)
def test_extreme_finite_cpu_values_do_not_publish_nonfinite_averages(cpu_seconds, duration, expected):
    participant = _participant()
    participant["duration_seconds"] = duration
    participant["resource_utilization"]["cpu_consumed"] = {"cpu_seconds": cpu_seconds}
    summary = _job_summary(participant)
    summary["kind"] = "nvflare.resource_stats.resource_summary"
    read_resource_report(canonical_json_bytes(summary))

    output = render_job_resources(summary)

    assert _site_cell(output, "AVG CPU USED") == expected
    assert "inf" not in output
    assert summary["participants"][0]["resource_utilization"]["cpu_consumed"]["cpu_seconds"] == cpu_seconds


@pytest.mark.parametrize("memory_bytes,table_value,detail", [(0, "0", "RAM: 0 GiB"), (None, "—", "RAM: —")])
def test_memory_capacity_distinguishes_a_zero_limit_from_an_unavailable_observation(memory_bytes, table_value, detail):
    participant = _participant()
    if memory_bytes is None:
        participant["resource_capacity"].pop("memory")
        participant["resource_capacity"]["status"] = "partial"
        participant["issues"].append({"metric": "resource_capacity", "code": "observation_incomplete"})
    else:
        participant["resource_capacity"]["memory"]["bytes"] = memory_bytes

    output = render_job_resources(_job_summary(participant), participant)

    assert _site_cell(output, "MEM VISIBLE GiB-h") == table_value
    assert detail in output


@pytest.mark.parametrize("duration", [None, 60])
def test_selected_worker_totals_use_explicit_time_and_optional_duration(duration):
    participant = _participant()
    participant["resource_capacity"]["gpu"] = {"count": 0}
    participant["resource_time"] = {
        "cpu": {"unit_seconds": 3600},
        "memory": {"byte_seconds": 0},
        "gpu": {"instance_seconds": 3600},
    }
    if duration is None:
        participant.pop("duration_seconds")
    else:
        participant["duration_seconds"] = duration
    summary = _job_summary(participant)
    other = deepcopy(summary["participants"][0])
    other["participant_name"] = "site-2"
    summary["participants"].append(other)
    output = render_job_resources(summary, participant)
    assert "GPUs available 1 GPU-h" in output
    assert "GPU: 0 GPUs" in output
    assert _site_cell(output, "AVG CPU USED") == ("—" if duration is None else "30")
