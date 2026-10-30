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

from nvflare.private.fed.resource_stats.contract import (
    SCHEMA_VERSION,
    capacity_to_resource_time,
    derive_job_totals,
    derive_study_totals,
)
from nvflare.tool.resources.resources_output import render_job_resources, render_study_resources


def _job_summary(participant):
    if "resource_time" not in participant:
        participant["resource_time"] = capacity_to_resource_time(
            participant["resource_capacity"], participant.get("duration_seconds")
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": "job-1",
        "participants": [
            {
                **({"duration_seconds": participant["duration_seconds"]} if "duration_seconds" in participant else {}),
                "participant_name": participant["participant_name"],
                "role": participant["role"],
                "received_at": "2026-09-17T12:00:00Z",
                "resource_capacity": participant["resource_capacity"],
                "resource_time": participant["resource_time"],
                "resource_utilization": {
                    "cpu_consumed": participant["resource_utilization"]["cpu_consumed"],
                    "retained_content": participant["resource_utilization"]["retained_content"],
                    "message_traffic": participant["resource_utilization"]["message_traffic"],
                },
            }
        ],
    }


def _participant():
    return {
        "duration_seconds": "3600",
        "participant_name": "site-1",
        "role": "client",
        "resource_capacity": {
            "cpu": {"groups": [{"units": "8", "model": "AMD EPYC 9654"}]},
            "memory": {"bytes": str(64 * 2**30)},
            "gpu": {"groups": [{"kind": "full_gpu", "count": "2", "model": "NVIDIA H100 80GB HBM3"}]},
        },
        "workspace_filesystem": {"capacity_bytes": str(2 * 2**40)},
        "resource_utilization": {
            "cpu_consumed": {"cpu_seconds": "1800"},
            "retained_content": {"status": "unavailable", "issues": ["observation_incomplete"]},
            "message_traffic": {"status": "unavailable", "issues": ["observation_incomplete"]},
        },
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
    participant["resource_utilization"]["retained_content"] = {"bytes": "0"}
    participant["resource_utilization"]["message_traffic"] = {"sent_bytes": "0", "received_bytes": "0"}
    if incomplete:
        participant["resource_capacity"].update(status="partial", issues=["observation_incomplete"])
    summary = _job_summary(participant)
    if scope == "job":
        output = render_job_resources(summary)
    else:
        job = {
            "job_id": "job-1",
            "job_name": "example",
            "job_status": "FINISHED:COMPLETED",
            "resource_data": "included",
            "totals": derive_job_totals(summary["participants"]),
        }
        output = render_study_resources(
            {
                "selection": {"study_name": "default"},
                "coverage": {
                    "selected_jobs": "1",
                    "included_jobs": "1",
                    "unavailable_jobs": "0",
                    "nonterminal_jobs": "0",
                },
                "jobs": [job],
                "totals": derive_study_totals([job]),
            }
        )

    assert "Notes:" not in output
    assert "Capacity-time totals multiply" not in output
    assert "TIME covers" not in output
    assert "Use --site" not in output
    if incomplete:
        assert "Data gaps:" in output
        assert output.endswith("* = incomplete; — = unavailable.")
    else:
        assert "Data gaps:" not in output
        assert "* = incomplete" not in output


@pytest.mark.parametrize(
    "measured, cpu_hours, average", [("3600", "8", "0.5"), ("7200", "16", "0.25"), ("0", "0", "—")]
)
def test_shows_execution_capacity_time_and_startup_detail(measured, cpu_hours, average):
    participant = _participant()
    participant["duration_seconds"] = measured
    summary = _job_summary(participant)

    output = render_job_resources(summary, job_name="hello-pt")
    detail = render_job_resources(summary, participant)

    assert "GPU count basis: CUDA Driver (schema 1.0)." in output
    assert "Recorded capacity-time per participant" in output
    assert "AVG CPU VISIBLE" not in output
    assert "AVG MEM" not in output
    assert "AVG CUDA DEVICE-h" not in output
    assert _site_cell(output, "CPU VISIBLE") == cpu_hours
    assert _site_cell(output, "AVG CPU USED") == average
    assert _site_cell(output, "MEM VISIBLE GiB-h") == {"3600": "64", "7200": "128", "0": "0"}[measured]
    assert _site_cell(output, "CUDA DEVICE-h") == {"3600": "2", "7200": "4", "0": "0"}[measured]
    assert _site_cell(output, "FULL GPU-h") == {"3600": "2", "7200": "4", "0": "0"}[measured]
    assert "Recorded capacity-time totals" in output
    assert f"CPU visible {cpu_hours} core-h" in output
    assert "Notes:" not in output
    assert "AMD EPYC 9654; 8 visible units" in detail
    assert "NVIDIA H100 80GB HBM3; 2 visible instances" in detail
    assert "average visible" not in detail
    assert "visible capacity unavailable" not in output


def test_partial_capacity_keeps_cpu_average_on_the_shared_window():
    participant = _participant()
    participant["resource_capacity"] = {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "cpu": {"groups": [{"units": "8"}]},
        "memory": {"bytes": str(64 * 2**30)},
    }
    participant["duration_seconds"] = "10"
    participant["resource_utilization"]["cpu_consumed"] = {"cpu_seconds": "2"}

    output = render_job_resources(_job_summary(participant))

    assert _site_cell(output, "CPU VISIBLE") == "0.02*"
    assert _site_cell(output, "MEM VISIBLE GiB-h") == "0.18*"
    assert _site_cell(output, "AVG CPU USED") == "0.2"
    assert _site_cell(output, "CUDA DEVICE-h") == "—"
    assert "capacity-time: GPU capacity could not be observed for the whole interval" in output


def test_zero_visible_devices_is_an_observed_zero():
    participant = _participant()
    participant["resource_capacity"]["gpu"] = {"groups": []}

    output = render_job_resources(_job_summary(participant))

    assert _site_cell(output, "CUDA DEVICE-h") == "0"
    assert _site_cell(output, "FULL GPU-h") == "0"
    assert "CUDA devices available 0 device-h" in output
    assert "visible capacity unavailable" not in output


def test_best_effort_capacity_keeps_values_and_explains_the_possible_overstatement():
    participant = _participant()
    participant["resource_capacity"].update(status="partial", issues=["observation_incomplete"])

    output = render_job_resources(_job_summary(participant))

    assert _site_cell(output, "CPU VISIBLE") == "8*"
    assert _site_cell(output, "MEM VISIBLE GiB-h") == "64*"
    assert _site_cell(output, "CUDA DEVICE-h") == "2*"
    assert _site_cell(output, "AVG CPU USED") == "0.5"
    assert "some execution measurements or limits could not be checked" in output
    assert "observation_incomplete" not in output
    assert "visible capacity partial" not in output


def test_file_scan_gap_explains_what_stopped_instead_of_showing_an_issue_code():
    participant = _participant()
    participant["resource_utilization"]["retained_content"] = {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "bytes": str(2**30),
    }

    output = render_job_resources(_job_summary(participant))

    assert "run-dir files: the file scan stopped early or could not read every file" in output
    assert "observation_incomplete" not in output


def test_site_detail_reports_observed_mig_and_unclassified_counts():
    participant = _participant()
    participant["resource_capacity"]["gpu"]["groups"].extend(
        [
            {
                "kind": "mig_compute_instance",
                "count": "1",
                "model": "NVIDIA H100 MIG 1g.10gb",
                "mig_profile": "1g.10gb",
            },
            {"kind": "cuda_device_unclassified", "count": "1", "model": "NVIDIA H100 80GB HBM3"},
        ]
    )

    output = render_job_resources(_job_summary(participant), participant)

    assert _site_cell(output, "CUDA DEVICE-h") == "4"
    assert _site_cell(output, "MIG instance-h") == "1"
    assert _site_cell(output, "TYPE-UNVERIFIED") == "1"
    assert "MIG compute instance: NVIDIA H100 MIG 1g.10gb (1g.10gb); 1 visible instances" in output
    assert "CUDA device, type unverified: NVIDIA H100 80GB HBM3; 1 visible instances" in output


def test_mig_column_is_hidden_when_no_participant_reports_mig():
    participant = _participant()

    output = render_job_resources(_job_summary(participant), job_name="hello-pt")

    assert output.startswith("Recorded resources for job hello-pt (ID: job-1).")
    assert "GPU count basis: CUDA Driver (schema 1.0)." in output
    assert "CPU VISIBLE" in output
    assert "AVG CPU USED" in output
    assert "MEM VISIBLE GiB-h" in output
    assert "CUDA DEVICE-h" in output
    assert "FULL GPU-h" in output
    assert "MIG instance-h" not in output
    assert "8" in output
    assert "0.5" in output  # 1,800 consumed CPU seconds / 3,600 measured seconds.
    assert "64" in output
    assert "2" in output
    assert "RUN-DIR FILES" not in output
    assert "MESSAGE PAYLOAD SENT" not in output
    assert "CPU visible 8 core-h" in output
    assert "CPU time used 0.5 core-h" in output
    assert "Memory visible 64 GiB-h" in output
    assert "CUDA devices available 2 device-h" in output
    assert "Verified full GPUs 2 GPU-h" in output
    assert "COMPUTE" not in output
    assert "CPU CONSUMED STATUS" not in output
    assert "accepted" not in output.lower()
    assert "reported" not in output.lower()


def test_job_header_falls_back_to_id_when_name_metadata_is_not_supplied():
    output = render_job_resources(_job_summary(_participant()))
    assert output.startswith("Recorded resources for job ID job-1.")


def test_mig_column_and_site_detail_are_shown_only_when_applicable():
    participant = _participant()
    participant["resource_capacity"]["gpu"]["groups"].append(
        {
            "kind": "mig_compute_instance",
            "count": "1",
            "model": "NVIDIA H100 MIG 1g.10gb",
            "mig_profile": "1g.10gb",
        }
    )
    summary = _job_summary(participant)

    job_output = render_job_resources(summary)
    site_output = render_job_resources(summary, participant)

    assert "MIG instance-h" in job_output
    assert "1" in job_output
    assert "CUDA devices available 3 device-h" in job_output
    assert "MIG compute instance: NVIDIA H100 MIG 1g.10gb (1g.10gb)" in site_output
    assert "1 visible instances" in site_output


def test_driver_counted_unclassified_time_is_complete_and_not_double_counted():
    participant = _participant()
    participant["resource_capacity"]["gpu"]["groups"].insert(
        0, {"kind": "cuda_device_unclassified", "count": "1", "model": "NVIDIA H100 80GB HBM3"}
    )
    summary = _job_summary(participant)

    output = render_job_resources(summary)
    detail = render_job_resources(summary, participant)

    assert _site_cell(output, "CUDA DEVICE-h") == "3"
    assert _site_cell(output, "FULL GPU-h") == "2"
    assert _site_cell(output, "TYPE-UNVERIFIED") == "1"
    assert "CUDA devices available 3 device-h" in output
    assert "Verified full GPUs 2 GPU-h" in output
    assert "Type-unverified CUDA devices 1 device-h" in output
    assert "CUDA device, type unverified: NVIDIA H100 80GB HBM3" in detail
    assert "visible capacity incomplete" not in output


def test_run_directory_files_and_message_traffic_columns_are_only_shown_when_bound():
    participant = _participant()
    summary = _job_summary(participant)

    initial_output = render_job_resources(summary)
    assert "RUN-DIR FILES" not in initial_output
    assert "MESSAGE PAYLOAD SENT" not in initial_output

    participant["resource_utilization"]["retained_content"] = {"bytes": str(3 * 2**30)}
    participant["resource_utilization"]["message_traffic"] = {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "sent_bytes": str(5 * 2**30),
        "received_bytes": str(2 * 2**30),
    }
    summary = _with_missing_server(_job_summary(participant))
    output = render_job_resources(summary)

    assert "RUN-DIR FILES GiB" in output
    assert "PAYLOAD SENT GiB" in output
    assert "PAYLOAD RECEIVED GiB" in output
    assert "Run-dir files 3* GiB" in output  # The server report is missing.
    assert "5*" in output  # Useful observed subtotal, not a complete send total.
    assert "2*" in output
    assert "RUN-DIR STATUS" not in output
    assert "MESSAGE TRAFFIC STATUS" not in output
    assert "Data gaps" in output
    assert "server" in output  # The expected server report is missing.

    participant["resource_utilization"]["retained_content"] = {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "bytes": str(2 * 2**30),
    }
    partial_output = render_job_resources(_with_missing_server(_job_summary(participant)))
    assert "2*" in partial_output
    assert "RUN-DIR STATUS" not in partial_output


def test_missing_report_does_not_invent_received_bytes_from_another_sender():
    site = _participant()
    site["resource_utilization"]["message_traffic"] = {
        "sent_bytes": str(2**30),
        "received_bytes": str(3 * 2**30),
    }
    summary = _with_missing_server(_job_summary(site))

    output = render_job_resources(summary)
    traffic_heading = next(index for index, line in enumerate(output.splitlines()) if "PAYLOAD RECEIVED GiB" in line)
    server_row = next(line for line in output.splitlines()[traffic_heading + 1 :] if line.startswith("server"))
    assert re.split(r"\s{2,}", server_row.strip()) == ["server", "—", "—", "—"]
    site_row = next(line for line in output.splitlines()[traffic_heading + 1 :] if line.startswith("site-1"))
    assert re.split(r"\s{2,}", site_row.strip()) == ["site-1", "0.5", "1", "3"]
    assert "Notes:" not in output


def test_cpu_used_average_is_marked_when_process_cpu_is_incomplete():
    participant = _participant()
    participant["resource_utilization"]["cpu_consumed"] = {
        "status": "partial",
        "issues": ["attribution_incomplete"],
        "cpu_seconds": "900",
    }

    output = render_job_resources(_job_summary(participant))

    assert _site_cell(output, "CPU VISIBLE") == "8"
    assert _site_cell(output, "AVG CPU USED") == "0.25*"  # 900 / 3,600; incomplete subtotal.
    assert "Data gaps" in output
    assert "CPU CONSUMED STATUS" not in output


def test_cpu_average_is_unavailable_when_shared_duration_is_missing():
    participant = _participant()
    participant["resource_capacity"] = {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "cpu": participant["resource_capacity"]["cpu"],
    }

    participant["duration_seconds"] = "10"
    participant["resource_utilization"]["cpu_consumed"] = {"cpu_seconds": "2"}

    participant.pop("duration_seconds")
    output = render_job_resources(_job_summary(participant))

    assert _site_cell(output, "AVG CPU USED") == "—"
    assert "Data gaps" in output


@pytest.mark.parametrize("cpu_partial", [False, True])
def test_cpu_used_average_uses_shared_duration_when_gpu_observation_is_missing(cpu_partial):
    participant = _participant()
    participant["resource_capacity"] = {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "cpu": {"groups": [{"units": "8"}]},
        "memory": {"bytes": str(64 * 2**30)},
    }
    participant["duration_seconds"] = "10"
    participant["resource_utilization"]["cpu_consumed"] = {"cpu_seconds": "2"}
    if cpu_partial:
        participant["resource_utilization"]["cpu_consumed"].update(status="partial", issues=["attribution_incomplete"])

    output = render_job_resources(_job_summary(participant))

    assert _site_cell(output, "AVG CPU USED") == ("0.2*" if cpu_partial else "0.2")
    assert _site_cell(output, "CUDA DEVICE-h") == "—"


def test_cpu_used_average_remains_available_without_any_capacity_measurement():
    participant = _participant()
    participant["resource_capacity"] = {"status": "unavailable", "issues": ["observation_incomplete"]}
    participant["duration_seconds"] = "10"
    participant["resource_utilization"]["cpu_consumed"] = {"cpu_seconds": "2"}

    output = render_job_resources(_job_summary(participant))

    assert _site_cell(output, "AVG CPU USED") == "0.2"
    assert _site_cell(output, "CPU VISIBLE") == "—"


def test_zero_shared_duration_does_not_produce_a_cpu_average():
    participant = _participant()
    participant["duration_seconds"] = "10"
    participant["resource_utilization"]["cpu_consumed"] = {"cpu_seconds": "2"}

    participant["duration_seconds"] = "0"
    output = render_job_resources(_job_summary(participant))

    assert _site_cell(output, "AVG CPU USED") == "—"


def test_unavailable_source_cause_is_shown_without_another_status_column():
    participant = _participant()
    participant["resource_utilization"]["retained_content"] = {"status": "unavailable", "issues": ["malformed_source"]}

    output = render_job_resources(_job_summary(participant))

    assert "run-dir files unavailable (measurement contained invalid values)" in output
    assert "RUN-DIR STATUS" not in output


def test_job_renderer_sorts_participants_and_accepts_equivalent_integer_spelling():
    summary = _with_missing_server(_job_summary(_participant()))
    summary["participants"][0]["resource_utilization"]["message_traffic"] = {
        "sent_bytes": "1073741824",
        "received_bytes": "2147483648",
    }
    alternate = deepcopy(summary)
    alternate["participants"].reverse()
    alternate["participants"][1]["resource_utilization"]["message_traffic"]["sent_bytes"] = "1073741824.0"
    alternate["participants"][1]["resource_utilization"]["message_traffic"]["received_bytes"] = "2147483648.0"

    assert render_job_resources(alternate) == render_job_resources(summary)


def test_study_mig_column_is_shown_only_when_an_included_job_reports_mig():
    participant = _participant()
    job_totals = derive_job_totals(_job_summary(participant)["participants"])
    study = {
        "selection": {"study_name": "default"},
        "coverage": {
            "selected_jobs": "1",
            "included_jobs": "1",
            "unavailable_jobs": "0",
            "nonterminal_jobs": "0",
        },
        "jobs": [
            {
                "job_id": "job-1",
                "job_name": "hello-pt",
                "job_status": "FINISHED_COMPLETED",
                "resource_data": "included",
                "totals": job_totals,
            }
        ],
        "totals": job_totals,
    }

    initial_output = render_study_resources(study)
    assert "JOB ID" in initial_output
    assert "NAME" in initial_output
    assert "hello-pt" in initial_output
    assert "GPU BASIS" not in initial_output
    assert "Recorded capacity-time totals" in initial_output
    assert "CUDA DEVICE-h" in initial_output
    assert "CPU USED core-h" in initial_output
    assert "0.5" in initial_output
    assert "MIG instance-h" not in initial_output
    assert "RUN-DIR FILES" not in initial_output
    assert "MESSAGE PAYLOAD SENT" not in initial_output
    assert "QUALITY" not in initial_output
    assert "RESOURCE DATA" not in initial_output
    assert "CPU CONSUMED STATUS" not in initial_output

    mig_group = {
        "kind": "mig_compute_instance",
        "instance_seconds": "3600",
        "model": "NVIDIA H100 MIG 1g.10gb",
    }
    job_totals["resource_time"]["gpu"]["groups"].append(mig_group)

    output = render_study_resources(study)
    assert "MIG instance-h" in output
    assert "MIG instances visible 1 instance-h" in output
    assert "CUDA devices available 3 device-h" in output

    job_totals["resource_utilization"]["message_traffic"] = {
        "status": "partial",
        "sent_bytes": str(5 * 2**30),
        "received_bytes": str(2 * 2**30),
    }
    output = render_study_resources(study)
    assert "PAYLOAD SENT GiB" in output
    assert "PAYLOAD RECEIVED GiB" in output
    assert "5*" in output
    assert "2*" in output
    assert "MESSAGE TRAFFIC STATUS" not in output


def test_cpu_consumed_job_and_study_cli_reconcile_without_changing_capacity_time():
    first = _job_summary(_participant())
    second_participant = _participant()
    second_participant["resource_utilization"]["cpu_consumed"] = {"cpu_seconds": "900"}
    second = _job_summary(second_participant)
    job_totals = [derive_job_totals(summary["participants"]) for summary in (first, second)]
    assert [total["resource_utilization"]["cpu_consumed"]["cpu_seconds"] for total in job_totals] == ["1800", "900"]

    jobs = [
        {
            "job_id": f"job-{index}",
            "job_name": f"run-{index}",
            "job_status": "FINISHED:COMPLETED",
            "resource_data": "included",
            "totals": total,
        }
        for index, total in enumerate(job_totals, 1)
    ]
    study_totals = derive_study_totals(jobs)
    assert study_totals["resource_utilization"]["cpu_consumed"] == {"cpu_seconds": "2700"}
    assert study_totals["resource_time"]["cpu"]["groups"][0]["unit_seconds"] == "57600"

    study = {
        "selection": {"study_name": "default"},
        "coverage": {"selected_jobs": "2", "included_jobs": "2", "unavailable_jobs": "0", "nonterminal_jobs": "0"},
        "jobs": jobs,
        "totals": study_totals,
    }
    output = render_study_resources(study)
    assert "CPU time used 0.75 core-h" in output
    assert "CPU visible 16 core-h" in output


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
        "totals": derive_job_totals(summary["participants"]),
    }
    study = {
        "selection": {"study_name": "default"},
        "coverage": {"selected_jobs": "1", "included_jobs": "1", "unavailable_jobs": "0", "nonterminal_jobs": "0"},
        "jobs": [row],
        "totals": derive_study_totals([row]),
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
        "coverage": {"selected_jobs": "1", "included_jobs": "0", "unavailable_jobs": "0", "nonterminal_jobs": "1"},
        "jobs": [row],
        "totals": derive_study_totals([row]),
    }

    output = render_study_resources(study)

    assert "not finalized (excluded)" in output
    assert "still running" not in output


def test_fractional_cpu_capacity_and_gpu_hours_do_not_become_fractional_device_counts():
    participant = _participant()
    participant["resource_capacity"]["cpu"]["groups"][0]["units"] = "1.5"
    participant["resource_capacity"]["gpu"]["groups"][0]["count"] = "1"
    participant["duration_seconds"] = "1800"
    participant["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = "225"
    output = render_job_resources(_job_summary(participant))
    assert _site_cell(output, "CPU VISIBLE") == "0.75"
    assert _site_cell(output, "AVG CPU USED") == "0.12"
    assert _site_cell(output, "CUDA DEVICE-h") == "0.5"
    assert "Verified full GPUs 0.5 GPU-h" in output
    assert "CPU visible 0.75 core-h" in output


def test_positive_small_quantities_are_not_displayed_as_zero():
    participant = _participant()
    participant["duration_seconds"] = "1"
    participant["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = "0.000001"
    output = render_job_resources(_job_summary(participant))
    assert _site_cell(output, "AVG CPU USED") == "<0.01"
    assert _site_cell(output, "CUDA DEVICE-h") == "<0.01"
    assert "Verified full GPUs <0.01 GPU-h" in output
    assert "CPU time used <0.01 core-h" in output


def test_capacity_and_utilization_are_separate_in_job_tables_and_totals():
    participant = _participant()
    participant["resource_utilization"]["retained_content"] = {"bytes": "1024"}
    participant["resource_utilization"]["message_traffic"] = {"sent_bytes": "1", "received_bytes": "2"}
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
