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

"""Numeric resource reports shared by record and query tests."""

from nvflare.private.fed.resource_stats.measurements import (
    SCHEMA_VERSION,
    capacity_to_resource_time,
    participant_measurements,
)
from nvflare.private.fed.resource_stats.report_reader import KIND_PARTICIPANT_SUMMARY, KIND_RESOURCE_SUMMARY


def _participant_summary(job_id="job-1", participant_name="site-1", reported_at="2026-09-17T12:00:00Z"):
    capacity = {"cpu": {"units": 2}, "memory": {"bytes": 4096}, "gpu": {"count": 1}}
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND_PARTICIPANT_SUMMARY,
        "job_id": job_id,
        "participant_name": participant_name,
        "reported_at": reported_at,
        "issues": [],
        "duration_seconds": 2,
        "resource_capacity": capacity,
        "resource_time": capacity_to_resource_time(capacity, 2),
        "resource_utilization": {
            "cpu_consumed": {"cpu_seconds": 1.5},
            "message_traffic": {"sent_bytes": 3, "received_bytes": 4},
        },
    }


def _accepted_entry(participant_name="site-1", role="client", received_at="2026-09-17T12:00:00Z"):
    record = _participant_summary(participant_name=participant_name)
    return {
        "participant_name": participant_name,
        "role": role,
        "received_at": received_at,
        **participant_measurements(record),
    }


def _resource_summary(job_id="job-1", participants=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND_RESOURCE_SUMMARY,
        "job_id": job_id,
        "participants": [_accepted_entry()] if participants is None else participants,
    }
