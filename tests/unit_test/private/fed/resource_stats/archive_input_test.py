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

"""Archive queries validate requested reports, not an exclusive directory."""

import io
import json
import struct
import zipfile
from copy import deepcopy
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from nvflare.private.fed.resource_stats import archive_reader
from nvflare.private.fed.resource_stats.archive_reader import WorkspaceResourceStatsError, WorkspaceResourceStatsReader
from nvflare.private.fed.resource_stats.contract import (
    SCHEMA_VERSION,
    ContractError,
    canonical_json_bytes,
    capacity_to_resource_time,
    normalize_record,
    participant_measurements,
    validate_bundle,
)


def _reports():
    participant = {
        "duration_seconds": "10",
        "schema_version": SCHEMA_VERSION,
        "kind": "nvflare.resource_stats.participant_summary",
        "job_id": "job-1",
        "participant_name": "site-1",
        "reported_at": "2026-09-17T12:00:00Z",
        "resource_capacity": {
            "status": "partial",
            "issues": ["attribution_incomplete", "observation_incomplete"],
            "cpu": {"groups": [{"units": "2", "model": "CPU A"}, {"units": "1", "model": "CPU B"}]},
            "memory": {"bytes": "4096"},
            "gpu": {
                "groups": [
                    {"kind": "cuda_device_unclassified", "count": "1"},
                    {"kind": "full_gpu", "count": "1", "memory_bytes": "4096"},
                ]
            },
        },
        "workspace_filesystem": {"capacity_bytes": "4096"},
        "resource_utilization": {
            "cpu_consumed": {"cpu_seconds": "2"},
            "retained_content": {"bytes": "20"},
            "message_traffic": {"sent_bytes": "0", "received_bytes": "0"},
        },
    }
    participant["resource_time"] = capacity_to_resource_time(
        participant["resource_capacity"], participant["duration_seconds"]
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "nvflare.resource_stats.resource_summary",
        "job_id": "job-1",
        "participants": [
            {
                "participant_name": "site-1",
                "role": "client",
                "received_at": "2026-09-17T12:00:01Z",
                **participant_measurements(participant),
            },
            {"participant_name": "server", "role": "server", "status": "missing"},
        ],
    }
    return normalize_record(summary), participant


def _archive(summary, participant=None, extra=()):
    stream = io.BytesIO()
    with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
        archive.writestr("resource_stats/resource_summary.json", canonical_json_bytes(summary))
        if participant is not None:
            archive.writestr("resource_stats/participants/site-1.json", canonical_json_bytes(participant))
        for name, data in extra:
            archive.writestr(name, data)
    return stream.getvalue()


def test_summary_does_not_open_unrequested_participant_body():
    summary, _ = _reports()
    reader = WorkspaceResourceStatsReader(
        _archive(summary, extra=[("resource_stats/participants/site-1.json", b"not JSON")])
    )
    assert reader.read_resource_summary() == summary
    with pytest.raises(WorkspaceResourceStatsError, match="participant summary is invalid"):
        reader.read_participant_summary("site-1")


def test_requested_detail_can_use_equivalent_numeric_strings_and_list_order():
    summary, participant = _reports()
    participant["resource_capacity"]["issues"].reverse()
    participant["resource_capacity"]["cpu"]["groups"].reverse()
    participant["resource_capacity"]["gpu"]["groups"].reverse()
    participant["duration_seconds"] = "10.0"
    participant["resource_capacity"]["cpu"]["groups"][0]["units"] = "1.000"
    participant["resource_capacity"]["memory"]["bytes"] = "4096.0"
    participant["resource_capacity"]["gpu"]["groups"][0]["count"] = "1.0"
    participant["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = "2.000"
    participant["resource_utilization"]["retained_content"]["bytes"] = "20.0"
    participant["resource_capacity"]["gpu"]["groups"][0]["memory_bytes"] = "4096.0"
    reader = WorkspaceResourceStatsReader(_archive(summary, participant))
    detail = reader.read_participant_summary("site-1")
    assert detail["resource_utilization"]["cpu_consumed"]["cpu_seconds"] == "2"
    assert detail["resource_utilization"]["retained_content"]["bytes"] == "20"
    assert detail["resource_capacity"] == summary["participants"][0]["resource_capacity"]


@pytest.mark.parametrize("mismatch", ["missing_detail_duration", "missing_summary_duration", "different_duration"])
def test_shared_duration_must_match_in_summary_and_detail(mismatch):
    summary, participant = _reports()
    entry = summary["participants"][0]
    if mismatch == "missing_detail_duration":
        participant.pop("duration_seconds")
    elif mismatch == "missing_summary_duration":
        entry.pop("duration_seconds")
    else:
        entry["duration_seconds"] = "20"
    reader = WorkspaceResourceStatsReader(_archive(summary, participant))
    with pytest.raises(WorkspaceResourceStatsError, match="values do not match"):
        reader.read_participant_summary("site-1")
    files = {
        "resource_summary.json": canonical_json_bytes(summary),
        "participants/site-1.json": canonical_json_bytes(participant),
    }
    with pytest.raises(ContractError, match="duration_seconds"):
        validate_bundle(summary, {"site-1": participant}, files)


@pytest.mark.parametrize("field,value", [("job_id", "other-job"), ("participant_name", "other-site")])
def test_requested_detail_identity_must_match_summary_and_slot(field, value):
    summary, participant = _reports()
    participant[field] = value
    reader = WorkspaceResourceStatsReader(_archive(summary, participant))
    assert reader.read_resource_summary() == summary
    with pytest.raises(WorkspaceResourceStatsError, match="identity does not match"):
        reader.read_participant_summary("site-1")


def test_requested_detail_size_limit_is_checked_only_for_detail(monkeypatch):
    summary, participant = _reports()
    monkeypatch.setattr(archive_reader, "MAX_PARTICIPANT_SUMMARY_BYTES", 1)
    reader = WorkspaceResourceStatsReader(_archive(summary, participant))
    assert reader.read_resource_summary() == summary
    with pytest.raises(WorkspaceResourceStatsError, match="size limit"):
        reader.read_participant_summary("site-1")


def test_detail_ignores_other_missing_participant_reports():
    summary, participant = _reports()
    other = deepcopy(summary["participants"][0])
    other["participant_name"] = "site-2"
    summary["participants"].append(other)
    reader = WorkspaceResourceStatsReader(_archive(summary, participant))
    assert reader.read_resource_summary() == normalize_record(summary)
    assert reader.read_participant_summary("site-1") == participant
    with pytest.raises(WorkspaceResourceStatsError, match="found 0"):
        reader.read_participant_summary("site-2")


def test_summary_accepts_compact_json():
    summary, _ = _reports()
    stream = io.BytesIO()
    with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
        archive.writestr("resource_stats/resource_summary.json", json.dumps(summary, separators=(",", ":")))
    assert WorkspaceResourceStatsReader(stream.getvalue()).read_resource_summary() == summary


@pytest.mark.parametrize("version", ["0.0", "2.0", "3.0", "4.0"])
def test_summary_rejects_unsupported_formats(version):
    summary, _ = _reports()
    summary["schema_version"] = version

    with pytest.raises(WorkspaceResourceStatsError, match="schema_version"):
        WorkspaceResourceStatsReader(_archive(summary)).read_resource_summary()


@pytest.mark.parametrize("source_kind", ["bytes", "path", "stream"])
def test_archive_index_is_bounded_before_large_directory_read(tmp_path, monkeypatch, source_kind):
    summary, _ = _reports()
    raw = _archive(summary, extra=[(f"unrelated/{i}.txt", b"") for i in range(100)])
    source = raw
    if source_kind == "path":
        source = tmp_path / "workspace.zip"
        source.write_bytes(raw)
    elif source_kind == "stream":
        source = io.BytesIO(raw)
    monkeypatch.setattr(archive_reader, "_MAX_ZIP_METADATA_BYTES", 1024)
    with pytest.raises(WorkspaceResourceStatsError, match="ZIP metadata exceeds"):
        WorkspaceResourceStatsReader(source).read_resource_summary()
    if source_kind == "stream":
        assert not source.closed


def test_archive_rejects_forged_directory_size_before_allocating(monkeypatch):
    summary, _ = _reports()
    raw = bytearray(b"x" * 4096 + _archive(summary))
    monkeypatch.setattr(archive_reader, "_MAX_ZIP_METADATA_BYTES", 1024)
    # The classic end record stores central-directory byte length at offset 12.
    struct.pack_into("<I", raw, len(raw) - 22 + 12, 2048)
    reads = []

    class TrackedStream(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    with pytest.raises(WorkspaceResourceStatsError, match="ZIP metadata exceeds"):
        WorkspaceResourceStatsReader(TrackedStream(raw)).read_resource_summary()
    assert max(reads) <= archive_reader._MAX_ZIP_METADATA_BYTES + 1
    assert -1 not in reads


def test_member_body_uses_its_own_limit_after_index_construction(monkeypatch):
    summary, _ = _reports()
    # The JSON body is larger than the directory budget. Member limits remain
    # independent; the budget does not cap the size of the entire ZIP.
    monkeypatch.setattr(archive_reader, "_MAX_ZIP_METADATA_BYTES", 256)
    stream = io.BytesIO(_archive(summary))
    assert WorkspaceResourceStatsReader(stream).read_resource_summary() == summary
    assert not stream.closed


def test_commented_zip64_archive_and_repeated_detail_queries(monkeypatch):
    summary, participant = _reports()
    # Exercise ZIP64 end records without a multi-gigabyte fixture.
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1)
    raw = _archive(summary, participant)
    stream = io.BytesIO(raw)
    with ZipFile(stream, "a") as archive:
        archive.comment = b"workspace comment" * 100
    reader = WorkspaceResourceStatsReader(stream)
    assert reader.read_resource_summary() == summary
    assert reader.read_participant_summary("site-1") == participant
    assert reader.read_participant_summary("site-1") == participant
    assert not stream.closed


def test_archive_comment_and_directory_share_metadata_budget(monkeypatch):
    summary, _ = _reports()
    stream = io.BytesIO(_archive(summary))
    with ZipFile(stream, "a") as archive:
        archive.comment = b"x" * 2048
    monkeypatch.setattr(archive_reader, "_MAX_ZIP_METADATA_BYTES", 1024)
    with pytest.raises(WorkspaceResourceStatsError, match="ZIP metadata exceeds"):
        WorkspaceResourceStatsReader(stream).read_resource_summary()
    assert not stream.closed
