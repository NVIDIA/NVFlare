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
import struct
import zipfile
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from nvflare.private.fed.resource_stats import archive_reader
from nvflare.private.fed.resource_stats.archive_reader import WorkspaceResourceStatsError, WorkspaceResourceStatsReader
from tests.resource_stats_report_test_utils import (
    _accepted_entry,
    _participant_summary,
    _resource_archive,
    _resource_summary,
)


@pytest.fixture
def archive_path(tmp_path):
    return tmp_path / "workspace.zip"


def test_archive_index_is_bounded_before_large_directory_read(archive_path, monkeypatch):
    summary = _resource_summary()
    raw = _resource_archive(summary, extra=[(f"unrelated/{i}.txt", b"") for i in range(100)])
    archive_path.write_bytes(raw)
    monkeypatch.setattr(archive_reader, "_MAX_ZIP_METADATA_BYTES", 1024)
    with pytest.raises(WorkspaceResourceStatsError, match="ZIP metadata exceeds"):
        WorkspaceResourceStatsReader(archive_path).read_resource_summary()


def test_archive_rejects_forged_directory_size_before_allocating(monkeypatch):
    summary = _resource_summary()
    raw = bytearray(b"x" * 4096 + _resource_archive(summary))
    monkeypatch.setattr(archive_reader, "_MAX_ZIP_METADATA_BYTES", 1024)
    # The classic end record stores central-directory byte length at offset 12.
    struct.pack_into("<I", raw, len(raw) - 22 + 12, 2048)
    reads = []

    class TrackedStream(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    with pytest.raises(WorkspaceResourceStatsError, match="ZIP metadata exceeds"):
        ZipFile(archive_reader._MetadataBoundedReader(TrackedStream(raw)))
    assert max(reads) <= archive_reader._MAX_ZIP_METADATA_BYTES + 1
    assert -1 not in reads


def test_member_body_uses_its_own_limit_after_index_construction(archive_path, monkeypatch):
    summary = _resource_summary()
    # The JSON body is larger than the directory budget. Member limits remain
    # independent; the budget does not cap the size of the entire ZIP.
    monkeypatch.setattr(archive_reader, "_MAX_ZIP_METADATA_BYTES", 256)
    archive_path.write_bytes(_resource_archive(summary))
    assert WorkspaceResourceStatsReader(archive_path).read_resource_summary() == summary


def test_commented_zip64_archive_and_repeated_detail_queries(archive_path, monkeypatch):
    summary, participant = _resource_summary(), _participant_summary()
    # Exercise ZIP64 end records without a multi-gigabyte fixture.
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1)
    archive_path.write_bytes(_resource_archive(summary))
    with ZipFile(archive_path, "a") as archive:
        archive.comment = b"workspace comment" * 100
    reader = WorkspaceResourceStatsReader(archive_path)
    assert reader.read_resource_summary() == summary
    assert reader.read_participant_summary("site-1") == participant
    assert reader.read_participant_summary("site-1") == participant


def test_archive_comment_and_directory_share_metadata_budget(archive_path, monkeypatch):
    archive_path.write_bytes(_resource_archive())
    with ZipFile(archive_path, "a") as archive:
        archive.comment = b"x" * 2048
    monkeypatch.setattr(archive_reader, "_MAX_ZIP_METADATA_BYTES", 1024)
    with pytest.raises(WorkspaceResourceStatsError, match="ZIP metadata exceeds"):
        WorkspaceResourceStatsReader(archive_path).read_resource_summary()


@pytest.mark.parametrize("failure", ["malformed", "oversized", "duplicate", "identity"])
def test_invalid_workspace_summary_never_falls_back_to_outputs(archive_path, monkeypatch, failure):
    summary = _resource_summary()
    if failure == "identity":
        summary["job_id"] = "../invalid"
    raw = _resource_archive(summary)
    if failure == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            raw = _resource_archive(summary, extra=[("resource_stats/resource_summary.json", b"{}")])
    elif failure == "oversized":
        monkeypatch.setattr(archive_reader, "MAX_RESOURCE_SUMMARY_BYTES", 1)
    archive_path.write_bytes(raw)
    if failure == "malformed":
        with ZipFile(archive_path, "w") as archive:
            archive.writestr("resource_stats/resource_summary.json", b"invalid")
    fallback = Mock(side_effect=AssertionError("unexpected OUTPUTS staging"))
    with pytest.raises(WorkspaceResourceStatsError):
        WorkspaceResourceStatsReader(archive_path, outputs_archive=fallback).read_resource_summary()
    fallback.assert_not_called()


def test_site_query_projects_original_report_without_another_archive_read(archive_path):
    participant = _participant_summary(reported_at="2026-09-17T12:00:00.123456789Z")
    summary = _resource_summary(participants=[_accepted_entry(record=participant)])
    fallback = Mock(side_effect=AssertionError("unexpected OUTPUTS staging"))
    archive_path.write_bytes(
        _resource_archive(summary, extra=[("resource_stats/participants/site-1.json", b"obsolete detail")])
    )
    reader = WorkspaceResourceStatsReader(archive_path, outputs_archive=fallback)
    assert reader.read_resource_summary() == summary
    reader._read_exact_member = Mock(side_effect=AssertionError("unexpected member read"))
    assert reader.read_participant_summary("site-1") == participant
    fallback.assert_not_called()
    with pytest.raises(WorkspaceResourceStatsError, match="no accepted resource report"):
        reader.read_participant_summary("unknown-site")
