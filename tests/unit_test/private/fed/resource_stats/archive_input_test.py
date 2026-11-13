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


@pytest.mark.parametrize("source_kind", ["bytes", "path", "stream"])
def test_archive_index_is_bounded_before_large_directory_read(tmp_path, monkeypatch, source_kind):
    summary = _resource_summary()
    raw = _resource_archive(summary, extra=[(f"unrelated/{i}.txt", b"") for i in range(100)])
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
        WorkspaceResourceStatsReader(TrackedStream(raw)).read_resource_summary()
    assert max(reads) <= archive_reader._MAX_ZIP_METADATA_BYTES + 1
    assert -1 not in reads


def test_member_body_uses_its_own_limit_after_index_construction(monkeypatch):
    summary = _resource_summary()
    # The JSON body is larger than the directory budget. Member limits remain
    # independent; the budget does not cap the size of the entire ZIP.
    monkeypatch.setattr(archive_reader, "_MAX_ZIP_METADATA_BYTES", 256)
    stream = io.BytesIO(_resource_archive(summary))
    assert WorkspaceResourceStatsReader(stream).read_resource_summary() == summary
    assert not stream.closed


def test_commented_zip64_archive_and_repeated_detail_queries(monkeypatch):
    summary, participant = _resource_summary(), _participant_summary()
    # Exercise ZIP64 end records without a multi-gigabyte fixture.
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1)
    raw = _resource_archive(summary)
    stream = io.BytesIO(raw)
    with ZipFile(stream, "a") as archive:
        archive.comment = b"workspace comment" * 100
    reader = WorkspaceResourceStatsReader(stream)
    assert reader.read_resource_summary() == summary
    assert reader.read_participant_summary("site-1") == participant
    assert reader.read_participant_summary("site-1") == participant
    assert not stream.closed


def test_archive_comment_and_directory_share_metadata_budget(monkeypatch):
    summary = _resource_summary()
    stream = io.BytesIO(_resource_archive(summary))
    with ZipFile(stream, "a") as archive:
        archive.comment = b"x" * 2048
    monkeypatch.setattr(archive_reader, "_MAX_ZIP_METADATA_BYTES", 1024)
    with pytest.raises(WorkspaceResourceStatsError, match="ZIP metadata exceeds"):
        WorkspaceResourceStatsReader(stream).read_resource_summary()
    assert not stream.closed


@pytest.mark.parametrize("failure", ["malformed", "oversized", "duplicate", "identity"])
def test_invalid_workspace_summary_never_falls_back_to_outputs(monkeypatch, failure):
    summary = _resource_summary()
    if failure == "identity":
        summary["job_id"] = "../invalid"
    raw = _resource_archive(summary)
    if failure == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            raw = _resource_archive(summary, extra=[("resource_stats/resource_summary.json", b"{}")])
    elif failure == "oversized":
        monkeypatch.setattr(archive_reader, "MAX_RESOURCE_SUMMARY_BYTES", 1)
    elif failure == "malformed":
        stream = io.BytesIO()
        with ZipFile(stream, "w") as archive:
            archive.writestr("resource_stats/resource_summary.json", b"invalid")
        raw = stream.getvalue()
    fallback = Mock(return_value=_resource_archive())
    with pytest.raises(WorkspaceResourceStatsError):
        WorkspaceResourceStatsReader(raw, outputs_archive=fallback).read_resource_summary()
    fallback.assert_not_called()


def test_site_query_projects_original_report_without_another_archive_read():
    participant = _participant_summary(reported_at="2026-09-17T12:00:00.123456789Z")
    summary = _resource_summary(participants=[_accepted_entry(record=participant)])
    fallback = Mock(side_effect=AssertionError("unexpected OUTPUTS staging"))
    reader = WorkspaceResourceStatsReader(
        _resource_archive(summary, extra=[("resource_stats/participants/site-1.json", b"obsolete detail")]),
        outputs_archive=fallback,
    )
    assert reader.read_resource_summary() == summary
    reader._read_exact_member = Mock(side_effect=AssertionError("unexpected member read"))
    assert reader.read_participant_summary("site-1") == participant
    fallback.assert_not_called()
    with pytest.raises(WorkspaceResourceStatsError, match="no accepted resource report"):
        reader.read_participant_summary("unknown-site")
