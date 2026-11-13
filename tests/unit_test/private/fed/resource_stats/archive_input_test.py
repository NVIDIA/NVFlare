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
from copy import deepcopy
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from nvflare.private.fed.resource_stats import archive_reader
from nvflare.private.fed.resource_stats.archive_reader import WorkspaceResourceStatsError, WorkspaceResourceStatsReader
from nvflare.private.fed.resource_stats.measurements import canonical_json_bytes
from nvflare.private.fed.resource_stats.report_reader import read_resource_report
from tests.resource_stats_report_test_utils import _participant_summary, _resource_summary


def _reports():
    return _resource_summary(), _participant_summary()


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
    assert reader.read_resource_summary() == read_resource_report(canonical_json_bytes(summary))
    assert reader.read_participant_summary("site-1") == participant
    with pytest.raises(WorkspaceResourceStatsError, match="found 0"):
        reader.read_participant_summary("site-2")


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
