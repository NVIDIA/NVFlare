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

"""Read validated resource-statistics records from the normal WORKSPACE ZIP."""

from __future__ import annotations

import io
import os
from contextlib import ExitStack, contextmanager
from typing import Any, BinaryIO
from zipfile import BadZipFile, ZipFile

from .contract import (
    KIND_PARTICIPANT_SUMMARY,
    KIND_RESOURCE_SUMMARY,
    MAX_PARTICIPANT_SUMMARY_BYTES,
    MAX_RESOURCE_SUMMARY_BYTES,
    ContractError,
    load_and_validate,
    participant_measurements,
)

RESOURCE_STATS_PREFIX = "resource_stats/"
RESOURCE_SUMMARY_MEMBER = f"{RESOURCE_STATS_PREFIX}resource_summary.json"
# Includes the end records, archive comment, and central directory. Bound
# reads before ZipFile allocates its member index, regardless of archive size.
_MAX_ZIP_METADATA_BYTES = 4 * 1024 * 1024


class WorkspaceResourceStatsError(ValueError):
    """Raised when a WORKSPACE archive has no valid resource bundle."""


class _MetadataBoundedReader:
    """Limit index construction without parsing ZIP/ZIP64 structures ourselves."""

    def __init__(self, source: BinaryIO):
        self._source = source
        self.remaining: int | None = _MAX_ZIP_METADATA_BYTES

    def __getattr__(self, name):
        return getattr(self._source, name)

    def read(self, size: int = -1) -> bytes:
        if self.remaining is None:
            return self._source.read(size)
        if size > self.remaining:
            raise WorkspaceResourceStatsError("workspace ZIP metadata exceeds its size limit")
        # ZipFile also uses read() for its bounded end-of-file search. Never
        # forward an unbounded read while it is constructing the index.
        data = self._source.read(self.remaining + 1 if size < 0 else size)
        self.remaining -= len(data)
        if self.remaining < 0:
            raise WorkspaceResourceStatsError("workspace ZIP metadata exceeds its size limit")
        return data


class WorkspaceResourceStatsReader:
    """Bounded fixed-member reader; arbitrary workspace paths are never accepted."""

    def __init__(self, workspace_zip: bytes | str | os.PathLike[str] | BinaryIO):
        if isinstance(workspace_zip, bytes):
            if not workspace_zip:
                raise WorkspaceResourceStatsError("workspace archive is unavailable")
        elif isinstance(workspace_zip, (str, os.PathLike)):
            if not os.fspath(workspace_zip):
                raise WorkspaceResourceStatsError("workspace archive is unavailable")
        elif not (hasattr(workspace_zip, "read") and hasattr(workspace_zip, "seek")):
            raise WorkspaceResourceStatsError("workspace archive is unavailable")
        self._workspace_zip = workspace_zip
        self._summary: dict[str, Any] | None = None

    def read_resource_summary(self) -> dict[str, Any]:
        """Read the self-contained job rollup, independently of detail files.

        Participant bodies are validated only when requested by
        ``read_participant_summary``. Job and study queries avoid reading every
        participant file from the archive and can use the summary even when
        detail files are missing. Other archive members are not inspected.
        """

        data = self._read_exact_member(RESOURCE_SUMMARY_MEMBER, MAX_RESOURCE_SUMMARY_BYTES)
        try:
            summary = load_and_validate(data, KIND_RESOURCE_SUMMARY)
        except ContractError as exc:
            raise WorkspaceResourceStatsError(f"resource summary is invalid: {exc}") from exc
        self._summary = summary
        return summary

    def read_participant_summary(self, participant_name: str) -> dict[str, Any]:
        summary = self._summary or self.read_resource_summary()
        accepted = {
            entry["participant_name"]: entry
            for entry in summary["participants"]
            if entry.get("status", "accepted") == "accepted"
        }
        summary_entry = accepted.get(participant_name)
        if summary_entry is None:
            raise WorkspaceResourceStatsError(f"participant has no accepted resource report: {participant_name}")
        member = f"{RESOURCE_STATS_PREFIX}participants/{participant_name}.json"
        data = self._read_exact_member(member, MAX_PARTICIPANT_SUMMARY_BYTES)
        try:
            record = load_and_validate(data, KIND_PARTICIPANT_SUMMARY)
        except ContractError as exc:
            raise WorkspaceResourceStatsError(f"participant summary is invalid: {exc}") from exc
        if (
            record["schema_version"] != summary["schema_version"]
            or record["job_id"] != summary["job_id"]
            or record["participant_name"] != participant_name
        ):
            raise WorkspaceResourceStatsError("participant summary identity does not match its archive slot")
        copied = participant_measurements(record)
        if any(summary_entry.get(field) != copied.get(field) for field in copied.keys() | {"duration_seconds"}):
            raise WorkspaceResourceStatsError("participant summary values do not match the resource summary")
        return record

    def _read_exact_member(self, member_name: str, max_bytes: int) -> bytes:
        try:
            with self._open_archive() as archive:
                matches = [info for info in archive.infolist() if info.filename == member_name]
                if len(matches) != 1:
                    raise WorkspaceResourceStatsError(
                        f"workspace must contain exactly one {member_name!r} member; found {len(matches)}"
                    )
                info = matches[0]
                if info.is_dir() or info.flag_bits & 0x1:
                    raise WorkspaceResourceStatsError(f"workspace member is not a readable regular file: {member_name}")
                if info.file_size > max_bytes:
                    raise WorkspaceResourceStatsError(f"workspace member exceeds its size limit: {member_name}")
                with archive.open(info, "r") as stream:
                    data = stream.read(max_bytes + 1)
                if len(data) > max_bytes or len(data) != info.file_size:
                    raise WorkspaceResourceStatsError(f"workspace member length is inconsistent: {member_name}")
                return data
        except WorkspaceResourceStatsError:
            raise
        except (BadZipFile, KeyError, OSError, RuntimeError, EOFError, TypeError) as exc:
            raise WorkspaceResourceStatsError(f"cannot read workspace archive: {exc}") from exc

    @contextmanager
    def _open_archive(self):
        with ExitStack() as stack:
            source = self._workspace_zip
            if isinstance(source, bytes):
                source = stack.enter_context(io.BytesIO(source))
            elif isinstance(source, (str, os.PathLike)):
                source = stack.enter_context(open(source, "rb"))
            guarded = _MetadataBoundedReader(source)
            with ZipFile(guarded, "r") as archive:
                guarded.remaining = None
                yield archive
