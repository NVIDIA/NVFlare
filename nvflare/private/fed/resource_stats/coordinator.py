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

"""Root-server ownership of terminal participant resource reports."""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .measurements import SCHEMA_VERSION, canonical_json_bytes, participant_measurements, utc_timestamp
from .report_reader import (
    JOB_ID_PATTERN,
    KIND_PARTICIPANT_SUMMARY,
    KIND_RESOURCE_SUMMARY,
    MAX_PARTICIPANT_SUMMARY_BYTES,
    MAX_PARTICIPANTS,
    MAX_RESOURCE_SUMMARY_BYTES,
    PARTICIPANT_NAME_PATTERN,
    read_resource_report,
)

_logger = logging.getLogger(__name__)

RESOURCE_REPORT_ACCEPTED = "accepted"
RESOURCE_REPORT_DUPLICATE = "duplicate"
RESOURCE_REPORT_INVALID = "invalid"
RESOURCE_REPORT_CONFLICT = "conflict"
RESOURCE_REPORT_TOO_LATE = "too_late"
RESOURCE_REPORT_NOT_PROVIDED = "not_provided"
RESOURCE_REPORT_NOT_EXPECTED = "not_expected"
RESOURCE_REPORT_SERVER_ERROR = "server_error"

RESOURCE_STATS_DIR = "resource_stats"
RESOURCE_SUMMARY_FILE = "resource_summary.json"
# The schema permits many individually bounded reports, but the root process
# must not retain an unbounded aggregate while it waits for job finalization.
# Keep the total accepted canonical payload no larger than the maximum public
# summary that can ultimately be published for the job.
MAX_ACCEPTED_REPORT_BYTES_PER_JOB = MAX_RESOURCE_SUMMARY_BYTES
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)
_SECURE_DIR_FD = (
    hasattr(os, "O_NOFOLLOW")
    and bool(getattr(os, "O_DIRECTORY", 0))
    and os.open in os.supports_dir_fd
    and os.mkdir in os.supports_dir_fd
    and os.rename in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and os.rmdir in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.listdir in os.supports_fd
)


@dataclass
class _AcceptedReport:
    received_at: str
    data: bytes


@dataclass
class _JobState:
    run_dir: Path
    expected: dict[str, str]
    disabled: set[str] = field(default_factory=set)
    accepted: dict[str, _AcceptedReport] = field(default_factory=dict)
    accepted_bytes: int = 0
    invalid_at: dict[str, str] = field(default_factory=dict)
    closed: bool = False
    finalized: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class ResourceStatsCoordinator:
    """Accept reports into trusted slots and finalize one archived bundle per job."""

    def __init__(self):
        self._jobs: dict[str, _JobState] = {}
        # This lock protects only the job-state registry.  Each state has its
        # own lock for acceptance, closure, and filesystem publication so I/O
        # for one job never blocks reports for an unrelated job.
        self._jobs_lock = threading.Lock()

    def has_job(self, job_id: str) -> bool:
        with self._jobs_lock:
            return job_id in self._jobs

    def _get_job_state(self, job_id: str) -> _JobState | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def start_job(
        self,
        job_id: str,
        client_names: Iterable[str],
        run_dir: str | Path,
        reset_existing: bool = False,
    ) -> None:
        if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
            raise ValueError("job_id has invalid resource-statistics syntax")
        expected = {"server": "server"}
        for name in client_names:
            if not isinstance(name, str) or not PARTICIPANT_NAME_PATTERN.fullmatch(name):
                raise ValueError("client names must use valid resource-statistics participant syntax")
            if name in expected:
                raise ValueError(f"duplicate resource participant name: {name}")
            if len(expected) >= MAX_PARTICIPANTS:
                raise ValueError(f"resource participant count cannot exceed {MAX_PARTICIPANTS}")
            expected[name] = "client"
        run_dir = Path(run_dir)
        while True:
            with self._jobs_lock:
                state = self._jobs.get(job_id)
                if state is None:
                    state = _JobState(run_dir=run_dir, expected=expected)
                    # Publish an already-locked state so same-job operations
                    # wait for setup while unrelated jobs keep using the short
                    # registry lock independently.
                    state.lock.acquire()
                    self._jobs[job_id] = state
                    created = True
                else:
                    created = False

            if created:
                try:
                    self._prepare_job_directory(run_dir, reset_existing)
                except Exception:
                    state.closed = True
                    with self._jobs_lock:
                        if self._jobs.get(job_id) is state:
                            self._jobs.pop(job_id, None)
                    raise
                finally:
                    state.lock.release()
                return

            with state.lock:
                # A concurrent forget may have removed this state between the
                # registry lookup and lock acquisition. Retry with the current
                # registry entry rather than mutating an orphaned state.
                with self._jobs_lock:
                    if self._jobs.get(job_id) is not state:
                        continue
                if not reset_existing:
                    if state.expected != expected or state.run_dir != run_dir:
                        raise RuntimeError(f"resource participants already registered differently for job {job_id}")
                    return

                state.closed = True
                try:
                    self._prepare_job_directory(run_dir, reset_existing=True)
                except Exception:
                    with self._jobs_lock:
                        if self._jobs.get(job_id) is state:
                            self._jobs.pop(job_id, None)
                    raise
                state.run_dir = run_dir
                state.expected = expected
                state.disabled.clear()
                state.accepted.clear()
                state.accepted_bytes = 0
                state.invalid_at.clear()
                state.finalized = False
                state.closed = False
                return

    @classmethod
    def _prepare_job_directory(cls, run_dir: Path, reset_existing: bool) -> None:
        run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if reset_existing:
            cls._reset_resource_directory(run_dir)
        directory_fd = cls._open_resource_directory(run_dir)
        os.close(directory_fd)

    def disable_clients(self, job_id: str, client_names: Iterable[str]) -> None:
        state = self._get_job_state(job_id)
        if state is None:
            return
        disabled = []
        with state.lock:
            if state.closed:
                return
            for name in client_names:
                if state.expected.get(name) == "client" and name not in state.accepted:
                    state.disabled.add(name)
                    disabled.append(name)
        if disabled:
            _logger.info(f"job {job_id}: excluded {disabled} from resource-statistics reporting")

    def accept_resource_report(
        self, job_id: str, participant_name: str, resource_report: Mapping[str, Any] | None
    ) -> str:
        """Validate and retain one report in the bounded parent-owned ledger."""

        status = self._accept_resource_report(job_id, participant_name, resource_report)
        self._log_report_status(job_id, participant_name, status)
        return status

    @staticmethod
    def _log_report_status(job_id: str, participant_name: str, status: str) -> None:
        if status in (RESOURCE_REPORT_CONFLICT, RESOURCE_REPORT_SERVER_ERROR):
            _logger.warning(f"job {job_id}: resource report from '{participant_name}' rejected: {status}")
        elif status in (RESOURCE_REPORT_ACCEPTED, RESOURCE_REPORT_DUPLICATE):
            _logger.debug(f"job {job_id}: resource report from '{participant_name}': {status}")
        else:
            _logger.info(f"job {job_id}: resource report from '{participant_name}': {status}")

    def _accept_resource_report(
        self, job_id: str, participant_name: str, resource_report: Mapping[str, Any] | None
    ) -> str:
        if resource_report is None:
            return RESOURCE_REPORT_NOT_PROVIDED
        state = self._get_job_state(job_id)
        if state is None:
            return RESOURCE_REPORT_NOT_EXPECTED
        with state.lock:
            if participant_name not in state.expected or participant_name in state.disabled:
                return RESOURCE_REPORT_NOT_EXPECTED
            existing = state.accepted.get(participant_name)
            if state.closed and existing is None:
                return RESOURCE_REPORT_TOO_LATE
            received_at = utc_timestamp()
            try:
                if not isinstance(resource_report, Mapping) or set(resource_report) != {"participant_summary"}:
                    raise ValueError("expected a participant summary")
                data = resource_report["participant_summary"]
                if isinstance(data, bytes) and existing and existing.data == data:
                    return RESOURCE_REPORT_DUPLICATE
                record = read_resource_report(data, KIND_PARTICIPANT_SUMMARY)
                if record["job_id"] != job_id or record["participant_name"] != participant_name:
                    raise ValueError("report identity does not match its authenticated slot")
                data = canonical_json_bytes(record)
                if len(data) > MAX_PARTICIPANT_SUMMARY_BYTES:
                    raise ValueError("normalized report exceeds its size limit")
            except ValueError:
                if state.closed:
                    return RESOURCE_REPORT_TOO_LATE
                if existing is None:
                    state.invalid_at.setdefault(participant_name, received_at)
                return RESOURCE_REPORT_INVALID
            if existing:
                # Numeric equivalence and formatting changes are still retries.
                if existing.data == data or json.loads(existing.data) == record:
                    return RESOURCE_REPORT_DUPLICATE
                return RESOURCE_REPORT_TOO_LATE if state.closed else RESOURCE_REPORT_CONFLICT
            if len(data) > MAX_ACCEPTED_REPORT_BYTES_PER_JOB - state.accepted_bytes:
                return RESOURCE_REPORT_SERVER_ERROR
            state.accepted[participant_name] = _AcceptedReport(received_at, data)
            state.accepted_bytes += len(data)
            state.invalid_at.pop(participant_name, None)
            return RESOURCE_REPORT_ACCEPTED

    def finalize_job(self, job_id: str) -> dict[str, Any]:
        """Close acceptance and publish one self-contained summary of accepted reports."""

        state = self._get_job_state(job_id)
        if state is None:
            raise RuntimeError(f"resource participants are not registered for job {job_id}")
        with state.lock:
            state.closed = True
            participants = []
            for participant_name, role in state.expected.items():
                base = {"participant_name": participant_name, "role": role}
                accepted = state.accepted.get(participant_name)
                if accepted is not None:
                    # These immutable bytes already passed the receiving reader.
                    # Decode them without replaying the input checks.
                    record = json.loads(accepted.data)
                    participants.append(
                        {
                            **base,
                            "received_at": accepted.received_at,
                            "reported_at": record["reported_at"],
                            **participant_measurements(record),
                        }
                    )
                elif participant_name in state.disabled:
                    participants.append({**base, "status": "disabled"})
                elif participant_name in state.invalid_at:
                    participants.append(
                        {
                            **base,
                            "status": "invalid",
                            "received_at": state.invalid_at[participant_name],
                        }
                    )
                else:
                    participants.append({**base, "status": "missing"})
            participants.sort(key=lambda item: (item["role"], item["participant_name"]))
            summary = {
                "schema_version": SCHEMA_VERSION,
                "kind": KIND_RESOURCE_SUMMARY,
                "job_id": job_id,
                "participants": participants,
            }
            if state.finalized:
                return summary
            summary_data = canonical_json_bytes(summary)
            if len(summary_data) > MAX_RESOURCE_SUMMARY_BYTES:
                raise ValueError("resource summary exceeds its size limit")

            self._remove_empty_staging_directory(state.run_dir)
            self._atomic_write_resource(state.run_dir, RESOURCE_SUMMARY_FILE, summary_data)
            state.finalized = True
            status_counts: dict[str, int] = {}
            for participant in participants:
                status = participant.get("status", "accepted")
                status_counts[status] = status_counts.get(status, 0) + 1
            _logger.info(f"job {job_id}: resource statistics finalized, participants by status: {status_counts}")
            return summary

    def discard_job_artifacts(self, job_id: str) -> None:
        state = self._get_job_state(job_id)
        if state is not None:
            with state.lock:
                state.closed = True
                self.discard_run_artifacts(state.run_dir)

    @classmethod
    def discard_run_artifacts(cls, run_dir: str | Path) -> None:
        """Remove the reserved bundle path without following a replacement symlink."""

        cls._reset_resource_directory(Path(run_dir))

    def forget_job(self, job_id: str) -> None:
        state = self._get_job_state(job_id)
        if state is None:
            return
        with state.lock:
            state.closed = True
        with self._jobs_lock:
            if self._jobs.get(job_id) is state:
                self._jobs.pop(job_id, None)

    @classmethod
    def _atomic_write_resource(
        cls,
        run_dir: Path,
        file_name: str,
        data: bytes,
    ) -> None:
        directory_fd = cls._open_resource_directory(run_dir)
        try:
            cls._atomic_write_at(directory_fd, file_name, data)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _atomic_write_at(directory_fd: int, file_name: str, data: bytes) -> None:
        temp_name = f".{file_name}-{secrets.token_hex(16)}.tmp"
        fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory_fd)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(fd)
            os.rename(temp_name, file_name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            os.close(fd)
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass

    @classmethod
    def _open_resource_directory(cls, run_dir: Path) -> int:
        """Open each server-owned path component without following symlinks."""

        run_dir = Path(run_dir)
        if not _SECURE_DIR_FD:
            raise OSError("resource publication requires directory-relative filesystem operations")

        current_fd = os.open(run_dir, _DIRECTORY_OPEN_FLAGS)
        try:
            try:
                os.mkdir(RESOURCE_STATS_DIR, 0o700, dir_fd=current_fd)
            except FileExistsError:
                pass
            result = os.open(RESOURCE_STATS_DIR, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            return result
        finally:
            os.close(current_fd)

    @staticmethod
    def _ensure_plain_directory(path: Path) -> None:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise OSError(f"resource directory is unavailable: {path}") from exc
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise OSError(f"resource path is not a plain directory: {path}")

    @classmethod
    def _reset_resource_directory(cls, run_dir: Path) -> None:
        cls._ensure_plain_directory(run_dir)
        stats_path = run_dir / RESOURCE_STATS_DIR
        try:
            mode = stats_path.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            stats_path.unlink()
            return
        if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            raise OSError("cannot safely reset the resource statistics directory on this platform")
        shutil.rmtree(stats_path)

    @classmethod
    def _remove_empty_staging_directory(cls, run_dir: Path) -> None:
        stats_fd = cls._open_resource_directory(run_dir)
        try:
            try:
                os.rmdir("staging", dir_fd=stats_fd)
            except FileNotFoundError:
                return
            os.fsync(stats_fd)
        finally:
            os.close(stats_fd)
