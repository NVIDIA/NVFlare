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

import logging
import os
import secrets
import shutil
import stat
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contract import (
    JOB_ID_PATTERN,
    KIND_PARTICIPANT_SUMMARY,
    KIND_RESOURCE_SUMMARY,
    MAX_PARTICIPANT_SUMMARY_BYTES,
    MAX_PARTICIPANTS,
    MAX_RESOURCE_SUMMARY_BYTES,
    PARTICIPANT_NAME_PATTERN,
    SCHEMA_VERSION,
    ContractError,
    canonical_json_bytes,
    load_and_validate,
    participant_measurements,
    utc_timestamp,
    validate_record,
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
PARTICIPANTS_DIR = "participants"
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
        directory_fd, _ = cls._open_resource_directory(run_dir, include_participants=True)
        if directory_fd is not None:
            os.close(directory_fd)

    def disable_clients(self, job_id: str, client_names: Iterable[str]) -> None:
        state = self._get_job_state(job_id)
        if state is None:
            return
        disabled = []
        with state.lock:
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
        # "conflict" and "server_error" are the two outcomes most likely to indicate
        # a real bug or tampering rather than a benign retry/race; everything else
        # is either expected (accepted/duplicate) or informational for an operator
        # tracking why a participant's utilization did not make it into the summary.
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
        if not isinstance(resource_report, Mapping) or set(resource_report) != {"participant_summary"}:
            return self._mark_invalid(job_id, participant_name)
        data = resource_report.get("participant_summary")
        if not isinstance(data, bytes) or len(data) > MAX_PARTICIPANT_SUMMARY_BYTES:
            return self._mark_invalid(job_id, participant_name)

        received_at = utc_timestamp()
        state = self._get_job_state(job_id)
        if state is None:
            return RESOURCE_REPORT_NOT_EXPECTED
        with state.lock:
            if participant_name not in state.expected or participant_name in state.disabled:
                return RESOURCE_REPORT_NOT_EXPECTED
            existing = state.accepted.get(participant_name)
            if existing and existing.data == data:
                return RESOURCE_REPORT_DUPLICATE
            if state.closed and existing is None:
                return RESOURCE_REPORT_TOO_LATE

        try:
            record = load_and_validate(data, KIND_PARTICIPANT_SUMMARY)
        except ContractError:
            return self._mark_invalid(job_id, participant_name, received_at)
        if record["job_id"] != job_id or record["participant_name"] != participant_name:
            return self._mark_invalid(job_id, participant_name, received_at)
        # Normalize validated values and set-like lists before comparing retries
        # or retaining the report. Sender whitespace, object-key order, numeric
        # formatting, and list presentation order are not part of its identity.
        data = canonical_json_bytes(record)
        if len(data) > MAX_PARTICIPANT_SUMMARY_BYTES:
            return self._mark_invalid(job_id, participant_name, received_at)

        state = self._get_job_state(job_id)
        if state is None:
            return RESOURCE_REPORT_NOT_EXPECTED
        with state.lock:
            if participant_name not in state.expected or participant_name in state.disabled:
                return RESOURCE_REPORT_NOT_EXPECTED
            existing = state.accepted.get(participant_name)
            if existing:
                if existing.data == data:
                    return RESOURCE_REPORT_DUPLICATE
                return RESOURCE_REPORT_TOO_LATE if state.closed else RESOURCE_REPORT_CONFLICT
            if state.closed:
                return RESOURCE_REPORT_TOO_LATE
            if len(data) > MAX_ACCEPTED_REPORT_BYTES_PER_JOB - state.accepted_bytes:
                return RESOURCE_REPORT_SERVER_ERROR
            state.accepted[participant_name] = _AcceptedReport(
                received_at=received_at,
                data=data,
            )
            state.accepted_bytes += len(data)
            state.invalid_at.pop(participant_name, None)
            return RESOURCE_REPORT_ACCEPTED

    def _mark_invalid(self, job_id: str, participant_name: str, received_at: str | None = None) -> str:
        state = self._get_job_state(job_id)
        if state is None:
            return RESOURCE_REPORT_NOT_EXPECTED
        with state.lock:
            if participant_name not in state.expected or participant_name in state.disabled:
                return RESOURCE_REPORT_NOT_EXPECTED
            if state.closed:
                return RESOURCE_REPORT_TOO_LATE
            if participant_name not in state.accepted:
                state.invalid_at.setdefault(participant_name, received_at or utc_timestamp())
        return RESOURCE_REPORT_INVALID

    def finalize_job(self, job_id: str) -> dict[str, Any]:
        """Close acceptance, write participant records, then publish the summary last."""

        state = self._get_job_state(job_id)
        if state is None:
            raise RuntimeError(f"resource participants are not registered for job {job_id}")
        with state.lock:
            if state.finalized:
                return self._read_resource_file(state.run_dir, RESOURCE_SUMMARY_FILE, MAX_RESOURCE_SUMMARY_BYTES)
            state.closed = True
            participants = []
            for participant_name, role in state.expected.items():
                base = {"participant_name": participant_name, "role": role}
                accepted = state.accepted.get(participant_name)
                if accepted is not None:
                    # The long-lived ledger keeps only the exact canonical
                    # bytes. Revalidate them when constructing the roll-up
                    # instead of retaining a second decoded representation.
                    record = load_and_validate(accepted.data, KIND_PARTICIPANT_SUMMARY)
                    participants.append(
                        {
                            **base,
                            "received_at": accepted.received_at,
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
            validate_record(summary)
            summary_data = canonical_json_bytes(summary)
            if len(summary_data) > MAX_RESOURCE_SUMMARY_BYTES:
                raise ContractError("resource summary exceeds its size limit")

            self._remove_empty_staging_directory(state.run_dir)
            # The job child shares the run directory. Rewrite the exact accepted
            # bytes through parent-owned descriptors immediately before publishing
            # the summary marker, so child-side mutation is not retained.
            for participant_name, accepted in state.accepted.items():
                self._atomic_write_resource(
                    state.run_dir,
                    f"{participant_name}.json",
                    accepted.data,
                    include_participants=True,
                )
            self._atomic_write_resource(state.run_dir, RESOURCE_SUMMARY_FILE, summary_data)
            self._verify_final_inventory(state.run_dir, set(state.accepted))
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

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", delete=False
            ) as stream:
                temp_path = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
            ResourceStatsCoordinator._fsync_directory(path.parent)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

    @classmethod
    def _atomic_write_resource(
        cls,
        run_dir: Path,
        file_name: str,
        data: bytes,
        include_participants: bool = False,
    ) -> None:
        directory_fd, directory_path = cls._open_resource_directory(run_dir, include_participants)
        if directory_fd is None:
            cls._atomic_write(directory_path / file_name, data)
            return
        try:
            cls._atomic_write_at(directory_fd, file_name, data)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _atomic_write_at(directory_fd: int, file_name: str, data: bytes) -> None:
        temp_name = None
        file_fd = None
        try:
            for _ in range(10):
                candidate = f".{file_name}-{secrets.token_hex(8)}.tmp"
                try:
                    file_fd = os.open(
                        candidate,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        0o600,
                        dir_fd=directory_fd,
                    )
                    temp_name = candidate
                    break
                except FileExistsError:
                    continue
            if file_fd is None or temp_name is None:
                raise OSError(f"could not create a temporary resource file for {file_name}")
            remaining = memoryview(data)
            while remaining:
                written = os.write(file_fd, remaining)
                if written <= 0:
                    raise OSError(f"short write while storing {file_name}")
                remaining = remaining[written:]
            os.fsync(file_fd)
            os.close(file_fd)
            file_fd = None
            os.rename(temp_name, file_name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            temp_name = None
            os.fsync(directory_fd)
        finally:
            if file_fd is not None:
                os.close(file_fd)
            if temp_name is not None:
                try:
                    os.unlink(temp_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass

    @classmethod
    def _open_resource_directory(cls, run_dir: Path, include_participants: bool = False) -> tuple[int | None, Path]:
        """Open each server-owned path component without following symlinks."""

        run_dir = Path(run_dir)
        target_path = run_dir / RESOURCE_STATS_DIR
        if include_participants:
            target_path /= PARTICIPANTS_DIR
        if not _SECURE_DIR_FD:
            cls._ensure_plain_directory(run_dir, create=False)
            cls._ensure_plain_directory(run_dir / RESOURCE_STATS_DIR, create=True)
            if include_participants:
                cls._ensure_plain_directory(target_path, create=True)
            return None, target_path

        opened = []
        try:
            current_fd = os.open(run_dir, _DIRECTORY_OPEN_FLAGS)
            opened.append(current_fd)
            for component in (RESOURCE_STATS_DIR, PARTICIPANTS_DIR if include_participants else None):
                if component is None:
                    continue
                try:
                    os.mkdir(component, 0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
                opened.append(next_fd)
                current_fd = next_fd
            result_fd = opened.pop()
            return result_fd, target_path
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)

    @staticmethod
    def _ensure_plain_directory(path: Path, create: bool) -> None:
        if create:
            try:
                path.mkdir(mode=0o700)
            except FileExistsError:
                pass
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise OSError(f"resource directory is unavailable: {path}") from exc
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise OSError(f"resource path is not a plain directory: {path}")

    @classmethod
    def _reset_resource_directory(cls, run_dir: Path) -> None:
        cls._ensure_plain_directory(run_dir, create=False)
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
        stats_fd, stats_path = cls._open_resource_directory(run_dir)
        if stats_fd is None:
            staging_path = stats_path / "staging"
            try:
                mode = staging_path.lstat().st_mode
            except FileNotFoundError:
                return
            if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode) or any(staging_path.iterdir()):
                raise OSError("resource staging path is not an empty plain directory")
            staging_path.rmdir()
            return
        try:
            try:
                staging_fd = os.open("staging", _DIRECTORY_OPEN_FLAGS, dir_fd=stats_fd)
            except FileNotFoundError:
                return
            try:
                if os.listdir(staging_fd):
                    raise OSError("resource staging directory is not empty")
            finally:
                os.close(staging_fd)
            os.rmdir("staging", dir_fd=stats_fd)
            os.fsync(stats_fd)
        finally:
            os.close(stats_fd)

    @classmethod
    def _read_resource_file(cls, run_dir: Path, file_name: str, max_bytes: int) -> dict[str, Any]:
        directory_fd, directory_path = cls._open_resource_directory(run_dir)
        if directory_fd is None:
            with (directory_path / file_name).open("rb") as stream:
                data = stream.read(max_bytes + 1)
        else:
            try:
                file_fd = os.open(
                    file_name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=directory_fd,
                )
                try:
                    file_mode = os.fstat(file_fd).st_mode
                    if not stat.S_ISREG(file_mode):
                        raise OSError(f"resource member is not a regular file: {file_name}")
                    chunks = []
                    remaining = max_bytes + 1
                    while remaining:
                        chunk = os.read(file_fd, remaining)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    data = b"".join(chunks)
                finally:
                    os.close(file_fd)
            finally:
                os.close(directory_fd)
        if len(data) > max_bytes:
            raise ContractError(f"resource member exceeds its size limit: {file_name}")
        return load_and_validate(data, KIND_RESOURCE_SUMMARY)

    @classmethod
    def _verify_final_inventory(cls, run_dir: Path, accepted_names: set[str]) -> None:
        stats_fd, stats_path = cls._open_resource_directory(run_dir)
        expected_stats = {PARTICIPANTS_DIR, RESOURCE_SUMMARY_FILE}
        expected_participants = {f"{name}.json" for name in accepted_names}
        if stats_fd is None:
            if {path.name for path in stats_path.iterdir()} != expected_stats:
                raise OSError("resource statistics directory contains unexpected entries")
            participant_path = stats_path / PARTICIPANTS_DIR
            if {path.name for path in participant_path.iterdir()} != expected_participants:
                raise OSError("resource participant directory contains unexpected entries")
            return
        try:
            if set(os.listdir(stats_fd)) != expected_stats:
                raise OSError("resource statistics directory contains unexpected entries")
            participant_fd = os.open(PARTICIPANTS_DIR, _DIRECTORY_OPEN_FLAGS, dir_fd=stats_fd)
            try:
                if set(os.listdir(participant_fd)) != expected_participants:
                    raise OSError("resource participant directory contains unexpected entries")
                for name in expected_participants:
                    mode = os.stat(name, dir_fd=participant_fd, follow_symlinks=False).st_mode
                    if not stat.S_ISREG(mode):
                        raise OSError(f"resource participant member is not a regular file: {name}")
            finally:
                os.close(participant_fd)
        finally:
            os.close(stats_fd)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Best-effort persistence of an atomic rename's directory entry."""

        descriptor = None
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            os.fsync(descriptor)
        except OSError:
            # Some supported platforms do not permit opening/fsyncing a directory.
            # The file itself was fsynced and atomically replaced before this point.
            pass
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
