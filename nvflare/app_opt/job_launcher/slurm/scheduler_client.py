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
"""Bounded Slurm commands, exact lookup, ownership checks, and cancellation."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from typing import Callable, Optional

from nvflare.app_opt.job_launcher.slurm.config import (
    _MAX_STDERR_BYTES,
    _MAX_STDOUT_BYTES,
    _SUBMIT_ID_RE,
    SACCT_FORMAT,
    SQUEUE_FORMAT,
    CommandResult,
    LookupStatus,
    QueryResult,
    SlurmProtocolError,
    SlurmRecord,
    SubmissionResult,
)

_STDERR_TRUNCATION_MARKER = "\n...[stderr truncated]"
_STRAY_ROW_WARNING_INTERVAL = 60.0


def _command_diagnostic(command: CommandResult) -> str:
    if command.timed_out:
        rc = "timeout"
    elif command.returncode is None:
        rc = "unavailable"
    else:
        rc = str(command.returncode)
    stderr = "".join(char if char.isprintable() else " " for char in command.stderr)
    return f"rc={rc} stderr={' '.join(stderr.split()) or '<empty>'}"


def _scheduler_env() -> dict:
    result = {}
    for name, value in os.environ.items():
        if name.startswith(("SBATCH_", "SQUEUE_", "SACCT_", "SCANCEL_")):
            continue
        if name in {
            "SLURM_CLUSTERS",
            "SLURM_EXPORT_ENV",
            "SLURM_HINT",
            "SLURM_MEM_PER_CPU",
            "SLURM_MEM_PER_GPU",
            "SLURM_MEM_PER_NODE",
            "LC_ALL",
        }:
            continue
        result[name] = value
    result["LC_ALL"] = "C"
    return result


def _parse_state(value: str) -> str:
    tokens = value.strip().split(maxsplit=1)
    return tokens[0].removesuffix("+") if tokens else ""


class _SlurmCliAdapter:
    def __init__(
        self,
        executables: dict,
        uid: int,
        logger: logging.Logger,
        runner: Optional[Callable] = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ):
        self.executables = executables
        self.uid = uid
        try:
            import pwd

            self.user = pwd.getpwuid(uid).pw_name
        except (ImportError, KeyError) as e:
            raise SlurmProtocolError(f"cannot resolve Slurm submit user for UID {uid}") from e
        self.logger = logger
        self._runner = runner
        self._monotonic = monotonic_clock
        self._warning_lock = threading.Lock()
        self._stray_warning_at = float("-inf")

    def _run(self, argv: list[str], timeout: float) -> CommandResult:
        timeout = max(timeout, 0.001)
        if self._runner is not None:
            return self._runner(argv, timeout)
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                timeout=timeout,
                env=_scheduler_env(),
            )
        except subprocess.TimeoutExpired:
            return CommandResult(None, "", "", timed_out=True)
        except OSError as e:
            return CommandResult(None, "", type(e).__name__)

        if len(completed.stdout) > _MAX_STDOUT_BYTES:
            self.logger.error("Slurm command stdout exceeded %d bytes: %s", _MAX_STDOUT_BYTES, argv[0])
            raise SlurmProtocolError(f"Slurm command stdout exceeded the {_MAX_STDOUT_BYTES}-byte hard limit")
        stderr = completed.stderr[:_MAX_STDERR_BYTES]
        stderr_text = stderr.decode("utf-8", errors="replace")
        if len(completed.stderr) > _MAX_STDERR_BYTES:
            stderr_text += _STDERR_TRUNCATION_MARKER
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout.decode("utf-8", errors="replace"),
            stderr=stderr_text,
        )

    @staticmethod
    def _rows(command: CommandResult, source: str, width: int):
        for line in command.stdout.splitlines():
            if not line.strip():
                continue
            fields = line.split("|", width - 1)
            if len(fields) != width:
                raise SlurmProtocolError(f"malformed {source} row")
            yield fields

    def submit(self, argv: list[str], timeout: float) -> SubmissionResult:
        command = self._run(argv, timeout)
        lines = [line.strip() for line in command.stdout.splitlines() if line.strip()]
        matched = _SUBMIT_ID_RE.fullmatch(lines[0]) if len(lines) == 1 else None
        return SubmissionResult(
            command=command,
            job_id=matched.group(1) if matched else None,
            cluster=matched.group(2) if matched else None,
        )

    def accounting_probe(self, timeout: float) -> CommandResult:
        return self._run(
            [self.executables["sacct"], "-X", "-n", "--starttime=now", "--format=JobIDRaw"],
            timeout,
        )

    def version_probe(self, timeout: float) -> CommandResult:
        return self._run([self.executables["sbatch"], "--version"], timeout)

    def _warn_stray_row(self, job_name: str, job_id: str) -> None:
        now = self._monotonic()
        with self._warning_lock:
            if now - self._stray_warning_at < _STRAY_ROW_WARNING_INTERVAL:
                return
            self._stray_warning_at = now
        self.logger.warning("ignoring Slurm job %s sharing name %s", job_id, job_name)

    def active_by_id(
        self,
        job_id: str,
        job_name: str,
        marker: str,
        timeout: float,
    ) -> QueryResult:
        command = self._run(
            [
                self.executables["squeue"],
                "-h",
                f"--user={self.uid}",
                f"--name={job_name}",
                f"--format={SQUEUE_FORMAT}",
            ],
            timeout,
        )
        if not command.available:
            return QueryResult(LookupStatus.UNAVAILABLE)
        records = []
        for raw_id, raw_state, raw_uid, comment, returned_name in self._rows(command, "squeue", 5):
            returned_id = raw_id.strip()
            if returned_id != job_id:
                self._warn_stray_row(job_name, returned_id)
                continue
            try:
                uid = int(raw_uid.strip())
            except ValueError as e:
                raise SlurmProtocolError("malformed squeue UID") from e
            state = _parse_state(raw_state)
            if not state:
                raise SlurmProtocolError("malformed squeue row")
            if uid != self.uid or comment != marker or returned_name.strip() != job_name:
                raise SlurmProtocolError(f"squeue ownership mismatch for job name {job_name}")
            records.append(SlurmRecord(job_id=returned_id, state=state))
        if len(records) > 1:
            raise SlurmProtocolError(f"duplicate squeue rows for job {job_id}")
        return QueryResult(LookupStatus.FOUND if records else LookupStatus.NOT_FOUND, tuple(records))

    def accounting_by_id(self, job_id: str, job_name: str, timeout: float) -> QueryResult:
        command = self._run(
            [
                self.executables["sacct"],
                "-X",
                "-n",
                "-P",
                f"--jobs={job_id}",
                f"--format={SACCT_FORMAT}",
            ],
            timeout,
        )
        if not command.available:
            return QueryResult(LookupStatus.UNAVAILABLE)
        records = []
        for raw_id, raw_name, raw_user, raw_state, raw_exit in self._rows(command, "sacct", 5):
            returned_id = raw_id.strip()
            try:
                status, signal = (int(value) for value in raw_exit.strip().split(":", 1))
            except ValueError as e:
                raise SlurmProtocolError("malformed sacct numeric field") from e
            state = _parse_state(raw_state)
            if (
                not returned_id.isdigit()
                or not state
                or min(status, signal) < 0
                or raw_user.strip() != self.user
                or raw_name.strip() != job_name
            ):
                raise SlurmProtocolError(f"sacct ownership or protocol mismatch for job name {job_name}")
            records.append(
                SlurmRecord(
                    job_id=returned_id,
                    state=state,
                    exit_status=status,
                    exit_signal=signal,
                )
            )
        if not records:
            return QueryResult(LookupStatus.NOT_FOUND)
        if len(records) != 1 or records[0].job_id != job_id:
            raise SlurmProtocolError(f"sacct identity mismatch for job {job_id}")
        return QueryResult(LookupStatus.FOUND, tuple(records))

    def cancel(self, job_id: str, timeout: float) -> QueryResult:
        command = self._run(
            [self.executables["scancel"], "-Q", "--me", job_id],
            timeout,
        )
        return QueryResult(LookupStatus.FOUND if command.available else LookupStatus.UNAVAILABLE)

    def cancel_suffix(self, job_id: str, suffix: str, timeout: float) -> QueryResult:
        command = self._run(
            [self.executables["scancel"], "-M", suffix, "-Q", "--me", job_id],
            timeout,
        )
        return QueryResult(LookupStatus.FOUND if command.available else LookupStatus.UNAVAILABLE)
