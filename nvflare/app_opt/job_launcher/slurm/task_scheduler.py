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
"""Task-specific accounting; resident JobLauncher queries remain unchanged."""

import re
from dataclasses import dataclass

from nvflare.app_opt.job_launcher.slurm.config import LookupStatus, QueryResult, SlurmProtocolError, SlurmRecord
from nvflare.app_opt.job_launcher.slurm.scheduler_client import _parse_state, _SlurmCliAdapter

_TASK_FORMAT = "JobIDRaw%64,JobName%128,UID%32,Comment%128,State%64,ExitCode%32,DerivedExitCode%32"


@dataclass(frozen=True)
class TaskSlurmRecord(SlurmRecord):
    derived_exit_signal: int = 0


class SlurmTaskCliAdapter(_SlurmCliAdapter):
    def task_accounting(self, job_id: str, job_name: str, marker: str, timeout: float) -> QueryResult:
        # -X deliberately omitted: a finished batch leader is insufficient.
        # Duplicates must be visible so ID recycling/requeue cannot look settled.
        command = self._run(
            [
                self.executables["sacct"],
                "--local",
                "--duplicates",
                "-n",
                "-P",
                f"--jobs={job_id}",
                f"--format={_TASK_FORMAT}",
            ],
            timeout,
        )
        if not command.available:
            return QueryResult(LookupStatus.UNAVAILABLE)
        records = {}
        for raw_id, name, uid, comment, state, exit_code, derived_code in self._rows(command, "sacct task", 7):
            returned_id = raw_id.strip()
            allocation = returned_id == job_id
            if not allocation and not re.fullmatch(re.escape(job_id) + r"\.(?:batch|extern|[0-9]+)", returned_id):
                raise SlurmProtocolError("unexpected Slurm task allocation/step identity")
            if returned_id in records:
                raise SlurmProtocolError("duplicate Slurm task accounting identity")
            if allocation:
                if uid.strip() != str(self.uid) or name.strip() != job_name or comment != marker:
                    raise SlurmProtocolError("Slurm task accounting ownership mismatch")
            elif uid.strip() not in ("", str(self.uid)):
                # Some Slurm versions omit allocation-only fields on step rows.
                raise SlurmProtocolError("Slurm task step ownership mismatch")
            if allocation and not derived_code.strip():
                raise SlurmProtocolError("Slurm task allocation is missing its derived exit code")
            try:
                status, sig = (int(value) for value in exit_code.strip().split(":", 1))
                derived, derived_sig = (int(value) for value in (derived_code.strip() or "0:0").split(":", 1))
            except ValueError as e:
                raise SlurmProtocolError("malformed Slurm task exit code") from e
            state = _parse_state(state)
            if not state or min(status, sig, derived, derived_sig) < 0:
                raise SlurmProtocolError("malformed Slurm task state/exit code")
            records[returned_id] = TaskSlurmRecord(returned_id, state, status, sig, derived, derived_sig)
        if not records:
            return QueryResult(LookupStatus.NOT_FOUND)
        if job_id not in records:
            raise SlurmProtocolError("Slurm task accounting is missing its allocation")
        return QueryResult(LookupStatus.FOUND, tuple(records.values()))
