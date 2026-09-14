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

"""Shared classification of current and legacy job statuses."""

from typing import Optional

from nvflare.apis.job_def import RunStatus

_LEGACY_TERMINAL_JOB_STATUSES = {
    "FINISHED_OK",
    "FINISHED_EXCEPTION",
    "ABORTED",
    "ABANDONED",
    "FAILED",
}


def is_terminal_job_status(status: str) -> bool:
    return isinstance(status, str) and (status.startswith("FINISHED") or status in _LEGACY_TERMINAL_JOB_STATUSES)


def job_status_outcome(status: str) -> Optional[str]:
    """Return the presentation outcome for a current or legacy terminal status."""
    if not is_terminal_job_status(status):
        return None
    if status == RunStatus.FINISHED_CANT_SCHEDULE.value:
        return "not_scheduled"
    return "completed" if status in (RunStatus.FINISHED_COMPLETED.value, "FINISHED_OK") else "failed"
