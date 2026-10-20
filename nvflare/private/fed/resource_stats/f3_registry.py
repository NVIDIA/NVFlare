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

"""Per-job :class:`F3Counter` ownership for a long-lived parent process.

A job process (SJ/CJ) only ever runs one job, so it can own a single
process-global counter. A parent process (SP: the root server; CP: a client
parent) is long-lived and handles many jobs over its lifetime -- often
concurrently on the server side -- so it needs one counter per job_id, kept
alive from job start through finalization and discarded afterward. This is
that per-job registry; it owns no traffic-classification logic itself (that
lives at the trusted NVFlare call sites that use ``get()``).
"""

from __future__ import annotations

import threading
from typing import Optional

from .f3_counter import F3_PARENT_DRAIN_TIMEOUT_SECONDS, F3Counter


class F3CounterRegistry:
    """Thread-safe job_id -> F3Counter registry for one parent process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, F3Counter] = {}

    def start_job(self, job_id: str, *, prior_history_incomplete: bool = False) -> F3Counter:
        """Create (or return the existing) counter for a job.

        Idempotent so a caller does not need to track whether it already
        started this job's counter. ``prior_history_incomplete`` is used when
        a parent restores a job whose traffic before the restart is unknown.
        """

        with self._lock:
            counter = self._counters.get(job_id)
            if counter is None:
                counter = F3Counter()
                self._counters[job_id] = counter
            if prior_history_incomplete:
                counter.mark_prior_history_incomplete()
            return counter

    def get(self, job_id: str) -> Optional[F3Counter]:
        """Return the job's counter, or ``None`` if it was never started (or already forgotten)."""

        with self._lock:
            return self._counters.get(job_id)

    def mark_prior_history_incomplete(self, job_id: str) -> bool:
        """Mark an existing job counter partial after a parent restart."""

        counter = self.get(job_id)
        return counter.mark_prior_history_incomplete() if counter is not None else False

    def close_and_freeze(
        self, job_id: str, *, drain_timeout_seconds: float = F3_PARENT_DRAIN_TIMEOUT_SECONDS
    ) -> Optional[dict]:
        """Close, condition-drain for a bounded time, and freeze a job counter."""

        counter = self.get(job_id)
        if counter is None:
            return None
        counter.close_and_drain(drain_timeout_seconds)
        return counter.freeze()

    def forget_job(self, job_id: str) -> None:
        """Drop the registry's reference to a job's counter. Safe to call on an unknown job_id."""

        with self._lock:
            self._counters.pop(job_id, None)
