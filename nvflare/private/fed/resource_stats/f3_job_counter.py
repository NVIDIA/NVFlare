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

"""Optional process-local counter shared by one job's send call sites.

Importing this module does not activate accounting. Reporting integration owns
counter startup, final draining and publication. Without that consumer, send
hooks receive None and do not create accounting contexts or transfer ledgers.
The long-lived server parent uses the per-job registry in f3_registry.py instead.
"""

from __future__ import annotations

import threading
from typing import Optional

from .f3_counter import F3Counter

_lock = threading.Lock()
_counter: Optional[F3Counter] = None


def start_job_f3_counter() -> F3Counter:
    """Create this process's counter once and return it.

    Startup paths may converge or retry. Returning the existing counter keeps
    observations already recorded in this one-job process instead of silently
    replacing them.
    """

    global _counter
    with _lock:
        if _counter is None:
            _counter = F3Counter()
        return _counter


def get_job_f3_counter() -> Optional[F3Counter]:
    """Return this process's counter, or ``None`` if it was never started."""

    with _lock:
        return _counter


def clear_job_f3_counter() -> None:
    """Drop the reference so a later, unrelated import of this module starts clean.

    Job processes are short-lived (one job per process), so this mainly
    matters for test isolation.
    """

    global _counter
    with _lock:
        _counter = None
