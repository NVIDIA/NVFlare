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

"""One-boundary CPU accounting for a job execution process.

On Unix, SELF covers the process and its threads; CHILDREN covers only
terminated children that have been waited for, including descendants only
when every intervening process waited. No site-parent CPU or cgroup-wide
counter is used. This module does not sample or change worker teardown.
"""

from __future__ import annotations

import math
import os
import threading
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Any

from .contract import U128_MAX

try:
    import resource
except ImportError:
    resource = None

_NANOSECOND = Decimal("0.000000001")
_active: CpuConsumedAccountant | None = None
_active_lock = threading.Lock()


def _reset_accountant_after_fork() -> None:
    # The child has different process counters, and a copied thread lock may
    # have been held by a thread that did not survive the fork. Never acquire
    # an inherited lock here or use the parent's measurement baseline.
    global _active, _active_lock
    _active = None
    _active_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_accountant_after_fork)


def _cpu_seconds(usage: Any) -> Decimal:
    user = usage.ru_utime
    system = usage.ru_stime
    if not all(isinstance(value, (float, int)) and math.isfinite(value) and value >= 0 for value in (user, system)):
        raise ValueError("invalid process CPU counter")
    return Decimal(str(user)) + Decimal(str(system))


def _read_usage() -> tuple[Decimal, Decimal]:
    if resource is None:
        raise OSError("process CPU counters are unavailable on this platform")
    return (
        _cpu_seconds(resource.getrusage(resource.RUSAGE_SELF)),
        _cpu_seconds(resource.getrusage(resource.RUSAGE_CHILDREN)),
    )


def _seconds_text(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 100
        rounded = value.quantize(_NANOSECOND, rounding=ROUND_HALF_EVEN)
    rendered = format(rounded, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


class CpuConsumedAccountant:
    """Measure CPU used by one job process and its waited descendants.

    The job collector owns the shared participant measurement duration.
    This accountant records only the CPU counter delta over that window.
    """

    def __init__(
        self,
        *,
        usage_reader=_read_usage,
        prior_observation_incomplete: bool = False,
        known_attribution_incomplete: bool = False,
        include_children: bool = True,
    ):
        self._usage_reader = usage_reader
        self._start: tuple[Decimal, Decimal] | None = None
        self._children: list[Any] = []
        self._lock = threading.Lock()
        self._prior_observation_incomplete = prior_observation_incomplete
        self._known_attribution_incomplete = known_attribution_incomplete
        self._include_children = include_children
        try:
            self._start = self._usage_reader()
        except (OSError, ValueError, AttributeError, TypeError):
            pass

    def register_child(self, process: Any, *, descendants_may_be_unwaited: bool = False) -> None:
        """Track a platform-managed child without touching its normal lifecycle."""

        with self._lock:
            if descendants_may_be_unwaited:
                # We already know that a grandchild can escape CHILDREN usage.
                # Do not retain many completed trainers and their pipe handles
                # merely to reconfirm a status that must remain partial.
                self._known_attribution_incomplete = True
            else:
                self._children.append(process)

    def mark_attribution_incomplete(self) -> None:
        """Record a lifecycle path that may outlive the final boundary."""

        with self._lock:
            self._known_attribution_incomplete = True

    def exclude_children(self) -> None:
        """Use SELF over the existing baseline when workers report separately."""

        with self._lock:
            self._include_children = False

    def finish(self) -> dict[str, Any]:
        # poll() is nonblocking and reaps an already-exited child. It cannot
        # make an un-waited live worker appear in RUSAGE_CHILDREN.
        with self._lock:
            children = tuple(self._children)
            known_attribution_incomplete = self._known_attribution_incomplete
        issues = set()
        if self._prior_observation_incomplete:
            issues.add("observation_incomplete")
        if known_attribution_incomplete:
            issues.add("attribution_incomplete")
        for process in children:
            try:
                # subprocess.Popen.poll() and multiprocessing.Process.exitcode
                # each perform at most one nonblocking completion check.
                completed = process.poll() if hasattr(process, "poll") else process.exitcode
                if completed is None:
                    issues.add("attribution_incomplete")
            except Exception:
                issues.add("attribution_incomplete")

        if self._start is None:
            return {"status": "unavailable", "issues": ["observation_incomplete"]}
        try:
            end = self._usage_reader()
            with localcontext() as context:
                context.prec = 100
                seconds = end[0] - self._start[0]
                if self._include_children:
                    seconds += end[1] - self._start[1]
            if not seconds.is_finite() or seconds < 0 or seconds > U128_MAX:
                raise ValueError("invalid CPU counter delta")
        except (OSError, ValueError, AttributeError, TypeError):
            return {"status": "unavailable", "issues": ["observation_incomplete"]}
        result = {"cpu_seconds": _seconds_text(seconds)}
        if issues:
            result["status"] = "partial"
            result["issues"] = sorted(issues)
        return result


def install_cpu_accountant(accountant: CpuConsumedAccountant) -> None:
    """Make the current job process's collector available to managed launchers."""

    global _active
    with _active_lock:
        _active = accountant


def register_managed_child(process: Any, *, descendants_may_be_unwaited: bool = False) -> None:
    """Mark a known child; no-op in a shared site parent or unrelated process."""

    with _active_lock:
        accountant = _active
    if accountant is not None:
        accountant.register_child(process, descendants_may_be_unwaited=descendants_may_be_unwaited)


def mark_unwaited_managed_children() -> None:
    """Mark a managed worker pool whose members are not reliably waited for."""

    with _active_lock:
        accountant = _active
    if accountant is not None:
        accountant.mark_attribution_incomplete()
