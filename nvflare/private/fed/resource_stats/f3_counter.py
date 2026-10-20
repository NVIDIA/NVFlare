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

"""Thread-safe, process-local F3 accounting for one job.

F3 counts originating logical sends, not transport frames or intermediary
forwarding hops. A trusted call site admits an included operation before it
starts, then either completes it after the transport accepts the operation or
abandons it when acceptance fails. The accounting API never owns or wraps
the send, so an observability failure cannot change send behaviour.

The counter has a one-way ``collecting`` -> ``closing`` -> ``frozen`` state
machine. Closing stops new admissions while allowing admitted operations to
finish. A bounded, condition-based drain is available before freezing. Any
operation still pending at the cutoff makes the result partial.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .communication_limits import PARTICIPANT_NAME_PATTERN, U128_MAX

_COLLECTING = "collecting"
_CLOSING = "closing"
_FROZEN = "frozen"

_ISSUE_ATTRIBUTION_INCOMPLETE = "attribution_incomplete"
_ISSUE_COUNTER_GAP = "counter_gap"

# A fixed child-process cutoff, not user configuration. It leaves a small,
# bounded interval for an accepted final stream to settle without allowing a
# stuck accounting operation to hold finalization forever.
F3_DRAIN_TIMEOUT_SECONDS = 5.0

# Parent processes originate no late task traffic. The CP has no included
# origin class, and the SP's blocking job-application sends finish during
# deployment. A parent admission still pending at terminal cleanup is therefore
# an accounting gap, not useful work for terminal cleanup to wait on.
F3_PARENT_DRAIN_TIMEOUT_SECONDS = 0.0

# A participant report is capped at 1 MiB. Bound distinct recipient groups so
# a large fanout can degrade to a partial subtotal instead of growing without
# limit or making the terminal report unpublishable.
MAX_F3_SENT_TO_GROUPS = 2048


class F3TrafficClass(str, Enum):
    """The closed allowlist of logical operations included in F3 v1.

    Only the trusted NVFlare call site that originates an operation may assign
    a class. A wire header, topic, relaying process, or job-supplied value is
    not classification authority.
    """

    JOB_APPLICATION = "job_application"
    TASK_RESPONSE = "task_response"
    TASK_RESULT = "task_result"


INCLUDED_F3_TRAFFIC_CLASSES = frozenset(F3TrafficClass)


@dataclass(frozen=True)
class _FrozenState:
    """Immutable canonical state retained after the publication cutoff."""

    status: str
    issues: tuple[str, ...]
    sent_to: tuple[tuple[str, int, int], ...]

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.status != "reported":
            result["status"] = self.status
        if self.issues:
            result["issues"] = list(self.issues)
        result["sent_to"] = [
            {"participant_name": name, "payload_bytes": str(payload_bytes), "messages": str(messages)}
            for name, payload_bytes, messages in self.sent_to
        ]
        return result


class F3Admission:
    """Opaque token for one originating logical send."""

    __slots__ = ("_completed", "_extra_payload_bytes", "_invalid", "_owner", "_recipient_name", "_traffic_class")

    def __init__(self, owner: object, traffic_class: F3TrafficClass, recipient_name: str):
        self._owner = owner
        self._recipient_name = recipient_name
        self._traffic_class = traffic_class
        self._completed = False
        self._extra_payload_bytes = 0
        self._invalid = False


def _valid_payload_bytes(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= U128_MAX


def _valid_timeout(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0 <= value <= threading.TIMEOUT_MAX
    if isinstance(value, float):
        return math.isfinite(value) and 0 <= value <= threading.TIMEOUT_MAX
    return False


class F3Counter:
    """One process-local, job-scoped F3 sender counter.

    All public mutation methods are fail-safe: malformed or duplicate
    observability calls return ``False``/``None`` instead of raising into the
    job's send path. A malformed admitted observation makes the final result
    partial rather than publishing a value known to be incomplete.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._owner = object()
        self._state = _COLLECTING
        self._pending = 0
        self._payload_bytes = 0
        self._messages = 0
        self._sent_to: dict[str, tuple[int, int]] = {}
        self._issues: set[str] = set()
        self._frozen_state: Optional[_FrozenState] = None

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    @property
    def pending_count(self) -> int:
        """Return the number of admitted operations awaiting an outcome."""

        with self._condition:
            return self._pending

    def try_begin(self, traffic_class: F3TrafficClass, recipient_name: str) -> Optional[F3Admission]:
        """Try to admit one originating logical send.

        ``None`` means the observation was not admitted. This is expected
        after :meth:`close`; callers must still perform the real send.
        Supplying anything except an allowlisted enum is treated as an
        instrumentation gap, but is never allowed to raise into send logic.
        """

        with self._condition:
            if self._state != _COLLECTING:
                return None
            if not isinstance(traffic_class, F3TrafficClass) or traffic_class not in INCLUDED_F3_TRAFFIC_CLASSES:
                self._issues.add(_ISSUE_COUNTER_GAP)
                return None
            if not isinstance(recipient_name, str) or not PARTICIPANT_NAME_PATTERN.fullmatch(recipient_name):
                self._issues.add(_ISSUE_COUNTER_GAP)
                return None
            self._pending += 1
            return F3Admission(self._owner, traffic_class, recipient_name)

    def add_accepted_payload_bytes(self, admission: Optional[F3Admission], payload_bytes: int) -> bool:
        """Add accepted out-of-band bytes to a pending logical operation.

        FOBS can replace a large object with a DownloadService reference. The
        successful source bytes for that object still belong to the same
        logical operation, but must not increment ``messages``. They are kept
        on the admission and published atomically by
        :meth:`complete_remote_accepted` with the main post-FOBS payload.
        """

        with self._condition:
            if not isinstance(admission, F3Admission) or admission._owner is not self._owner:
                return False
            if admission._completed:
                if self._frozen_state is None:
                    self._issues.add(_ISSUE_COUNTER_GAP)
                return False
            if self._frozen_state is not None:
                return False
            if not _valid_payload_bytes(payload_bytes):
                admission._invalid = True
                self._issues.add(_ISSUE_COUNTER_GAP)
                return False
            if admission._extra_payload_bytes > U128_MAX - payload_bytes:
                admission._invalid = True
                self._issues.add(_ISSUE_COUNTER_GAP)
                return False
            admission._extra_payload_bytes += payload_bytes
            return True

    def complete_remote_accepted(self, admission: Optional[F3Admission], payload_bytes: int) -> bool:
        """Record one admitted operation accepted by the local transport.

        ``payload_bytes`` is the serialized logical payload size at the
        agreed F3 boundary. Each admission represents one logical message to
        one destination, so a successful completion increments ``messages``
        by exactly one.

        The return value is diagnostic only. ``False`` means the observation
        was not applied (for example, because it was malformed, duplicated,
        foreign, or completed after the cutoff).
        """

        with self._condition:
            if not self._owns_incomplete(admission):
                return False

            admission._completed = True
            self._pending -= 1
            self._condition.notify_all()

            if self._frozen_state is not None:
                return False
            if admission._invalid or not _valid_payload_bytes(payload_bytes):
                self._issues.add(_ISSUE_COUNTER_GAP)
                return False
            if admission._extra_payload_bytes > U128_MAX - payload_bytes:
                self._issues.add(_ISSUE_COUNTER_GAP)
                return False
            operation_bytes = admission._extra_payload_bytes + payload_bytes
            if self._payload_bytes > U128_MAX - operation_bytes or self._messages == U128_MAX:
                self._issues.add(_ISSUE_COUNTER_GAP)
                return False

            recipient_name = admission._recipient_name
            if recipient_name not in self._sent_to and len(self._sent_to) >= MAX_F3_SENT_TO_GROUPS:
                self._issues.add(_ISSUE_COUNTER_GAP)
                return False
            recipient_bytes, recipient_messages = self._sent_to.get(recipient_name, (0, 0))

            self._payload_bytes += operation_bytes
            self._messages += 1
            self._sent_to[recipient_name] = (recipient_bytes + operation_bytes, recipient_messages + 1)
            return True

    def abandon(self, admission: Optional[F3Admission]) -> bool:
        """Release an admission whose operation was not transport-accepted.

        A known rejection is not a counter gap and contributes no public F3
        value. This method is also safe after close/freeze and never raises.
        """

        with self._condition:
            if not self._owns_incomplete(admission):
                return False
            admission._completed = True
            self._pending -= 1
            self._condition.notify_all()
            return True

    def mark_incomplete(self, admission: Optional[F3Admission]) -> bool:
        """Release an admitted operation whose observation cannot be finalized.

        Unlike :meth:`abandon`, this means the accounting outcome is unknown,
        not that the transport definitely rejected the operation. Any bytes
        accumulated on the admission are discarded and the final snapshot is
        marked ``partial/counter_gap``.
        """

        with self._condition:
            if not self._owns_incomplete(admission):
                return False
            admission._completed = True
            self._pending -= 1
            self._issues.add(_ISSUE_COUNTER_GAP)
            self._condition.notify_all()
            return True

    def mark_counter_gap(self) -> bool:
        """Mark an instrumentation failure that occurred before admission.

        This is distinct from missing history after a process restart. It is
        intended for fail-safe integration boundaries that could not create
        or attach an accounting context but must not disrupt the real send.
        """

        with self._condition:
            if self._frozen_state is not None:
                return False
            self._issues.add(_ISSUE_COUNTER_GAP)
            return True

    def mark_prior_history_incomplete(self) -> bool:
        """Record that traffic before this counter was created is unavailable.

        Parent processes use this after restoring a running job. The method
        is idempotent and returns ``False`` only after the publication cutoff.
        """

        with self._condition:
            if self._frozen_state is not None:
                return False
            self._issues.add(_ISSUE_ATTRIBUTION_INCOMPLETE)
            return True

    def close(self) -> None:
        """Stop admitting new operations; already-admitted ones may finish."""

        with self._condition:
            if self._state == _COLLECTING:
                self._state = _CLOSING

    def close_and_drain(self, timeout_seconds: float) -> bool:
        """Close admission and wait at most ``timeout_seconds`` for pending work.

        Waiting uses a condition rather than polling. ``True`` means every
        admitted operation completed before return. An invalid timeout is
        treated as a zero-length drain and marks the observation partial;
        observability mistakes never raise into lifecycle code.
        """

        with self._condition:
            if self._state == _COLLECTING:
                self._state = _CLOSING
            if not _valid_timeout(timeout_seconds):
                self._issues.add(_ISSUE_COUNTER_GAP)
                timeout_seconds = 0.0
            return self._condition.wait_for(lambda: self._pending == 0, timeout=float(timeout_seconds))

    def freeze(self) -> dict[str, Any]:
        """Fix the publication cutoff and return a defensive snapshot copy."""

        with self._condition:
            if self._frozen_state is None:
                if self._pending:
                    self._issues.add(_ISSUE_COUNTER_GAP)
                self._state = _FROZEN
                self._frozen_state = self._make_state_locked(include_pending=False)
            return self._frozen_state.as_dict()

    def snapshot(self) -> dict[str, Any]:
        """Return a defensive diagnostic snapshot without fixing the cutoff."""

        with self._condition:
            if self._frozen_state is not None:
                return self._frozen_state.as_dict()
            return self._make_state_locked(include_pending=True).as_dict()

    def _owns_incomplete(self, admission: object) -> bool:
        return isinstance(admission, F3Admission) and admission._owner is self._owner and not admission._completed

    def _make_state_locked(self, *, include_pending: bool) -> _FrozenState:
        issues = set(self._issues)
        if include_pending and self._pending:
            issues.add(_ISSUE_COUNTER_GAP)
        ordered_issues = tuple(sorted(issues))
        return _FrozenState(
            status="partial" if ordered_issues else "reported",
            issues=ordered_issues,
            sent_to=tuple((name, *self._sent_to[name]) for name in sorted(self._sent_to)),
        )
