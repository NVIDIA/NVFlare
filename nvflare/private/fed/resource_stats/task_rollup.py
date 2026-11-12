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

"""Bounded, participant-owned reduction of settled task-worker observations."""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from dataclasses import dataclass, replace

from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity

from .measurements import (
    _sum_numbers,
    aggregate_cpu_consumed,
    aggregate_message_traffic,
    aggregate_scope_issues,
    capacity_to_resource_time,
)

MAX_SIDECAR_BYTES = 64 * 1024
_MAX_SCOPES = 10_000
_MAX_STATE_BYTES = 4 * 1024 * 1024
_MAX_PENDING_BYTES = 8 * 1024 * 1024
_UNAVAILABLE = {"status": "unavailable"}
_ACTIVE = None
_ACTIVE_LOCK = threading.Lock()


def _reset_after_fork():
    global _ACTIVE, _ACTIVE_LOCK
    _ACTIVE = None
    _ACTIVE_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def install_task_resource_reducer(reducer):
    """Install the current physical Client Job's owner; never a site parent."""
    global _ACTIVE
    with _ACTIVE_LOCK:
        _ACTIVE = reducer


def get_task_resource_reducer():
    with _ACTIVE_LOCK:
        return _ACTIVE


def _encoded(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _key(identity):
    return (identity.job_id, identity.site_name, identity.task_id, identity.attempt_id, "worker")


@dataclass
class _Scope:
    identity: TaskAttemptIdentity
    execution_id: str | None = None
    disposition: str = "unresolved"
    record: str | None = None
    reserved: bool = True


def _scope_size(scope):
    # Count the serialized ledger, not an asserted bound on Python heap usage.
    # _encoded escapes non-ASCII characters, so character and byte counts agree.
    return (
        len(
            _encoded(
                {
                    "identity": scope.identity.to_dict(),
                    "execution_id": scope.execution_id,
                    "disposition": "settled_missing",
                    "record": scope.record,
                    "reserved": False,
                }
            )
        )
        + 2
    )


class TaskResourceReducer:
    """Count each trusted singleton session once without controlling its lifecycle.

    The launcher owns all handles, waiting, cancellation and settlement. This
    ledger only consumes optional final files after that owner confirms exit.
    No identities are evicted: losing one could let a replay add work twice.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._scopes = {}
        self._state_bytes = 0
        self._pending_bytes = 0
        self._totals = {"resource_time": None, "cpu_consumed": None, "message_traffic": None, "issues": []}
        self._incomplete = False
        self._frozen = False
        self._containment_lost = False

    def _fits(self, entry_bytes, totals):
        return entry_bytes + len(_encoded(totals)) + 1024 <= _MAX_STATE_BYTES

    def _freeze(self):
        self._frozen = True
        self._incomplete = True

    def mark_incomplete(self):
        """Keep optional collection failures out of the runtime control path."""
        with self._lock:
            self._incomplete = True

    def register(self, identity):
        """Reserve one sidecar before launch; false disables only statistics."""
        key = _key(identity)
        with self._lock:
            previous = self._scopes.get(key)
            if previous is not None:
                if previous.identity != identity:
                    self._incomplete = True
                return previous.identity == identity and previous.reserved and previous.disposition == "unresolved"
            if self._frozen:
                self._containment_lost = True
                return False
            scope = _Scope(identity)
            size = _scope_size(scope)
            if len(self._scopes) >= _MAX_SCOPES or not self._fits(self._state_bytes + size, self._totals):
                self._freeze()
                self._containment_lost = True
                return False
            if self._pending_bytes + MAX_SIDECAR_BYTES > _MAX_PENDING_BYTES:
                self._incomplete = True
                self._containment_lost = True
                return False
            self._scopes[key] = scope
            self._state_bytes += size
            self._pending_bytes += MAX_SIDECAR_BYTES
            return True

    def bind(self, identity, execution_id):
        """Keep the runtime's accepted handle identity in its private owner row."""
        with self._lock:
            scope = self._scopes.get(_key(identity))
            if scope is None:
                return
            if scope.identity != identity:
                self._incomplete = True
                return
            if scope.execution_id is not None and scope.execution_id != execution_id:
                self._incomplete = True
                return
            candidate = replace(scope, execution_id=execution_id)
            size = self._state_bytes - _scope_size(scope) + _scope_size(candidate)
            if not self._fits(size, self._totals):
                self._freeze()
                return
            self._scopes[_key(identity)] = candidate
            self._state_bytes = size

    @staticmethod
    def merge_resource_time(previous, contribution, issues=None):
        """Keep independent dimension subtotals when another one exceeds bounds."""
        values = [value for value in (previous, contribution) if value is not None]
        result = {}
        incomplete = any("status" in value for value in values)
        for dimension, field in (("cpu", "unit_seconds"), ("memory", "byte_seconds"), ("gpu", "instance_seconds")):
            amounts = [value[dimension][field] for value in values if dimension in value]
            if not amounts:
                continue
            try:
                result[dimension] = {field: _sum_numbers(amounts)}
            except (OverflowError, ValueError):
                # An overflowing dimension cannot erase other usable subtotals.
                if previous is not None and dimension in previous:
                    result[dimension] = deepcopy(previous[dimension])
                incomplete = True
                if issues is not None:
                    issues.append({"metric": "resource_time", "code": "observation_incomplete"})
        if not result:
            if issues is not None:
                issues[:] = [issue for issue in issues if issue["metric"] != "resource_time"]
                issues.append({"metric": "resource_time", "code": "observation_incomplete"})
            return dict(_UNAVAILABLE)
        if incomplete or set(result) != {"cpu", "memory", "gpu"}:
            result["status"] = "partial"
        return result

    def _updated_totals(self, record):
        totals = deepcopy(self._totals)
        record_issues = list(record["issues"])
        resource_time = capacity_to_resource_time(
            record["resource_capacity"], record.get("duration_seconds"), record_issues
        )
        issues = totals["issues"]
        issues.extend(
            issue for issue in record_issues if issue["metric"] in {"resource_time", "cpu_consumed", "message_traffic"}
        )
        totals["resource_time"] = self.merge_resource_time(totals["resource_time"], resource_time, issues)
        for field, aggregate in (
            ("cpu_consumed", aggregate_cpu_consumed),
            ("message_traffic", aggregate_message_traffic),
        ):
            sources = [record[field]]
            if totals[field] is not None:
                sources.insert(0, totals[field])
            try:
                totals[field] = aggregate(sources)
            except (OverflowError, ValueError):
                issues.append({"metric": field, "code": "observation_incomplete"})
                previous = totals[field]
                totals[field] = aggregate([previous], complete=False) if previous is not None else dict(_UNAVAILABLE)
        totals["issues"] = aggregate_scope_issues(
            issues, {field: value for field, value in totals.items() if field != "issues" and value is not None}
        )
        return totals

    def settle(self, identity, store, *, started=None, settled=False):
        """Consume independently of result validity, only after physical settlement."""
        key = _key(identity)
        with self._lock:
            scope = self._scopes.get(key)
            if scope is None or scope.identity != identity:
                return
            if not settled and started is not False:
                # The unresolved row itself marks snapshots incomplete. A later
                # normal cleanup proof can still supply its final contribution.
                return
        record = None
        invalid = False
        if started is not False:
            try:
                record = store.read_resource_stats(identity)
            except Exception:
                invalid = True
        with self._lock:
            scope = self._scopes[key]
            previous_size = _scope_size(scope)
            candidate = replace(scope)
            if scope.disposition != "unresolved":
                if record is not None and (scope.record is None or json.loads(scope.record) != record):
                    self._incomplete = True
            elif started is False:
                candidate.disposition = "never_started"
            elif record is None or self._frozen:
                candidate.disposition = "invalid" if invalid else "settled_missing"
                self._incomplete = True
            else:
                try:
                    candidate.disposition = "reported"
                    candidate.record = _encoded(record)
                    totals = self._updated_totals(record)
                    size = self._state_bytes - previous_size + _scope_size(candidate)
                    if self._fits(size, totals):
                        self._totals = totals
                    else:
                        self._freeze()
                        candidate = replace(candidate, record=None, disposition="invalid")
                except (KeyError, TypeError, ValueError, OverflowError):
                    # A failed optional contribution leaves earlier totals intact
                    # and still retires its file and reservation below.
                    self._incomplete = True
                    candidate = replace(candidate, record=None, disposition="invalid")
            self._state_bytes += _scope_size(candidate) - previous_size
            self._scopes[key] = candidate
        retired = started is False
        if started is not False:
            try:
                # This optional file is outside checkpoint/result commit and receipts.
                store.remove_resource_stats(identity)
                retired = True
            except Exception:
                with self._lock:
                    self._incomplete = True
        if retired:
            with self._lock:
                scope = self._scopes[key]
                if scope.reserved:
                    # A failed deletion still occupies this generation's disk budget.
                    # Successful retries release it exactly once without recounting.
                    scope.reserved = False
                    self._pending_bytes -= MAX_SIDECAR_BYTES

    def snapshot(self):
        """Return worker-only subtotals; None means no contribution and no gap."""
        with self._lock:
            result = deepcopy(self._totals)
            # Rejected registrations may still execute without a tracked settlement.
            contained = not self._containment_lost and all(
                scope.disposition != "unresolved" for scope in self._scopes.values()
            )
            incomplete = self._incomplete or not contained
        issues = result["issues"]
        if incomplete:
            for field in ("resource_time", "cpu_consumed", "message_traffic"):
                value = result[field]
                if value is None:
                    result[field] = dict(_UNAVAILABLE)
                elif value.get("status") != "unavailable":
                    value["status"] = "partial"
                issues.append({"metric": field, "code": "observation_incomplete"})
        result["issues"] = aggregate_scope_issues(
            issues, {field: value for field, value in result.items() if field != "issues" and value is not None}
        )
        result["duration_contained"] = contained
        return result
