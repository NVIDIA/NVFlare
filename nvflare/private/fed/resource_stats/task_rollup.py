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

from .contract import (
    ContractError,
    aggregate_cpu_consumed,
    aggregate_message_traffic,
    aggregate_resource_time,
    capacity_to_resource_time,
    normalize_duration,
    normalize_measurement,
)

MAX_SIDECAR_BYTES = 64 * 1024
_MAX_SCOPES = 10_000
_MAX_GROUPS = 128
_MAX_STATE_BYTES = 4 * 1024 * 1024
_MAX_PENDING_BYTES = 8 * 1024 * 1024
_UNAVAILABLE = {"status": "unavailable", "issues": ["observation_incomplete"]}
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
    if not isinstance(identity, TaskAttemptIdentity):
        raise TypeError("task resource identity must be a TaskAttemptIdentity")
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
            ).encode("ascii")
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
        self._totals = {"resource_time": None, "cpu_consumed": None, "message_traffic": None}
        self._incomplete = False
        self._frozen = False

    def _fits(self, entry_bytes, totals):
        return entry_bytes + len(_encoded(totals).encode("ascii")) + 1024 <= _MAX_STATE_BYTES

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
                return False
            scope = _Scope(identity)
            size = _scope_size(scope)
            if len(self._scopes) >= _MAX_SCOPES or not self._fits(self._state_bytes + size, self._totals):
                self._freeze()
                return False
            if self._pending_bytes + MAX_SIDECAR_BYTES > _MAX_PENDING_BYTES:
                self._incomplete = True
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
            if not isinstance(execution_id, str) or not execution_id or len(execution_id) > 4096:
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
    def _normalize(record, identity):
        fields = {"schema_version", "identity", "scope", "resource_capacity", "cpu_consumed", "message_traffic"}
        if not isinstance(record, dict) or not fields <= set(record) or set(record) - fields - {"duration_seconds"}:
            raise ValueError("resource sidecar has missing or unknown fields")
        if record["schema_version"] != "1.0" or record["scope"] != "worker":
            raise ValueError("unsupported execution sidecar")
        if TaskAttemptIdentity.from_dict(record["identity"]) != identity:
            raise ValueError("resource sidecar identity does not match its owned execution")
        normalized = {
            "schema_version": "1.0",
            "identity": identity.to_dict(),
            "scope": "worker",
        }
        for field in ("resource_capacity", "cpu_consumed", "message_traffic"):
            normalized[field] = normalize_measurement(field, record[field])
        if "duration_seconds" in record:
            normalized["duration_seconds"] = normalize_duration(record["duration_seconds"])
        return normalized

    @staticmethod
    def merge_resource_time(previous, contribution):
        """Keep independent dimension subtotals when another one exceeds bounds."""
        values = [value for value in (previous, contribution) if value is not None]
        result = {}
        incomplete = any("status" in value for value in values)
        issues = {
            issue if issue == "attribution_incomplete" else "observation_incomplete"
            for value in values
            for issue in value.get("issues", ())
        }
        for dimension in ("cpu", "memory", "gpu"):
            sources = [
                {"status": "partial", "issues": ["observation_incomplete"], dimension: value[dimension]}
                for value in values
                if dimension in value
            ]
            if not sources:
                continue
            try:
                combined = aggregate_resource_time(sources)[dimension]
                if dimension in ("cpu", "gpu") and len(combined["groups"]) > _MAX_GROUPS:
                    raise ValueError("task hardware group limit exceeded")
                result[dimension] = combined
            except (ContractError, ValueError):
                # An overflowing dimension cannot erase other usable subtotals.
                if previous is not None and dimension in previous:
                    result[dimension] = deepcopy(previous[dimension])
                incomplete = True
                issues.add("observation_incomplete")
        if not result:
            return dict(_UNAVAILABLE)
        if incomplete or set(result) != {"cpu", "memory", "gpu"}:
            result.update(status="partial", issues=sorted(issues or {"observation_incomplete"}))
        return normalize_measurement("resource_time", result)

    def _updated_totals(self, record):
        totals = deepcopy(self._totals)
        try:
            resource_time = capacity_to_resource_time(record["resource_capacity"], record.get("duration_seconds"))
        except (ContractError, ValueError):
            # Preserve a valid dimension if another capacity-time product overflows.
            resource_time = {}
            for dimension in ("cpu", "memory", "gpu"):
                if dimension not in record["resource_capacity"]:
                    continue
                capacity = {
                    "status": "partial",
                    "issues": ["observation_incomplete"],
                    dimension: record["resource_capacity"][dimension],
                }
                try:
                    partial = capacity_to_resource_time(capacity, record.get("duration_seconds"))
                    if dimension in partial:
                        resource_time[dimension] = partial[dimension]
                except (ContractError, ValueError):
                    pass
            resource_time.update(
                status="partial" if resource_time else "unavailable",
                issues=["observation_incomplete"],
            )
        totals["resource_time"] = self.merge_resource_time(totals["resource_time"], resource_time)
        for field, aggregate in (
            ("cpu_consumed", aggregate_cpu_consumed),
            ("message_traffic", aggregate_message_traffic),
        ):
            sources = [record[field]]
            if totals[field] is not None:
                sources.insert(0, totals[field])
            try:
                totals[field] = aggregate(sources)
            except (ContractError, ValueError):
                previous = totals[field]
                totals[field] = aggregate([previous], complete=False) if previous is not None else dict(_UNAVAILABLE)
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
                raw = store.read_resource_stats(identity)
                record = self._normalize(raw, identity) if raw is not None else None
            except Exception:
                invalid = True
        with self._lock:
            scope = self._scopes[key]
            candidate = replace(scope)
            if scope.disposition != "unresolved":
                if record is not None and scope.record != _encoded(record):
                    self._incomplete = True
            elif started is False:
                candidate.disposition = "never_started"
            elif record is None or self._frozen:
                candidate.disposition = "invalid" if invalid else "settled_missing"
                self._incomplete = True
            else:
                candidate.disposition = "reported"
                candidate.record = _encoded(record)
                totals = self._updated_totals(record)
                size = self._state_bytes - _scope_size(scope) + _scope_size(candidate)
                if self._fits(size, totals):
                    self._totals = totals
                else:
                    self._freeze()
                    candidate = replace(candidate, record=None, disposition="invalid")
            self._state_bytes += _scope_size(candidate) - _scope_size(scope)
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
            contained = all(scope.disposition != "unresolved" for scope in self._scopes.values())
            incomplete = self._incomplete or not contained
        if incomplete:
            for field, aggregate in (
                ("resource_time", aggregate_resource_time),
                ("cpu_consumed", aggregate_cpu_consumed),
                ("message_traffic", aggregate_message_traffic),
            ):
                value = result[field]
                result[field] = aggregate([value], complete=False) if value is not None else dict(_UNAVAILABLE)
        result["duration_contained"] = contained
        return result
