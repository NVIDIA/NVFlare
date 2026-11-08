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

"""Applies one client's catalog job result to the server's stored state."""

import re
from datetime import datetime, timedelta
from enum import Enum

from nvflare.app_common.fdn.catalog.common.payload import ReasonCode, ResultStatus, Selection, parse_result
from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo, ReadLimits, ValidationError, validate
from nvflare.app_common.fdn.catalog.server.store import CatalogStore

OBSERVATION_SKEW = timedelta(minutes=5)
HEALTH_OK = "OK"
HEALTH_DEGRADED = "DEGRADED"
HEALTH_FAILED = "FAILED"


class Outcome(str, Enum):
    APPLIED = "APPLIED"
    UNSUPPORTED = "UNSUPPORTED"  # an unknown result format; the job's state is left untouched


_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


def _parse_time(value: str | None) -> datetime | None:
    """An RFC 3339 UTC time ending in Z, or None."""
    if not isinstance(value, str) or not _RFC3339_UTC.match(value):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def valid_observation(value: str | None, now: str) -> str | None:
    """The source observation if it is an RFC 3339 time that is not in the future, else None (unknown)."""
    observed, current = _parse_time(value), _parse_time(now)
    if observed is None or current is None or observed > current + OBSERVATION_SKEW:
        return None
    return value


def _stored_catalog(name: str, row: dict) -> CatalogInfo:
    return CatalogInfo.from_wire({"dataset": name, "metadata": row["metadata"], "observed_at": None})


class Reconciler:
    """Applies complete dataset updates atomically. The last applied valid update wins, regardless of job order.
    Datasets with no update in a result are unchanged."""

    def __init__(self, store: CatalogStore, read_limits: ReadLimits):
        self._store = store
        self._limits = read_limits

    def apply(
        self,
        study: str,
        client: str,
        selection: Selection,
        raw_result: dict,
        now: str,
        application_id: str | None = None,
        pending_applications: set[str] | None = None,
    ) -> Outcome:
        with self._store.lock:
            return self._apply(study, client, selection, raw_result, now, application_id, pending_applications)

    def _apply(self, study, client, selection, raw_result, now, application_id, pending_applications) -> Outcome:
        state = self._store.client_state(study, client)
        row = state["client"]
        parsed = parse_result(raw_result, selection)
        if parsed.unsupported:
            return Outcome.UNSUPPORTED
        datasets = state["datasets"]

        # 1. restrictions first; they need no check time and survive anything below
        for name in parsed.withheld:
            if name in datasets:
                datasets[name]["withheld"] = True

        # 2. each catalog independently
        rejected = set(parsed.rejected)
        accepted = 0
        for catalog in parsed.catalogs:
            try:
                validate(catalog, self._limits)
            except ValidationError:
                rejected.add(catalog.dataset)
                continue
            prev = datasets.get(catalog.dataset)
            digest = catalog.digest()
            changed = prev is None or prev["metadata_digest"] != digest
            catalog_version = prev["catalog_version"] if prev else 0
            schema_version = prev["schema_version"] if prev else 0
            published_at = prev["published_at"] if prev else None
            if changed:
                old = _stored_catalog(catalog.dataset, prev) if prev else None
                catalog_version += 1
                if old is None or (old.tables, old.resources) != (catalog.tables, catalog.resources):
                    schema_version += 1
                published_at = now
            datasets[catalog.dataset] = {
                "metadata": catalog.metadata_wire() if changed else prev["metadata"],
                "metadata_digest": digest,
                "catalog_version": catalog_version,
                "schema_version": schema_version,
                "observed_at": valid_observation(catalog.observed_at, now),
                "published_at": published_at,
                "available": True,
                "last_checked_at": now,
                "withheld": False,
            }
            accepted += 1

        # A rejected replacement cannot confirm that cached fields remain permitted by the current policy.
        # Hide that entry while preserving its metadata, versions, availability and last source-check time.
        for name in rejected:
            if name in datasets:
                datasets[name]["withheld"] = True

        for name in parsed.unavailable:
            if name in datasets:
                datasets[name].update(available=False, last_checked_at=now)

        # 3. health
        reason = parsed.reason_code.value if parsed.reason_code else None
        if rejected:
            reason = ReasonCode.INTERNAL.value
        row.update(last_attempt_at=now, reason_code=reason)
        if parsed.status == ResultStatus.OK and (not rejected or accepted or parsed.unavailable):
            row.update(last_success_at=now, failures=0, health=HEALTH_DEGRADED if reason else HEALTH_OK)
        else:
            row.update(health=HEALTH_FAILED, failures=row["failures"] + 1)
        self._store.commit_application(study, client, state, application_id, pending_applications)
        return Outcome.APPLIED
