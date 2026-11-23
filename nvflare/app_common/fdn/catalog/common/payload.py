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

"""The catalog_read task payload, a site's result, and the assembled researcher snapshot.

Dataset identifiers follow the Study Config API naming rule.
"""

import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from nvflare.apis.dataset_config_spec import is_valid_dataset_name
from nvflare.app_common.fdn.catalog.common.limits import ReadLimits
from nvflare.app_common.fdn.catalog.common.odcs import check_contract

CATALOG_READ_KEY = "catalog_read"  # the Shareable key of the task payload and the site result
TASK_NAME = "catalog_read"
RESULTS_FILE = "catalog/index.json"  # in the job's server workspace, saved with the job in the job store
CATALOG_JOB_META = "fdn_catalog_read"  # a job meta property; a site executes the task only for a job that has it
FORMAT_VERSION = 2


class ResultStatus(str, Enum):
    OK = "OK"
    FAILED = "FAILED"


class SiteStatus(str, Enum):
    """A site's status in catalog_results.json."""

    OK = "OK"
    FAILED = "FAILED"
    NO_RESULT = "NO_RESULT"  # timeout, disconnect, abort, or a non-OK framework return code


class ReasonCode(str, Enum):
    CONFIG_UNAVAILABLE = "CONFIG_UNAVAILABLE"
    INTERNAL = "INTERNAL"


class PayloadError(ValueError):
    """A payload is malformed. The message names fields, never values."""


@dataclass(frozen=True)
class TaskPayload:
    datasets: tuple[str, ...] | None  # None selects every dataset the Study's configuration lists
    time_budget_s: float | None = None  # seconds the site may spend reading; ends before the task timeout


MAX_SITE_REPLY_BYTES = 16 * 1024 * 1024
MIN_TIME_BUDGET_S = 0.1


def json_size(value):
    """Count encoded bytes, stopping once the site ceiling has been exceeded."""
    size = 0
    for chunk in json.JSONEncoder(ensure_ascii=False).iterencode(value):
        size += len(chunk.encode("utf-8"))
        if size > MAX_SITE_REPLY_BYTES:
            break
    return size


def task_time_budget(task_timeout_s: float, elapsed_s: float = 0.0) -> float:
    """The reading time left to a site when it receives its task.

    The task timeout runs from when the server schedules the task, so time spent before the site picks it up is
    subtracted. A reserve of 20% of the timeout, between 1 and 30 seconds, is kept to return completed catalogs.
    """
    reserve = min(30.0, max(1.0, task_timeout_s * 0.2))
    return max(MIN_TIME_BUDGET_S, task_timeout_s - elapsed_s - reserve)


@dataclass(frozen=True)
class ParsedResult:
    status: ResultStatus = ResultStatus.FAILED
    reason_code: ReasonCode | None = None
    catalogs: tuple[dict, ...] = ()
    unavailable: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    unsupported: bool = False


def _is_supported_version(value) -> bool:
    return type(value) is int and value == FORMAT_VERSION


def _names(value, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(is_valid_dataset_name(x) for x in value):
        raise PayloadError(f"{path} must be a list of names")
    if len(set(value)) != len(value):
        raise PayloadError(f"{path} has duplicate entries")
    return tuple(value)


def build_task_payload(datasets: list[str] | None, time_budget_s: float | None = None) -> dict:
    payload = {"format_version": FORMAT_VERSION}
    if datasets is not None:
        payload["datasets"] = list(datasets)
    if time_budget_s is not None:
        payload["time_budget_s"] = time_budget_s
    return payload


def parse_task_payload(d: dict) -> TaskPayload:
    if not isinstance(d, dict) or not _is_supported_version(d.get("format_version")):
        raise PayloadError("unsupported task payload")
    if set(d) - {"format_version", "datasets", "time_budget_s"}:
        raise PayloadError("task payload has unsupported fields")
    datasets = _names(d["datasets"], "datasets") if "datasets" in d else None
    if datasets is not None and not datasets:
        raise PayloadError("datasets must not be empty; omit it to select all")
    budget = d.get("time_budget_s")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, float)) or not budget > 0):
        raise PayloadError("time_budget_s must be a positive number")
    return TaskPayload(datasets=datasets, time_budget_s=budget)


def build_result(
    status: ResultStatus, catalogs=(), unavailable=(), skipped=(), reason_code: ReasonCode | None = None
) -> dict:
    result = {
        "format_version": FORMAT_VERSION,
        "status": ResultStatus(status).value,
        "catalogs": list(catalogs),
        "unavailable": list(unavailable),
        "skipped": list(skipped),
    }
    if reason_code is not None:
        result["reason_code"] = ReasonCode(reason_code).value
    return result


def _selected_names(value, wanted: tuple[str, ...] | None) -> list[str]:
    """Valid, selected, unique names in order; anything else is dropped."""
    if not isinstance(value, list):
        return []
    out = []
    for name in value:
        if is_valid_dataset_name(name) and (wanted is None or name in wanted) and name not in out:
            out.append(name)
    return out


def _enum_or_none(enum_cls, value):
    try:
        return enum_cls(value)
    except ValueError:
        return None


def parse_result(d: dict, wanted: tuple[str, ...] | None, limits: ReadLimits) -> ParsedResult:
    """Parse and validate one site's result with the server's own limits.

    Names outside the site's selection are dropped. A catalog entry that fails the ODCS envelope or budget checks, or a
    dataset name that appears in more than one entry, is rejected and its dataset is reported unavailable, so the
    site's other entries still count. The three lists are made disjoint. Catalogs count only for an OK result.
    """
    if not isinstance(d, dict) or not _is_supported_version(d.get("format_version")):
        return ParsedResult(unsupported=True)
    entries = d.get("catalogs")
    entries = entries if isinstance(entries, list) else []
    try:
        # Measure entries separately so a non-JSON contract cannot discard a valid sibling.
        size = json_size({**d, "catalogs": []} if isinstance(d.get("catalogs"), list) else d) + 2 * max(
            0, len(entries) - 1
        )
        for entry in entries:
            try:
                size += json_size(entry)
            except (TypeError, ValueError, RecursionError):
                continue  # check_contract below rejects malformed contracts individually
            if size > MAX_SITE_REPLY_BYTES:
                break
        if size > MAX_SITE_REPLY_BYTES:
            return ParsedResult(reason_code=ReasonCode.INTERNAL)
    except (TypeError, ValueError, RecursionError):
        return ParsedResult(reason_code=ReasonCode.INTERNAL)
    status = _enum_or_none(ResultStatus, d.get("status")) or ResultStatus.FAILED
    unavailable = _selected_names(d.get("unavailable"), wanted)
    skipped = [n for n in _selected_names(d.get("skipped"), wanted) if n not in unavailable]
    catalogs = []
    if status == ResultStatus.OK:
        names = [e.get("dataset") if isinstance(e, dict) else None for e in entries]
        counts = Counter(n for n in names if is_valid_dataset_name(n))
        remaining = limits.max_records
        for entry, name in zip(entries, names):
            if not (is_valid_dataset_name(name) and (wanted is None or name in wanted)):
                continue
            if name in unavailable or name in skipped:
                continue
            try:
                if counts[name] > 1:
                    raise ValueError("dataset appears more than once")
                records = check_contract(entry["contract"])
                if records > remaining:
                    skipped.append(name)
                    continue
                remaining -= records
                catalogs.append({"dataset": name, "contract": entry["contract"]})
            except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
                if name not in unavailable:
                    unavailable.append(name)
    return ParsedResult(
        status=status,
        reason_code=_enum_or_none(ReasonCode, d.get("reason_code")),
        catalogs=tuple(catalogs),
        unavailable=tuple(unavailable),
        skipped=tuple(skipped),
    )


@dataclass(frozen=True)
class SiteEntry:
    status: SiteStatus
    reason_code: ReasonCode | None = None
    catalogs: tuple[dict, ...] = ()
    unavailable: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()


def build_catalog_results(study: str, job_id: str, completed_at: str, sites: dict[str, SiteEntry]) -> dict:
    """The catalog_results.json document: one entry per targeted site, and every accepted catalog with its site."""
    datasets = []
    for site in sorted(sites):
        for catalog in sites[site].catalogs:
            datasets.append({"site": site, **catalog})
    return {
        "format_version": FORMAT_VERSION,
        "study": study,
        "job_id": job_id,
        "completed_at": completed_at,
        "sites": {
            site: {
                "status": SiteStatus(e.status).value,
                "reason_code": None if e.reason_code is None else ReasonCode(e.reason_code).value,
                "unavailable": list(e.unavailable),
                "skipped": list(e.skipped),
            }
            for site, e in sorted(sites.items())
        },
        "datasets": datasets,
    }


def check_catalog_results(d) -> dict:
    """Light shape check of a loaded catalog_results.json; raises PayloadError."""
    if not isinstance(d, dict) or not _is_supported_version(d.get("format_version")):
        raise PayloadError("unsupported catalog results")
    if not isinstance(d.get("sites"), dict) or not isinstance(d.get("datasets"), list):
        raise PayloadError("catalog results need sites and datasets")
    return d
