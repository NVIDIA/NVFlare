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

"""Task and result payloads of the catalog.read task, carried under CATALOG_READ_KEY in a Shareable.

This module has no FLARE or I/O imports.
"""

import hashlib
import hmac
import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from nvflare.app_common.fdn.catalog.common.schema import MAX_TEXT_LEN, CatalogInfo, ValidationError

CATALOG_READ_KEY = "catalog_read"
TASK_NAME = "catalog.read"
MANIFEST_FILE = "catalog_results.json"  # in the job's run dir, written by the controller, read by the server
CONTEXT_FILE = (
    "catalog_context.json"  # in the job's run dir, written by the server before launch, read by the controller
)
CATALOG_JOB_META = "fdn_catalog_read"  # a job meta property: the server applies the result of a job that has it
FORMAT_VERSION = 1


class ResultStatus(str, Enum):
    OK = "OK"
    FAILED = "FAILED"


class ReasonCode(str, Enum):
    POLICY_UNAVAILABLE = "POLICY_UNAVAILABLE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    CANCELLED = "CANCELLED"
    INTERNAL = "INTERNAL"


class PayloadError(ValueError):
    """A task payload is malformed. The message names fields, never values."""


@dataclass(frozen=True)
class TaskPayload:
    datasets: tuple[str, ...] | None  # None selects all current bindings
    previous: tuple[str, ...]  # names the server already indexes in this selection


@dataclass(frozen=True)
class Selection:
    """The dataset names a catalog job filters on; each site reads those it has. None selects all."""

    datasets: tuple[str, ...] | None

    def contains(self, name: str) -> bool:
        return self.datasets is None or name in self.datasets


@dataclass(frozen=True)
class ParsedResult:
    status: ResultStatus = ResultStatus.FAILED
    reason_code: ReasonCode | None = None
    catalogs: tuple[CatalogInfo, ...] = ()
    withheld: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()  # named replacements that could not be parsed; never a wire field
    unsupported: bool = False


def _is_supported_version(value) -> bool:
    return type(value) is int and value == FORMAT_VERSION


def _is_name(value) -> bool:
    return isinstance(value, str) and 0 < len(value) <= MAX_TEXT_LEN


def _names(value, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(_is_name(x) for x in value):
        raise PayloadError(f"{path} must be a list of names")
    if len(set(value)) != len(value):
        raise PayloadError(f"{path} has duplicate entries")
    return tuple(value)


@dataclass(frozen=True)
class ManifestEntry:
    rc: str  # the framework's return code for the client's task; TIMEOUT when nothing arrived
    result: dict | None  # the client's catalog_read payload, or None
    datasets: tuple[str, ...] | None = None  # the names asked of this client; None asked for all it shares


def build_manifest(entries: dict[str, ManifestEntry]) -> dict:
    return {
        "results": {
            name: {"rc": e.rc, "result": e.result, "datasets": None if e.datasets is None else list(e.datasets)}
            for name, e in entries.items()
        }
    }


def job_key(master: bytes, job_id: str) -> bytes:
    """The key of one job, derived from the server's master key. Only this key leaves the server process."""
    return hmac.new(master, b"job\n" + job_id.encode(), hashlib.sha256).digest()


def manifest_mac(key: bytes, job_id: str, manifest: dict) -> str:
    """Binds a manifest to its job with a key only the server holds, so a job's own files cannot pose as the result."""
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(key, job_id.encode() + b"\n" + body, hashlib.sha256).hexdigest()


def parse_manifest(d: dict) -> dict[str, ManifestEntry]:
    if not isinstance(d, dict) or set(d) != {"results"} or not isinstance(d["results"], dict):
        raise PayloadError("unsupported manifest")
    entries = {}
    for name, e in d["results"].items():
        if (
            not _is_name(name)
            or not isinstance(e, dict)
            or set(e) != {"rc", "result", "datasets"}
            or not isinstance(e["rc"], str)
            or not (e["result"] is None or isinstance(e["result"], dict))
        ):
            raise PayloadError("manifest entry is invalid")
        datasets = None if e["datasets"] is None else _names(e["datasets"], "datasets")
        entries[name] = ManifestEntry(e["rc"], e["result"], datasets)
    return entries


def build_task_payload(datasets: list[str] | None, previous: list[str]) -> dict:
    payload = {"format_version": FORMAT_VERSION, "previous_datasets": list(previous)}
    if datasets is not None:
        payload["datasets"] = list(datasets)
    return payload


def parse_task_payload(d: dict) -> TaskPayload:
    if not isinstance(d, dict) or not _is_supported_version(d.get("format_version")):
        raise PayloadError("unsupported task payload")
    if set(d) - {"format_version", "datasets", "previous_datasets"}:
        raise PayloadError("task payload has unsupported fields")
    datasets = _names(d["datasets"], "datasets") if "datasets" in d else None
    if datasets is not None and not datasets:
        raise PayloadError("datasets must not be empty; omit it to select all")
    return TaskPayload(datasets=datasets, previous=_names(d.get("previous_datasets", []), "previous_datasets"))


def build_result(
    status: ResultStatus,
    catalogs=(),
    withheld=(),
    unavailable=(),
    reason_code: ReasonCode | None = None,
) -> dict:
    result = {
        "format_version": FORMAT_VERSION,
        "status": ResultStatus(status).value,
        "catalogs": [c.to_wire() for c in catalogs],
        "withheld": list(withheld),
        "unavailable": list(unavailable),
    }
    if reason_code is not None:
        result["reason_code"] = ReasonCode(reason_code).value
    return result


def _selected_names(value, selection: Selection) -> list[str]:
    """Valid, selected, unique names in order; anything else is dropped."""
    if not isinstance(value, list):
        return []
    out = []
    for name in value:
        if _is_name(name) and selection.contains(name) and name not in out:
            out.append(name)
    return out


def _enum_or_none(enum_cls, value):
    try:
        return enum_cls(value)
    except ValueError:
        return None


def parse_result(d: dict, selection: Selection) -> ParsedResult:
    """Parse one client's result. Tolerant per section, strict within a section.

    The restriction lists are extracted first and are returned even when catalogs are malformed. A name in
    several lists resolves to withheld, then unavailable, and its catalog entry is dropped; a dataset that
    appears in more than one catalog entry has every copy dropped, as does a malformed entry. Catalogs count
    only for an OK result.
    """
    if not isinstance(d, dict) or not _is_supported_version(d.get("format_version")):
        return ParsedResult(unsupported=True)

    status = _enum_or_none(ResultStatus, d.get("status")) or ResultStatus.FAILED
    withheld = _selected_names(d.get("withheld"), selection)
    unavailable = [n for n in _selected_names(d.get("unavailable"), selection) if n not in withheld]
    catalogs, rejected = [], []
    if status == ResultStatus.OK:
        entries = d.get("catalogs")
        entries = entries if isinstance(entries, list) else []
        names = [e.get("dataset") if isinstance(e, dict) else None for e in entries]
        counts = Counter(n for n in names if _is_name(n))
        for entry, name in zip(entries, names):
            if not (_is_name(name) and selection.contains(name)):
                continue
            if name in withheld or name in unavailable:
                continue
            if counts[name] > 1:
                if name not in rejected:
                    rejected.append(name)
                continue
            try:
                catalogs.append(CatalogInfo.from_wire(entry))
            except ValidationError:
                rejected.append(name)
    return ParsedResult(
        status=status,
        reason_code=_enum_or_none(ReasonCode, d.get("reason_code")),
        catalogs=tuple(catalogs),
        withheld=tuple(withheld),
        unavailable=tuple(unavailable),
        rejected=tuple(rejected),
    )
