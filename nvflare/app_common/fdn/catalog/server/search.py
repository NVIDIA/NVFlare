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

"""Search over the server's accepted catalog entries. Pure functions; the store supplies the entries."""

import base64
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo

MAX_PAGE_SIZE = 100


@dataclass(frozen=True)
class SearchQuery:
    client: str | None = None
    dataset: str | None = None
    text: str | None = None
    modality: str | None = None
    standard: str | None = None
    standard_version: str | None = None
    operation: str | None = None
    available: bool | None = None
    schema_version: int | None = None
    mapping_version: str | None = None
    cursor: str | None = None
    page_size: int = 50


@dataclass(frozen=True)
class CatalogEntry:
    """One search result. catalog is None while the dataset is unavailable."""

    client: str
    dataset: str
    available: bool
    last_checked_at: str | None
    catalog: CatalogInfo | None
    catalog_version: int | None
    schema_version: int | None
    published_at: str | None


@dataclass(frozen=True)
class SearchPage:
    entries: tuple[CatalogEntry, ...]
    next_cursor: str | None
    total: int
    facets: Mapping[str, Mapping[str, int]]


def entry_to_dict(entry: CatalogEntry) -> dict:
    """The JSON form of a search result. An unavailable entry carries no metadata."""
    out = {
        "client": entry.client,
        "dataset": entry.dataset,
        "available": entry.available,
        "last_checked_at": entry.last_checked_at,
    }
    if entry.catalog is not None:
        out.update(
            metadata=entry.catalog.metadata_wire(),
            observed_at=entry.catalog.observed_at,
            catalog_version=entry.catalog_version,
            schema_version=entry.schema_version,
            published_at=entry.published_at,
        )
    return out


def entry_from_dict(d: dict) -> CatalogEntry:
    """Rebuild a search result; the catalog part goes through CatalogInfo.from_wire."""
    catalog = None
    if "metadata" in d:
        catalog = CatalogInfo.from_wire(
            {"dataset": d["dataset"], "metadata": d["metadata"], "observed_at": d.get("observed_at")}
        )
    return CatalogEntry(
        client=d["client"],
        dataset=d["dataset"],
        available=d["available"],
        last_checked_at=d.get("last_checked_at"),
        catalog=catalog,
        catalog_version=d.get("catalog_version"),
        schema_version=d.get("schema_version"),
        published_at=d.get("published_at"),
    )


_METADATA_FILTERS = (
    "text",
    "modality",
    "standard",
    "standard_version",
    "operation",
    "schema_version",
    "mapping_version",
)


def _filter_hash(study: str, query: SearchQuery) -> str:
    filters = {k: v for k, v in asdict(query).items() if k not in ("cursor", "page_size")}
    return hashlib.sha256(json.dumps([study, filters], sort_keys=True).encode()).hexdigest()[:16]


def _decode_cursor(cursor: str, expected_hash: str) -> tuple[str, str]:
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        if data["h"] == expected_hash and len(data["k"]) == 2 and all(isinstance(x, str) for x in data["k"]):
            return data["k"][0], data["k"][1]
    except (ValueError, KeyError, TypeError):
        pass
    raise ValueError("invalid cursor")


def _encode_cursor(filter_hash: str, key: tuple[str, str]) -> str:
    return base64.urlsafe_b64encode(json.dumps({"h": filter_hash, "k": list(key)}).encode()).decode()


def _matches(entry: CatalogEntry, q: SearchQuery) -> bool:
    if q.client is not None and entry.client != q.client:
        return False
    if q.dataset is not None and entry.dataset != q.dataset:
        return False
    if q.available is not None and entry.available != q.available:
        return False
    if any(getattr(q, f) is not None for f in _METADATA_FILTERS):
        c = entry.catalog
        if c is None:  # unavailable entries expose no metadata, so metadata filters cannot match them
            return False
        if q.text is not None and q.text.casefold() not in c.display_name.casefold():
            return False
        if q.modality is not None and q.modality not in c.modalities:
            return False
        if q.operation is not None and q.operation not in c.operations:
            return False
        if q.standard is not None and c.semantic_profile.standard != q.standard:
            return False
        if q.standard_version is not None and c.semantic_profile.version != q.standard_version:
            return False
        if q.mapping_version is not None and c.semantic_profile.mapping_version != q.mapping_version:
            return False
        if q.schema_version is not None and entry.schema_version != q.schema_version:
            return False
    return True


def run_search(study: str, entries: Iterable[CatalogEntry], query: SearchQuery) -> SearchPage:
    """Filter, order by (client, dataset) and page. The caller has already removed withheld entries."""
    if not 1 <= query.page_size <= MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")
    filter_hash = _filter_hash(study, query)
    matched = sorted((e for e in entries if _matches(e, query)), key=lambda e: (e.client, e.dataset))
    facets = {"modality": Counter(), "operation": Counter(), "standard": Counter()}
    for e in matched:
        if e.catalog is not None:
            facets["modality"].update(e.catalog.modalities)
            facets["operation"].update(e.catalog.operations)
            if e.catalog.semantic_profile.standard:
                facets["standard"][e.catalog.semantic_profile.standard] += 1
    after = _decode_cursor(query.cursor, filter_hash) if query.cursor else None
    remaining = [e for e in matched if after is None or (e.client, e.dataset) > after]
    page = remaining[: query.page_size]
    more = len(remaining) > len(page)
    return SearchPage(
        entries=tuple(page),
        next_cursor=_encode_cursor(filter_hash, (page[-1].client, page[-1].dataset)) if more else None,
        total=len(matched),
        facets={k: dict(v) for k, v in facets.items()},
    )
